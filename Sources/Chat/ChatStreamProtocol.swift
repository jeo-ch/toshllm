// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct ChatStreamEvent {
    var receivedContent = false
    var receivedToolCall = false
    var progress: Double?
    var completed = false
}

struct ChatStreamAccumulator {
    var reasoning = ""
    var visible = ""
    /// Length of `visible` as it grows. The flush cadence reads it on every
    /// event and `visible.count` walks the whole answer each time.
    private(set) var visibleCount = 0
    var usage: (prompt: Int, completion: Int)?
    var timings: ChatTimings?
    var mtpAccept: Double?
    var finishReason: String?
    private(set) var toolCalls: [ChatToolCall] = []

    /// Ceiling on parallel tool calls in one turn. Well above anything the agent
    /// loop issues, and low enough that a malformed index cannot allocate.
    private static let maxParallelToolCalls = 64

    mutating func consume(_ line: String) throws -> ChatStreamEvent? {
        guard line.hasPrefix("data: ") else { return nil }
        let payload = line.dropFirst(6)
        if payload == "[DONE]" { return ChatStreamEvent(completed: true) }
        // Bytes straight off the substring: materialising a String first and then
        // encoding it copied every line of the answer twice, once per token.
        guard let object = try? JSONSerialization.jsonObject(
            with: Data(payload.utf8)) as? [String: Any]
        else { return nil }

        if let message = ChatStore.streamedError(from: object) {
            throw StreamError(message: message)
        }

        if let value = object["usage"] as? [String: Any],
           let prompt = (value["prompt_tokens"] as? NSNumber)?.intValue,
           let completion = (value["completion_tokens"] as? NSNumber)?.intValue {
            usage = (prompt, completion)
        }

        if let value = object["timings"] as? [String: Any] {
            if let parsed = ChatTimings(json: value) { timings = parsed }
            if let drafted = (value["draft_n"] as? NSNumber)?.intValue, drafted > 0,
               let accepted = (value["draft_n_accepted"] as? NSNumber)?.intValue {
                mtpAccept = Double(accepted) / Double(drafted)
            }
        }

        var event = ChatStreamEvent()
        if let value = object["prompt_progress"] as? [String: Any],
           let processed = (value["processed"] as? NSNumber)?.intValue,
           let total = (value["total"] as? NSNumber)?.intValue, total > 0 {
            event.progress = min(1, Double(processed) / Double(total))
        }

        guard let choice = (object["choices"] as? [[String: Any]])?.first else { return event }
        if let reason = choice["finish_reason"] as? String, !reason.isEmpty {
            finishReason = reason
        }
        if let delta = choice["delta"] as? [String: Any] {
            if let text = delta["reasoning_content"] as? String, !text.isEmpty {
                reasoning += text
                event.receivedContent = true
            }
            if let text = delta["content"] as? String, !text.isEmpty {
                visible += text
                visibleCount += text.count
                event.receivedContent = true
            }
            if let fragments = delta["tool_calls"] as? [[String: Any]] {
                mergeToolCalls(fragments)
                event.receivedToolCall = !fragments.isEmpty
            }
        }
        return event
    }

    private mutating func mergeToolCalls(_ fragments: [[String: Any]]) {
        for fragment in fragments {
            let index = (fragment["index"] as? NSNumber)?.intValue ?? toolCalls.count
            // Bounded on both ends. A negative index skipped the growth loop and
            // then indexed out of range; a huge one asked for that many empty
            // tool calls. The endpoint is unauthenticated unless the user turned
            // the API key on, so the field is not necessarily well-behaved.
            guard index >= 0, index < Self.maxParallelToolCalls else { continue }
            while toolCalls.count <= index {
                toolCalls.append(ChatToolCall(name: "", arguments: ""))
            }
            if let id = fragment["id"] as? String, !id.isEmpty { toolCalls[index].serverID = id }
            guard let function = fragment["function"] as? [String: Any] else { continue }
            if let name = function["name"] as? String { toolCalls[index].name += name }
            if let arguments = function["arguments"] as? String { toolCalls[index].arguments += arguments }
        }
    }
}

/// Splits a byte stream into lines while counting exactly what was consumed.
///
/// `URLSession.AsyncBytes.lines` strips the terminator, so a resume offset built
/// from it assumes every line ended in one byte. Under CRLF — anything that puts
/// a proxy or an HTTP filter in front — the real cost is two, and the count
/// drifts low by one per line. The engine then resumed from before what it had
/// already sent and replayed it, so the answer contained a duplicated passage
/// with nothing in the UI to say why. Counting the terminator we actually saw
/// keeps the offset a position in the stream.
struct ByteLineSplitter {
    private var pending: [UInt8] = []

    struct Line {
        let text: String
        /// Bytes consumed, terminator included.
        let bytes: Int
    }

    /// Appends a chunk and returns whatever complete lines it finished.
    mutating func push(_ chunk: some Sequence<UInt8>) -> [Line] {
        pending.append(contentsOf: chunk)
        var lines: [Line] = []
        var start = 0
        var index = 0
        while index < pending.count {
            guard pending[index] == 0x0A else { index += 1; continue }
            var end = index
            if end > start, pending[end - 1] == 0x0D { end -= 1 }   // CRLF
            let text = String(decoding: pending[start..<end], as: UTF8.self)
            lines.append(Line(text: text, bytes: index + 1 - start))
            start = index + 1
            index += 1
        }
        if start > 0 { pending.removeFirst(start) }
        return lines
    }

    /// What is left once the stream ended without a final newline.
    mutating func flush() -> [Line] {
        guard !pending.isEmpty else { return [] }
        let text = String(decoding: pending, as: UTF8.self)
        let bytes = pending.count
        pending.removeAll()
        return [Line(text: text, bytes: bytes)]
    }
}

enum ChatStreamIdentity {
    static func value(conversationID: UUID, model: String?) -> String {
        guard let model, !model.isEmpty else { return conversationID.uuidString }
        return "\(conversationID.uuidString)::\(model)"
    }

    /// The engine keys a session by the conv_id query parameter, not by a path segment.
    static func resumeURL(port: Int, identity: String, from offset: Int) -> URL? {
        // A negative offset would ask the engine to rewind past the start, which
        // it answers by replaying the whole answer rather than by failing.
        guard offset >= 0 else { return nil }
        var comps = URLComponents(string: "http://127.0.0.1:\(port)/v1/stream")
        comps?.percentEncodedQueryItems = [URLQueryItem(name: "conv_id", value: encode(identity)),
                                           URLQueryItem(name: "from", value: String(offset))]
        return comps?.url
    }

    /// DELETE here cancels the generation. Closing the connection does not: the engine
    /// keeps a keyed stream running so a dropped client can pick it up again.
    static func stopURL(port: Int, identity: String) -> URL? {
        var comps = URLComponents(string: "http://127.0.0.1:\(port)/v1/stream")
        comps?.percentEncodedQueryItems = [URLQueryItem(name: "conv_id", value: encode(identity))]
        return comps?.url
    }

    private static func encode(_ identity: String) -> String {
        var allowed = CharacterSet.urlQueryAllowed
        allowed.remove(charactersIn: "/&=+?#")
        return identity.addingPercentEncoding(withAllowedCharacters: allowed) ?? identity
    }
}
