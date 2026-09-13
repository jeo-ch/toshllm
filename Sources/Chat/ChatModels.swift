// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

// MARK: - Chat data models

/// A text file attached to a user message: sent to the model as a fenced
/// block, rendered in the transcript as a compact chip.
struct ChatAttachment: Identifiable, Codable, Equatable {
    var id = UUID()
    var name: String
    var content: String
    var mimeType: String? = nil
    var dataURI: String? = nil
    var byteCount: Int? = nil
    var durationSeconds: Double? = nil
    var videoFrameArea: Int? = nil

    // mtmd samples video at ~4 fps with no cap; Qwen-VL bills each frame by its
    // pixels (≈1 token / 32x32) up to a per-frame cap.
    private static let videoFPS = 4.0
    private static let videoPixelsPerToken = 1024
    private static let videoMaxTokensPerFrame = 1120

    /// Rough token estimate for context budgeting in the UI: chars/4 for text,
    /// duration-and-resolution based for video (its text content is empty).
    var estimatedTokens: Int {
        if mediaKind == "video", let seconds = durationSeconds, seconds > 0 {
            let area = videoFrameArea ?? Self.videoMaxTokensPerFrame * Self.videoPixelsPerToken
            let perFrame = min(area / Self.videoPixelsPerToken, Self.videoMaxTokensPerFrame)
            return max(1, Int(seconds * Self.videoFPS) * perFrame)
        }
        return max(1, content.count / 4)
    }

    var fenceHint: String { (name as NSString).pathExtension.lowercased() }

    var mediaKind: String? {
        guard let mimeType else { return nil }
        if mimeType.hasPrefix("audio/") { return "audio" }
        if mimeType.hasPrefix("video/") { return "video" }
        return nil
    }

    var base64Payload: String? {
        guard let dataURI, let comma = dataURI.firstIndex(of: ",") else { return nil }
        return String(dataURI[dataURI.index(after: comma)...])
    }

    var audioInputFormat: String {
        let normalized = (mimeType ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let waveTypes: Set<String> = [
            "audio/wav", "audio/wave", "audio/x-wav", "audio/x-wave",
            "audio/vnd.wave", "audio/x-pn-wav",
        ]
        return waveTypes.contains(normalized) ? "wav" : "mp3"
    }

    var videoInputFormat: String {
        let normalized = (mimeType ?? "").lowercased()
        if normalized.contains("mp4") { return "mp4" }
        if normalized.contains("ogg") { return "ogg" }
        return "auto"
    }
}

struct ChatMessage: Identifiable, Codable, Equatable {
    var id = UUID()
    let role: String          // user | assistant
    var content: String
    var date = Date()
    var genSpeed: Double?     // t/s for this response
    var mtpAccept: Double?    // MTP acceptance 0-1, when speculation ran
    var timings: ChatTimings? = nil
    var model: String? = nil
    var rawOutput: String? = nil
    var toolCalls: [ChatToolCall]? = nil
    var toolCallID: String? = nil
    // Optional keeps pre-attachment JSON decodable.
    var attachments: [ChatAttachment]? = nil
    // Attached images as data URIs (data:image/jpeg;base64,…) for vision models.
    var imageURIs: [String]? = nil

    var estimatedTokens: Int {
        let text = role == "assistant" ? parts.body : wireContent
        let attached = (attachments ?? []).reduce(0) { $0 + $1.estimatedTokens }
        return max(1, text.count / 4) + attached
    }

    var wireContent: String {
        guard let attachments, !attachments.isEmpty else { return content }
        let blocks = attachments.filter { $0.mediaKind == nil }.map { a in
            "File: \(a.name)\n```\(a.fenceHint)\n\(a.content)\n```"
        }
        return ((blocks.isEmpty ? "" : blocks.joined(separator: "\n\n") + "\n\n") + content)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Splits the <think>…</think> block from the visible content.
    var parts: (thinking: String?, body: String) {
        guard role == "assistant", content.hasPrefix("<think>") else { return (nil, content) }
        if let end = content.range(of: "</think>") {
            let think = String(content[content.index(content.startIndex, offsetBy: 7)..<end.lowerBound])
            let body = String(content[end.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            return (think.trimmingCharacters(in: .whitespacesAndNewlines), body)
        }
        return (String(content.dropFirst(7)).trimmingCharacters(in: .whitespacesAndNewlines), "")
    }
}

struct Conversation: Identifiable, Codable {
    var id = UUID()
    var title: String
    var messages: [ChatMessage] = []
    var created = Date()
    var updated = Date()
    var summary: String?
    var summarizedCount: Int?
    /// Pinned conversations sort first. Optional for backward compatibility
    /// with conversations.json saved by older builds.
    var pinned: Bool? = nil
    /// Project this conversation belongs to; nil = ungrouped. Optionals keep
    /// pre-projects JSON decodable, and older builds ignore the extra keys.
    var projectID: UUID? = nil
    /// Per-conversation system prompt. Empty/nil falls back to the project's,
    /// then to the global one.
    var systemPrompt: String? = nil
    var draft: ChatDraft? = nil
    var branches: [ChatBranch]? = nil
    var activeBranchID: UUID? = nil
    var enabledToolNames: [String]? = nil
    /// Working directory the server tools run in. Empty/nil falls back to the project's.
    var workingDirectory: String? = nil
    /// Ranges the model set aside with memory_archive: still in `messages` and on
    /// disk, just not sent with each request. Optional so older JSON still decodes.
    var archived: [ArchivedBlock]? = nil
    /// Last exchange's context tokens, per chat so the bar follows the open
    /// conversation and persists with the KV cache. Optional for old JSON.
    var contextUsed: Int? = nil
}

struct ChatBranch: Identifiable, Codable, Equatable {
    var id = UUID()
    var name: String
    var messages: [ChatMessage]
    var created = Date()
}

extension Conversation {
    mutating func beginAlternativeBranch(messages newMessages: [ChatMessage]) {
        var values = branches ?? []
        if values.isEmpty {
            let original = ChatBranch(name: "Branch 1", messages: messages)
            values.append(original)
            activeBranchID = original.id
        } else if let activeBranchID,
                  let index = values.firstIndex(where: { $0.id == activeBranchID }) {
            values[index].messages = messages
        }
        let branch = ChatBranch(name: "Branch \(values.count + 1)", messages: newMessages)
        values.append(branch)
        branches = values
        activeBranchID = branch.id
        messages = newMessages
    }

    mutating func activateBranch(_ id: UUID) -> Bool {
        guard var values = branches,
              let target = values.firstIndex(where: { $0.id == id }) else { return false }
        if let activeBranchID,
           let current = values.firstIndex(where: { $0.id == activeBranchID }) {
            values[current].messages = messages
        }
        branches = values
        activeBranchID = id
        messages = values[target].messages
        return true
    }
}

/// Folder in the chat sidebar grouping conversations, with its own system
/// prompt inherited by every chat inside.
struct ChatProject: Identifiable, Codable, Equatable {
    var id = UUID()
    var name: String
    var systemPrompt: String = ""
    var workingDirectory: String? = nil
    var pinned: Bool? = nil
    /// Sidebar disclosure state, persisted so folders keep their fold.
    var collapsed: Bool? = nil
    var created = Date()
}

struct StreamError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// The live answer as it stood at one instant. Rendering this instead of the
/// live properties keeps the bubble's layout still while generation continues.
struct StreamSnapshot: Equatable {
    let visible: String
    let reasoning: String
    let reasoningTail: String
}
