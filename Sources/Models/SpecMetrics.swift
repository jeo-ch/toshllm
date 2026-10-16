// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Speculative-decoding counters from the server's Prometheus endpoint. The chat
/// already shows per-answer acceptance; what only lives here is the split per draft
/// position, which says how deep the draft is still worth verifying.
struct SpecDecodeMetrics: Equatable, Sendable {
    var draftTokens = 0
    var acceptedTokens = 0
    var drafts = 0
    /// Accepted count per draft position, index 0 being the first drafted token.
    var acceptedPerPosition: [Int] = []

    var ran: Bool { drafts > 0 && draftTokens > 0 }

    var acceptance: Double? {
        guard draftTokens > 0 else { return nil }
        // Clamped: a counter that accepted more than it drafted would otherwise
        // render as 150%.
        return min(1, Double(acceptedTokens) / Double(draftTokens))
    }

    /// Mean tokens accepted per verification step, the figure that decides whether
    /// speculation pays for its extra pass.
    var meanAccepted: Double? {
        drafts > 0 ? Double(acceptedTokens) / Double(drafts) : nil
    }

    /// Fraction of drafts whose token at `position` was accepted.
    func acceptance(atPosition position: Int) -> Double? {
        guard drafts > 0, acceptedPerPosition.indices.contains(position) else { return nil }
        return Double(acceptedPerPosition[position]) / Double(drafts)
    }

    private static let prefix = "llamacpp:spec_decode_"

    /// Parses the Prometheus text exposition. Unknown lines, comments and label sets
    /// we don't recognise are skipped, so a new upstream counter cannot break this.
    static func parse(_ text: String) -> SpecDecodeMetrics {
        var out = SpecDecodeMetrics()
        var byPosition: [Int: Int] = [:]

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.hasPrefix("#"), line.hasPrefix(prefix) else { continue }
            guard let space = line.lastIndex(of: " ") else { continue }
            let name = line[line.startIndex..<space].trimmingCharacters(in: .whitespaces)
            // Prometheus writes any float here — "1.0", "1e+03", "+Inf", "NaN".
            // Int() rejected all of those, so a single counter emitted as 337.0
            // read as absent and the panel hid itself as "speculation was off".
            let parsed = Double(line[line.index(after: space)...].trimmingCharacters(in: .whitespaces))
            guard let parsed, parsed.isFinite, parsed >= 0, parsed < Double(Int32.max) else { continue }
            let value = Int(parsed)

            switch name {
            case prefix + "num_draft_tokens_total":    out.draftTokens = value
            case prefix + "num_accepted_tokens_total": out.acceptedTokens = value
            case prefix + "num_drafts_total":          out.drafts = value
            default:
                guard name.hasPrefix(prefix + "num_accepted_tokens_per_pos_total{"),
                      let position = positionLabel(in: name) else { continue }
                byPosition[position] = value
            }
        }

        if let highest = byPosition.keys.max(), highest >= 0, highest < Self.maxDraftPositions {
            // The guard matters: 0...highest traps when highest is negative, and
            // the label is a number off the wire with nothing constraining it.
            out.acceptedPerPosition = (0...highest).map { byPosition[$0] ?? 0 }
        }
        return out
    }

    /// Ceiling on the position label, so a malformed metric cannot ask for a very
    /// large array. Well past any real draft depth.
    private static let maxDraftPositions = 1024

    private static func positionLabel(in name: String) -> Int? {
        guard let open = name.firstIndex(of: "{"), let close = name.lastIndex(of: "}") else { return nil }
        let labels = name[name.index(after: open)..<close]
        for label in labels.split(separator: ",") {
            let parts = label.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, parts[0].trimmingCharacters(in: .whitespaces) == "position" else { continue }
            return Int(parts[1].trimmingCharacters(in: CharacterSet(charactersIn: "\" ")))
        }
        return nil
    }

    /// Reads the endpoint of the local server. Returns nil when it is unreachable or
    /// was started without `--metrics`, so callers can just hide the readout.
    static func fetch(port: Int, timeout: TimeInterval = 2) async -> SpecDecodeMetrics? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/metrics") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        let data: Data?
        let response: URLResponse?
        do {
            let result = try await NetworkManager.session.data(for: request)
            data = result.0
            response = result.1
        } catch {
            AppLog.models.error("failed to fetch spec decode metrics: \(error.localizedDescription)")
            return nil
        }
        guard let data, let response,
              (response as? HTTPURLResponse)?.statusCode == 200,
              let text = String(data: data, encoding: .utf8) else { return nil }
        let parsed = parse(text)
        return parsed.ran ? parsed : nil
    }
}
