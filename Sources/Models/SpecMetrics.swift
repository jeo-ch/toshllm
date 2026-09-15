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
        draftTokens > 0 ? Double(acceptedTokens) / Double(draftTokens) : nil
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
            guard let value = Int(line[line.index(after: space)...].trimmingCharacters(in: .whitespaces))
            else { continue }

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

        if let highest = byPosition.keys.max() {
            out.acceptedPerPosition = (0...highest).map { byPosition[$0] ?? 0 }
        }
        return out
    }

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

// MARK: - Four-Dimensional Model Scoring

/// Four-dimensional scoring system for evaluating model quality on specific hardware.
/// Inspired by llmfit's scoring model with dimensions: Fit, Speed, Quality, Context.
struct ModelScore: Equatable, Sendable {
    /// Fit: How well the model fits in VRAM (0-100).
    /// 100 = fully on GPU, 0 = doesn't fit at all.
    let fit: Double
    
    /// Speed: Estimated tokens per second (0-100).
    /// 100 = fastest possible, 0 = unusably slow.
    let speed: Double
    
    /// Quality: Quantization quality loss (0-100).
    /// 100 = no quality loss, 0 = severe quality loss.
    let quality: Double
    
    /// Context: Effective context length (0-100).
    /// 100 = full context available, 0 = no context available.
    let context: Double
    
    /// Overall score (weighted average).
    var overall: Double {
        fit * 0.35 + speed * 0.30 + quality * 0.20 + context * 0.15
    }
    
    /// Human-readable grade based on overall score.
    var grade: String {
        switch overall {
        case 90...100: return "Excellent"
        case 75..<90: return "Good"
        case 60..<75: return "Fair"
        case 40..<60: return "Poor"
        default: return "Unusable"
        }
    }
    
    /// Color name for UI display.
    var gradeColor: String {
        switch overall {
        case 90...100: return "green"
        case 75..<90: return "blue"
        case 60..<75: return "yellow"
        case 40..<60: return "orange"
        default: return "red"
        }
    }
    
    /// Create a score from a model spec and hardware info.
    static func score(
        spec: ModelSpec,
        hw: HardwareInfo,
        ctx: Int = 16384,
        kvScale: Double = 1.0
    ) -> ModelScore {
        let estimate = Estimator.estimate(spec: spec, hw: hw, ctx: ctx, kvScale: kvScale)
        
        // Fit score: based on FitLevel
        let fitScore: Double
        switch estimate.level {
        case .ideal: fitScore = 100
        case .good: fitScore = 75
        case .slow: fitScore = 40
        case .no: fitScore = 0
        }
        
        // Speed score: based on expected tokens/second
        let speedScore: Double
        let speedStr = estimate.expectedSpeed.replacingOccurrences(of: " t/s", with: "")
        let components = speedStr.components(separatedBy: "-")
        if components.count == 2,
           let minSpeed = Double(components[0]),
           let maxSpeed = Double(components[1]) {
            let avgSpeed = (minSpeed + maxSpeed) / 2
            // Normalize: 100 t/s = 100 score, 0 t/s = 0 score
            speedScore = min(100, avgSpeed)
        } else {
            speedScore = 0
        }
        
        // Quality score: based on quantization tier
        let qualityScore: Double
        let fileName = spec.fileGB > 0 ? "model" : ""  // Placeholder for actual file name
        if fileName.contains("Q8_0") || fileName.contains("F16") {
            qualityScore = 100
        } else if fileName.contains("Q6_K") {
            qualityScore = 92
        } else if fileName.contains("Q5_K_M") {
            qualityScore = 85
        } else if fileName.contains("Q4_K_M") {
            qualityScore = 75
        } else if fileName.contains("Q4_0") {
            qualityScore = 60
        } else {
            qualityScore = 70  // Default for unknown quantization
        }
        
        // Context score: based on available context vs requested
        let contextScore: Double
        let maxContext = Double(ctx)
        let usedContext = estimate.vramGB > 0 ? maxContext * 0.8 : maxContext * 0.5  // Estimate context usage
        contextScore = min(100, (usedContext / maxContext) * 100)
        
        return ModelScore(
            fit: fitScore,
            speed: speedScore,
            quality: qualityScore,
            context: contextScore
        )
    }
    
    /// Create a score from explicit values (for testing or custom calculations).
    static func create(fit: Double, speed: Double, quality: Double, context: Double) -> ModelScore {
        ModelScore(fit: fit, speed: speed, quality: quality, context: context)
    }
}

// MARK: - Model Score Cache

/// Caches model scores to avoid recomputation.
@MainActor
final class ModelScoreCache: ObservableObject {
    static let shared = ModelScoreCache()
    
    @Published var scores: [String: ModelScore] = [:]
    
    private init() {}
    
    /// Get or compute a score for a model.
    func score(
        for modelPath: String,
        spec: ModelSpec,
        hw: HardwareInfo,
        ctx: Int = 16384,
        kvScale: Double = 1.0
    ) -> ModelScore {
        if let cached = scores[modelPath] {
            return cached
        }
        
        let score = ModelScore.score(spec: spec, hw: hw, ctx: ctx, kvScale: kvScale)
        scores[modelPath] = score
        return score
    }
    
    /// Clear the cache (e.g., when hardware changes).
    func clear() {
        scores.removeAll()
    }
}
