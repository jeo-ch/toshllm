// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

struct ModelModalities: Codable, Equatable, Sendable {
    var vision: Bool
    var audio: Bool
    var video: Bool
    var thinking: Bool?
    var reasoning: ReasoningEffortSupport?

    init(vision: Bool, audio: Bool, video: Bool, thinking: Bool? = nil,
         reasoning: ReasoningEffortSupport? = nil) {
        self.vision = vision
        self.audio = audio
        self.video = video
        self.thinking = thinking
        self.reasoning = reasoning
    }

    static let textOnly = ModelModalities(vision: false, audio: false, video: false)
}

enum ModelCapabilitiesService {
    struct Props: Decodable {
        let modalities: ModelModalities?
        let chatTemplate: String?
        /// The context the engine is actually running with. Two places carry it:
        /// the default generation settings, and the top level. Read both, in that
        /// order, because which one is present depends on the build.
        let nCtx: Int?
        let defaultGenerationSettings: [String: Int]?

        enum CodingKeys: String, CodingKey {
            case modalities
            case chatTemplate = "chat_template"
            case nCtx = "n_ctx"
            case defaultGenerationSettings = "default_generation_settings"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            modalities = try container.decodeIfPresent(ModelModalities.self, forKey: .modalities)
            chatTemplate = try container.decodeIfPresent(String.self, forKey: .chatTemplate)
            // Numbers come through as whatever JSONSerialization or the decoder
            // chose; a context of 16384.0 is not a decoding failure.
            nCtx = Self.number(try container.decodeIfPresent(Double.self, forKey: .nCtx))
            // Decoded one value at a time, and with a typed attempt that is
            // allowed to fail per key. As `[String: Double]` the whole dictionary
            // failed if any single value was not a number, and because this is
            // `Props.init`, that took the context *and* the modalities with it —
            // one string field the engine might add to its own settings object
            // would leave the app unable to read either.
            var settings: [String: Int] = [:]
            if let raw = try? container.nestedContainer(
                keyedBy: DynamicKey.self, forKey: .defaultGenerationSettings) {
                for key in raw.allKeys {
                    if let value = try? raw.decode(Double.self, forKey: key),
                       let value = Self.number(value) {
                        settings[key.stringValue] = value
                    }
                }
            }
            defaultGenerationSettings = settings.isEmpty ? nil : settings
        }

        /// Coding keys are not known ahead of time for a free-form object, so the
        /// dictionary's own keys are reconstructed from their decoded form.
        private struct DynamicKey: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }

        private static func number(_ value: Double?) -> Int? {
            guard let value, value.isFinite, value >= 0, value < Double(Int32.max) else { return nil }
            return Int(value)
        }

        /// The context in force, or nil when the engine did not say.
        var contextTokens: Int? {
            defaultGenerationSettings?["n_ctx"] ?? nCtx
        }
    }

    /// One read of `/props`, shared by everything that needs it.
    ///
    /// The context recall card used to build its own request and read a different
    /// set of fields from the same endpoint, so the two could disagree — and the
    /// card's version did not pass the model, which under the router meant it
    /// described whichever model happened to be resident.
    nonisolated static func fetchProps(port: Int, model: String?) async -> Props? {
        guard let baseURL = URL(string: "http://127.0.0.1:\(port)/props") else { return nil }
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        if let model, !model.isEmpty {
            components?.queryItems = [
                URLQueryItem(name: "model", value: model),
                URLQueryItem(name: "autoload", value: "false"),
            ]
        }
        guard let url = components?.url else { return nil }
        var request = URLRequest(url: url, timeoutInterval: 15)
        if let key = ServerSettings.activeAPIKey() {
            request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
        }
        guard let (data, response) = try? await NetworkManager.session.data(for: request) else {
            return nil
        }
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            return nil
        }
        return try? JSONDecoder().decode(Props.self, from: data)
    }

    static func fetch(port: Int, model: String?) async throws -> ModelModalities? {
        guard let props = await fetchProps(port: port, model: model) else { return nil }
        return modalities(from: props)
    }

    /// Split out so a caller that needs both the context and the modalities reads
    /// `/props` once: they come from the same response, and asking twice meant two
    /// round trips to the engine on every model change.
    nonisolated static func modalities(from props: Props) -> ModelModalities? {
        var capabilities = props.modalities ?? .textOnly
        capabilities.thinking = props.chatTemplate.map(ThinkingSupportDetector.supportsThinking)
        capabilities.reasoning = props.chatTemplate.map(ReasoningEffortDetector.detect)
        capabilities.video = capabilities.video && VideoRuntimeAvailability.isAvailable
        return capabilities
    }
}

enum ThinkingSupportDetector {
    static func supportsThinking(_ template: String) -> Bool {
        guard !template.isEmpty else { return false }
        let lowered = template.lowercased()
        for variable in ["enable_thinking", "reasoning_effort", "thinking_budget"]
        where lowered.contains(variable) {
            return true
        }
        for pair in [("<think>", "</think>"),
                     ("<|think|>", "</|think|>"),
                     ("<seed:think|>", "</seed:think|>")]
        where lowered.contains(pair.0) && lowered.contains(pair.1) {
            return true
        }
        return lowered.contains("<|channel>thought") || lowered.contains("<think></think>")
    }
}

/// What a chat template accepts in `reasoning_effort`. Templates that validate
/// the value raise a Jinja exception on anything else, which reaches the app as
/// an HTTP 500.
struct ReasoningEffortSupport: Codable, Equatable, Sendable {
    /// Empty when the template never validates the value: anything is accepted.
    var levels: [String] = []
    var modelDefault: String?
}

enum ReasoningEffortDetector {
    // Only a membership test guarding a raise_exception is the accepted set:
    // templates also branch on the value to word the prompt, with partial lists.
    private static let guardPattern = try! NSRegularExpression(
        pattern: "[\\w.]*reasoning_effort\\s+not\\s+in\\s*[\\[(]([^\\])]*)[\\])]")
    private static let assignmentPattern = try! NSRegularExpression(
        pattern: "set\\s+[\\w.]*reasoning_effort\\s*=([^\\n]*)")
    // Some templates only name the default in the error they raise.
    private static let defaultMarkerPattern = try! NSRegularExpression(
        pattern: "([A-Za-z_][\\w-]*)\\s*\\(default\\)")
    private static let literalPattern = try! NSRegularExpression(pattern: "['\"]([^'\"]+)['\"]")

    /// Ladder of the levels llama.cpp names in --reasoning-effort, so the picker
    /// can order them and fall back to a lower one the model does accept.
    private static let ranks = ["none": 0, "no_think": 0, "minimal": 1, "low": 2,
                                "medium": 3, "high": 4, "xhigh": 5, "max": 6]

    static func rank(of level: String) -> Int? { ranks[level.lowercased()] }

    /// The level to use when the model rejects `effort`: the closest one it
    /// accepts at or below it, so its token budget is not lifted along the way.
    static func closest(to effort: String, in levels: [String]) -> String? {
        if levels.contains(effort) { return effort }
        guard let wanted = rank(of: effort) else { return nil }
        let ranked = levels.compactMap { level in rank(of: level).map { (level: level, rank: $0) } }
        let below = ranked.filter { $0.rank <= wanted }.max { $0.rank < $1.rank }
        return (below ?? ranked.min { $0.rank < $1.rank })?.level
    }

    static func detect(_ template: String) -> ReasoningEffortSupport {
        guard template.contains("reasoning_effort") else { return ReasoningEffortSupport() }
        let levels = validatedLevels(template)
        var fallback = defaultLevel(template)
        if let value = fallback, !levels.isEmpty, !levels.contains(value) { fallback = nil }
        return ReasoningEffortSupport(levels: levels, modelDefault: fallback)
    }

    private static func validatedLevels(_ template: String) -> [String] {
        let text = template as NSString
        let whole = NSRange(location: 0, length: text.length)
        for match in guardPattern.matches(in: template, range: whole) {
            let tail = NSRange(location: match.range.upperBound,
                               length: min(500, text.length - match.range.upperBound))
            guard text.substring(with: tail).contains("raise_exception") else { continue }
            let levels = literals(in: text.substring(with: match.range(at: 1)))
            if !levels.isEmpty { return ordered(levels) }
        }
        return []
    }

    private static func defaultLevel(_ template: String) -> String? {
        let text = template as NSString
        let whole = NSRange(location: 0, length: text.length)
        for match in assignmentPattern.matches(in: template, range: whole) {
            var statement = text.substring(with: match.range(at: 1))
            if let end = statement.range(of: "%}") { statement = String(statement[..<end.lowerBound]) }
            let before = NSRange(location: max(0, match.range.location - 160),
                                 length: min(160, match.range.location))
            // Only an assignment filling in a missing value is the default.
            guard text.substring(with: before).contains("defined")
                    || statement.contains("defined") || statement.contains("default")
            else { continue }
            if let literal = literals(in: statement).first { return literal }
        }
        if let match = defaultMarkerPattern.firstMatch(in: template, range: whole) {
            return text.substring(with: match.range(at: 1))
        }
        return nil
    }

    private static func literals(in fragment: String) -> [String] {
        let text = fragment as NSString
        return literalPattern
            .matches(in: fragment, range: NSRange(location: 0, length: text.length))
            .map { text.substring(with: $0.range(at: 1)) }
    }

    private static func ordered(_ levels: [String]) -> [String] {
        var seen = Set<String>()
        return levels.filter { seen.insert($0).inserted }
            .enumerated()
            .sorted {
                let left = ranks[$0.element.lowercased()] ?? 99
                let right = ranks[$1.element.lowercased()] ?? 99
                return left == right ? $0.offset < $1.offset : left < right
            }
            .map(\.element)
    }
}
