// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

enum SettingsArchiveError: LocalizedError {
    case invalidFormat

    var errorDescription: String? { "The settings file is not valid." }
}

enum SettingsArchive {
    static func exportData(defaults: UserDefaults = .standard) throws -> Data {
        let all = defaults.dictionaryRepresentation()
        let config = Dictionary(SettingsKeys.archiveOptionKeys.compactMap { key in
            all[key].map { (key, archiveValue($0)) }
        }, uniquingKeysWith: { first, _ in first })
        return try JSONSerialization.data(withJSONObject: [
            "format": "toshllm-settings",
            "version": 1,
            "exported_at": ISO8601DateFormatter().string(from: Date()),
            "config": config
        ], options: [.prettyPrinted, .sortedKeys])
    }

    @discardableResult
    static func importData(_ data: Data, defaults: UserDefaults = .standard) throws -> Int {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let config = root["config"] as? [String: Any] else {
            throw SettingsArchiveError.invalidFormat
        }
        let allowed = Set(SettingsKeys.archiveOptionKeys)
        // Say why a file that carries an executable path or MCP definitions
        // imports fewer settings than it appears to, instead of dropping them
        // without a trace.
        let excluded = SettingsKeys.archiveExcludedKeys.filter { config[$0] != nil }
        if !excluded.isEmpty {
            AppLog.app.notice("settings archive ignored executable/MCP keys: \(excluded.sorted().joined(separator: ", "), privacy: .public)")
        }
        var count = 0
        for (key, value) in config where allowed.contains(key) {
            if value is NSNull { defaults.removeObject(forKey: key) }
            else { defaults.set(restoredValue(value), forKey: key) }
            count += 1
        }
        return count
    }

    private static func archiveValue(_ value: Any) -> Any {
        if let data = value as? Data { return ["$data": data.base64EncodedString()] }
        if let array = value as? [Any] { return array.map(archiveValue) }
        if let dictionary = value as? [String: Any] {
            return dictionary.mapValues(archiveValue)
        }
        return value
    }

    private static func restoredValue(_ value: Any) -> Any {
        if let tagged = value as? [String: Any], tagged.count == 1,
           let encoded = tagged["$data"] as? String, let data = Data(base64Encoded: encoded) {
            return data
        }
        if let array = value as? [Any] { return array.map(restoredValue) }
        if let dictionary = value as? [String: Any] {
            return dictionary.mapValues(restoredValue)
        }
        return value
    }
}
