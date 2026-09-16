// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Supported download sources for model files.
enum DownloadSource: String, CaseIterable, Identifiable {
    case huggingface = "huggingface"
    case hfMirror = "hf-mirror"
    case modelScope = "modelscope"
    case custom = "custom"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .huggingface: return "HuggingFace"
        case .hfMirror: return "hf-mirror.com"
        case .modelScope: return "魔塔社区 (ModelScope)"
        case .custom: return "自定义镜像"
        }
    }

    /// Base URL for API calls (model tree listing, file metadata).
    var apiBase: String {
        switch self {
        case .huggingface: return "https://huggingface.co"
        case .hfMirror: return "https://hf-mirror.com"
        case .modelScope: return "https://modelscope.cn"
        case .custom:
            let custom = UserDefaults.standard.string(forKey: SettingsKeys.customMirrorURL) ?? ""
            return custom.isEmpty ? "https://huggingface.co" : custom
        }
    }

    /// Build a download URL for a file in a repository.
    /// - Parameters:
    ///   - repo: Repository path (e.g. "Qwen/Qwen3-4B-GGUF")
    ///   - file: File path within the repo (e.g. "Qwen3-4B-Q4_K_M.gguf")
    ///   - branch: Branch name (default: "main")
    /// - Returns: Full download URL string
    func downloadURL(repo: String, file: String, branch: String = "main") -> String {
        switch self {
        case .huggingface, .hfMirror, .custom:
            return "\(apiBase)/\(repo)/resolve/\(branch)/\(file)"
        case .modelScope:
            // ModelScope uses "master" as default branch for most models
            let effectiveBranch = branch == "main" ? "master" : branch
            return "\(apiBase)/models/\(repo)/resolve/\(effectiveBranch)/\(file)"
        }
    }

    /// Build an API URL to list files in a repository.
    func treeAPI(repo: String, branch: String = "main") -> String {
        switch self {
        case .huggingface, .hfMirror, .custom:
            return "\(apiBase)/api/models/\(repo)/tree/\(branch)"
        case .modelScope:
            let effectiveBranch = branch == "main" ? "master" : branch
            return "\(apiBase)/api/v1/models/\(repo)/repository/tree?Revision=\(effectiveBranch)"
        }
    }

/// Build an API URL to search for models.
    func searchAPI(query: String, limit: Int = 20, sort: String = "trendingScore") -> String {
        switch self {
        case .huggingface, .hfMirror, .custom:
            var comps = URLComponents(string: "\(apiBase)/api/models")
            comps?.queryItems = [
                URLQueryItem(name: "search", value: query),
                URLQueryItem(name: "filter", value: "gguf"),
                URLQueryItem(name: "sort", value: sort),
                URLQueryItem(name: "direction", value: "-1"),
                URLQueryItem(name: "limit", value: String(limit)),
            ]
            return comps?.url?.absoluteString ?? "\(apiBase)/api/models"
        case .modelScope:
            // ModelScope search API
            var comps = URLComponents(string: "\(apiBase)/api/v1/models")
            comps?.queryItems = [
                URLQueryItem(name: "Query", value: query),
                URLQueryItem(name: "PageSize", value: String(limit)),
                URLQueryItem(name: "SortBy", value: "Downloads"),
            ]
            return comps?.url?.absoluteString ?? "\(apiBase)/api/v1/models"
        }
    }
    
    /// Build an API URL to list trending/popular models.
    func trendingAPI(limit: Int = 20, sort: String = "trendingScore") -> String {
        switch self {
        case .huggingface, .hfMirror, .custom:
            var comps = URLComponents(string: "\(apiBase)/api/models")
            comps?.queryItems = [
                URLQueryItem(name: "filter", value: "gguf"),
                URLQueryItem(name: "sort", value: sort),
                URLQueryItem(name: "direction", value: "-1"),
                URLQueryItem(name: "limit", value: String(limit)),
            ]
            return comps?.url?.absoluteString ?? "\(apiBase)/api/models"
        case .modelScope:
            var comps = URLComponents(string: "\(apiBase)/api/v1/models")
            comps?.queryItems = [
                URLQueryItem(name: "PageSize", value: String(limit)),
                URLQueryItem(name: "SortBy", value: "Downloads"),
            ]
            return comps?.url?.absoluteString ?? "\(apiBase)/api/v1/models"
        }
    }

    /// Check if a URL belongs to this download source.
    func matchesURL(_ url: URL) -> Bool {
        guard let host = url.host else { return false }
        switch self {
        case .huggingface: return host == "huggingface.co" || host.contains("huggingface.co")
        case .hfMirror: return host == "hf-mirror.com" || host.contains("hf-mirror.com")
        case .modelScope: return host == "modelscope.cn" || host.contains("modelscope.cn")
        case .custom:
            guard let customBase = UserDefaults.standard.string(forKey: SettingsKeys.customMirrorURL),
                  let customURL = URL(string: customBase),
                  let customHost = customURL.host else { return false }
            return host == customHost || host.contains(customHost)
        }
    }

    // MARK: - Current Source

    /// The currently selected download source from user settings.
    static var current: DownloadSource {
        let raw = UserDefaults.standard.string(forKey: SettingsKeys.downloadSource) ?? ""
        return DownloadSource(rawValue: raw) ?? .huggingface
    }
}
