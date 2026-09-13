// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Centralized network configuration: proxy, timeout, and shared URLSession.
/// Replace ad-hoc `URLSession.shared` usage with `NetworkManager.session` so
/// proxy settings and timeouts apply consistently across the app.
enum NetworkManager {
    /// Shared session with proxy and timeout settings applied.
    static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 600
        config.waitsForConnectivity = true
        applyProxy(to: config)
        return URLSession(configuration: config)
    }()

    /// Creates a new URLSession inheriting the global proxy settings.
    static func makeSession(timeout: TimeInterval = 600) -> URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = timeout
        config.waitsForConnectivity = true
        applyProxy(to: config)
        return URLSession(configuration: config)
    }

    /// Creates a URLSessionConfiguration with proxy settings applied.
    static func makeConfiguration(timeout: TimeInterval = 600) -> URLSessionConfiguration {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = timeout
        config.waitsForConnectivity = true
        applyProxy(to: config)
        return config
    }

    /// Applies HTTP/HTTPS proxy from UserDefaults if configured.
    static func applyProxy(to config: URLSessionConfiguration) {
        let defaults = UserDefaults.standard
        guard defaults.bool(forKey: SettingsKeys.proxyEnabled) else { return }
        let host = defaults.string(forKey: SettingsKeys.proxyHost) ?? ""
        let port = defaults.integer(forKey: SettingsKeys.proxyPort)
        guard !host.isEmpty, port > 0 else { return }

        let proxyDict: [String: Any] = [
            kCFNetworkProxiesHTTPEnable as String: true,
            kCFNetworkProxiesHTTPProxy as String: host,
            kCFNetworkProxiesHTTPPort as String: port,
            kCFNetworkProxiesHTTPSEnable as String: true,
            kCFNetworkProxiesHTTPSProxy as String: host,
            kCFNetworkProxiesHTTPSPort as String: port,
        ]
        config.connectionProxyDictionary = proxyDict
    }
}
