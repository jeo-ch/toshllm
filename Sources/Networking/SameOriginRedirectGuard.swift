// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Refuses a redirect that leaves the origin of the address the user configured.
///
/// URLSession re-sends the body and the custom headers on a 307/308, so a
/// receiver that answers with a redirect elsewhere could take a conversation
/// (and its bearer token) to a host the user never named. A plain `http`
/// endpoint may still be upgraded to `https` on the same host.
final class SameOriginRedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    struct Origin: Equatable {
        let scheme: String
        let host: String
        let port: Int

        /// Nil for a URL that is not plain http(s) or carries credentials.
        init?(_ url: URL) {
            guard url.user == nil, url.password == nil,
                  let host = url.host?.lowercased(), !host.isEmpty else { return nil }
            switch url.scheme?.lowercased() {
            case "http": scheme = "http"
            case "https": scheme = "https"
            default: return nil
            }
            self.host = host
            port = url.port ?? (scheme == "https" ? 443 : 80)
        }
    }

    private let origin: Origin?
    private let lock = NSLock()
    private var refusedTarget: String?

    init(_ url: URL) {
        origin = Origin(url)
    }

    /// Same origin, or the same host upgraded from http to https.
    func allows(_ target: URL?) -> Bool {
        guard let target, let candidate = Origin(target), let origin else { return false }
        if candidate == origin { return true }
        return origin.scheme == "http" && candidate.scheme == "https"
            && candidate.host == origin.host && candidate.port == 443
    }

    /// Where the refused redirect pointed, for the caller's log.
    var refusal: String? {
        lock.withLock { refusedTarget }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard allows(request.url) else {
            let target = request.url.flatMap(Origin.init).map(\.description)
                ?? request.url?.absoluteString
                ?? "an address with credentials or no valid host"
            lock.withLock { refusedTarget = target }
            completionHandler(nil)
            return
        }
        completionHandler(request)
    }
}

private extension SameOriginRedirectGuard.Origin {
    var description: String { "\(scheme)://\(host):\(port)" }
}
