// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Host checks for addresses that arrive from outside the app.
///
/// A substring test is not a host check: `huggingface.co.evil.example` and
/// `evil-huggingface.co` both contain the trusted name, so the comparison is
/// made on whole labels and only a real subdomain is accepted.
enum HostMatch {
    /// True when `host` is `domain` itself or a subdomain of it.
    static func isHost(_ host: String?, _ domain: String) -> Bool {
        guard var host = host?.lowercased(), !host.isEmpty else { return false }
        // A fully qualified name may carry a trailing dot.
        if host.hasSuffix(".") { host.removeLast() }
        let domain = domain.lowercased()
        return host == domain || host.hasSuffix("." + domain)
    }

    static func isHost(_ url: URL, _ domain: String) -> Bool {
        isHost(url.host, domain)
    }
}
