// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A download source used to be recognised by `host.contains(name)`, so
/// `huggingface.co.evil.example` and `evil-huggingface.co` both counted as
/// Hugging Face. The check has to be on whole labels.
final class HostMatchTests: XCTestCase {
    private var previousMirror: String?

    override func setUp() {
        previousMirror = UserDefaults.standard.string(forKey: SettingsKeys.customMirrorURL)
    }

    override func tearDown() {
        if let previousMirror {
            UserDefaults.standard.set(previousMirror, forKey: SettingsKeys.customMirrorURL)
        } else {
            UserDefaults.standard.removeObject(forKey: SettingsKeys.customMirrorURL)
        }
    }

    func testTheHostItselfMatches() {
        XCTAssertTrue(HostMatch.isHost("huggingface.co", "huggingface.co"))
        XCTAssertTrue(HostMatch.isHost("HUGGINGFACE.CO", "huggingface.co"))
    }

    func testARealSubdomainMatches() {
        XCTAssertTrue(HostMatch.isHost("cdn-lfs.huggingface.co", "huggingface.co"))
        XCTAssertTrue(HostMatch.isHost("a.b.huggingface.co", "huggingface.co"))
    }

    func testLookalikeHostsDoNotMatch() {
        for host in ["evil-huggingface.co",
                     "huggingface.co.evil.example",
                     "huggingface.coevil.example",
                     "nothuggingface.co"] {
            XCTAssertFalse(HostMatch.isHost(host, "huggingface.co"), "\(host) must not match")
        }
    }

    func testATrailingDotIsIgnored() {
        XCTAssertTrue(HostMatch.isHost("huggingface.co.", "huggingface.co"))
        XCTAssertTrue(HostMatch.isHost("cdn.huggingface.co.", "huggingface.co"))
    }

    func testAnEmptyOrNilHostNeverMatches() {
        XCTAssertFalse(HostMatch.isHost(nil, "huggingface.co"))
        XCTAssertFalse(HostMatch.isHost("", "huggingface.co"))
    }

    func testTheDownloadSourcesUseWholeLabels() {
        let lookalike = URL(string: "https://huggingface.co.evil.example/owner/repo/resolve/main/m.gguf")!
        let mirrorLookalike = URL(string: "https://evil-hf-mirror.com/owner/repo/resolve/main/m.gguf")!
        XCTAssertFalse(DownloadSource.huggingface.matchesURL(lookalike))
        XCTAssertFalse(DownloadSource.hfMirror.matchesURL(mirrorLookalike))
        XCTAssertTrue(DownloadSource.huggingface.matchesURL(
            URL(string: "https://cdn-lfs.huggingface.co/owner/repo/resolve/main/m.gguf")!))
    }

    func testACustomMirrorAlsoRejectsLookalikes() {
        UserDefaults.standard.set("https://hub.example.com", forKey: SettingsKeys.customMirrorURL)
        XCTAssertTrue(DownloadSource.custom.matchesURL(URL(string: "https://hub.example.com/a")!))
        XCTAssertTrue(DownloadSource.custom.matchesURL(URL(string: "https://cdn.hub.example.com/a")!))
        XCTAssertFalse(DownloadSource.custom.matchesURL(URL(string: "https://hub.example.com.evil/a")!))
        XCTAssertFalse(DownloadSource.custom.matchesURL(URL(string: "https://evil-hub.example.com/a")!))
    }
}
