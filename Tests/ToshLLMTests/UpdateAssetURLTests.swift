// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The release API reply names the DMG, the checksums and the page to open. A
/// reply that points at another host must not be fetched or opened.
final class UpdateAssetURLTests: XCTestCase {
    func testGitHubOverHTTPSIsAccepted() {
        XCTAssertTrue(UpdateChecker.isReleaseAsset(
            URL(string: "https://github.com/jeo-ch/toshllm/releases/download/v1/ToshLLM.dmg")!))
        XCTAssertTrue(UpdateChecker.isReleaseAsset(
            URL(string: "https://objects.githubusercontent.com/asset")!))
    }

    func testAnotherHostIsRejected() {
        XCTAssertFalse(UpdateChecker.isReleaseAsset(URL(string: "https://evil.example.com/ToshLLM.dmg")!))
        XCTAssertFalse(UpdateChecker.isReleaseAsset(
            URL(string: "https://github.com.evil.example/ToshLLM.dmg")!))
        XCTAssertFalse(UpdateChecker.isReleaseAsset(
            URL(string: "https://evil-github.com/ToshLLM.dmg")!))
    }

    func testAnotherSchemeIsRejected() {
        XCTAssertFalse(UpdateChecker.isReleaseAsset(URL(string: "http://github.com/ToshLLM.dmg")!))
        XCTAssertFalse(UpdateChecker.isReleaseAsset(URL(string: "file:///tmp/ToshLLM.dmg")!))
    }
}
