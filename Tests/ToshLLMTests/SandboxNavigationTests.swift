// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The JavaScript sandbox keeps a message bridge in the page. A script that
/// navigates the view would load foreign code next to that bridge, so only the
/// document `loadHTMLString` creates may load.
final class SandboxNavigationTests: XCTestCase {
    func testTheHarnessDocumentIsAllowed() {
        XCTAssertTrue(JavaScriptSandboxService.isHarnessNavigation(nil))
        XCTAssertTrue(JavaScriptSandboxService.isHarnessNavigation(URL(string: "about:blank")!))
    }

    func testEveryOtherDocumentIsRefused() {
        for address in ["https://example.com/", "http://example.com/",
                        "file:///etc/passwd", "data:text/html,<script></script>",
                        "toshmap://app/map.html"] {
            let url = URL(string: address)!
            XCTAssertFalse(JavaScriptSandboxService.isHarnessNavigation(url),
                           "\(address) must not load in the sandbox")
        }
    }
}
