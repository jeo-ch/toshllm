// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// Model output is interpolated into a `<script>` element as a JSON string. If a
/// closing tag can be formed the HTML parser ends the element early and the rest
/// of the model's text becomes live markup in the transcript webview.
final class RichContentScriptEscapingTests: XCTestCase {

    private let breakout = "</script><img src=x onerror=alert(1)>"

    func testScriptJSONNeverEmitsAClosingTag() {
        let encoded = RichWebView.scriptJSON(breakout)
        XCTAssertFalse(encoded.contains("</"), "an unescaped `<` would let the parser close the element")
        XCTAssertTrue(encoded.contains("\\u003C"))
    }

    func testScriptJSONRoundTripsTheOriginalText() throws {
        let encoded = RichWebView.scriptJSON(breakout)
        let decoded = try JSONDecoder().decode(String.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded, breakout, "\\u003C must decode back to `<` for the renderer")
    }

    func testEveryKindEscapesThePayload() {
        // These three interpolate the model's text directly as a JSON string.
        for kind in [RichContentKind.math, .inlineMath, .mermaid] {
            let html = RichWebView.html(source: breakout, kind: kind)
            XCTAssertFalse(html.contains("</script><img"),
                           "\(kind) let the payload close the script element")
            XCTAssertTrue(html.contains("\\u003C"),
                          "\(kind) no longer escapes `<` in the payload")
        }
        // `.svg` base64-encodes into a data URL first, so the payload never
        // reaches the parser as markup; the same input must still be harmless.
        let svg = RichWebView.html(source: breakout, kind: .svg)
        XCTAssertFalse(svg.contains("</script><img"))
    }

    func testOrdinaryTextStillEmbedsAsAJsonString() throws {
        let text = "a \"quoted\" path: C:\\tmp\\model.gguf"
        let encoded = RichWebView.scriptJSON(text)
        XCTAssertEqual(try JSONDecoder().decode(String.self, from: Data(encoded.utf8)), text)
    }
}
