// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The startup banner and the diagnostics export both write the argument list to
/// a file the user is then encouraged to share, so a key placed in Extra
/// arguments must never survive either path.
final class ServerRedactionTests: XCTestCase {

    func testSeparateFormRedactsTheFollowingToken() {
        let out = ServerController.redactArguments(["-m", "model.gguf", "--api-key", "sk-secret", "-ngld", "99"])
        XCTAssertEqual(out, ["-m", "model.gguf", "--api-key", "***", "-ngld", "99"])
    }

    func testJoinedFormRedactsTheValue() {
        let out = ServerController.redactArguments(["--api-key=sk-secret"])
        XCTAssertEqual(out, ["--api-key=***"])
    }

    /// Extra arguments can carry a second key; redacting only the first would
    /// still export the rest.
    func testEveryOccurrenceIsRedactedNotJustTheFirst() {
        let out = ServerController.redactArguments(["--api-key", "sk-one", "--api-key=sk-two", "-ngld", "99"])
        XCTAssertEqual(out, ["--api-key", "***", "--api-key=***", "-ngld", "99"])
    }

    func testSymPyTrustKeyIsRedacted() {
        let key = SymPyToolsService.trustKey
        let out = ServerController.redactArguments(["--tosh-sympy-token", key])
        XCTAssertEqual(out, ["--tosh-sympy-token", "***"])
    }

    func testUnrelatedArgumentsAreLeftAlone() {
        let args = ["-m", "model.gguf", "-c", "8192", "-ngld", "99", "--jinja"]
        XCTAssertEqual(ServerController.redactArguments(args), args)
    }

    func testKeyLookalikeInAPathIsNotMangled() {
        let args = ["-m", "/Users/me/--api-key=notes.gguf"]
        XCTAssertEqual(ServerController.redactArguments(args), args)
    }
}
