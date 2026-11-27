// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The memory archive hook posts conversation turns to a URL the user names.
/// URLSession re-sends a POST body on a 307/308, so a redirect must not be able
/// to take that body (and the bearer token) to another host.
final class SameOriginRedirectGuardTests: XCTestCase {
    private func guardFor(_ address: String) -> SameOriginRedirectGuard {
        SameOriginRedirectGuard(URL(string: address)!)
    }

    func testTheSameOriginIsAllowed() {
        let guardrail = guardFor("https://hooks.example.com/archive")
        XCTAssertTrue(guardrail.allows(URL(string: "https://hooks.example.com/other")!))
        XCTAssertTrue(guardrail.allows(URL(string: "https://hooks.example.com:443/other")!))
    }

    func testAnotherHostIsRefused() {
        let guardrail = guardFor("https://hooks.example.com/archive")
        XCTAssertFalse(guardrail.allows(URL(string: "https://evil.example.com/archive")!))
        XCTAssertFalse(guardrail.allows(URL(string: "https://hooks.example.com.evil.example/archive")!))
    }

    func testAnotherPortIsRefused() {
        let guardrail = guardFor("https://hooks.example.com/archive")
        XCTAssertFalse(guardrail.allows(URL(string: "https://hooks.example.com:8443/archive")!))
    }

    func testADowngradeToPlainHTTPIsRefused() {
        let guardrail = guardFor("https://hooks.example.com/archive")
        XCTAssertFalse(guardrail.allows(URL(string: "http://hooks.example.com/archive")!))
    }

    func testAPlainHTTPEndpointMayBeUpgradedOnTheSameHost() {
        let guardrail = guardFor("http://hooks.example.com/archive")
        XCTAssertTrue(guardrail.allows(URL(string: "https://hooks.example.com/archive")!))
        XCTAssertFalse(guardrail.allows(URL(string: "https://other.example.com/archive")!))
    }

    func testCredentialsAndUnknownSchemesNeverMatch() {
        let guardrail = guardFor("https://hooks.example.com/archive")
        XCTAssertFalse(guardrail.allows(URL(string: "https://user:pass@hooks.example.com/archive")!))
        XCTAssertFalse(guardrail.allows(URL(string: "file:///etc/passwd")!))
        XCTAssertFalse(guardrail.allows(nil))
    }

    func testTheRefusalIsRememberedForTheLog() {
        let guardrail = guardFor("https://hooks.example.com/archive")
        XCTAssertNil(guardrail.refusal)
        XCTAssertFalse(guardrail.allows(URL(string: "https://evil.example.com/archive")!))
    }
}
