// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Darwin
import XCTest
@testable import ToshLLM

final class ProcessEscalationTests: XCTestCase {
    func testTerminateWithEscalationKillsAProcessThatIgnoresSIGTERM() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        // The shell ignores TERM, so only the delayed SIGKILL can stop it.
        process.arguments = ["-c", "trap '' TERM; while :; do :; done"]
        try process.run()
        let pid = process.processIdentifier
        XCTAssertGreaterThan(pid, 0)

        process.terminateWithEscalation(after: 0.2)

        let deadline = Date().addingTimeInterval(5)
        while kill(pid, 0) == 0 && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if kill(pid, 0) == 0 {
            kill(pid, SIGKILL)
            XCTFail("process \(pid) survived SIGTERM and the escalation grace period")
        }
    }
}
