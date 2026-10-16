// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A PID is only unique while its process lives. Every escalation path here
/// holds a number across a grace period and then signals it, so the executable
/// has to be compared again at that moment or the signal can reach whatever the
/// kernel handed the number to next.
final class ProcessIdentityTests: XCTestCase {

    func testTheCurrentProcessResolvesToAnAbsolutePath() {
        let path = ProcessIdentity.executablePath(of: getpid())
        XCTAssertNotNil(path)
        XCTAssertTrue(path?.hasPrefix("/") == true, "got \(path ?? "nil")")
    }

    func testAMissingProcessResolvesToNil() {
        XCTAssertNil(ProcessIdentity.executablePath(of: Int32.max - 1))
    }

    private func spawnSleep() throws -> Process {
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/sleep")
        child.arguments = ["30"]
        try child.run()
        return child
    }

    /// The number is alive but names something else. Signalling it is exactly the
    /// mistake the check exists to prevent.
    func testDoesNotSignalWhenTheExecutableDoesNotMatch() throws {
        let child = try spawnSleep()
        defer { if child.isRunning { child.terminate() } }

        ProcessIdentity.killIfStill(pid: child.processIdentifier, running: "/bin/sh", signal: SIGTERM)
        Thread.sleep(forTimeInterval: 0.2)

        XCTAssertTrue(child.isRunning, "a live pid with a different executable must not be signalled")
    }

    func testSignalsWhenTheExecutableStillMatches() throws {
        let child = try spawnSleep()
        let pid = child.processIdentifier
        let path = ProcessIdentity.executablePath(of: pid)

        ProcessIdentity.killIfStill(pid: pid, running: path, signal: SIGTERM)
        child.waitUntilExit()

        XCTAssertFalse(child.isRunning)
    }

    func testDoesNotSignalAPidThatIsGone() {
        // A nil path means the process was already gone when it was remembered;
        // nothing may be signalled on the strength of the number alone.
        ProcessIdentity.killIfStill(pid: Int32.max - 1, running: nil, signal: SIGKILL)
        ProcessIdentity.killIfStill(pid: Int32.max - 1, running: "/bin/sleep", signal: SIGKILL)
    }
}
