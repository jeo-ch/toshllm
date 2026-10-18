// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// Every request to a local MCP server returns by its deadline, once, with its own answer.
final class MCPStdioTransportTests: XCTestCase {
    /// Answers at once, except `ignore` (never) and `defer` (late, just before the next answer);
    /// each answer counts the cancellations received so far.
    private static let responder = #"""
    held=""; cancelled=0
    while IFS= read -r line; do
      case "$line" in *cancelled*) cancelled=$((cancelled+1)); continue;; esac
      id=$(printf '%s\n' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
      [ -n "$id" ] || continue
      case "$line" in
        *'"method":"ignore"'*) continue;;
        *'"method":"defer"'*) held=$id; continue;;
      esac
      if [ -n "$held" ]; then printf '{"jsonrpc":"2.0","id":%s,"result":{"late":true}}\n' "$held"; held=""; fi
      printf '{"jsonrpc":"2.0","id":%s,"result":{"answered":%s,"cancelled":%s}}\n' "$id" "$id" "$cancelled"
    done
    """#

    /// Reads every request and answers none, with its output left open.
    private static let silent = "while IFS= read -r line; do :; done"

    private var transports: [MCPStdioTransport] = []

    override func tearDown() async throws {
        for transport in transports { await transport.stop() }
    }

    private func launch(_ script: String) async throws -> MCPStdioTransport {
        let transport = try MCPStdioTransport(server: MCPServer(name: "probe", url: "", transport: .stdio,
                                                                command: "/bin/sh", arguments: ["-c", script]))
        try await transport.start()
        transports.append(transport)
        return transport
    }

    private static func request(_ method: String, _ id: Int) -> [String: Any] {
        ["jsonrpc": "2.0", "id": id, "method": method]
    }

    /// What `body` gave, or nil with a recorded failure when it outlived `seconds`.
    private func bounded<T>(_ seconds: TimeInterval, _ body: @escaping @Sendable () async throws -> T)
        async -> Result<T, Error>? {
        let outcome = Outcome<T>()
        let done = expectation(description: "returned within \(seconds) s")
        Task {
            do { outcome.set(.success(try await body())) } catch { outcome.set(.failure(error)) }
            done.fulfill()
        }
        await fulfillment(of: [done], timeout: seconds)
        return outcome.get()
    }

    private func failure<T>(_ outcome: Result<T, Error>?, _ what: String) -> String? {
        guard case let .failure(error)? = outcome else {
            XCTFail("\(what) returned instead of failing")
            return nil
        }
        return error.localizedDescription
    }

    func testASilentServerTimesOutByItsDeadline() async throws {
        let transport = try await launch(Self.silent)
        let start = ContinuousClock.now
        let outcome = await bounded(5) {
            try await transport.send(Self.request("ping", 1), id: 1, timeout: .milliseconds(300)).count
        }
        XCTAssertEqual(failure(outcome, "a silent request")?.contains("did not answer in time"), true)
        XCTAssertLessThan(ContinuousClock.now - start, .seconds(3))
        let left = await transport.waiting
        XCTAssertEqual(left, 0)
    }

    func testALateAnswerIsDroppedAndTheNextRequestGetsItsOwn() async throws {
        let transport = try await launch(Self.responder)
        let deferred = await bounded(5) {
            try await transport.send(Self.request("defer", 1), id: 1, timeout: .milliseconds(300)).count
        }
        guard let reason = failure(deferred, "the deferred request") else { return }
        XCTAssertTrue(reason.contains("did not answer in time"), reason)
        let next = await bounded(10) { () async throws -> [Int] in
            let result = try await transport.send(Self.request("ping", 2), id: 2, timeout: .seconds(10))
            return [result["answered"] as? Int ?? -1, result["cancelled"] as? Int ?? -1, result["late"] == nil ? 0 : 1]
        }
        XCTAssertEqual(try next?.get(), [2, 1, 0])
        let left = await transport.waiting
        XCTAssertEqual(left, 0)
    }

    func testCancellingAWaitingRequestReturnsAtOnceAndTellsTheServer() async throws {
        let transport = try await launch(Self.responder)
        let ignored = Task { try await transport.send(Self.request("ignore", 1), id: 1, timeout: .seconds(30)).count }
        let answered = await bounded(10) {
            try await transport.send(Self.request("ping", 2), id: 2, timeout: .seconds(10))["answered"] as? Int
        }
        XCTAssertEqual(try answered?.get(), 2)
        ignored.cancel()
        let cancelled = await bounded(5) { () async -> Bool in
            do { _ = try await ignored.value; return false } catch { return error is CancellationError }
        }
        XCTAssertEqual(try cancelled?.get(), true)
        let next = await bounded(10) {
            try await transport.send(Self.request("ping", 3), id: 3, timeout: .seconds(10))["cancelled"] as? Int
        }
        XCTAssertEqual(try next?.get(), 1)
    }

    func testARequestCancelledBeforeItStartsReturnsAtOnce() async throws {
        let transport = try await launch(Self.silent)
        let task = Task { try await transport.send(Self.request("ping", 1), id: 1, timeout: .seconds(30)).count }
        task.cancel()
        let outcome = await bounded(5) { () async -> Bool in
            do { _ = try await task.value; return false } catch { return error is CancellationError }
        }
        XCTAssertEqual(try outcome?.get(), true)
    }

    func testClosedOutputFailsTheWaitingRequestAndTheNextOneAtOnce() async throws {
        let transport = try await launch("IFS= read -r line; exec >&-; exec cat >/dev/null")
        for id in 1...2 {
            let outcome = await bounded(5) {
                try await transport.send(Self.request("ping", id), id: id, timeout: .seconds(30)).count
            }
            guard let reason = failure(outcome, "request \(id)") else { return }
            XCTAssertTrue(reason.contains("closed its output"), "\(id): \(reason)")
        }
    }

    func testAWriteToAServerThatClosedItsInputFailsTheRequest() async throws {
        let transport = try await launch(
            #"IFS= read -r line; exec 0<&-; printf '{"jsonrpc":"2.0","id":1,"result":{}}\n'; exec sleep 30"#)
        let first = await bounded(5) { try await transport.send(Self.request("ping", 1), id: 1, timeout: .seconds(10)).count }
        XCTAssertEqual(try first?.get(), 0)
        let second = await bounded(5) {
            try await transport.send(Self.request("ping", 2), id: 2, timeout: .seconds(30)).count
        }
        guard let reason = failure(second, "a write to a closed input") else { return }
        XCTAssertFalse(reason.contains("did not answer in time"), reason)
    }

    func testIdleServersLeaveTheConcurrencyPoolFree() async throws {
        for _ in 0..<ProcessInfo.processInfo.activeProcessorCount { _ = try await launch(Self.silent) }
        let outcome = await bounded(10) {
            try await Task.detached(priority: .utility) {
                try await Task.sleep(for: .milliseconds(50))
                return await Task.detached(priority: .utility) { 1 }.value
            }.value
        }
        XCTAssertEqual(try outcome?.get(), 1)
    }

    /// Average cost of the `/bin/sh` + `sed` pair the responder spawns per line, over 20 tries.
    private static func spawnCost() -> Duration {
        let clock = ContinuousClock()
        let start = clock.now
        for _ in 0..<20 {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/sh")
            process.arguments = ["-c", "printf '{\"id\":1}' | sed -n 's/.*\"id\":\\([0-9]*\\).*/\\1/p'"]
            process.standardOutput = Pipe()
            process.standardError = Pipe()
            try? process.run()
            process.waitUntilExit()
        }
        return (clock.now - start) / 20
    }

    func testAnswersRacingTheirDeadlinesResolveEachRequestOnce() async throws {
        // The responder spawns one `sed` per line, so 200 requests cost 200 process launches.
        // Where a launch is slow, that alone outlasts every deadline (all at most 1.5 s) and no
        // request is ever answered: a property of the host, not of the transport.
        let perSpawn = await Task.detached { Self.spawnCost() }.value
        try XCTSkipIf(perSpawn * 200 > .milliseconds(1000),
                      "200 process launches cost \(perSpawn * 200); the responder cannot outrun the deadlines here")
        let transport = try await launch(Self.responder)
        let codes = await bounded(30) { () async throws -> [Int] in
            try await withThrowingTaskGroup(of: Int.self) { group in
                for id in 1...200 {
                    group.addTask {
                        do {
                            // Deadlines spread over the time the server takes to answer all 200.
                            let result = try await transport.send(Self.request("ping", id), id: id,
                                                                  timeout: .milliseconds((id * 37) % 1500))
                            return result["answered"] as? Int == id ? 1 : -1
                        } catch {
                            return error.localizedDescription.contains("did not answer in time") ? 0 : -2
                        }
                    }
                }
                return try await group.reduce(into: []) { $0.append($1) }
            }
        }
        let resolved = try XCTUnwrap(try codes?.get())
        XCTAssertEqual(resolved.count, 200)
        XCTAssertEqual(resolved.filter { $0 < 0 }, [], "a request got another's answer or failed for another reason")
        XCTAssertTrue(resolved.contains(1) && resolved.contains(0), "answers and deadlines did not both occur")
        let next = await bounded(10) {
            try await transport.send(Self.request("ping", 201), id: 201, timeout: .seconds(10))["answered"] as? Int
        }
        XCTAssertEqual(try next?.get(), 201)
        let left = await transport.waiting
        XCTAssertEqual(left, 0)
    }
}

private final class Outcome<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<T, Error>?

    func set(_ result: Result<T, Error>) { lock.withLock { value = result } }
    func get() -> Result<T, Error>? { lock.withLock { value } }
}
