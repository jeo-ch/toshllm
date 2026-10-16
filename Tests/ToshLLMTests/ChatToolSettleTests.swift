// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A turn that ends by hand or by stalling still leaves tool calls in the
/// transcript. If they are not settled their cards spin forever, and the next
/// request sends the engine a tool call with no result.
@MainActor
final class ChatToolSettleTests: XCTestCase {

    private func makeStore() throws -> (ChatStore, URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-chat-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return (ChatStore(storageDirectory: dir), dir)
    }

    /// Leaves a conversation holding one call in each of the three live states.
    private func install(_ store: ChatStore) -> UUID {
        var conversation = Conversation(title: "t")
        var message = ChatMessage(role: "assistant", content: "")
        message.toolCalls = [
            ChatToolCall(name: "read_file", arguments: "{}", state: .pending),
            ChatToolCall(name: "write_file", arguments: "{}", state: .awaitingPermission),
            ChatToolCall(name: "finished", arguments: "{}", state: .completed),
        ]
        conversation.messages = [message]
        store.conversations = [conversation]
        store.currentID = conversation.id
        return conversation.id
    }

    func testStopSettlesTheCallsStillWaiting() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let id = install(store)

        store.stop()

        let calls = store.conversations[0].messages[0].toolCalls!
        XCTAssertEqual(calls[0].state, .denied, "a pending call must not outlive the turn")
        XCTAssertEqual(calls[1].state, .denied, "a call awaiting permission must not outlive the turn")
        XCTAssertNotNil(calls[0].result)
        XCTAssertNotNil(calls[1].finishedAt)
        XCTAssertNil(store.pendingToolPermission)
        XCTAssertEqual(id, store.currentID)
    }

    /// A call that already finished carries a result the model read; rewriting it
    /// would lose that.
    func testStopLeavesFinishedCallsAlone() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        install(store)
        store.conversations[0].messages[0].toolCalls?[2].result = "already answered"

        store.stop()

        let finished = store.conversations[0].messages[0].toolCalls![2]
        XCTAssertEqual(finished.state, .completed)
        XCTAssertEqual(finished.result, "already answered")
    }

    /// Each settled call needs a tool message, or the next request reaches the
    /// engine with a call that has no result.
    func testEachSettledCallGetsAResultMessage() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        install(store)

        store.stop()

        let tools = store.conversations[0].messages.filter { $0.role == "tool" }
        XCTAssertEqual(tools.count, 2)
        XCTAssertEqual(Set(tools.compactMap(\.toolCallID)).count, 2, "each result must name a distinct call")
    }

    func testStopOnAConversationWithoutToolsIsHarmless() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        store.conversations = [Conversation(title: "empty")]
        store.currentID = store.conversations[0].id

        store.stop()

        XCTAssertEqual(store.conversations[0].messages.count, 0)
    }
}
