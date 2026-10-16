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

    // MARK: a turn is more than the stream

    /// Leaves a user message and an assistant reply ending in a live tool call.
    private func installPendingPermission(_ store: ChatStore) -> PendingToolPermission {
        let messageID = UUID()
        var call = ChatToolCall(name: "read_file", arguments: "{}", state: .pending)
        call.id = UUID()
        var assistant = ChatMessage(role: "assistant", content: "")
        assistant.id = messageID
        assistant.toolCalls = [call]
        var conversation = Conversation(title: "t")
        conversation.messages = [ChatMessage(role: "user", content: "hi"), assistant]
        store.conversations = [conversation]
        store.currentID = conversation.id
        let request = PendingToolPermission(
            conversationID: conversation.id, messageID: messageID, callID: call.id,
            name: "read_file", displayName: "read_file", arguments: "{}",
            writesData: false, serverID: nil, serverName: nil)
        store.pendingToolPermission = request
        return request
    }

    /// `generating` is false while a tool waits for permission, but the turn is
    /// not over. The old guards looked only at `generating`, so these buttons
    /// stayed live and started a second turn on top of the agent's context.
    func testAToolAwaitingPermissionCountsAsAnActiveFlow() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        installPendingPermission(store)
        XCTAssertTrue(store.agentFlowActive)
        store.pendingToolPermission = nil
        XCTAssertFalse(store.agentFlowActive)
    }

    func testRegenerateIsRefusedWhileAToolAwaitsPermission() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        installPendingPermission(store)
        let before = store.conversations[0].messages.map(\.id)

        store.regenerate(port: 1, temperature: 0.7, maxTokens: 128, system: "", thinking: false)

        XCTAssertEqual(store.conversations[0].messages.map(\.id), before,
                       "regenerating mid-turn would strand the agent's context")
    }

    func testEditIsRefusedWhileAToolAwaitsPermission() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        let request = installPendingPermission(store)

        XCTAssertNil(store.editMessage(request.messageID),
                     "cutting the branch mid-turn would discard the round in flight")
    }

    /// The guard must not stick: once nothing is streaming and nothing is
    /// waiting, the conversation is ordinary again.
    func testTheFlowEndsWhenThePermissionIsResolved() throws {
        let (store, dir) = try makeStore()
        defer { try? FileManager.default.removeItem(at: dir) }
        installPendingPermission(store)
        store.pendingToolPermission = nil

        XCTAssertFalse(store.agentFlowActive)
    }
}
