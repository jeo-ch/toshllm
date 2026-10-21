// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A store file this build cannot read must not cost the user the rest of the history, and must
/// not be overwritten by the first save after the failed read.
@MainActor
final class ChatStoreRecoveryTests: XCTestCase {
    private func directory() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-store-recovery-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: dir)
        }
        return dir
    }

    private func encoded(_ conversations: [Conversation]) throws -> Data {
        try JSONEncoder().encode(conversations)
    }

    private func conversation(_ title: String) -> Conversation {
        Conversation(title: title, messages: [ChatMessage(role: "user", content: "hola")])
    }

    private func corruptCopies(in dir: URL) -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?
            .filter { $0.contains(".corrupt-") } ?? []
    }

    func testOneUnreadableEntryDoesNotDropTheRest() throws {
        let dir = directory()
        let kept = conversation("kept")

        // A valid record next to one this build cannot parse: the old whole-array decode returned
        // nil and the store showed no conversations at all.
        var array = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded([kept])) as? [Any]
        )
        array.append(["bogus": true])
        try JSONSerialization.data(withJSONObject: array)
            .write(to: dir.appendingPathComponent("conversations.json"))

        let store = ChatStore(storageDirectory: dir)
        ChatStore.flushPendingSave()

        XCTAssertEqual(store.conversations.filter { $0.title == "kept" }.count, 1)
        XCTAssertEqual(corruptCopies(in: dir).count, 1, "the entries left out stay recoverable")
    }

    func testUnreadablePrimaryIsKeptAndTheBackupIsUsed() throws {
        let dir = directory()
        let backup = conversation("from backup")
        let primary = Data("{ not json".utf8)

        try primary.write(to: dir.appendingPathComponent("conversations.json"))
        try encoded([backup]).write(to: dir.appendingPathComponent("conversations.json.backup"))

        let store = ChatStore(storageDirectory: dir)
        ChatStore.flushPendingSave()

        XCTAssertEqual(store.conversations.filter { $0.title == "from backup" }.count, 1)
        let copies = corruptCopies(in: dir)
        XCTAssertEqual(copies.count, 1)
        let kept = try Data(contentsOf: dir.appendingPathComponent(copies[0]))
        XCTAssertEqual(kept, primary, "the unreadable bytes are preserved verbatim")
    }

    func testEmptyDirectoryStartsEmptyWithoutCorruptCopies() {
        let dir = directory()
        let store = ChatStore(storageDirectory: dir)
        ChatStore.flushPendingSave()

        XCTAssertEqual(store.conversations.count, 1)
        XCTAssertTrue(store.conversations.allSatisfy(\.messages.isEmpty))
        XCTAssertTrue(corruptCopies(in: dir).isEmpty)
    }
}
