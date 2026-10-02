// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The listing cache exists only to stop a models scan from re-reading the same
/// directory once per model, so the property that matters is that a cached read
/// is indistinguishable from a fresh one.
final class ModelDirectoryListingTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dir-listing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in ["a.gguf", "b.gguf", "notes.txt", "MTP"] {
            if name == "MTP" {
                try FileManager.default.createDirectory(
                    at: dir.appendingPathComponent(name), withIntermediateDirectories: true)
            } else {
                try Data("x".utf8).write(to: dir.appendingPathComponent(name))
            }
        }
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
        dir = nil
    }

    /// Inside a cached pass, the second read of a directory must return exactly
    /// what the first one saw — same entries, same order.
    func testCachedReadMatchesAFreshOne() {
        let uncached = ModelDirectoryListing.entries(in: dir)
        XCTAssertFalse(uncached.isEmpty)

        ModelDirectoryListing.withCache {
            let first = ModelDirectoryListing.entries(in: dir)
            let second = ModelDirectoryListing.entries(in: dir)
            XCTAssertEqual(first.map(\.lastPathComponent), uncached.map(\.lastPathComponent))
            XCTAssertEqual(second.map(\.lastPathComponent), uncached.map(\.lastPathComponent))
        }
    }

    /// A file added mid-pass is not visible, which is the trade the pass makes. The
    /// point of the test is both halves: inside the pass the listing is taken once,
    /// and outside it every read is fresh. A projector paired against a stale
    /// listing would be a wrong answer, not just a slow one.
    func testOutsideAPassEveryReadIsFresh() throws {
        ModelDirectoryListing.withCache {
            XCTAssertFalse(ModelDirectoryListing.entries(in: dir)
                .contains { $0.lastPathComponent == "late.gguf" })
            try? Data("x".utf8).write(to: dir.appendingPathComponent("late.gguf"))
            XCTAssertFalse(ModelDirectoryListing.entries(in: dir)
                .contains { $0.lastPathComponent == "late.gguf" },
                           "the pass reads the directory once, on purpose")
        }
        XCTAssertTrue(ModelDirectoryListing.entries(in: dir)
            .contains { $0.lastPathComponent == "late.gguf" },
            "outside the pass the listing is taken fresh")
    }

    /// A directory that cannot be read yields an empty array, the same fallback
    /// the call sites already handled.
    func testAnUnreadableDirectoryYieldsNothing() {
        let missing = dir.appendingPathComponent("nope", isDirectory: true)
        XCTAssertTrue(ModelDirectoryListing.entries(in: missing).isEmpty)
    }

    func testSubdirectoryPresenceIsAnsweredFromTheSameListing() {
        ModelDirectoryListing.withCache {
            XCTAssertTrue(ModelDirectoryListing.contains("MTP", in: dir))
            XCTAssertFalse(ModelDirectoryListing.contains("absent", in: dir))
        }
    }
}
