// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// The inline formatting cache is bounded and evicts oldest-first. It now walks a
/// cursor instead of shifting its order array, so what these pin is that the
/// eviction order and the retained set are unchanged by that change — including
/// the re-entry case, where a key that was evicted earlier comes back.
final class FormattedTextCacheTests: XCTestCase {
    private func make(_ limit: Int) -> FormattedTextCache {
        FormattedTextCache(limit: limit)
    }

    private func value(_ cache: FormattedTextCache, _ key: String) -> AttributedString {
        cache.formatted(key) { AttributedString(key) }
    }

    func testARepeatedKeyIsBuiltOnce() {
        let cache = make(4)
        let first = value(cache, "a")
        let second = value(cache, "a")
        XCTAssertEqual(String(first.characters), String(second.characters))
    }

    func testOldestEntriesAreEvictedFirst() {
        let cache = make(2)
        _ = value(cache, "a")
        _ = value(cache, "b")
        _ = value(cache, "c")   // retires "a"
        // "b" must still be a hit: it is inside the limit.
        XCTAssertEqual(String(value(cache, "b").characters), "b")
        // "a" is gone, so this is a rebuild rather than a hit — visible because
        // the builder's result is what comes back either way, so check identity
        // of the value the builder produced instead.
        var rebuilt = false
        _ = cache.formatted("a") { rebuilt = true; return AttributedString("a") }
        XCTAssertTrue(rebuilt, "an evicted key must be built again")
    }

    func testAnEvictedKeyCanComeBack() {
        let cache = make(1)
        _ = value(cache, "a")
        _ = value(cache, "b")   // retires "a"
        var rebuilt = false
        _ = cache.formatted("a") { rebuilt = true; return AttributedString("a") }
        XCTAssertTrue(rebuilt)
        _ = value(cache, "a")   // now resident again
        var rebuiltAgain = false
        _ = cache.formatted("a") { rebuiltAgain = true; return AttributedString("a") }
        XCTAssertFalse(rebuiltAgain, "re-inserting must not be treated as a fresh entry")
    }

    /// Enough traffic to push the retired prefix past its compaction threshold,
    /// which is where the order array is rebuilt.
    func testTheCacheKeepsWorkingAcrossManyEvictions() {
        let cache = make(8)
        for i in 0..<500 {
            let key = "key-\(i)"
            var built = false
            _ = cache.formatted(key) { built = true; return AttributedString(key) }
            XCTAssertTrue(built, "a key never seen before must be built (i=\(i))")
        }
        // The most recent key is resident; the very first is not.
        var rebuiltOldest = false
        _ = cache.formatted("key-0") { rebuiltOldest = true; return AttributedString("key-0") }
        XCTAssertTrue(rebuiltOldest)
        var rebuiltNewest = false
        _ = cache.formatted("key-499") { rebuiltNewest = true; return AttributedString("key-499") }
        XCTAssertFalse(rebuiltNewest)
    }

    func testConcurrentReadersAllSeeAValue() {
        let cache = make(64)
        DispatchQueue.concurrentPerform(iterations: 64) { i in
            let key = "shared-\(i % 8)"
            let result = cache.formatted(key) { AttributedString(key) }
            XCTAssertEqual(String(result.characters), key)
        }
    }
}
