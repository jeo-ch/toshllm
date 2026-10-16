// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// Regression tests for two integer traps in the GGUF header scanner. Both are
/// reachable from a hostile or merely truncated file, which the app mmaps from
/// anywhere the user points the library at.
final class GGUFHeaderGuardTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("toshllm-gguf-guard-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func write(_ bytes: [UInt8], to url: URL) throws {
        try Data(bytes).write(to: url)
    }

    private func littleEndian(_ value: UInt64) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }

    private func littleEndian(_ value: UInt32) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian) { Array($0) }
    }

    // MARK: - scanForNextNTensor descending-range trap

    /// `.nextn.` landing inside the first eight bytes used to build `8...lowerBound`,
    /// a descending closed range that traps. Byte 4 is the GGUF version field, so a
    /// crafted header reaches it without breaking the magic.
    func testNextNNeedleInsideHeaderPrefixDoesNotTrap() throws {
        let dir = try temporaryDirectory()
        let url = dir.appendingPathComponent("needle-in-prefix.gguf")
        var bytes: [UInt8] = Array("GGUF".utf8)
        bytes += Array(".nextn.".utf8)      // offset 4 — inside the first eight bytes
        bytes += Array(repeating: 0, count: 64)
        try write(bytes, to: url)

        XCTAssertFalse(GGUFMetadataCache.hasNextNTensor(at: url.path))
    }

    /// Same trap with the needle at offset 0, where `readPrefix` still returns the
    /// mapping because it does not validate the magic.
    func testNextNNeedleAtFileStartDoesNotTrap() throws {
        let dir = try temporaryDirectory()
        let url = dir.appendingPathComponent("needle-at-zero.gguf")
        var bytes: [UInt8] = Array(".nextn.".utf8)
        bytes += Array(repeating: 0, count: 64)
        try write(bytes, to: url)

        XCTAssertFalse(GGUFMetadataCache.hasNextNTensor(at: url.path))
    }

    /// The guard must not blind the scanner: a well-formed name preceded by its
    /// 8-byte length prefix is still detected.
    func testWellFormedNextNTensorStillDetected() throws {
        let dir = try temporaryDirectory()
        let url = dir.appendingPathComponent("well-formed.gguf")
        let name = Array("blk.0.nextn.x".utf8)          // contains ".nextn."
        let nameStart = 100

        var bytes: [UInt8] = []
        bytes += Array("GGUF".utf8)
        bytes += littleEndian(UInt32(3))
        bytes += littleEndian(UInt64(0))                 // tensor count (unused here)
        bytes += littleEndian(UInt64(0))                 // metadata count
        bytes += Array(repeating: 0, count: nameStart - 8 - bytes.count)
        bytes += littleEndian(UInt64(UInt64(name.count))) // 8-byte length prefix
        bytes += name
        bytes += littleEndian(UInt32(2))                 // dimensions
        bytes += Array(repeating: 0, count: 64)          // descriptor + slack
        try write(bytes, to: url)

        XCTAssertTrue(GGUFMetadataCache.hasNextNTensor(at: url.path))
    }

    // MARK: - parseTensorFlagsOne byte-estimate trap

    /// `UInt64(Double(elements) * bytesPerElement)` traps once the product leaves
    /// UInt64's range. Here the element count is representable but the product is not.
    func testOversizedTensorByteEstimateDoesNotTrap() throws {
        let dir = try temporaryDirectory()
        let url = dir.appendingPathComponent("oversized-tensor.gguf")
        let name = Array("blk.0.attn_output.weight".utf8)

        var bytes: [UInt8] = []
        bytes += Array("GGUF".utf8)
        bytes += littleEndian(UInt32(3))
        bytes += littleEndian(UInt64(1))                 // one tensor
        bytes += littleEndian(UInt64(0))                 // no metadata
        bytes += littleEndian(UInt64(UInt64(name.count))) // tensor-name length prefix
        bytes += Array(name)
        bytes += littleEndian(UInt32(2))                 // two dimensions
        bytes += littleEndian(UInt64(1) << 62)           // 2^62
        bytes += littleEndian(UInt64(3))                 // 2^62 * 3, still < 2^64
        bytes += littleEndian(UInt32(0))                 // F32 => 4 bytes each
        bytes += littleEndian(UInt64(0))                 // offset
        try write(bytes, to: url)

        let flags = GGUFMetadataCache.tensorFlags(at: url.path)
        // Saturated, not trapped: the estimate is absurd by construction.
        XCTAssertEqual(flags.expertBytesPerLayer, 0)
        XCTAssertEqual(flags.otherBytesPerLayer, UInt64.max)
    }

    /// A well-formed tensor still yields the exact byte estimate.
    func testOrdinaryTensorByteEstimateUnchanged() throws {
        let dir = try temporaryDirectory()
        let url = dir.appendingPathComponent("ordinary-tensor.gguf")
        let name = Array("blk.0.attn_output.weight".utf8)

        var bytes: [UInt8] = []
        bytes += Array("GGUF".utf8)
        bytes += littleEndian(UInt32(3))
        bytes += littleEndian(UInt64(1))
        bytes += littleEndian(UInt64(0))
        bytes += littleEndian(UInt64(UInt64(name.count)))
        bytes += Array(name)
        bytes += littleEndian(UInt32(2))
        bytes += littleEndian(UInt64(4096))
        bytes += littleEndian(UInt64(4096))
        bytes += littleEndian(UInt32(0))                 // F32
        bytes += littleEndian(UInt64(0))
        try write(bytes, to: url)

        let flags = GGUFMetadataCache.tensorFlags(at: url.path)
        XCTAssertEqual(flags.otherBytesPerLayer, 4096 * 4096 * 4, "F32 4096x4096 is 64 MiB")
        XCTAssertEqual(flags.expertBytesPerLayer, 0)
    }
}
