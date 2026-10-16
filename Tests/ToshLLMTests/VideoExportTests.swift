// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AVFoundation
import XCTest
@testable import ToshLLM

final class VideoExportTests: XCTestCase {

    /// The loop that waits for the input to drain used to look only at readiness.
    /// A writer that has failed never becomes ready again, so the wait never ended.
    func testAFailedWriterEndsTheWait() {
        XCTAssertFalse(VideoExporter.shouldWaitForInput(writerStatus: .failed, inputReady: false),
                       "a failed writer would never become ready; waiting hangs the export")
        XCTAssertFalse(VideoExporter.shouldWaitForInput(writerStatus: .failed, inputReady: true))
    }

    func testAHealthyWriterStillWaitsForRoom() {
        XCTAssertTrue(VideoExporter.shouldWaitForInput(writerStatus: .writing, inputReady: false))
        XCTAssertFalse(VideoExporter.shouldWaitForInput(writerStatus: .writing, inputReady: true))
    }

    /// Only failure ends the wait early. Every other state keeps the original
    /// readiness-based behaviour, so the fix cannot silently shorten a healthy
    /// export.
    func testOnlyFailureEndsTheWaitEarly() {
        for status in [AVAssetWriter.Status.writing, .completed, .cancelled, .unknown] {
            XCTAssertTrue(VideoExporter.shouldWaitForInput(writerStatus: status, inputReady: false),
                          "state \(status.rawValue) must keep waiting for readiness")
            XCTAssertFalse(VideoExporter.shouldWaitForInput(writerStatus: status, inputReady: true))
        }
    }

    private func frame(_ size: Int = 64) -> NSImage {
        let image = NSImage(size: NSSize(width: size, height: size))
        image.lockFocus()
        NSColor.red.setFill()
        NSRect(x: 0, y: 0, width: size, height: size).fill()
        image.unlockFocus()
        return image
    }

    /// The destination is unwritable, so the writer fails after starting. The
    /// export has to report that instead of writing a file it never filled.
    func testAnUnwritableDestinationThrows() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-mp4-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
            try? FileManager.default.removeItem(at: dir)
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir.path)
        let out = dir.appendingPathComponent("out.mp4")

        XCTAssertThrowsError(try VideoExporter.writeMP4(frames: [frame()], fps: 8, to: out))
    }

    func testAWritableDestinationProducesAFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-mp4-ok-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = dir.appendingPathComponent("out.mp4")

        try VideoExporter.writeMP4(frames: [frame(), frame(), frame()], fps: 8, to: out)

        let size = try FileManager.default.attributesOfItem(atPath: out.path)[.size] as? Int64 ?? 0
        XCTAssertGreaterThan(size, 0, "the export produced no container")
    }
}
