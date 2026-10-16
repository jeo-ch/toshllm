// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A model can be many gigabytes. Removing the old file and then moving the new
/// one is not atomic, so a failed move used to leave the user with neither.
final class ModelInstallTests: XCTestCase {

    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-install-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ name: String, _ body: String) throws -> URL {
        let url = dir.appendingPathComponent(name)
        try Data(body.utf8).write(to: url)
        return url
    }

    private func read(_ url: URL) throws -> String {
        String(decoding: try Data(contentsOf: url), as: UTF8.self)
    }

    func testMovesWhenThereIsNothingToReplace() throws {
        let staging = try write("staging.gguf", "new")
        let destination = dir.appendingPathComponent("model.gguf")
        try DownloadItem.installStaged(staging, at: destination)
        XCTAssertEqual(try read(destination), "new")
        XCTAssertFalse(FileManager.default.fileExists(atPath: staging.path))
    }

    func testReplacesAnExistingFileInPlace() throws {
        let staging = try write("staging.gguf", "new")
        let destination = try write("model.gguf", "old")
        try DownloadItem.installStaged(staging, at: destination)
        XCTAssertEqual(try read(destination), "new")
    }

    /// The whole point of the atomic swap: when it cannot complete, the file the
    /// user already had must still be there and still be theirs.
    func testAFailureLeavesTheExistingFileUntouched() throws {
        let staging = dir.appendingPathComponent("never-written.gguf")
        let destination = try write("model.gguf", "old")
        XCTAssertThrowsError(try DownloadItem.installStaged(staging, at: destination))
        XCTAssertEqual(try read(destination), "old")
    }

    func testRepeatedInstallsKeepReplacing() throws {
        let destination = dir.appendingPathComponent("model.gguf")
        for body in ["one", "two", "three"] {
            let staging = try write("staging-\(body).gguf", body)
            try DownloadItem.installStaged(staging, at: destination)
        }
        XCTAssertEqual(try read(destination), "three")
    }
}
