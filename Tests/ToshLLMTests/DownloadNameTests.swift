// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// A model download writes next to its destination and then moves the finished
/// file into place. If the name is `..`, or carries a separator, both paths land
/// outside the folder the app manages.
final class DownloadNameTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-name-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testAnOrdinaryNamePasses() {
        XCTAssertEqual(DownloadName.safe("Qwen3-4B-Q4_K_M.gguf"), "Qwen3-4B-Q4_K_M.gguf")
        XCTAssertEqual(DownloadName.safe("  model.gguf  "), "model.gguf")
        XCTAssertEqual(DownloadName.safe("mmproj-F16.gguf"), "mmproj-F16.gguf")
    }

    func testDotSegmentsAreRejected() {
        for name in ["", " ", ".", "..", ".hidden.gguf"] {
            XCTAssertNil(DownloadName.safe(name), "\(name.debugDescription) must be rejected")
        }
    }

    func testSeparatorsAndControlCharactersAreRejected() {
        for name in ["a/b.gguf", "../escape.gguf", "..\\escape.gguf",
                     "a:b.gguf", "a\u{0}b.gguf", "a\nb.gguf"] {
            XCTAssertNil(DownloadName.safe(name), "\(name.debugDescription) must be rejected")
        }
    }

    /// `URL.lastPathComponent` decodes percent-escapes before the name reaches
    /// the downloader, so a pasted `%2e%2e` is really `..`.
    func testPercentDecodedTraversalIsRejected() {
        let url = URL(string: "https://huggingface.co/owner/repo/resolve/main/%2e%2e")!
        XCTAssertEqual(url.lastPathComponent, "..")
        XCTAssertNil(DownloadName.safe(url.lastPathComponent))
    }

    func testANameTooLongForTheFilesystemIsRejected() {
        XCTAssertNil(DownloadName.safe(String(repeating: "a", count: 256) + ".gguf"))
        XCTAssertNotNil(DownloadName.safe(String(repeating: "a", count: 200) + ".gguf"))
    }

    func testTheDestinationStaysDirectlyInsideTheFolder() throws {
        let dest = try XCTUnwrap(DownloadName.destination("model.gguf", in: root))
        XCTAssertEqual(dest.deletingLastPathComponent().standardizedFileURL.path,
                       root.standardizedFileURL.path)

        XCTAssertNil(DownloadName.destination("..", in: root))
        XCTAssertNil(DownloadName.destination("../escape.gguf", in: root))
        XCTAssertNil(DownloadName.destination("sub/model.gguf", in: root))
    }

    func testAProjectorPathFromAListingMustBeRelativeAndContained() {
        XCTAssertEqual(DownloadName.safeRelativePath("mmproj-F16.gguf"), "mmproj-F16.gguf")
        XCTAssertEqual(DownloadName.safeRelativePath("sub dir/mmproj-Q8_0.gguf"),
                       "sub dir/mmproj-Q8_0.gguf")

        for path in ["", "/mmproj.gguf", "../mmproj.gguf", "a/../mmproj.gguf",
                     "a//mmproj.gguf", "a?x", "a#x", "a%2e%2e", "..\\mmproj.gguf"] {
            XCTAssertNil(DownloadName.safeRelativePath(path), "\(path.debugDescription) must be rejected")
        }
    }
}
