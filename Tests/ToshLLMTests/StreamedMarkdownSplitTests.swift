// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// While an answer streams, the renderer splits it into a part that will not
/// change again and a tail that still might. The split used to be recomputed from
/// the whole text on every tick; it is now resumed from where the previous pass
/// stopped, which is only sound if the two agree at every step. These tests pin
/// that agreement over the awkward cases: an unterminated fence, a blank line
/// that a later fence could have swallowed, and text replaced rather than grown.
final class StreamedMarkdownSplitTests: XCTestCase {
    /// The straightforward definition the incremental one has to reproduce.
    private func reference(_ text: String) -> (settled: String, tail: String) {
        let lines = text.components(separatedBy: "\n")
        var inFence = false
        var fenceChar: Character = "`"
        var fenceLength = 0
        var settledLines = 0
        for (i, line) in lines.enumerated() {
            if RichText.fenceInfoForTest(line) != nil {
                let f = RichText.fenceInfoForTest(line)!
                if !inFence {
                    inFence = true; fenceChar = f.char; fenceLength = f.length
                } else if f.info.isEmpty, f.char == fenceChar, f.length >= fenceLength {
                    inFence = false
                }
            } else if !inFence, line.trimmingCharacters(in: .whitespaces).isEmpty {
                settledLines = i + 1
            }
        }
        let settled = lines[..<settledLines].joined(separator: "\n")
        let tail = lines[settledLines...].joined(separator: "\n")
        return (settled, tail)
    }

    /// Grows the text one line at a time, the way tokens arrive, and compares at
    /// every step against a fresh full scan.
    private func assertAgreesAtEveryStep(_ lines: [String], file: StaticString = #filePath, line: UInt = #line) {
        let boundary = RichText.settledBoundaryForTest()
        var text = ""
        for fragment in lines {
            text = text.isEmpty ? fragment : text + "\n" + fragment
            let incremental = boundary.split(text)
            let expected = reference(text)
            XCTAssertEqual(incremental.settled, expected.settled, "settled after \(text.debugDescription)", file: file, line: line)
            XCTAssertEqual(incremental.tail, expected.tail, "tail after \(text.debugDescription)", file: file, line: line)
        }
    }

    func testPlainProseSettlesUpToTheLastBlankLine() {
        assertAgreesAtEveryStep(["one", "", "two", "three", "", "four"])
    }

    /// A fence that opens and never closes swallows the blank lines after it:
    /// nothing after it may be treated as settled.
    func testAnUnterminatedFenceSettlesNothingAfterIt() {
        assertAgreesAtEveryStep(["intro", "", "```swift", "let x = 1", "", "still code"])
    }

    func testAClosedFenceLetsSettlingResumeAfterIt() {
        assertAgreesAtEveryStep(["intro", "", "```", "code", "```", "", "after"])
    }

    /// The case the incremental scan could plausibly get wrong: a blank line is
    /// settled, and only afterwards does a fence open below it.
    func testAFenceOpeningAfterABlankLineDoesNotRetractIt() {
        assertAgreesAtEveryStep(["para", "", "```", "code", "", "tail"])
    }

    func testLongerFenceInsideShorterOneIsContent() {
        assertAgreesAtEveryStep(["a", "", "```", "````", "", "b", "````", "", "c"])
    }

    func testTildeFences() {
        assertAgreesAtEveryStep(["a", "", "~~~", "code", "~~~", "", "b"])
    }

    /// Text can be replaced wholesale (a regeneration, a switch of model), which
    /// is shorter than what was there before. The scan has to notice and restart.
    func testReplacingTheTextWithShorterContentIsHandled() {
        let boundary = RichText.settledBoundaryForTest()
        let long = (1...40).map { "line \($0)" }.joined(separator: "\n\n")
        _ = boundary.split(long)
        let short = "just one line"
        let after = boundary.split(short)
        XCTAssertEqual(after.settled, reference(short).settled)
        XCTAssertEqual(after.tail, reference(short).tail)
    }

    func testAnEmptyStringYieldsNothingSettled() {
        let boundary = RichText.settledBoundaryForTest()
        let result = boundary.split("")
        XCTAssertEqual(result.settled, "")
        XCTAssertEqual(result.tail, "")
    }
}
