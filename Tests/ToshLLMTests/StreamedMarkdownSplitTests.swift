// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// While an answer streams, the renderer splits it into a part that will not
/// change again and a tail that still might. The split walks every line of the
/// answer on each tick, so the fence rule it follows is pinned here against an
/// independent implementation — the awkward cases are an unterminated fence, a
/// fence opening below a blank line, a longer fence inside a shorter one, and a
/// message that opens with a fence at all.
///
/// An incremental version of this scan was written and withdrawn; these cases are
/// what it failed on, and they are kept so the rule cannot drift.
final class StreamedMarkdownSplitTests: XCTestCase {
    /// An independent statement of the rule, written from the prose above rather
    /// than from the code, so the two can disagree.
    private func reference(_ text: String) -> (settled: String, tail: String) {
        let lines = text.components(separatedBy: "\n")
        var inFence = false
        var fenceChar: Character = "`"
        var fenceLength = 0
        var settledLines = 0
        for (i, line) in lines.enumerated() {
            guard let f = RichText.fenceInfo(line) else {
                if !inFence, line.trimmingCharacters(in: .whitespaces).isEmpty {
                    settledLines = i + 1
                }
                continue
            }
            if !inFence {
                inFence = true
                fenceChar = f.char
                fenceLength = f.length
            } else if f.info.isEmpty, f.char == fenceChar, f.length >= fenceLength {
                inFence = false
            }
        }
        return (lines[..<settledLines].joined(separator: "\n"),
                lines[settledLines...].joined(separator: "\n"))
    }

    /// Grows the text one line at a time, the way tokens arrive, and compares at
    /// every step.
    private func assertAgreesAtEveryStep(_ fragments: [String],
                                         file: StaticString = #filePath, line: UInt = #line) {
        var text = ""
        for fragment in fragments {
            text = text.isEmpty ? fragment : text + "\n" + fragment
            let result = RichText.splitSettled(text)
            let expected = reference(text)
            XCTAssertEqual(result.settled, expected.settled,
                           "settled after \(text.debugDescription)", file: file, line: line)
            XCTAssertEqual(result.tail, expected.tail,
                           "tail after \(text.debugDescription)", file: file, line: line)
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

    /// The case an incremental scan gets wrong: a blank line is settled, and only
    /// afterwards does a fence open below it.
    func testAFenceOpeningAfterABlankLineDoesNotRetractIt() {
        assertAgreesAtEveryStep(["para", "", "```", "code", "", "tail"])
    }

    func testLongerFenceInsideShorterOneIsContent() {
        assertAgreesAtEveryStep(["a", "", "```", "````", "", "b", "````", "", "c"])
    }

    func testTildeFences() {
        assertAgreesAtEveryStep(["a", "", "~~~", "code", "~~~", "", "b"])
    }

    /// A message that opens with a fence has every blank line inside it.
    func testAMessageOpeningWithAFenceSettlesNothing() {
        assertAgreesAtEveryStep(["```", "", "code", "", "more code"])
        assertAgreesAtEveryStep(["```swift", "let x = 1", "```", "", "after"])
        assertAgreesAtEveryStep(["~~~", "", "code"])
    }

    func testAnEmptyStringYieldsNothingSettled() {
        let result = RichText.splitSettled("")
        XCTAssertEqual(result.settled, "")
        XCTAssertEqual(result.tail, "")
    }

    func testASingleLineWithoutABreakIsAllTail() {
        let result = RichText.splitSettled("just one line")
        XCTAssertEqual(result.settled, "")
        XCTAssertEqual(result.tail, "just one line")
    }
}
