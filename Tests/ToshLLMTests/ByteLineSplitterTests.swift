import XCTest
@testable import ToshLLM

/// The resume offset has to be a position in the byte stream. It used to be built
/// from `AsyncBytes.lines`, which strips the terminator, so every line was counted
/// as one byte longer than it was not — under CRLF the count drifted low, the
/// engine resumed from before what it had already sent, and the answer came back
/// with a duplicated passage. The offset is the whole point of the resume, so it
/// is pinned here against the raw bytes for both terminators.
final class ByteLineSplitterTests: XCTestCase {
    private func lines(_ text: String) -> [ByteLineSplitter.Line] {
        var splitter = ByteLineSplitter()
        return splitter.push(Data(text.utf8)) + splitter.flush()
    }

    func testCountsTheTerminatorItActuallySaw() {
        let lf = lines("data: one\ndata: two\n")
        XCTAssertEqual(lf.map(\.text), ["data: one", "data: two"])
        XCTAssertEqual(lf.map(\.bytes), [10, 10])   // 9 characters + LF

        // Same content, two-byte terminators: every line is exactly one byte more.
        // This is the whole point — the old count assumed one byte for both.
        let crlf = lines("data: one\r\ndata: two\r\n")
        XCTAssertEqual(crlf.map(\.text), ["data: one", "data: two"])
        XCTAssertEqual(crlf.map(\.bytes), [11, 11])
        XCTAssertEqual(crlf.map(\.bytes), lf.map(\.bytes).map { $0 + 1 })
    }

    func testTheRunningTotalIsThePositionInTheStream() {
        for terminator in ["\n", "\r\n"] {
            let stream = (0..<50).map { "data: line \($0)\(terminator)" }.joined()
            let total = lines(stream).reduce(0) { $0 + $1.bytes }
            XCTAssertEqual(total, Data(stream.utf8).count,
                           "the offset must land exactly at the end of a \(terminator == "\n" ? "LF" : "CRLF") stream")
        }
    }

    func testASingleUnterminatedTailIsStillReturned() {
        let result = lines("data: [DONE]")
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].text, "data: [DONE]")
        XCTAssertEqual(result[0].bytes, Data("data: [DONE]".utf8).count)
    }

    func testEmptyLinesAreKept() {
        let result = lines("a\n\nb\n")
        XCTAssertEqual(result.map(\.text), ["a", "", "b"])
        XCTAssertEqual(result.map(\.bytes), [2, 1, 2])
    }

    func testALoneCarriageReturnIsNotATerminator() {
        // Only CRLF counts; a bare CR stays in the text.
        let result = lines("a\rb\n")
        XCTAssertEqual(result.map(\.text), ["a\rb"])
        XCTAssertEqual(result[0].bytes, 4)
    }

    /// Splitting across chunk boundaries is the normal case, not an edge one: the
    /// bytes arrive in whatever sizes the network chose.
    func testASplitAcrossChunksProducesTheSameLines() {
        let stream = "data: alpha\r\ndata: beta\r\ndata: gamma"
        let whole = lines(stream)
        for chunkSize in 1...7 {
            var splitter = ByteLineSplitter()
            var split: [ByteLineSplitter.Line] = []
            let bytes = Array(stream.utf8)
            var start = 0
            while start < bytes.count {
                let end = min(start + chunkSize, bytes.count)
                split += splitter.push(Data(bytes[start..<end]))
                start = end
            }
            split += splitter.flush()
            XCTAssertEqual(split.map(\.text), whole.map(\.text), "chunk size \(chunkSize)")
            XCTAssertEqual(split.map(\.bytes), whole.map(\.bytes), "chunk size \(chunkSize)")
        }
    }

    func testMultibyteTextSurvivesTheSplit() {
        // Byte counting is byte counting: the lengths are utf8, not characters.
        let result = lines("data: 日本語\n")
        XCTAssertEqual(result[0].text, "data: 日本語")
        XCTAssertEqual(result[0].bytes, Data("data: 日本語\n".utf8).count)
    }

    func testResumeURLRejectsANegativeOffset() {
        XCTAssertNil(ChatStreamIdentity.resumeURL(port: 8080, identity: "id", from: -1))
        XCTAssertNotNil(ChatStreamIdentity.resumeURL(port: 8080, identity: "id", from: 0))
        XCTAssertNotNil(ChatStreamIdentity.resumeURL(port: 8080, identity: "id", from: 4096))
    }
}