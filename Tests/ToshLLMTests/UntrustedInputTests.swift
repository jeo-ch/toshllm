import XCTest
@testable import ToshLLM

/// The fixes in this file are all about values arriving from outside the app:
/// a JSON null where a nil was assumed, an index with no bound, a project file
/// carrying a duration no clock could hold, and a Prometheus value that is not an
/// integer. Each one used to crash, abort a stream, or silently read as absent.
final class UntrustedInputTests: XCTestCase {

    // MARK: - An explicit JSON null is not an error

    /// `"error": null` reaches Swift as NSNull, not nil, so the guard fell through
    /// to the generic message and threw — ending the stream on every chunk of any
    /// OpenAI-compatible server that always includes the field.
    func testStreamedErrorIgnoresAnExplicitNull() {
        XCTAssertNil(ChatStore.streamedError(from: ["error": NSNull()]))
        XCTAssertNil(ChatStore.streamedError(from: [:]))
        XCTAssertEqual(ChatStore.streamedError(from: ["error": "boom"]), "boom")
        XCTAssertEqual(ChatStore.streamedError(from: ["error": ["message": "boom"]]), "boom")
    }

    // MARK: - Tool call indices are bounded

    func testANegativeOrHugeToolCallIndexIsIgnored() throws {
        var accumulator = ChatStreamAccumulator()
        func consume(_ index: Int) throws {
            let json = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":"
                + "\(index),\"function\":{\"name\":\"t\",\"arguments\":\"{}\"}}]}}]}"
            try accumulator.consume(json)
        }
        try consume(-1)
        XCTAssertTrue(accumulator.toolCalls.isEmpty, "a negative index must not index out of range")
        try consume(100_000_000)
        XCTAssertTrue(accumulator.toolCalls.isEmpty, "a huge index must not allocate for it")
        try consume(0)
        XCTAssertEqual(accumulator.toolCalls.count, 1)
        XCTAssertEqual(accumulator.toolCalls.first?.name, "t")
    }

    // MARK: - Durations out of range

    /// Int(Double) traps on NaN, on infinity and past Int64. Cue times come
    /// straight out of a project file, so "start": 1e17 was enough to crash the
    /// export.
    func testTimestampSurvivesValuesNoClockCouldHold() {
        for seconds in [Double.nan, .infinity, -.infinity, 1e17, -1e17, 1e300] {
            let text = SubtitleCue.timestamp(seconds, decimal: ",")
            XCTAssertFalse(text.isEmpty, "\(seconds) produced no timestamp")
        }
        XCTAssertEqual(SubtitleCue.timestamp(0, decimal: ","), "00:00:00,000")
        XCTAssertEqual(SubtitleCue.timestamp(3661.5, decimal: ","), "01:01:01,500")
    }

    // MARK: - Prometheus values are not integers

    func testMetricCountersWrittenAsFloatsStillParse() {
        let text = """
        # HELP llamacpp:spec_decode_num_draft_tokens_total drafted
        # TYPE llamacpp:spec_decode_num_draft_tokens_total counter
        llamacpp:spec_decode_num_draft_tokens_total 337.0
        llamacpp:spec_decode_num_accepted_tokens_total 101.0
        llamacpp:spec_decode_num_drafts_total 12
        llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 50.0
        llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 51.0
        """
        let metrics = SpecDecodeMetrics.parse(text)
        XCTAssertEqual(metrics.draftTokens, 337)
        XCTAssertEqual(metrics.acceptedTokens, 101)
        XCTAssertEqual(metrics.drafts, 12)
        XCTAssertEqual(metrics.acceptedPerPosition, [50, 51])
        XCTAssertTrue(metrics.ran)
    }

    func testNonFiniteAndNegativeCounterValuesAreSkipped() {
        let text = """
        llamacpp:spec_decode_num_draft_tokens_total +Inf
        llamacpp:spec_decode_num_accepted_tokens_total NaN
        llamacpp:spec_decode_num_drafts_total -5
        """
        let metrics = SpecDecodeMetrics.parse(text)
        XCTAssertEqual(metrics.draftTokens, 0)
        XCTAssertFalse(metrics.ran)
    }

    /// 0...highest traps when the label is negative, and nothing constrains a
    /// number that came off the wire.
    func testANegativeOrHugePositionLabelDoesNotCrash() {
        for position in ["-1", "999999"] {
            let text = """
            llamacpp:spec_decode_num_draft_tokens_total 10
            llamacpp:spec_decode_num_accepted_tokens_total 5
            llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="\(position)"} 1
            """
            let metrics = SpecDecodeMetrics.parse(text)
            XCTAssertTrue(metrics.acceptedPerPosition.allSatisfy { $0 >= 0 },
                          "position=\(position) produced \(metrics.acceptedPerPosition.count) slots")
        }
    }

    func testAcceptanceIsNeverAboveOneHundredPercent() {
        var metrics = SpecDecodeMetrics()
        metrics.draftTokens = 10
        metrics.acceptedTokens = 15
        XCTAssertEqual(metrics.acceptance, 1)
    }

    // MARK: - The recall check asks for the same thing chat asks for

    /// Sending only enable_thinking left 32 tokens of prefill to be consumed by
    /// reasoning, so a reasoning model returned an empty answer and the grid
    /// reported a lost context — the one conclusion that card must not draw
    /// wrongly.
    @MainActor
    func testTheRecallCheckClosesTheReasoningBlockLikeChatDoes() {
        let body = NeedleTest.requestBody(prompt: "needle")
        XCTAssertEqual((body["chat_template_kwargs"] as? [String: Bool])?["enable_thinking"], false)
        XCTAssertEqual(body["thinking_budget_tokens"] as? Int, 0)
        XCTAssertEqual(body["max_tokens"] as? Int, 32)
    }
}