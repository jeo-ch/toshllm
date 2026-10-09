import XCTest
@testable import ToshLLM

/// Copying the live transcript onto each conversation's active branch used to walk
/// a per-conversation `firstIndex`, which made every keystroke's save
/// O(conversations x branches) on the main thread. A map built over the whole list
/// fixed that — and the map is global, so its indices belong to whichever
/// conversation wrote them last.
final class BranchSlotTests: XCTestCase {

    private let shared = UUID()
    private let other = UUID()

    private func branches(_ names: [String], ids: [UUID]) -> [ChatBranch] {
        zip(ids, names).map { ChatBranch(id: $0.0, name: $0.1, messages: [], created: Date()) }
    }

    /// The whole point of the hint: the common case costs one comparison.
    func testACorrectHintIsUsed() {
        let b = branches(["one", "two", "three"], ids: [UUID(), shared, UUID()])
        XCTAssertEqual(ChatStore.branchSlot(shared, in: b, hint: 1), 1)
    }

    /// The case a bounds check alone missed. The hint is in range but holds a
    /// different branch, because another conversation wrote it: taking it would
    /// have copied the transcript onto the wrong branch.
    func testAHintInRangeButHoldingAnotherBranchIsNotUsed() {
        let b = branches(["other", "real"], ids: [other, shared])
        XCTAssertEqual(ChatStore.branchSlot(shared, in: b, hint: 0), 1,
                       "index 0 holds a different branch; the answer must come from the scan")
    }

    /// And the other direction: out of range used to skip the conversation entirely,
    /// leaving its transcript unsynced without a word.
    func testAHintOutOfRangeFallsBackToTheScan() {
        let b = branches(["a", "b"], ids: [shared, UUID()])
        XCTAssertEqual(ChatStore.branchSlot(shared, in: b, hint: 7), 0)
    }

    func testNoHintMeansAScan() {
        let b = branches(["a", "b", "c"], ids: [UUID(), UUID(), shared])
        XCTAssertEqual(ChatStore.branchSlot(shared, in: b, hint: nil), 2)
    }

    func testAnAbsentBranchIsNilEitherWay() {
        let b = branches(["a"], ids: [UUID()])
        XCTAssertNil(ChatStore.branchSlot(shared, in: b, hint: nil))
        XCTAssertNil(ChatStore.branchSlot(shared, in: b, hint: 0))
        XCTAssertNil(ChatStore.branchSlot(shared, in: [], hint: 0))
    }
}