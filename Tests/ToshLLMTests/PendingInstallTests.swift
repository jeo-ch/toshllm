import XCTest
@testable import ToshLLM

/// The updater used to delete the copy it replaced as soon as the new one was on
/// disk. An app that would not start then left nothing to go back to, and the
/// user had to fetch and install the previous release by hand.
final class PendingInstallTests: XCTestCase {

    private var recordURL: URL { UpdateChecker.pendingInstallURL() }
    private var scratch: URL!

    override func setUpWithError() throws {
        try? FileManager.default.removeItem(at: recordURL)
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-pending-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: recordURL)
        try? FileManager.default.removeItem(at: scratch)
    }

    @discardableResult
    private func writeRecord(installed: String, replaced: String) -> String {
        let record = UpdateChecker.PendingInstall(
            installedVersion: installed, replaced: replaced, target: scratch.path + "/ToshLLM.app")
        let data = try! JSONEncoder().encode(record)
        let url = recordURL
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try! data.write(to: url)
        return replaced
    }

    private func makeFile(_ name: String) -> String {
        let url = scratch.appendingPathComponent(name)
        try! "old version".write(to: url, atomically: true, encoding: .utf8)
        return url.path
    }

    /// The point of the whole record: the copy survives until the new build is
    /// seen running.
    func testThePreviousCopyIsKeptWhileTheRecordStands() {
        let replaced = makeFile("ToshLLM.old.app")
        writeRecord(installed: "0.87.19", replaced: replaced)
        XCTAssertTrue(FileManager.default.fileExists(atPath: replaced))
        XCTAssertEqual(UpdateChecker.readPendingInstall()?.installedVersion, "0.87.19")
    }

    func testTheVersionNowRunningClearsBothTheRecordAndTheCopy() throws {
        let replaced = makeFile("ToshLLM.old.app")
        writeRecord(installed: "0.87.19", replaced: replaced)

        XCTAssertTrue(UpdateChecker.resolvePendingInstall(running: "0.87.19"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: replaced),
                       "a build that started no longer needs the copy it replaced")
        XCTAssertNil(UpdateChecker.readPendingInstall())
    }

    /// The state the record exists to describe: the install did not take.
    func testADifferentRunningVersionLeavesTheCopyAndTheRecordAlone() throws {
        let replaced = makeFile("ToshLLM.old.app")
        writeRecord(installed: "0.87.19", replaced: replaced)

        XCTAssertFalse(UpdateChecker.resolvePendingInstall(running: "0.87.18"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: replaced),
                      "an app that would not start must still be recoverable")
        XCTAssertEqual(UpdateChecker.readPendingInstall()?.installedVersion, "0.87.19")
    }

    /// A build with no version string must not count as proof it started, or the
    /// copy would be deleted on the strength of a missing Info.plist key.
    func testAnEmptyInstalledVersionIsNeverTreatedAsRunning() throws {
        let replaced = makeFile("ToshLLM.old.app")
        writeRecord(installed: "", replaced: replaced)

        XCTAssertFalse(UpdateChecker.resolvePendingInstall(running: ""))
        XCTAssertTrue(FileManager.default.fileExists(atPath: replaced))
    }

    func testRestoringPutsTheOldCopyBackInPlace() throws {
        let replaced = makeFile("ToshLLM.old.app")
        // Stand-ins for the two bundles: a file where the .app path stands for a
        // directory, which is enough since the swap only moves paths around.
        let target = scratch.appendingPathComponent("ToshLLM.app")
        try "new version".write(to: target, atomically: true, encoding: .utf8)
        writeRecord(installed: "0.87.19", replaced: replaced)

        try UpdateChecker.restorePreviousVersion()
        XCTAssertEqual(try String(contentsOf: target, encoding: .utf8), "old version")
        XCTAssertFalse(FileManager.default.fileExists(atPath: replaced))
        XCTAssertNil(UpdateChecker.readPendingInstall(), "the record is spent once restored")
    }

    func testRestoringWithNothingToRestoreIsAnErrorRatherThanALie() {
        XCTAssertThrowsError(try UpdateChecker.restorePreviousVersion())
    }
}