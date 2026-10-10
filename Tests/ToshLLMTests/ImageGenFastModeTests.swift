import XCTest
@testable import ToshLLM

/// 0.87.21 clears a fast mode that the newly selected model cannot serve. `stepScale`
/// is the only thing that reads it and has no effect there, so a selection left over
/// from the previous model would look as though it still applied.
///
/// This classifies on the same predicate `supports(_:)` does: a distilled schedule
/// (`sigmas` non-empty) reuses no step, so it accepts `.off` alone, while a plain
/// transformer still takes cacheDit/easycache/spectrum. The method is what
/// `ImageGenTab`'s `onChange(of: cfg.modelID)` calls.
final class ClearUnsupportedFastModeTests: XCTestCase {

    /// A distilled model that rejects every fast mode but off.
    private var turbo: ImageGenModel { ImageGenCatalog.qwenImage21TurboQ4 }
    /// The same architecture without the distilled schedule, so it keeps them.
    private var plain: ImageGenModel { ImageGenCatalog.qwenImage21Q4 }

    /// The tests below classify by `sigmas`; if a catalog change makes both models
    /// alike they would still pass while testing nothing in particular.
    override func setUpWithError() throws {
        try XCTSkipIf(turbo.sigmas.isEmpty,
                      "qwenImage21TurboQ4 no longer ships a schedule; the fixtures no longer describe two cases")
        try XCTSkipIf(!plain.sigmas.isEmpty,
                      "qwenImage21Q4 gained a schedule; the fixture no longer describes a plain transformer")
        try XCTSkipIf(!plain.isTransformer || !turbo.isTransformer)
    }

    func testClearsAModeTheNewModelCannotServe() {
        var cfg = ImageInstanceConfig()
        cfg.fastMode = ImageFastMode.easycache.rawValue
        XCTAssertEqual(cfg.fastModeValue, .easycache, "precondition: the mode is on before the switch")

        cfg.clearUnsupportedFastMode(turbo)

        XCTAssertEqual(cfg.fastModeValue, .off,
                       "a distilled schedule has no redundant step to reuse, so the leftover mode must go")
    }

    /// The half that makes the test real. An unconditional `fastMode = .off.rawValue`
    /// clears the first case and still fails this one.
    func testKeepsAModeTheNewModelCanServe() {
        var cfg = ImageInstanceConfig()
        cfg.fastMode = ImageFastMode.easycache.rawValue

        cfg.clearUnsupportedFastMode(plain)

        XCTAssertEqual(cfg.fastModeValue, .easycache,
                       "a plain transformer reuses steps; clearing here would be a different bug")
    }

    func testKeepsTheOtherSupportedModesOnAPlainTransformer() {
        for mode in [ImageFastMode.cacheDit, .spectrum] {
            var cfg = ImageInstanceConfig()
            cfg.fastMode = mode.rawValue
            cfg.clearUnsupportedFastMode(plain)
            XCTAssertEqual(cfg.fastModeValue, mode)
        }
    }

    func testAnAlreadyOffSelectionIsLeftAlone() {
        var cfg = ImageInstanceConfig()
        cfg.fastMode = ImageFastMode.off.rawValue

        cfg.clearUnsupportedFastMode(turbo)

        XCTAssertEqual(cfg.fastModeValue, .off)
    }

    /// The mode is cleared, not merely ignored downstream: `args(for:)` is what builds
    /// the engine command line, so an uncleared selection would ship `--cache-mode`.
    func testTheClearedSelectionStopsReachingTheCommandLine() {
        var cfg = ImageInstanceConfig()
        cfg.fastMode = ImageFastMode.easycache.rawValue
        cfg.clearUnsupportedFastMode(turbo)
        XCTAssertTrue(cfg.fastModeValue.args(for: turbo).isEmpty,
                      "an unsupported mode must not reach sd-cli as --cache-mode")
    }
}