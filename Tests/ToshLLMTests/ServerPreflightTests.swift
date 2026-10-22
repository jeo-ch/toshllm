// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

/// Startup preflight for the measured 4 GB GCN case: router context, micro-batch,
/// expert recommendation and projector overrides. No GPU or model file is needed.
final class ServerPreflightTests: XCTestCase {
    private func hardware(vramMB: Int = 4_096, ramGB: Double = 64) -> HardwareInfo {
        HardwareInfo(
            cpuBrand: "Intel Core i5-10500",
            physicalCores: 6,
            logicalCores: 12,
            ramGB: ramGB,
            arch: "x86_64",
            model: "iMac20,1",
            osVersion: "macOS",
            gpus: [GPUDevice(index: 0, name: "AMD Radeon Pro 575X", vramMB: vramMB)])
    }

    private func moeSpec() -> ModelSpec {
        ModelSpec(
            fileGB: 17.3,
            paramsB: 30,
            layers: 48,
            isMoE: true,
            activeParamsB: 3.6,
            kvBytesPerToken: 98_304,
            expertGB: 15.5)
    }

    private func routerInput(requested: Int = 262_144, vramMB: Int = 4_096,
                             reserveMB: Int = 1_024, ngl: Int = 10,
                             ncmoe: Int = 6) -> ServerSettings.RouterContextInput {
        ServerSettings.RouterContextInput(
            requested: requested,
            vramMB: vramMB,
            reserveMB: reserveMB,
            modelBytes: Int64(17.3 * 1_073_741_824),
            layers: 48,
            isMoE: true,
            expertGB: 15.5,
            ngl: ngl,
            ncmoe: ncmoe,
            kvBytesPerToken: 98_304,
            kvScale: 1)
    }

    /// The old 45% fixed resident-weight estimate returned 2048 for this input.
    /// The measured 10/48 layers + 6 expert layers on CPU serves 16k.
    func testRouterContextUsesResidentLayersInsteadOfFixed45Percent() {
        let ctx = ServerSettings.routerCtx(routerInput())
        XCTAssertEqual(ctx, 16_384)
        XCTAssertGreaterThanOrEqual(ctx, 8_192)
    }

    func testRouterContextOn12GBKeepsRequestedContext() {
        XCTAssertEqual(ServerSettings.routerCtx(routerInput(vramMB: 12_288)), 262_144)
    }

    func testClampedUbatchUsesMeasuredBoundary() {
        let low = ServerSettings.clampedUbatch(4_096, vramMB: 4_096, reserveMB: 1_024)
        XCTAssertEqual(low.value, 1_024)
        XCTAssertTrue(low.clamped)

        let mid = ServerSettings.clampedUbatch(4_096, vramMB: 12_288, reserveMB: 1_024)
        XCTAssertEqual(mid.value, 2_048)
        XCTAssertTrue(mid.clamped)

        let safe = ServerSettings.clampedUbatch(1_024, vramMB: 4_096, reserveMB: 1_024)
        XCTAssertEqual(safe.value, 1_024)
        XCTAssertFalse(safe.clamped)

        let cpu = ServerSettings.clampedUbatch(4_096, vramMB: 0, reserveMB: 1_024)
        XCTAssertEqual(cpu.value, 4_096)
        XCTAssertFalse(cpu.clamped)

        let impossible = ServerSettings.clampedUbatch(4_096, vramMB: 1_024, reserveMB: 512)
        XCTAssertEqual(impossible.value, 0)
        XCTAssertTrue(impossible.clamped)

        let off = ServerSettings.clampedUbatch(0, vramMB: 4_096, reserveMB: 1_024)
        XCTAssertEqual(off.value, 0)
        XCTAssertFalse(off.clamped)
    }

    /// Model selection and router presets must use the report's measured 6
    /// expert layers on the exact 4 GB profile, not the estimator's 10.
    func testMeasuredGCNProfileRecommendsSixExperts() {
        XCTAssertEqual(Estimator.measuredNcmoe(
            spec: moeSpec(), hw: hardware(), reserveMB: 1_024), 6)
        XCTAssertEqual(Estimator.measuredTuning(
            spec: moeSpec(), hw: hardware(), reserveMB: 1_024)?.ngl, 10)
        XCTAssertNil(Estimator.measuredNcmoe(
            spec: moeSpec(), hw: hardware(vramMB: 12_288), reserveMB: 1_024))
    }

    /// When Dynamic MoE previews its classic fallback, the app launches the
    /// measured profile instead of its 46/6-layer, Q8 KV fallback.
    func testMeasuredFallbackTurnsDynamicMoEOffAndUsesMeasuredValues() throws {
        let measured = try XCTUnwrap(HardwareTuningAdvisor.advise(HardwareTuningInput(
            gpuName: "AMD Radeon Pro 575X", vramMB: 4_096, ramGB: 64,
            physicalCores: 6, logicalCores: 12, reserveMB: 1_024,
            modelFileGB: 17.3, modelLayers: 48, isMoE: true,
            expertGB: 15.5, kvBytesPerToken: 98_304, trainedContext: nil,
            requestedCtx: 262_144, requestedUbatch: 4_096,
            currentNgl: 99, currentNcmoe: 48, currentCacheK: "q8_0",
            currentCacheV: "q8_0", currentThreads: 12,
            dynamicMoeActive: true, manualFullGPU: false, routerMode: false)))
        XCTAssertTrue(measured.measuredProfile)

        var base = ServerSettings.fromDefaults()
        base.dynamicMoeEnabled = true
        base.ngl = 99
        base.ncmoe = 48
        base.ctx = 262_144
        base.ubatch = 4_096
        base.cacheTypeK = "q8_0"
        base.cacheTypeV = "q8_0"
        base.threads = 12

        let manual = try XCTUnwrap(base.applyingMeasuredProfile(measured))
        XCTAssertFalse(manual.dynamicMoeEnabled)
        XCTAssertEqual(manual.ngl, 10)
        XCTAssertEqual(manual.ncmoe, 6)
        XCTAssertEqual(manual.ctx, 16_384)
        XCTAssertEqual(manual.ubatch, 1_024)
        XCTAssertEqual(manual.cacheTypeK, "f16")
        XCTAssertEqual(manual.cacheTypeV, "f16")
        XCTAssertEqual(manual.threads, 6)

        let wide = try XCTUnwrap(HardwareTuningAdvisor.advise(HardwareTuningInput(
            gpuName: "AMD Radeon Pro 575X", vramMB: 12_288, ramGB: 64,
            physicalCores: 6, logicalCores: 12, reserveMB: 1_024,
            modelFileGB: 17.3, modelLayers: 48, isMoE: true,
            expertGB: 15.5, kvBytesPerToken: 98_304, trainedContext: nil,
            requestedCtx: 262_144, requestedUbatch: 4_096,
            currentNgl: 99, currentNcmoe: 48, currentCacheK: "q8_0",
            currentCacheV: "q8_0", currentThreads: 12,
            dynamicMoeActive: true, manualFullGPU: false, routerMode: false)))
        XCTAssertFalse(wide.measuredProfile)
        XCTAssertNil(base.applyingMeasuredProfile(wide))
    }

    func testEstimatorCapsSuggestedNcmoeToNgl() {
        let estimate = Estimator.estimate(
            spec: moeSpec(), hw: hardware(), ctx: 262_144, kvScale: 1, nglLimit: 10)
        XCTAssertLessThanOrEqual(estimate.suggestedNcmoe, 10)
    }

    /// The over-RAM "does not fit" branch used to return block_count, which the
    /// engine then wrote as n-cpu-moe above n-gpu-layers. It must honor the cap too.
    func testEstimatorCapsSuggestionWhenModelDoesNotFitRAM() {
        let large = ModelSpec(
            fileGB: 70, paramsB: 70, layers: 80, isMoE: true,
            activeParamsB: 8, kvBytesPerToken: 163_840, expertGB: 65)
        let estimate = Estimator.estimate(
            spec: large, hw: hardware(vramMB: 4_096, ramGB: 8),
            ctx: 16_384, kvScale: 1, nglLimit: 10)
        XCTAssertEqual(estimate.level, .no)
        XCTAssertLessThanOrEqual(estimate.suggestedNcmoe, 10)
    }

    /// A 262144-token global setting must not change the estimator's recommendation;
    /// it uses the 16k estimation ceiling and the current ngl.
    func testEstimateCurrentIgnoresHugeGlobalContextForNcmoe() {
        let defaults = UserDefaults.standard
        let keys = [SettingsKeys.ctx, SettingsKeys.ngl, SettingsKeys.cacheTypeK,
                    SettingsKeys.cacheTypeV, SettingsKeys.multiGPU,
                    SettingsKeys.vramReserve]
        let old = keys.map { defaults.object(forKey: $0) }
        defer {
            for (key, value) in zip(keys, old) {
                if let value {
                    defaults.set(value, forKey: key)
                } else {
                    defaults.removeObject(forKey: key)
                }
            }
        }

        defaults.set(262_144, forKey: SettingsKeys.ctx)
        defaults.set(10, forKey: SettingsKeys.ngl)
        defaults.set("f16", forKey: SettingsKeys.cacheTypeK)
        defaults.set("f16", forKey: SettingsKeys.cacheTypeV)
        defaults.set(false, forKey: SettingsKeys.multiGPU)
        defaults.set(1_024, forKey: SettingsKeys.vramReserve)

        let current = Estimator.estimateCurrent(spec: moeSpec(), hw: hardware())
        // The measured 4 GB profile keeps 6 experts off the GPU; the huge
        // global context must neither push that to 10 nor back to block_count.
        XCTAssertEqual(current.suggestedNcmoe, 6)
        XCTAssertLessThanOrEqual(current.suggestedNcmoe, 10)

        defaults.set(5, forKey: SettingsKeys.ngl)
        XCTAssertLessThanOrEqual(
            Estimator.estimateCurrent(spec: moeSpec(), hw: hardware()).suggestedNcmoe, 5)
    }

    func testMmprojOverrideRejectsRecordedIncompatibility() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let projector = dir.appendingPathComponent("test.mmproj.gguf")
        XCTAssertTrue(FileManager.default.createFile(atPath: projector.path, contents: Data()))
        let model = dir.appendingPathComponent("model.gguf").path

        let old = UserDefaults.standard.stringArray(forKey: SettingsKeys.incompatibleMmproj)
        defer {
            if let old {
                UserDefaults.standard.set(old, forKey: SettingsKeys.incompatibleMmproj)
            } else {
                UserDefaults.standard.removeObject(forKey: SettingsKeys.incompatibleMmproj)
            }
        }

        XCTAssertTrue(ServerSettings.mmprojOverrideIsUsable(projector.path, model: model))
        ServerSettings.recordIncompatibleMmproj(model: model, projector: projector.path)
        XCTAssertFalse(ServerSettings.mmprojOverrideIsUsable(projector.path, model: model))
        XCTAssertFalse(ServerSettings.mmprojOverrideIsUsable(
            dir.appendingPathComponent("missing.gguf").path, model: model))
        XCTAssertFalse(ServerSettings.mmprojOverrideIsUsable("", model: model))
    }
}
