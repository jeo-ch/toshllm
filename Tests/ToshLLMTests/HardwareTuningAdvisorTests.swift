// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class HardwareTuningAdvisorTests: XCTestCase {
    private func input(
        gpu: String = "AMD Radeon Pro 575X",
        vramMB: Int = 4_096,
        ramGB: Double = 64,
        physicalCores: Int = 6,
        logicalCores: Int = 12,
        reserveMB: Int = 1_024,
        fileGB: Double = 17.3,
        layers: Int = 48,
        isMoE: Bool = true,
        expertGB: Double = 15.5,
        kvBytesPerToken: Double = 98_304,
        trainedContext: Int? = 262_144,
        ctx: Int = 16_384,
        ubatch: Int = 0,
        ngl: Int = 99,
        ncmoe: Int = 0,
        cacheK: String = "f16",
        cacheV: String = "f16",
        threads: Int = 6,
        dynamicMoe: Bool = false,
        manualFullGPU: Bool = false,
        router: Bool = false,
        gpuCount: Int = 1
    ) -> HardwareTuningInput {
        HardwareTuningInput(
            gpuName: gpu, vramMB: vramMB, ramGB: ramGB,
            physicalCores: physicalCores, logicalCores: logicalCores,
            reserveMB: reserveMB, modelFileGB: fileGB, modelLayers: layers,
            isMoE: isMoE, expertGB: expertGB, kvBytesPerToken: kvBytesPerToken,
            trainedContext: trainedContext, requestedCtx: ctx, requestedUbatch: ubatch,
            currentNgl: ngl, currentNcmoe: ncmoe,
            currentCacheK: cacheK, currentCacheV: cacheV,
            currentThreads: threads, dynamicMoeActive: dynamicMoe,
            manualFullGPU: manualFullGPU, routerMode: router, gpuCount: gpuCount)
    }

    /// The configuration measured in the optimization report: 4 GB gfx803,
    /// 17 GB 48-layer MoE, f16 KV, 16k context. The allocator must land on the
    /// measured 10/6 profile rather than the app's ngl=99 default.
    func testMeasuredLowVRAMProfile() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            ubatch: 4_096, cacheK: "q8_0", cacheV: "turbo4", threads: 12)))

        XCTAssertTrue(advice.measuredProfile)
        XCTAssertEqual(advice.ngl, 10)
        XCTAssertEqual(advice.ncmoe, 6)
        XCTAssertEqual(advice.ctx, 16_384)
        XCTAssertEqual(advice.ubatch, 1_024)
        XCTAssertEqual(advice.cacheTypeK, "f16")
        XCTAssertEqual(advice.cacheTypeV, "f16")
        XCTAssertEqual(advice.threads, 6)
        XCTAssertLessThanOrEqual(advice.ncmoe, advice.ngl)
        XCTAssertLessThanOrEqual(advice.estimatedVRAMGB, advice.usableVRAMGB)
        XCTAssertTrue(advice.warnings.contains {
            if case .ubatchAboveSafeLimit(4_096, 1_024) = $0 { return true }
            return false
        })
        XCTAssertTrue(advice.warnings.contains {
            if case .quantizedKVOnWave64 = $0 { return true }
            return false
        })
    }

    /// The report's ubatch boundary: usable 3072 MiB → 1024; 12 GB keeps 2048.
    func testUbatchIsCappedByUsableVRAM() {
        XCTAssertEqual(HardwareTuningAdvisor.safeUbatch(vramMB: 4_096, reserveMB: 1_024), 1_024)
        XCTAssertEqual(HardwareTuningAdvisor.safeUbatch(vramMB: 12_288, reserveMB: 1_024), 2_048)
        XCTAssertEqual(HardwareTuningAdvisor.safeUbatch(vramMB: 24_576, reserveMB: 1_024), 2_048)
    }

    /// A 12 GB card must keep its current behavior: full offload, current KV,
    /// untouched ncmoe, and only the over-large micro-batch is pulled back.
    func testLargeCardKeepsCurrentBehavior() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            gpu: "AMD Radeon RX 6700 XT", vramMB: 12_272, ramGB: 32,
            physicalCores: 8, logicalCores: 16, reserveMB: 0,
            fileGB: 7.0, layers: 32, isMoE: false, expertGB: 0,
            kvBytesPerToken: 131_072, trainedContext: 32_768,
            ubatch: 4_096, cacheK: "q8_0", cacheV: "q8_0", threads: 12)))

        XCTAssertTrue(advice.fullGPU)
        XCTAssertEqual(advice.ngl, 99)
        XCTAssertEqual(advice.ncmoe, 0)
        XCTAssertEqual(advice.ctx, 16_384)
        XCTAssertEqual(advice.ubatch, 2_048)
        XCTAssertEqual(advice.cacheTypeK, "q8_0")
        XCTAssertEqual(advice.cacheTypeV, "q8_0")
        XCTAssertEqual(advice.threads, 8)
        XCTAssertFalse(advice.warnings.contains {
            if case .quantizedKVOnWave64 = $0 { return true }
            return false
        })
    }

    /// An 8 GB card that cannot hold the whole model gets a hybrid split. The
    /// important invariant is that the suggestion never asks the engine to
    /// offload more expert layers than there are GPU layers.
    func testMidVRAMMoEIsHybridAndConsistent() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            gpu: "AMD Radeon RX 580", vramMB: 8_192, reserveMB: 0,
            trainedContext: 32_768,
            ubatch: 4_096, cacheK: "q8_0", cacheV: "turbo4")))

        XCTAssertFalse(advice.fullGPU)
        XCTAssertGreaterThan(advice.ngl, 0)
        XCTAssertLessThanOrEqual(advice.ncmoe, advice.ngl)
        XCTAssertEqual(advice.ubatch, 2_048)
        XCTAssertEqual(advice.cacheTypeK, "f16")
        XCTAssertEqual(advice.cacheTypeV, "f16")
        XCTAssertLessThanOrEqual(advice.estimatedVRAMGB, advice.usableVRAMGB)
    }

    /// Wave64 is the only architecture whose KV recommendation overrides the
    /// user's choice; every other card keeps the selected types.
    func testWave64UsesMeasuredF16KV() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            gpu: "AMD Radeon Pro 575X", vramMB: 4_096,
            fileGB: 1.0, layers: 28, isMoE: false, expertGB: 0,
            kvBytesPerToken: 32_768, cacheK: "q4_0", cacheV: "q4_0")))

        XCTAssertTrue(advice.fullGPU)
        XCTAssertEqual(advice.cacheTypeK, "f16")
        XCTAssertEqual(advice.cacheTypeV, "f16")
    }

    func testTrainedContextCapsTheRecommendation() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            trainedContext: 8_192, ctx: 32_768)))

        XCTAssertEqual(advice.ctx, 8_192)
        XCTAssertTrue(advice.reasons.contains(.trainedContextCap(8_192)))
    }

    func testNcmoeNeverExceedsNgl() throws {
        for fileGB in stride(from: 4.0, through: 32.0, by: 4.0) {
            for vram in [4_096, 6_144, 8_192, 12_288, 24_576] {
                let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
                    vramMB: vram, fileGB: fileGB, layers: 48,
                    isMoE: true, expertGB: fileGB * 0.8,
                    trainedContext: 32_768, ubatch: 4_096)))
                XCTAssertLessThanOrEqual(advice.ncmoe, advice.ngl,
                                         "file \(fileGB) GB, vram \(vram) MB")
                XCTAssertGreaterThanOrEqual(advice.ngl, 0)
            }
        }
    }

    func testNoGPUIsUnavailable() {
        XCTAssertNil(HardwareTuningAdvisor.advise(input(vramMB: 0)))
    }

    /// A tiny usable budget must not produce a negative layer count; it falls
    /// back to CPU-only instead.
    func testTinyVRAMNeverReturnsNegativeLayers() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            vramMB: 1_024, reserveMB: 512, fileGB: 7, layers: 32,
            isMoE: false, expertGB: 0, kvBytesPerToken: 65_536,
            trainedContext: 8_192)))

        XCTAssertEqual(advice.ngl, 0)
        XCTAssertEqual(advice.ncmoe, 0)
        XCTAssertTrue(advice.warnings.contains(.cpuOnly))
    }

    func testModelThatDoesNotFitFallsBackToCPU() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            fileGB: 200, layers: 80, isMoE: false, expertGB: 0,
            kvBytesPerToken: 262_144, ubatch: 0)))

        XCTAssertEqual(advice.ngl, 0)
        XCTAssertTrue(advice.warnings.contains(.cpuOnly))
    }

    /// Applying the advice must mark it applied only when every managed value
    /// matches; Dynamic MoE and Full GPU are never treated as applied.
    func testAppliedComparison() throws {
        let advice = try XCTUnwrap(HardwareTuningAdvisor.advise(input(
            fileGB: 17.3, cacheK: "f16", cacheV: "f16")))
        var settings = ServerSettings.fromDefaults()
        settings.modelPath = "/tmp/model.gguf"
        settings.ngl = advice.ngl
        settings.ncmoe = advice.ncmoe
        settings.ctx = advice.ctx
        settings.contextAutomatic = false
        settings.ubatch = advice.ubatch
        settings.cacheTypeK = advice.cacheTypeK
        settings.cacheTypeV = advice.cacheTypeV
        settings.threads = advice.threads
        settings.dynamicMoeEnabled = false

        XCTAssertTrue(advice.isApplied(to: settings))
        settings.ubatch = 4_096
        XCTAssertFalse(advice.isApplied(to: settings))
    }
}
