// ToshLLM Tests - Concurrency Estimator Tests
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ConcurrencyEstimatorTests: XCTestCase {
    
    // MARK: - Basic Estimation Tests
    
    func testEstimateForHighEndHardware() {
        let spec = ModelSpec(
            fileGB: 4.5,
            paramsB: 8.0,
            layers: 32,
            isMoE: false,
            activeParamsB: 0,
            kvBytesPerToken: 256.0
        )
        
        let hw = HardwareInfo(
            cpuBrand: "Intel Core i9-12900K",
            physicalCores: 12,
            logicalCores: 24,
            ramGB: 64.0,
            arch: "x86_64",
            model: "Mac Pro (MacPro7,1)",
            osVersion: "macOS 15.5 Sequoia",
            gpus: [
                GPUDevice(
                    index: 0,
                    name: "AMD Radeon RX 7900 XTX",
                    vramMB: 24576,
                    isExternal: false,
                    isIntegrated: false,
                    peerGroupID: 0,
                    peerCount: 0,
                    supportsBF16: true
                )
            ]
        )
        
        let result = ConcurrencyEstimator.estimate(spec: spec, hw: hw, ctx: 8192)
        
        XCTAssertGreaterThan(result.maxSessions, 0, "High-end hardware should support at least one session")
        XCTAssertGreaterThan(result.kvPerSession, 0, "KV per session should be positive")
        XCTAssertGreaterThan(result.tokensPerSecond, 0, "Tokens per second should be positive")
        XCTAssertTrue(result.isReliable, "Should be reliable for high-end hardware")
    }
    
    func testEstimateForLowEndHardware() {
        let spec = ModelSpec(
            fileGB: 35.0,
            paramsB: 70.0,
            layers: 80,
            isMoE: false,
            activeParamsB: 0,
            kvBytesPerToken: 512.0
        )
        
        let hw = HardwareInfo(
            cpuBrand: "Intel Core i5-8400",
            physicalCores: 4,
            logicalCores: 8,
            ramGB: 16.0,
            arch: "x86_64",
            model: "Mac mini (MacMini2018)",
            osVersion: "macOS 15.5 Sequoia",
            gpus: [
                GPUDevice(
                    index: 0,
                    name: "AMD Radeon RX 580",
                    vramMB: 8192,
                    isExternal: false,
                    isIntegrated: false,
                    peerGroupID: 0,
                    peerCount: 0,
                    supportsBF16: false
                )
            ]
        )
        
        let result = ConcurrencyEstimator.estimate(spec: spec, hw: hw, ctx: 8192)
        
        XCTAssertEqual(result.maxSessions, 0, "Low-end hardware should not support 70B model")
        XCTAssertFalse(result.isReliable, "Should not be reliable for this configuration")
    }
    
    // MARK: - MoE Model Tests
    
    func testEstimateForMoEModel() {
        let spec = ModelSpec(
            fileGB: 350.0,
            paramsB: 685.0,
            layers: 61,
            isMoE: true,
            activeParamsB: 37.0,
            kvBytesPerToken: 1024.0
        )
        
        let hw = HardwareInfo(
            cpuBrand: "Intel Core i9-12900K",
            physicalCores: 12,
            logicalCores: 24,
            ramGB: 128.0,
            arch: "x86_64",
            model: "Mac Pro (MacPro7,1)",
            osVersion: "macOS 15.5 Sequoia",
            gpus: [
                GPUDevice(
                    index: 0,
                    name: "AMD Radeon RX 7900 XTX",
                    vramMB: 24576,
                    isExternal: false,
                    isIntegrated: false,
                    peerGroupID: 0,
                    peerCount: 0,
                    supportsBF16: true
                )
            ]
        )
        
        let result = ConcurrencyEstimator.estimate(spec: spec, hw: hw, ctx: 8192)
        
        // MoE models have smaller active parameters
        XCTAssertGreaterThanOrEqual(result.maxSessions, 0, "MoE estimation should complete")
    }
    
    // MARK: - Context Length Tests
    
    func testEstimateWithLargerContext() {
        let spec = ModelSpec(
            fileGB: 4.5,
            paramsB: 8.0,
            layers: 32,
            isMoE: false,
            activeParamsB: 0,
            kvBytesPerToken: 256.0
        )
        
        let hw = HardwareInfo(
            cpuBrand: "Intel Core i7-10700K",
            physicalCores: 8,
            logicalCores: 16,
            ramGB: 32.0,
            arch: "x86_64",
            model: "iMac (iMac20,1)",
            osVersion: "macOS 15.5 Sequoia",
            gpus: [
                GPUDevice(
                    index: 0,
                    name: "AMD Radeon RX 5700 XT",
                    vramMB: 8192,
                    isExternal: false,
                    isIntegrated: false,
                    peerGroupID: 0,
                    peerCount: 0,
                    supportsBF16: true
                )
            ]
        )
        
        let smallCtx = ConcurrencyEstimator.estimate(spec: spec, hw: hw, ctx: 4096)
        let largeCtx = ConcurrencyEstimator.estimate(spec: spec, hw: hw, ctx: 16384)
        
        // Larger context should use more KV memory per session
        XCTAssertGreaterThanOrEqual(largeCtx.kvPerSession, smallCtx.kvPerSession, "Larger context should have larger or equal KV per session")
    }
    
    // MARK: - Quick Estimate Tests
    
    func testQuickEstimate() {
        let sessions = ConcurrencyEstimator.quickEstimate(vramGB: 24.0, modelGB: 4.5, ctx: 8192)
        
        XCTAssertGreaterThanOrEqual(sessions, 0, "Should estimate non-negative sessions")
    }
    
    func testQuickEstimateWithSmallVRAM() {
        let sessions = ConcurrencyEstimator.quickEstimate(vramGB: 4.0, modelGB: 4.5, ctx: 8192)
        
        XCTAssertEqual(sessions, 0, "Should estimate 0 sessions for insufficient VRAM")
    }
    
    // MARK: - Display Helper Tests
    
    func testConcurrencyDisplay() {
        let result = ConcurrencyEstimator.Result(
            maxSessions: 3,
            availableForKV: 10.0,
            kvPerSession: 2.0,
            tokensPerSecond: 25.0,
            memoryEfficiency: 0.8,
            isReliable: true
        )
        
        let display = ConcurrencyDisplay(result: result)
        
        XCTAssertEqual(display.color, "green", "3 sessions should be green")
        XCTAssertEqual(display.icon, "person.2.circle", "3 sessions should show person.2")
        XCTAssertFalse(display.recommendation.isEmpty, "Should have recommendation")
        XCTAssertNil(display.performanceWarning, "No warning for good performance")
    }
    
    func testConcurrencyDisplayWithSlowPerformance() {
        let result = ConcurrencyEstimator.Result(
            maxSessions: 1,
            availableForKV: 2.0,
            kvPerSession: 1.5,
            tokensPerSecond: 5.0,
            memoryEfficiency: 0.95,
            isReliable: true
        )
        
        let display = ConcurrencyDisplay(result: result)
        
        XCTAssertEqual(display.color, "yellow", "1 session should be yellow")
        XCTAssertNotNil(display.performanceWarning, "Should have warning for slow performance")
    }
    
    // MARK: - Router Estimation Tests
    
    func testEstimateRouter() {
        let specs = [
            (ModelSpec(fileGB: 4.5, paramsB: 8.0, layers: 32, isMoE: false, activeParamsB: 0, kvBytesPerToken: 256.0), 1),
            (ModelSpec(fileGB: 2.0, paramsB: 3.0, layers: 26, isMoE: false, activeParamsB: 0, kvBytesPerToken: 128.0), 1)
        ]
        
        let hw = HardwareInfo(
            cpuBrand: "Intel Core i7-10700K",
            physicalCores: 8,
            logicalCores: 16,
            ramGB: 32.0,
            arch: "x86_64",
            model: "iMac (iMac20,1)",
            osVersion: "macOS 15.5 Sequoia",
            gpus: [
                GPUDevice(
                    index: 0,
                    name: "AMD Radeon RX 5700 XT",
                    vramMB: 8192,
                    isExternal: false,
                    isIntegrated: false,
                    peerGroupID: 0,
                    peerCount: 0,
                    supportsBF16: true
                )
            ]
        )
        
        let results = ConcurrencyEstimator.estimateRouter(models: specs, hw: hw, ctx: 8192)
        
        XCTAssertFalse(results.isEmpty, "Router estimation should return results")
        for (key, result) in results {
            XCTAssertFalse(key.isEmpty, "Key should not be empty")
            XCTAssertGreaterThanOrEqual(result.maxSessions, 0, "Sessions should be non-negative")
        }
    }
}
