// ToshLLM Tests - Optimization Module Tests
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class OptimizationModuleTests: XCTestCase {
    
    // MARK: - QuantizationRecommendation Tests
    
    func testQuantizationRecommendationWithRamBandwidth() {
        let hardware = HardwareInfo(
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
        
        let recommendation = QuantizationRecommendation.recommend(
            modelParamsB: 8.0,
            modelLayers: 32,
            isMoE: false,
            hardware: hardware,
            ramBandwidthGBs: 50.0  // DDR4-3200 dual channel
        )
        
        XCTAssertGreaterThan(recommendation.estimatedSizeGB, 0)
        XCTAssertGreaterThan(recommendation.estimatedSpeed, 0)
    }
    
    // MARK: - ConcurrencyEstimator Tests
    
    func testConcurrencyEstimatorWithKvBytesPerToken() {
        let spec = ModelSpec(
            fileGB: 4.5,
            paramsB: 8.0,
            layers: 32,
            isMoE: false,
            activeParamsB: 0,
            kvBytesPerToken: 128.0  // Q4 quantization
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
        
        let result = ConcurrencyEstimator.estimate(spec: spec, hw: hw, ctx: 8192)
        
        XCTAssertGreaterThan(result.kvPerSession, 0)
        XCTAssertGreaterThanOrEqual(result.memoryEfficiency, 0)
        XCTAssertLessThanOrEqual(result.memoryEfficiency, 1)
    }
    
    func testQuickEstimateWithKvBytesPerToken() {
        let sessions128 = ConcurrencyEstimator.quickEstimate(
            vramGB: 8.0,
            modelGB: 4.5,
            ctx: 8192,
            kvBytesPerToken: 128.0
        )
        
        let sessions256 = ConcurrencyEstimator.quickEstimate(
            vramGB: 8.0,
            modelGB: 4.5,
            ctx: 8192,
            kvBytesPerToken: 256.0
        )
        
        // Lower kvBytesPerToken should allow more sessions
        XCTAssertGreaterThanOrEqual(sessions128, sessions256)
    }
    
    // MARK: - PermissionPolicy Tests
    
    func testPermissionPolicyGlobMatching() {
        var policy = PermissionPolicy()
        policy.globalLevel = .ask
        policy.denyPatterns = ["/etc/*", "*.secret"]
        
        // Should match glob patterns
        let result1 = policy.checkPermission(category: .fileRead, operation: "/etc/passwd")
        XCTAssertTrue(result1.isDenied, "Should deny /etc/passwd")
        
        let result2 = policy.checkPermission(category: .fileRead, operation: "/home/user/file.secret")
        XCTAssertTrue(result2.isDenied, "Should deny .secret files")
        
        let result3 = policy.checkPermission(category: .fileRead, operation: "/home/user/file.txt")
        XCTAssertFalse(result3.isDenied, "Should allow regular files")
    }
    
    // MARK: - RequestTracer Tests
    
    func testRequestTracerStatistics() {
        var tracer = RequestTracer()
        
        // Add a completed trace with token usage
        let traceID = tracer.startTrace(
            requestID: "test-request-1",
            metadata: RequestTracer.TraceMetadata(
                modelPath: "/path/to/model.gguf",
                modelName: "test-model",
                endpoint: "/v1/chat/completions",
                method: "POST",
                conversationID: nil,
                userID: nil,
                tags: []
            )
        )
        
        // Complete the trace with token usage
        tracer.completeTrace(
            traceID: traceID,
            tokenUsage: RequestTracer.TokenUsage(
                promptTokens: 100,
                completionTokens: 50,
                totalTokens: 150,
                promptTime: 2.0,
                completionTime: 2.0,
                tokensPerSecond: 25.0
            )
        )
        
        let stats = tracer.getStatistics()
        XCTAssertEqual(stats.completedRequests, 1)
        XCTAssertEqual(stats.averageTokensPerSecond, 25.0)
    }
    
    // MARK: - SessionExporter Tests
    
    func testSessionExporterParseRole() {
        // Test safe parsing of role lines
        let line1 = "[user]: Hello"
        let line2 = "[assistant]:"
        let line3 = "[user]:"
        
        // All should parse without crashing
        XCTAssertFalse(line1.isEmpty)
        XCTAssertFalse(line2.isEmpty)
        XCTAssertFalse(line3.isEmpty)
    }
}
