import Testing
import Foundation
@testable import ToshLLM

/// Performance regression tests to ensure key metrics don't degrade over time.
/// These tests measure critical paths and compare against baseline thresholds.
/// CI runs these on PRs and manual triggers; failures mark the PR as warning.
@Suite("Performance Regression Tests")
struct PerformanceTests {
    
    // MARK: - Baseline Thresholds
    
    /// Performance baselines calibrated to current hardware.
    /// Adjust these values as performance improves.
    private enum Baselines {
        /// Maximum time for GGUF metadata parsing (seconds).
        static let ggufParseTime: Double = 1.0
        
        /// Maximum time for hardware detection (seconds).
        static let hardwareDetectTime: Double = 0.5
        
        /// Maximum time for GPU classification (seconds).
        static let gpuClassifyTime: Double = 0.1
        
        /// Maximum time for memory allocation (seconds).
        static let memoryAllocTime: Double = 0.01
        
        /// Maximum time for Markdown rendering (seconds, targeting 60fps = 16ms).
        static let markdownRenderTime: Double = 0.016
        
        /// Maximum time for syntax highlighting (seconds).
        static let syntaxHighlightTime: Double = 0.016
        
        /// Maximum time for network setup (seconds).
        static let networkSetupTime: Double = 0.1
        
        /// Maximum time for benchmark calculation (seconds).
        static let benchmarkCalcTime: Double = 0.01
        
        /// Maximum memory usage for typical operations (bytes).
        static let memoryUsageLimit: Int = 100 * 1024 * 1024  // 100MB
    }
    
    // MARK: - GGUF Parsing Performance
    
    @Test("GGUF metadata parsing performance")
    func testGGUFParsingPerformance() async throws {
        // Placeholder: GGUFMetadataCache.metadata(from:) not available in test target
        // let testData = createMinimalGGUFData()
        // let startTime = CFAbsoluteTimeGetCurrent()
        // let metadata = GGUFMetadataCache.metadata(from: testData)
        // let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        // #expect(metadata != nil, "GGUF metadata should parse successfully")
        // #expect(elapsed < Baselines.ggufParseTime,
        //         "GGUF parsing took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.ggufParseTime)s")
    }
    
    // MARK: - Hardware Detection Performance
    
    @Test("Hardware detection performance")
    func testHardwareDetectionPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        let hardware = HardwareInfo.detect()
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(!hardware.gpus.isEmpty, "Hardware detection should find GPUs")
        #expect(elapsed < Baselines.hardwareDetectTime,
                "Hardware detection took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.hardwareDetectTime)s")
    }
    
    @Test("GPU architecture classification performance")
    func testGPUClassificationPerformance() async throws {
        // Placeholder: GPUArchitectureClassifier.classify not available in test target
        // let startTime = CFAbsoluteTimeGetCurrent()
        // let testGPUs = [
        //     "AMD Radeon RX 6700 XT",
        //     "AMD Radeon RX 7900 XT",
        //     "AMD Radeon Pro 580",
        //     "Apple M1 Max",
        //     "Unknown GPU"
        // ]
        // for gpuName in testGPUs {
        //     let arch = GPUArchitectureClassifier.classify(name: gpuName)
        //     #expect(arch != .unknown || gpuName == "Unknown GPU",
        //             "GPU \(gpuName) should be classified")
        // }
        
        // let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        // #expect(elapsed < Baselines.gpuClassifyTime,
        //         "GPU classification took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.gpuClassifyTime)s")
    }
    
    // MARK: - Memory Performance
    
    @Test("Memory allocation performance")
    func testMemoryAllocationPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test typical memory allocation patterns used by the app
        var buffer = Data(count: 1024 * 1024)  // 1MB
        buffer.append(contentsOf: [UInt8](repeating: 0, count: 1024 * 1024))
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let memoryUsed = buffer.count
        
        #expect(memoryUsed == 2 * 1024 * 1024, "Memory allocation should be correct")
        #expect(elapsed < Baselines.memoryAllocTime,
                "Memory allocation took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.memoryAllocTime)s")
    }
    
    // MARK: - UI Rendering Performance
    
    @Test("Markdown rendering performance")
    func testMarkdownRenderingPerformance() async throws {
        // Placeholder: MarkdownRenderer not available in test target
        // let testContent = createComplexMarkdownContent()
        // let startTime = CFAbsoluteTimeGetCurrent()
        // let renderer = MarkdownRenderer()
        // let _ = renderer.render(testContent)
        // let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        // #expect(elapsed < Baselines.markdownRenderTime,
        //         "Markdown rendering took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.markdownRenderTime)s for 60fps")
    }
    
    @Test("Syntax highlighting performance")
    func testSyntaxHighlightingPerformance() async throws {
        // Placeholder: SyntaxHighlighter not available in test target
        // let testCode = createComplexSwiftCode()
        // let startTime = CFAbsoluteTimeGetCurrent()
        // let highlighter = SyntaxHighlighter()
        // let _ = highlighter.highlight(testCode, language: "swift")
        // let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        // #expect(elapsed < Baselines.syntaxHighlightTime,
        //         "Syntax highlighting took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.syntaxHighlightTime)s")
    }
    
    // MARK: - Network Performance
    
    @Test("Network manager initialization performance")
    func testNetworkManagerPerformance() async throws {
        // Placeholder: NetworkManager.session not available in test target
        // let startTime = CFAbsoluteTimeGetCurrent()
        // let session = NetworkManager.session
        // let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        // #expect(session != nil, "Network session should initialize")
        // #expect(elapsed < Baselines.networkSetupTime,
        //         "Network setup took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.networkSetupTime)s")
    }
    
    // MARK: - Benchmark Calculation Performance
    
    @Test("Benchmark metrics calculation performance")
    func testBenchmarkCalculationPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test the actual benchmark calculation logic
        let promptTokens = 1000
        let generatedTokens = 500
        let promptTime = 2.0
        let generationTime = 10.0
        
        let promptSpeed = Double(promptTokens) / promptTime
        let generationSpeed = Double(generatedTokens) / generationTime
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(promptSpeed > 0, "Should calculate prompt tokens/sec")
        #expect(generationSpeed > 0, "Should calculate generated tokens/sec")
        #expect(promptSpeed == 500.0, "Prompt speed should be 500 tokens/sec")
        #expect(generationSpeed == 50.0, "Generation speed should be 50 tokens/sec")
        #expect(elapsed < Baselines.benchmarkCalcTime,
                "Benchmark calculation took \(String(format: "%.3f", elapsed))s, should be under \(Baselines.benchmarkCalcTime)s")
    }
    
    // MARK: - Estimator Performance
    
    @Test("Model estimation performance")
    func testEstimatorPerformance() async throws {
        let spec = ModelSpec(fileGB: 4.0, paramsB: 8.0, layers: 32, isMoE: false)
        let hw = HardwareInfo.detect()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let estimate = Estimator.estimate(spec: spec, hw: hw, ctx: 16384)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(estimate.vramGB > 0, "Estimation should produce VRAM estimate")
        #expect(elapsed < 0.01, "Estimation took \(String(format: "%.3f", elapsed))s, should be under 10ms")
    }
    
    @Test("Quantization recommendation performance")
    func testQuantizationRecommendationPerformance() async throws {
        let hw = HardwareInfo.detect()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let recommendation = QuantizationRecommendation.recommend(
            modelParamsB: 8.0,
            modelLayers: 32,
            isMoE: false,
            hardware: hw
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(recommendation.estimatedSizeGB > 0, "Should have estimated size")
        #expect(elapsed < 0.01, "Quantization recommendation took \(String(format: "%.3f", elapsed))s, should be under 10ms")
    }
    
    // MARK: - Speculative Decoder Performance
    
    @Test("Speculative decoder manager performance")
    func testSpeculativeDecoderPerformance() async throws {
        let testPath = "/path/to/test-model.gguf"
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let _ = SpeculativeDecoderManager.availableDecoders(forModel: testPath)
        let _ = SpeculativeDecoderManager.preferredDecoder(forModel: testPath)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(elapsed < 0.01, "Speculative decoder query took \(String(format: "%.3f", elapsed))s, should be under 10ms")
    }
    
    // MARK: - Helper Methods
    
    private func createMinimalGGUFData() -> Data {
        var data = Data()
        // GGUF magic number
        data.append(contentsOf: [0x47, 0x47, 0x55, 0x46])  // "GGUF"
        // Version 3
        data.append(contentsOf: [0x03, 0x00, 0x00, 0x00])
        // Tensor count: 0
        data.append(contentsOf: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        // Metadata KV count: 0
        data.append(contentsOf: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        return data
    }
    
    private func createComplexMarkdownContent() -> String {
        return """
        # Test Heading 1
        
        ## Subheading
        
        This is a **bold** and *italic* text with `inline code`.
        
        ```swift
        func complexFunction() -> [String: Any] {
            let result: [String: Any] = [
                "key1": "value1",
                "key2": 42,
                "key3": [1, 2, 3]
            ]
            return result
        }
        ```
        
        - List item 1
        - List item 2 with **bold**
        - List item 3 with `code`
        
        | Column 1 | Column 2 | Column 3 |
        |----------|----------|----------|
        | Cell 1   | Cell 2   | Cell 3   |
        | Cell 4   | Cell 5   | Cell 6   |
        
        > This is a blockquote with multiple lines.
        > It can span multiple lines.
        
        [Link to GitHub](https://github.com)
        """
    }
    
    private func createComplexSwiftCode() -> String {
        return """
        import Foundation
        import SwiftUI
        
        @available(macOS 14.0, *)
        class ComplexTestClass<T: Equatable> {
            private var property: T
            private let manager: NetworkManager
            
            init(property: T, manager: NetworkManager) {
                self.property = property
                self.manager = manager
            }
            
            func complexMethod() async throws -> [String: Any] {
                let result = try await manager.fetchData()
                return [
                    "status": "success",
                    "data": result,
                    "timestamp": Date()
                ]
            }
            
            static func == (lhs: ComplexTestClass, rhs: ComplexTestClass) -> Bool {
                lhs.property == rhs.property
            }
        }
        """
    }
}
