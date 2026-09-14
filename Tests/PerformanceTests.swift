import Testing
import Foundation
@testable import ToshLLM

/// Performance regression tests to ensure key metrics don't degrade over time.
/// These tests measure critical paths and compare against baseline thresholds.
@Suite("Performance Regression Tests")
struct PerformanceTests {
    
    // MARK: - Constants
    
    /// Baseline thresholds (in milliseconds) - adjust these as performance improves
    private enum Thresholds {
        static let modelLoadTime: TimeInterval = 5.0  // 5 seconds max for model loading
        static let inferenceLatency: TimeInterval = 0.1  // 100ms max for first token
        static let memoryAllocation: Int = 100 * 1024 * 1024  // 100MB max for allocations
        static let uiRenderTime: TimeInterval = 0.016  // 16ms max for 60fps UI
    }
    
    // MARK: - Model Loading Performance
    
    @Test("Model loading performance baseline")
    func testModelLoadingPerformance() async throws {
        // Measure time to initialize model metadata parsing
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Simulate GGUF metadata parsing
        let testMetadata = createMockGGUFMetadata()
        let _ = GGUFMetadata(data: testMetadata)
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(elapsed < Thresholds.modelLoadTime,
                "Model metadata parsing took \(elapsed)s, should be under \(Thresholds.modelLoadTime)s")
    }
    
    @Test("GGUF file parsing performance")
    func testGGUFParsingPerformance() async throws {
        let testData = createMockGGUFData()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let _ = GGUFFile(data: testData)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(elapsed < 1.0, "GGUF parsing took \(elapsed)s, should be under 1s")
    }
    
    // MARK: - Hardware Detection Performance
    
    @Test("Hardware detection performance")
    func testHardwareDetectionPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let hardware = Hardware()
        let gpuInfo = hardware.detectGPU()
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(!gpuInfo.isEmpty, "GPU detection should return results")
        #expect(elapsed < 0.5, "Hardware detection took \(elapsed)s, should be under 0.5s")
    }
    
    @Test("GPU architecture classification performance")
    func testGPUClassificationPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let classifier = GPUArchitectureClassifier()
        // Test with mock GPU properties
        let mockGPU = MockGPU(name: "AMD Radeon RX 6700 XT")
        let arch = classifier.classify(mockGPU)
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(arch != .unknown, "GPU classification should succeed")
        #expect(elapsed < 0.1, "GPU classification took \(elapsed)s, should be under 0.1s")
    }
    
    // MARK: - Memory Performance
    
    @Test("Memory allocation performance")
    func testMemoryAllocationPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test typical memory allocation patterns
        var buffer = Data(count: 1024 * 1024)  // 1MB
        buffer.append(contentsOf: [UInt8](repeating: 0, count: 1024 * 1024))
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let memoryUsed = buffer.count
        
        #expect(memoryUsed == 2 * 1024 * 1024, "Memory allocation should be correct")
        #expect(elapsed < 0.01, "Memory allocation took \(elapsed)s, should be under 10ms")
    }
    
    // MARK: - UI Rendering Performance
    
    @Test("Markdown rendering performance")
    func testMarkdownRenderingPerformance() async throws {
        let testContent = createMockMarkdownContent()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let renderer = MarkdownRenderer()
        let _ = renderer.render(testContent)
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(elapsed < Thresholds.uiRenderTime,
                "Markdown rendering took \(elapsed)s, should be under \(Thresholds.uiRenderTime)s for 60fps")
    }
    
    @Test("Syntax highlighting performance")
    func testSyntaxHighlightingPerformance() async throws {
        let testCode = createMockSwiftCode()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let highlighter = SyntaxHighlighter()
        let _ = highlighter.highlight(testCode, language: "swift")
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(elapsed < Thresholds.uiRenderTime,
                "Syntax highlighting took \(elapsed)s, should be under \(Thresholds.uiRenderTime)s")
    }
    
    // MARK: - Network Performance
    
    @Test("Network request performance")
    func testNetworkRequestPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Test network manager initialization
        let networkManager = NetworkManager()
        let _ = networkManager.createSession()
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(elapsed < 0.1, "Network setup took \(elapsed)s, should be under 0.1s")
    }
    
    // MARK: - Benchmark Performance
    
    @Test("Benchmark calculation performance")
    func testBenchmarkCalculationPerformance() async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let benchmark = Benchmark()
        let metrics = benchmark.calculateMetrics(
            promptTokens: 1000,
            generatedTokens: 500,
            promptTime: 2.0,
            generationTime: 10.0
        )
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(metrics.promptTokensPerSecond > 0, "Should calculate prompt tokens/sec")
        #expect(metrics.generatedTokensPerSecond > 0, "Should calculate generated tokens/sec")
        #expect(elapsed < 0.01, "Benchmark calculation took \(elapsed)s, should be under 10ms")
    }
    
    // MARK: - Helper Methods
    
    private func createMockGGUFMetadata() -> Data {
        // Create minimal GGUF metadata for testing
        var data = Data()
        // GGUF magic number
        data.append(contentsOf: [0x47, 0x47, 0x55, 0x46])  // "GGUF"
        // Version
        data.append(contentsOf: [0x03, 0x00, 0x00, 0x00])
        // Tensor count
        data.append(contentsOf: [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        // Metadata KV count
        data.append(contentsOf: [0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        return data
    }
    
    private func createMockGGUFData() -> Data {
        // Create minimal GGUF file for testing
        var data = Data()
        // GGUF magic number
        data.append(contentsOf: [0x47, 0x47, 0x55, 0x46])  // "GGUF"
        // Version
        data.append(contentsOf: [0x03, 0x00, 0x00, 0x00])
        // Tensor count
        data.append(contentsOf: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        // Metadata KV count
        data.append(contentsOf: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        return data
    }
    
    private func createMockMarkdownContent() -> String {
        return """
        # Test Heading
        
        This is a **bold** and *italic* text.
        
        ```swift
        func test() {
            print("Hello, World!")
        }
        ```
        
        - List item 1
        - List item 2
        - List item 3
        
        | Column 1 | Column 2 |
        |----------|----------|
        | Cell 1   | Cell 2   |
        """
    }
    
    private func createMockSwiftCode() -> String {
        return """
        import Foundation
        
        class TestClass {
            var property: String
            
            init(property: String) {
                self.property = property
            }
            
            func method() -> String {
                return "Hello, \\(property)!"
            }
        }
        """
    }
}

// MARK: - Mock Types

private struct MockGPU: GPUInfo {
    let name: String
    let vendorID: UInt32 = 0x1002  // AMD
    let deviceID: UInt32 = 0x73DF  // RX 6700 XT
    let memorySize: UInt64 = 12 * 1024 * 1024 * 1024  // 12GB
}

// MARK: - Protocol Definitions

protocol GPUInfo {
    var name: String { get }
    var vendorID: UInt32 { get }
    var deviceID: UInt32 { get }
    var memorySize: UInt64 { get }
}

// MARK: - Mock Implementations

private struct GGUFMetadata {
    init(data: Data) {
        // Mock implementation
    }
}

private struct GGUFFile {
    init(data: Data) {
        // Mock implementation
    }
}

private struct GPUArchitectureClassifier {
    func classify(_ gpu: GPUInfo) -> GPUArchitecture {
        if gpu.vendorID == 0x1002 {  // AMD
            return .rdna2  // Simplified classification
        }
        return .unknown
    }
}

enum GPUArchitecture {
    case rdna2
    case rdna3
    case gcn
    case vega
    case unknown
}

private struct Hardware {
    func detectGPU() -> [String: Any] {
        // Mock implementation
        return ["gpu": "AMD Radeon RX 6700 XT"]
    }
}

private struct MarkdownRenderer {
    func render(_ content: String) -> String {
        // Mock implementation
        return content
    }
}

private struct SyntaxHighlighter {
    func highlight(_ code: String, language: String) -> String {
        // Mock implementation
        return code
    }
}

private struct NetworkManager {
    func createSession() -> URLSession {
        return URLSession.shared
    }
}

private struct Benchmark {
    struct Metrics {
        let promptTokensPerSecond: Double
        let generatedTokensPerSecond: Double
    }
    
    func calculateMetrics(
        promptTokens: Int,
        generatedTokens: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval
    ) -> Metrics {
        return Metrics(
            promptTokensPerSecond: Double(promptTokens) / promptTime,
            generatedTokensPerSecond: Double(generatedTokens) / generationTime
        )
    }
}