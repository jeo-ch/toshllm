// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Structured request tracing for debugging and performance analysis.
/// Inspired by cc-haha's tracing system.
struct RequestTracer {
    
    // MARK: - Request Trace
    
    /// A single request trace with timing and metadata.
    struct Trace: Identifiable, Sendable {
        let id: UUID
        let requestID: String
        let startTime: Date
        var endTime: Date?
        var duration: TimeInterval? {
            guard let endTime else { return nil }
            return endTime.timeIntervalSince(startTime)
        }
        
        /// Request metadata.
        let metadata: TraceMetadata
        
        /// Request stages with timing.
        var stages: [TraceStage] = []
        
        /// Error information if request failed.
        var error: TraceError?
        
        /// Token usage statistics.
        var tokenUsage: TokenUsage?
        
        /// Human-readable status.
        var status: String {
            if error != nil { return "Failed" }
            if endTime == nil { return "In Progress" }
            return "Completed"
        }
        
        /// Status color for UI.
        var statusColor: String {
            if error != nil { return "red" }
            if endTime == nil { return "blue" }
            return "green"
        }
    }
    
    // MARK: - Trace Metadata
    
    /// Metadata about a request.
    struct TraceMetadata: Sendable {
        let modelPath: String
        let modelName: String
        let endpoint: String
        let method: String
        let conversationID: UUID?
        let userID: String?
        let tags: [String]
    }
    
    // MARK: - Trace Stage
    
    /// A stage in the request lifecycle.
    struct TraceStage: Identifiable, Sendable {
        let id: UUID
        let name: String
        let startTime: Date
        var endTime: Date?
        var duration: TimeInterval? {
            guard let endTime else { return nil }
            return endTime.timeIntervalSince(startTime)
        }
        
        /// Stage-specific metadata.
        var metadata: [String: String] = [:]
        
        /// Whether this stage completed successfully.
        var succeeded: Bool { error == nil }
        
        /// Error if stage failed.
        var error: TraceError?
    }
    
    // MARK: - Trace Error
    
    /// Error information for a trace or stage.
    struct TraceError: Sendable {
        let code: Int
        let message: String
        let domain: String
        let underlyingError: Error?
        
        var description: String {
            "[\(domain)] \(code): \(message)"
        }
    }
    
    // MARK: - Token Usage
    
    /// Token usage statistics for a request.
    struct TokenUsage: Sendable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int
        let promptTime: TimeInterval?
        let completionTime: TimeInterval?
        let tokensPerSecond: Double?
        
        var promptTokensPerSecond: Double? {
            guard let promptTime, promptTime > 0 else { return nil }
            return Double(promptTokens) / promptTime
        }
        
        var completionTokensPerSecond: Double? {
            guard let completionTime, completionTime > 0 else { return nil }
            return Double(completionTokens) / completionTime
        }
    }
    
    // MARK: - Request Tracer Storage
    
    /// In-memory storage for request traces.
    private var traces: [Trace] = []
    
    /// Maximum number of traces to keep.
    private let maxTraces: Int
    
    /// Lock for thread safety.
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    init(maxTraces: Int = 1000) {
        self.maxTraces = maxTraces
    }
    
    // MARK: - Public API
    
    /// Start a new request trace.
    mutating func startTrace(
        requestID: String,
        metadata: TraceMetadata
    ) -> UUID {
        lock.lock()
        defer { lock.unlock() }
        
        let trace = Trace(
            id: UUID(),
            requestID: requestID,
            startTime: Date(),
            metadata: metadata
        )
        
        traces.append(trace)
        
        // Evict old traces if necessary
        if traces.count > maxTraces {
            traces.removeFirst(traces.count - maxTraces)
        }
        
        return trace.id
    }
    
    /// Start a new stage in a trace.
    mutating func startStage(
        traceID: UUID,
        name: String,
        metadata: [String: String] = [:]
    ) -> UUID? {
        lock.lock()
        defer { lock.unlock() }
        
        guard let index = traces.firstIndex(where: { $0.id == traceID }) else {
            return nil
        }
        
        let stage = TraceStage(
            id: UUID(),
            name: name,
            startTime: Date(),
            metadata: metadata
        )
        
        traces[index].stages.append(stage)
        return stage.id
    }
    
    /// Complete a stage in a trace.
    mutating func completeStage(
        traceID: UUID,
        stageID: UUID,
        error: TraceError? = nil
    ) {
        lock.lock()
        defer { lock.unlock() }
        
        guard let traceIndex = traces.firstIndex(where: { $0.id == traceID }),
              let stageIndex = traces[traceIndex].stages.firstIndex(where: { $0.id == stageID })
        else {
            return
        }
        
        traces[traceIndex].stages[stageIndex].endTime = Date()
        traces[traceIndex].stages[stageIndex].error = error
    }
    
    /// Complete a request trace.
    mutating func completeTrace(
        traceID: UUID,
        error: TraceError? = nil,
        tokenUsage: TokenUsage? = nil
    ) {
        lock.lock()
        defer { lock.unlock() }
        
        guard let index = traces.firstIndex(where: { $0.id == traceID }) else {
            return
        }
        
        traces[index].endTime = Date()
        traces[index].error = error
        traces[index].tokenUsage = tokenUsage
    }
    
    /// Get a specific trace.
    func getTrace(id: UUID) -> Trace? {
        lock.lock()
        defer { lock.unlock() }
        
        return traces.first { $0.id == id }
    }
    
    /// Get all traces, optionally filtered.
    func getTraces(
        modelPath: String? = nil,
        conversationID: UUID? = nil,
        status: String? = nil,
        limit: Int = 100
    ) -> [Trace] {
        lock.lock()
        defer { lock.unlock() }
        
        var filtered = traces
        
        if let modelPath {
            filtered = filtered.filter { $0.metadata.modelPath == modelPath }
        }
        
        if let conversationID {
            filtered = filtered.filter { $0.metadata.conversationID == conversationID }
        }
        
        if let status {
            filtered = filtered.filter { $0.status == status }
        }
        
        return Array(filtered.suffix(limit))
    }
    
    /// Get trace statistics.
    func getStatistics() -> TraceStatistics {
        lock.lock()
        defer { lock.unlock() }
        
        let completed = traces.filter { $0.endTime != nil }
        let failed = traces.filter { $0.error != nil }
        let inProgress = traces.filter { $0.endTime == nil }
        
        let totalDuration = completed.compactMap(\.duration).reduce(0, +)
        let averageDuration = completed.isEmpty ? 0 : totalDuration / Double(completed.count)
        
        let totalTokens = completed.compactMap(\.tokenUsage).reduce(0) { $0 + $1.totalTokens }
        let averageTokensPerSecond = completed.compactMap(\.tokenUsage?.tokensPerSecond).reduce(0, +) / Double(max(1, completed.count))
        
        return TraceStatistics(
            totalRequests: traces.count,
            completedRequests: completed.count,
            failedRequests: failed.count,
            inProgressRequests: inProgress.count,
            averageDuration: averageDuration,
            totalTokens: totalTokens,
            averageTokensPerSecond: averageTokensPerSecond
        )
    }
    
    /// Clear all traces.
    mutating func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        traces.removeAll()
    }
}

// MARK: - Trace Statistics

/// Statistics for request traces.
struct TraceStatistics: Sendable {
    let totalRequests: Int
    let completedRequests: Int
    let failedRequests: Int
    let inProgressRequests: Int
    let averageDuration: TimeInterval
    let totalTokens: Int
    let averageTokensPerSecond: Double
    
    /// Success rate as a percentage.
    var successRate: Double {
        guard totalRequests > 0 else { return 0 }
        return Double(completedRequests) / Double(totalRequests) * 100
    }
    
    /// Human-readable summary.
    var summary: String {
        "\(totalRequests) requests, \(Int(successRate))% success, avg \(String(format: "%.1f", averageDuration))s"
    }
}

// MARK: - Request Tracer Manager

/// Manages request tracing for the application.
@MainActor
final class RequestTracerManager: ObservableObject {
    static let shared = RequestTracerManager()
    
    @Published var statistics: TraceStatistics = TraceStatistics(
        totalRequests: 0,
        completedRequests: 0,
        failedRequests: 0,
        inProgressRequests: 0,
        averageDuration: 0,
        totalTokens: 0,
        averageTokensPerSecond: 0
    )
    
    private var tracer = RequestTracer()
    
    private init() {}
    
    /// Start a new request trace.
    func startTrace(
        requestID: String,
        metadata: RequestTracer.TraceMetadata
    ) -> UUID {
        let id = tracer.startTrace(requestID: requestID, metadata: metadata)
        updateStatistics()
        return id
    }
    
    /// Start a new stage in a trace.
    func startStage(
        traceID: UUID,
        name: String,
        metadata: [String: String] = [:]
    ) -> UUID? {
        let id = tracer.startStage(traceID: traceID, name: name, metadata: metadata)
        return id
    }
    
    /// Complete a stage in a trace.
    func completeStage(
        traceID: UUID,
        stageID: UUID,
        error: RequestTracer.TraceError? = nil
    ) {
        tracer.completeStage(traceID: traceID, stageID: stageID, error: error)
        updateStatistics()
    }
    
    /// Complete a request trace.
    func completeTrace(
        traceID: UUID,
        error: RequestTracer.TraceError? = nil,
        tokenUsage: RequestTracer.TokenUsage? = nil
    ) {
        tracer.completeTrace(traceID: traceID, error: error, tokenUsage: tokenUsage)
        updateStatistics()
    }
    
    /// Get a specific trace.
    func getTrace(id: UUID) -> RequestTracer.Trace? {
        tracer.getTrace(id: id)
    }
    
    /// Get all traces, optionally filtered.
    func getTraces(
        modelPath: String? = nil,
        conversationID: UUID? = nil,
        status: String? = nil,
        limit: Int = 100
    ) -> [RequestTracer.Trace] {
        tracer.getTraces(modelPath: modelPath, conversationID: conversationID, status: status, limit: limit)
    }
    
    /// Clear all traces.
    func clear() {
        tracer.clear()
        updateStatistics()
    }
    
    /// Update published statistics.
    private func updateStatistics() {
        statistics = tracer.getStatistics()
    }
}
