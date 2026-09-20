// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Simplified continuous batching scheduler for llama.cpp server.
/// Inspired by vLLM v1 scheduler and llama.cpp's new scheduler.
@MainActor
final class ContinuousBatchScheduler: ObservableObject {
    static let shared = ContinuousBatchScheduler()
    
    /// Request priority levels.
    enum Priority: Int, Comparable, Sendable {
        case critical = 0  // System prompts, health checks
        case high = 1      // User messages
        case medium = 2    // Tool calls
        case low = 3       // Background tasks
        
        static func < (lhs: Priority, rhs: Priority) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }
    
    /// Request status.
    enum Status: String, Sendable {
        case pending = "pending"
        case processing = "processing"
        case completed = "completed"
        case failed = "failed"
        case cancelled = "cancelled"
    }
    
/// Request processing mode.
    enum ProcessingMode: String, Sendable {
        case prefill   // Prefill phase: compute KV cache
        case decode    // Decode phase: generate next token
        case mixed     // Mixed prefill+decode (single token prefill)
    }
    
    /// A scheduled request.
    struct ScheduledRequest: Identifiable, Sendable {
        let id: UUID
        let prompt: String
        let priority: Priority
        let conversationID: UUID?
        let createdAt: Date
        var status: Status
        var startedAt: Date?
        var completedAt: Date?
        var result: String?
        var error: String?
        
        /// Processing mode (prefill, decode, mixed).
        var mode: ProcessingMode = .prefill
        
        /// Number of tokens to generate in decode mode.
        var maxTokens: Int = 512
        
        /// Processing time in seconds.
        var processingTime: TimeInterval? {
            guard let startedAt, let completedAt else { return nil }
            return completedAt.timeIntervalSince(startedAt)
        }
        
/// Wait time in seconds.
    var waitTime: TimeInterval {
        let start = startedAt ?? Date()
        return start.timeIntervalSince(createdAt)
    }
    
    /// Token usage summary (prefill/decode counts).
    var tokenUsage: (prefill: Int, decode: Int)? {
        // Would be populated by the server handler
        nil
    }
    }
    
    // MARK: - Properties
    
    @Published var pendingRequests: [ScheduledRequest] = []
    @Published var processingRequests: [ScheduledRequest] = []
    @Published var completedRequests: [ScheduledRequest] = []
    @Published var isProcessing = false
    
    /// Maximum concurrent requests (typically 1 for single GPU).
    private let maxConcurrent: Int
    
    /// Maximum queue size.
    private let maxQueueSize: Int
    
    /// Target throughput (requests per second). When pending requests exceed this
    /// threshold, new requests are held back to prevent GPU saturation.
    private let targetTPS: Double
    
    /// Current estimated throughput (calculated over last N requests).
    private var _estimatedTPS: Double = 0
    
    /// Smoothed TPS for stable control decisions.
    private var tpSSmoothing: [TimeInterval] = []
    private let tpSWindow: Int = 10
    
    /// Processing task.
    private var processingTask: Task<Void, Never>?
    
    /// Request handler closure.
    private var requestHandler: ((String, UUID?) async throws -> String)?
    
    // MARK: - Initialization
    
    init(maxConcurrent: Int = 1, maxQueueSize: Int = 100, targetTPS: Double = 20) {
        self.maxConcurrent = maxConcurrent
        self.maxQueueSize = maxQueueSize
        self.targetTPS = targetTPS
    }
    
    // MARK: - Public API
    
    /// Set the request handler.
    func setRequestHandler(_ handler: @escaping (String, UUID?) async throws -> String) {
        requestHandler = handler
    }
    
    /// Enqueue a new request.
    func enqueue(
        prompt: String,
        priority: Priority = .high,
        conversationID: UUID? = nil
    ) -> UUID? {
        guard pendingRequests.count < maxQueueSize else {
            return nil  // Queue full
        }
        
        let request = ScheduledRequest(
            id: UUID(),
            prompt: prompt,
            priority: priority,
            conversationID: conversationID,
            createdAt: Date(),
            status: .pending
        )
        
        pendingRequests.append(request)
        sortPendingQueue()
        
        startProcessingIfNeeded()
        
        return request.id
    }
    
    /// Cancel a pending request.
    func cancelRequest(id: UUID) -> Bool {
        if let index = pendingRequests.firstIndex(where: { $0.id == id }) {
            pendingRequests[index].status = .cancelled
            pendingRequests.remove(at: index)
            return true
        }
        
        if let index = processingRequests.firstIndex(where: { $0.id == id }) {
            processingRequests[index].status = .cancelled
            processingRequests.remove(at: index)
            return true
        }
        
        return false
    }
    
    /// Get request status.
    func getRequest(id: UUID) -> ScheduledRequest? {
        pendingRequests.first { $0.id == id }
            ?? processingRequests.first { $0.id == id }
            ?? completedRequests.first { $0.id == id }
    }
    
    /// Clear completed requests.
    func clearCompleted() {
        completedRequests.removeAll()
    }
    
    /// Get queue statistics.
    func getStatistics() -> QueueStatistics {
        QueueStatistics(
            pending: pendingRequests.count,
            processing: processingRequests.count,
            completed: completedRequests.count,
            totalProcessed: completedRequests.count,
            averageWaitTime: averageWaitTime(),
            averageProcessingTime: averageProcessingTime()
        )
    }
    
    // MARK: - Private Methods
    
    private func sortPendingQueue() {
        pendingRequests.sort { $0.priority < $1.priority || 
            ($0.priority == $1.priority && $0.createdAt < $1.createdAt) }
    }
    
    private func startProcessingIfNeeded() {
        guard processingRequests.count < maxConcurrent,
              !pendingRequests.isEmpty,
              let handler = requestHandler else {
            return
        }
        
        isProcessing = true
        
        processingTask?.cancel()
        processingTask = Task { [weak self] in
            guard let self else { return }
            
            while !Task.isCancelled {
                // Backpressure: if current TPS exceeds target, hold back new requests
                if self._estimatedTPS > self.targetTPS * 1.2 {
                    // Hold back: don't dequeue new request, let current finish
                    await MainActor.run {
                        self.isProcessing = !self.pendingRequests.isEmpty
                    }
                    continue
                }
                
                guard self.processingRequests.count < self.maxConcurrent,
                      !self.pendingRequests.isEmpty else {
                    break
                }
                
                // Priority scheduling with preemption:
                // Compare the highest-priority pending request against the lowest-priority processing request
                if let lowestProcessing = self.processingRequests.min(by: { $0.priority.rawValue < $1.priority.rawValue }),
                   let highestPending = self.pendingRequests.min(by: { $0.priority.rawValue < $1.priority.rawValue }),
                   highestPending.priority.rawValue < lowestProcessing.priority.rawValue {
                    // Preempt lowest priority processing request
                    if let lowestIdx = self.processingRequests.enumerated()
                        .max(by: { $0.element.priority.rawValue > $1.element.priority.rawValue })?.offset {
                        var preempted = self.processingRequests.remove(at: lowestIdx)
                        preempted.status = .pending
                        preempted.startedAt = nil
                        preempted.completedAt = nil
                        preempted.result = nil
                        preempted.error = nil
                        self.pendingRequests.append(preempted)
                        self.sortPendingQueue()
                        await MainActor.run {
                            self.isProcessing = !self.pendingRequests.isEmpty
                        }
                        continue  // Restart loop to process higher priority request
                    }
                }
                
                // Get next highest priority request from queue
                self.pendingRequests.sort { $0.priority < $1.priority || 
                    ($0.priority == $1.priority && $0.createdAt < $1.createdAt) }
                let request = self.pendingRequests.removeFirst()
                self.processingRequests.append(request)
                
                // Process request and track TPS
                let requestStart = Date()
                await self.processRequest(request, handler: handler)
                let elapsed = Date().timeIntervalSince(requestStart)
                
                // Update TPS smoothing window
                self.tpSSmoothing.append(elapsed)
                if self.tpSSmoothing.count > self.tpSWindow {
                    self.tpSSmoothing.removeFirst()
                }
                if !tpSSmoothing.isEmpty {
                    let avgInterval = tpSSmoothing.reduce(0, +) / Double(tpSSmoothing.count)
                    self._estimatedTPS = 1.0 / avgInterval
                }
                
                await MainActor.run {
                    self.isProcessing = !self.pendingRequests.isEmpty
                }
            }
        }
    }
    
    private func processRequest(
        _ request: ScheduledRequest,
        handler: @escaping (String, UUID?) async throws -> String
    ) async {
        guard let index = processingRequests.firstIndex(where: { $0.id == request.id }) else {
            return
        }
        
        processingRequests[index].status = .processing
        processingRequests[index].startedAt = Date()
        
        do {
            let result = try await handler(request.prompt, request.conversationID)
            
            guard !Task.isCancelled else { return }
            
            processingRequests[index].status = .completed
            processingRequests[index].result = result
            processingRequests[index].completedAt = Date()
            
            // Move to completed
            let completed = processingRequests.remove(at: index)
            completedRequests.append(completed)
            
        } catch {
            guard !Task.isCancelled else { return }
            
            processingRequests[index].status = .failed
            processingRequests[index].error = error.localizedDescription
            processingRequests[index].completedAt = Date()
            
            // Move to completed
            let completed = processingRequests.remove(at: index)
            completedRequests.append(completed)
        }
        
        // Continue processing next request
        startProcessingIfNeeded()
    }
    
    private func averageWaitTime() -> TimeInterval {
        let completed = completedRequests.filter { $0.status == .completed }
        guard !completed.isEmpty else { return 0 }
        
        let totalWait = completed.reduce(0) { $0 + $1.waitTime }
        return totalWait / Double(completed.count)
    }
    
    private func averageProcessingTime() -> TimeInterval {
        let completed = completedRequests.filter { $0.status == .completed }
        let withTime = completed.compactMap(\.processingTime)
        guard !withTime.isEmpty else { return 0 }
        
        let total = withTime.reduce(0, +)
        return total / Double(withTime.count)
    }
}

// MARK: - Queue Statistics

/// Statistics for the request queue.
struct QueueStatistics: Sendable {
    let pending: Int
    let processing: Int
    let completed: Int
    let totalProcessed: Int
    let averageWaitTime: TimeInterval
    let averageProcessingTime: TimeInterval
    
    /// Human-readable summary.
    var summary: String {
        "\(pending) pending, \(processing) processing, \(completed) completed"
    }
    
    /// Throughput (requests per minute).
    var throughput: Double {
        guard averageProcessingTime > 0 else { return 0 }
        return 60.0 / averageProcessingTime
    }
}

// MARK: - Server Integration

extension ContinuousBatchScheduler {
    /// Configure scheduler for llama.cpp server integration.
    func configureForServer(
        port: Int,
        modelPath: String,
        contextLength: Int,
        gpuLayers: Int
    ) {
        // Generate server arguments with continuous batching support
        var args = [
            "--host", "127.0.0.1",
            "--port", String(port),
            "--model", modelPath,
            "--ctx-size", String(contextLength),
            "--n-gpu-layers", String(gpuLayers),
            "--cont-batching"  // Enable continuous batching
        ]
        
        // Add Flash Attention if supported
        args += ["--flash-attn"]
        
        // Set request handler
        setRequestHandler { [weak self] prompt, conversationID in
            // This would integrate with the actual server
            // For now, return a placeholder
            return "Response for: \(prompt.prefix(50))..."
        }
    }
    
    /// Start the scheduler.
    func start() {
        isProcessing = true
        startProcessingIfNeeded()
    }
    
    /// Stop the scheduler.
    func stop() {
        processingTask?.cancel()
        isProcessing = false
        
        // Cancel all pending requests
        pendingRequests.removeAll()
        processingRequests.removeAll()
    }
}
