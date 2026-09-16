// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Manages chunked prefill for long prompts.
/// Inspired by vLLM and llama.cpp upstream implementations.
@MainActor
final class ChunkedPrefillManager: ObservableObject {
    static let shared = ChunkedPrefillManager()
    
    /// Configuration for chunked prefill.
    struct Configuration: Sendable {
        /// Maximum tokens per chunk.
        let chunkSize: Int
        
        /// Whether chunked prefill is enabled.
        let enabled: Bool
        
        /// Minimum prompt length to trigger chunking.
        let minLength: Int
        
        /// Overlap tokens between chunks (for context continuity).
        let overlapTokens: Int
        
        /// Default configuration.
        static let `default` = Configuration(
            chunkSize: 512,
            enabled: true,
            minLength: 1024,
            overlapTokens: 32
        )
        
        /// Conservative configuration for low VRAM.
        static let conservative = Configuration(
            chunkSize: 256,
            enabled: true,
            minLength: 512,
            overlapTokens: 16
        )
        
        /// Aggressive configuration for high VRAM.
        static let aggressive = Configuration(
            chunkSize: 1024,
            enabled: true,
            minLength: 2048,
            overlapTokens: 64
        )
    }
    
    /// A chunk of the prompt.
    struct PromptChunk: Identifiable, Sendable {
        let id: UUID
        let index: Int
        let tokens: [String]
        let startTokenIndex: Int
        let endTokenIndex: Int
        var status: ChunkStatus
        var startTime: Date?
        var endTime: Date?
        var tokensPerSecond: Double?
        
        /// Duration in seconds.
        var duration: TimeInterval? {
            guard let startTime, let endTime else { return nil }
            return endTime.timeIntervalSince(startTime)
        }
        
        enum ChunkStatus: String, Sendable {
            case pending = "pending"
            case processing = "processing"
            case completed = "completed"
            case failed = "failed"
        }
    }
    
    /// Prefill statistics.
    struct PrefillStats: Sendable {
        let totalTokens: Int
        let chunksProcessed: Int
        let totalDuration: TimeInterval
        let averageTokensPerSecond: Double
        let firstTokenLatency: TimeInterval?
        let chunkDetails: [PromptChunk]
    }
    
    // MARK: - Properties
    
    @Published var configuration: Configuration = .default
    @Published var isProcessing = false
    @Published var currentChunks: [PromptChunk] = []
    @Published var stats: PrefillStats?
    
    /// Progress callback (0.0 to 1.0).
    var onProgress: ((Double) -> Void)?
    
    /// Completion callback.
    var onCompletion: ((PrefillStats) -> Void)?
    
    // MARK: - Public API
    
    /// Configure chunked prefill settings.
    func configure(_ config: Configuration) {
        configuration = config
    }
    
    /// Process a long prompt with chunked prefill.
    func processPrompt(
        _ prompt: String,
        tokenCount: Int,
        handler: @escaping ([String], Int, Int) async throws -> String
    ) async throws -> PrefillStats {
        guard configuration.enabled, tokenCount >= configuration.minLength else {
            // Process as single chunk
            return try await processSingleChunk(prompt, handler: handler)
        }
        
        // Split prompt into chunks
        let chunks = splitIntoChunks(prompt: prompt, tokenCount: tokenCount)
        currentChunks = chunks
        isProcessing = true
        
        let startTime = Date()
        var processedChunks: [PromptChunk] = []
        var firstTokenLatency: TimeInterval?
        var totalTokens = 0
        
        for (index, chunk) in chunks.enumerated() {
            guard !Task.isCancelled else { break }
            
            var updatedChunk = chunk
            updatedChunk.status = .processing
            updatedChunk.startTime = Date()
            currentChunks[index] = updatedChunk
            
            do {
                let result = try await handler(
                    chunk.tokens,
                    chunk.startTokenIndex,
                    chunk.endTokenIndex
                )
                
                guard !Task.isCancelled else { break }
                
                let endTime = Date()
                guard let chunkStartTime = updatedChunk.startTime else {
                    updatedChunk.status = .completed
                    updatedChunk.endTime = endTime
                    currentChunks[index] = updatedChunk
                    continue
                }
                let duration = endTime.timeIntervalSince(chunkStartTime)
                let tps = Double(chunk.tokens.count) / max(0.001, duration)
                
                updatedChunk.status = .completed
                updatedChunk.endTime = endTime
                updatedChunk.tokensPerSecond = tps
                currentChunks[index] = updatedChunk
                processedChunks.append(updatedChunk)
                totalTokens += chunk.tokens.count
                
                // Track first token latency
                if firstTokenLatency == nil, index == 0 {
                    firstTokenLatency = duration
                }
                
                // Report progress
                let progress = Double(index + 1) / Double(chunks.count)
                onProgress?(progress)
                
            } catch {
                updatedChunk.status = .failed
                updatedChunk.endTime = Date()
                currentChunks[index] = updatedChunk
                processedChunks.append(updatedChunk)
            }
        }
        
        let totalDuration = Date().timeIntervalSince(startTime)
        let avgTPS = totalDuration > 0 ? Double(totalTokens) / totalDuration : 0
        
        let stats = PrefillStats(
            totalTokens: totalTokens,
            chunksProcessed: processedChunks.count,
            totalDuration: totalDuration,
            averageTokensPerSecond: avgTPS,
            firstTokenLatency: firstTokenLatency,
            chunkDetails: processedChunks
        )
        
        self.stats = stats
        isProcessing = false
        onCompletion?(stats)
        
        return stats
    }
    
    /// Cancel current prefill operation.
    func cancel() {
        isProcessing = false
        currentChunks.removeAll()
    }
    
    // MARK: - Private Methods
    
    private func processSingleChunk(
        _ prompt: String,
        handler: @escaping ([String], Int, Int) async throws -> String
    ) async throws -> PrefillStats {
        let startTime = Date()
        
        let chunk = PromptChunk(
            id: UUID(),
            index: 0,
            tokens: prompt.components(separatedBy: .whitespaces),
            startTokenIndex: 0,
            endTokenIndex: prompt.count,
            status: .processing,
            startTime: startTime
        )
        
        _ = try await handler(chunk.tokens, 0, prompt.count)
        
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        
        let stats = PrefillStats(
            totalTokens: chunk.tokens.count,
            chunksProcessed: 1,
            totalDuration: duration,
            averageTokensPerSecond: Double(chunk.tokens.count) / max(0.001, duration),
            firstTokenLatency: duration,
            chunkDetails: [PromptChunk(
                id: chunk.id,
                index: 0,
                tokens: chunk.tokens,
                startTokenIndex: 0,
                endTokenIndex: prompt.count,
                status: .completed,
                startTime: startTime,
                endTime: endTime,
                tokensPerSecond: Double(chunk.tokens.count) / max(0.001, duration)
            )]
        )
        
        self.stats = stats
        onCompletion?(stats)
        
        return stats
    }
    
    private func splitIntoChunks(prompt: String, tokenCount: Int) -> [PromptChunk] {
        let words = prompt.components(separatedBy: .whitespaces)
        let chunkSize = configuration.chunkSize
        let overlap = configuration.overlapTokens
        
        var chunks: [PromptChunk] = []
        var startIndex = 0
        var chunkIndex = 0
        
        while startIndex < words.count {
            let endIndex = min(startIndex + chunkSize, words.count)
            let chunkWords = Array(words[startIndex..<endIndex])
            
            let chunk = PromptChunk(
                id: UUID(),
                index: chunkIndex,
                tokens: chunkWords,
                startTokenIndex: startIndex,
                endTokenIndex: endIndex,
                status: .pending
            )
            
            chunks.append(chunk)
            chunkIndex += 1
            
            // Move to next chunk with overlap
            startIndex = endIndex - overlap
            if startIndex >= words.count { break }
        }
        
        return chunks
    }
}

// MARK: - Server Integration

extension ChunkedPrefillManager {
    /// Configure for llama.cpp server.
    func configureForServer(
        contextLength: Int,
        vramGB: Double,
        modelGB: Double
    ) {
        // Adjust chunk size based on available VRAM
        let availableVRAM = vramGB - modelGB
        let config: Configuration
        
        if availableVRAM < 4 {
            config = .conservative
        } else if availableVRAM > 16 {
            config = .aggressive
        } else {
            config = .default
        }
        
        configure(config)
    }
    
    /// Get recommended chunk size for hardware.
    static func recommendedChunkSize(
        vramGB: Double,
        modelGB: Double,
        contextLength: Int
    ) -> Int {
        let availableVRAM = vramGB - modelGB
        let tokensPerGB: Double = 1024  // Approximate tokens per GB of VRAM
        
        let maxTokens = Int(availableVRAM * tokensPerGB)
        let chunkSize = min(maxTokens, 1024)  // Cap at 1024 tokens
        
        return max(128, chunkSize)  // Minimum 128 tokens
    }
}
