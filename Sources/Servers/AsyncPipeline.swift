// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Unified async pipeline for download and inference operations.
/// Replaces mixed Task/CombineLatest patterns with structured concurrency.
/// Inspired by jcode's mpsc + singleton proxy pattern.
@MainActor
final class AsyncPipeline: ObservableObject {
    static let shared = AsyncPipeline()
    
    /// Pipeline stage types.
    enum Stage: String, Sendable {
        case idle = "idle"
        case downloading = "downloading"
        case loading = "loading"
        case inferring = "inferring"
        case error = "error"
    }
    
    /// Pipeline status.
    @Published var status: Stage = .idle
    @Published var progress: Double = 0
    @Published var currentOperation: String = ""
    @Published var error: String?
    
    /// Download pipeline.
    private var downloadTask: Task<Void, Never>?
    
    /// Inference pipeline.
    private var inferenceTask: Task<Void, Never>?
    
    /// Cancellation support.
    private var currentTaskID: UUID?
    
    // MARK: - Download Pipeline
    
    /// Download a model with progress tracking and backpressure.
    func downloadModel(
        from url: URL,
        to destination: URL,
        progress: @escaping @MainActor (Double) -> Void
    ) async throws -> URL {
        // A fresh download must not inherit bytes from a previous attempt.
        try? FileManager.default.removeItem(at: destination.appendingPathExtension("partial"))
        return try await downloadWithResume(from: url, to: destination, progress: progress)
    }
    
    /// Download with resumption support.
    /// Bytes always land in a `.partial` file next to `destination`; only a
    /// completed transfer is moved onto the final path, so an interrupted
    /// download never leaves a truncated file where a real model is expected.
    func downloadWithResume(
        from url: URL,
        to destination: URL,
        existingBytes: Int64 = 0,
        progress: @escaping @MainActor (Double) -> Void
    ) async throws -> URL {
        let taskID = UUID()
        currentTaskID = taskID
        
        status = .downloading
        currentOperation = "Downloading \(url.lastPathComponent)"
        self.progress = 0
        
        defer {
            if currentTaskID == taskID {
                status = .idle
                self.progress = 0
                currentOperation = ""
            }
        }
        
        let fileManager = FileManager.default
        let partialURL = destination.appendingPathExtension("partial")
        
        // Resume from what is actually on disk, not from a caller-supplied
        // count: the partial file is the single source of truth for how many
        // bytes can be continued from.
        let partialSize = (try? fileManager.attributesOfItem(atPath: partialURL.path))?[.size] as? Int64 ?? 0
        let resumeFrom = existingBytes > 0 ? partialSize : 0
        
        // Create resumable download request
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        
        if resumeFrom > 0 {
            request.setValue("bytes=\(resumeFrom)-", forHTTPHeaderField: "Range")
        }
        
        // `bytes(for:)` streams instead of pulling the whole (GB-sized) model
        // into memory the way `data(for:)` did.
        let (bytes, response) = try await NetworkManager.session.bytes(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw PipelineError.downloadFailed("Invalid response")
        }
        
        // A 206 means the server honored the Range and we append; anything
        // else restarts the file from zero, so truncate the partial bytes.
        let appending = resumeFrom > 0 && httpResponse.statusCode == 206
        if !appending || !fileManager.fileExists(atPath: partialURL.path) {
            fileManager.createFile(atPath: partialURL.path, contents: nil)
        }
        let fileHandle = try FileHandle(forWritingTo: partialURL)
        defer { try? fileHandle.close() }
        if appending {
            fileHandle.seekToEndOfFile()
        }
        
        // expectedContentLength is the remaining size on a 206 response.
        let remaining = httpResponse.expectedContentLength
        let totalBytes = remaining > 0 ? remaining + (appending ? resumeFrom : 0) : -1
        var writtenBytes = appending ? resumeFrom : 0
        var lastReported = -1.0
        
        for try await chunk in bytes {
            try fileHandle.write(contentsOf: chunk)
            writtenBytes += Int64(chunk.count)
            
            // Report progress as bytes land instead of only at 100%.
            guard totalBytes > 0 else { continue }
            let fraction = min(1.0, Double(writtenBytes) / Double(totalBytes))
            if fraction - lastReported >= 0.01 {
                lastReported = fraction
                await progress(fraction)
            }
        }
        
        // Replace the destination only now that every byte arrived.
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: partialURL, to: destination)
        
        await progress(1.0)
        return destination
    }
    
    // MARK: - Inference Pipeline
    
    /// Send inference request with streaming support.
    /// - Parameters:
    ///   - maxTokens: Generation cap for the reply; kept separate from
    ///     `contextLength`, which only bounds how much can fit in the prompt.
    ///   - port: Server port, defaulting to the one in Settings.
    func infer(
        prompt: String,
        modelPath: String,
        contextLength: Int,
        maxTokens: Int = 2048,
        port: Int = ServerSettings.fromDefaults().port,
        stream: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let taskID = UUID()
        currentTaskID = taskID
        
        status = .inferring
        currentOperation = "Processing request"
        progress = 0
        
        defer {
            if currentTaskID == taskID {
                status = .idle
                self.progress = 0
                currentOperation = ""
            }
        }
        
        // Build inference request
        let request = try buildInferenceRequest(
            prompt: prompt,
            modelPath: modelPath,
            contextLength: contextLength,
            maxTokens: maxTokens,
            port: port
        )
        
        // Execute with streaming
        let result = try await executeInference(request: request, stream: stream)
        
        return result
    }
    
    /// Execute inference with backpressure control.
    private func executeInference(
        request: URLRequest,
        stream: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let (data, response) = try await NetworkManager.session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw PipelineError.inferenceFailed("Invalid response")
        }
        
        // Parse streaming response
        if let text = String(data: data, encoding: .utf8) {
            await stream(text)
            return text
        }
        
        throw PipelineError.inferenceFailed("Failed to parse response")
    }
    
    private func buildInferenceRequest(
        prompt: String,
        modelPath: String,
        contextLength: Int,
        maxTokens: Int,
        port: Int
    ) throws -> URLRequest {
        // Build OpenAI-compatible request
        let body: [String: Any] = [
            "model": (modelPath as NSString).lastPathComponent,
            "messages": [
                ["role": "user", "content": prompt]
            ],
            // The context length is the prompt budget, not a generation cap:
            // it only upper-bounds how many tokens may be requested.
            "max_tokens": min(maxTokens, contextLength),
            "stream": false
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: body) else {
            throw PipelineError.requestBuildFailed
        }
        
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.httpBody = jsonData
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        if let apiKey = ServerSettings.activeAPIKey() {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        return request
    }
    
    // MARK: - Pipeline Management
    
    /// Cancel current operation.
    func cancel() {
        downloadTask?.cancel()
        inferenceTask?.cancel()
        currentTaskID = nil
        status = .idle
        progress = 0
        currentOperation = ""
        error = nil
    }
    
    /// Check if pipeline is busy.
    var isBusy: Bool {
        status != .idle
    }
    
    /// Get pipeline statistics.
    func getStatistics() -> PipelineStatistics {
        PipelineStatistics(
            status: status,
            progress: progress,
            currentOperation: currentOperation,
            hasError: error != nil
        )
    }
}

// MARK: - Pipeline Error

enum PipelineError: LocalizedError {
    case downloadFailed(String)
    case inferenceFailed(String)
    case requestBuildFailed
    case cancelled
    
    var errorDescription: String? {
        switch self {
        case .downloadFailed(let reason): return "Download failed: \(reason)"
        case .inferenceFailed(let reason): return "Inference failed: \(reason)"
        case .requestBuildFailed: return "Failed to build request"
        case .cancelled: return "Operation cancelled"
        }
    }
}

// MARK: - Pipeline Statistics

struct PipelineStatistics: Sendable {
    let status: AsyncPipeline.Stage
    let progress: Double
    let currentOperation: String
    let hasError: Bool
    
    var summary: String {
        switch status {
        case .idle: return "Ready"
        case .downloading: return "Downloading \(Int(progress * 100))%"
        case .loading: return "Loading model..."
        case .inferring: return "Generating..."
        case .error: return "Error occurred"
        }
    }
}

// MARK: - Download Manager Integration

extension AsyncPipeline {
    /// Integrate with existing download manager.
    func downloadModel(
        _ model: LocalModel,
        from url: URL,
        progress: @escaping @MainActor (Double) -> Void
    ) async throws -> URL {
        let destination = ServerSettings.modelsDirectory.appendingPathComponent(url.lastPathComponent)
        
        // Check for partial download
        let tempURL = destination.appendingPathExtension("partial")
        var existingBytes: Int64 = 0
        
        if FileManager.default.fileExists(atPath: tempURL.path) {
            let attributes = try? FileManager.default.attributesOfItem(atPath: tempURL.path)
            existingBytes = attributes?[.size] as? Int64 ?? 0
        }
        
        // Resume download if possible
        if existingBytes > 0 {
            return try await downloadWithResume(
                from: url,
                to: destination,
                existingBytes: existingBytes,
                progress: progress
            )
        } else {
            return try await downloadModel(from: url, to: destination, progress: progress)
        }
    }
}

// MARK: - Server Integration

extension AsyncPipeline {
    /// Configure for server operations.
    func configureForServer(port: Int) {
        // Update default server URL
        // This would integrate with the actual server configuration
    }
    
    /// Start inference pipeline.
    func startInference(
        prompt: String,
        modelPath: String,
        completion: @escaping @MainActor (Result<String, Error>) -> Void
    ) {
        inferenceTask?.cancel()
        
        inferenceTask = Task { [weak self] in
            guard let self else { return }
            
            do {
                let result = try await self.infer(
                    prompt: prompt,
                    modelPath: modelPath,
                    contextLength: 16384
                ) { _ in }
                
                await completion(.success(result))
            } catch {
                await completion(.failure(error))
            }
        }
    }
}
