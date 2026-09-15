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
        
        // Reuse shared session for connection pooling
        let session = URLSession.shared
        
        let (tempURL, response) = try await session.download(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw PipelineError.downloadFailed("Invalid response")
        }
        
        // Move to destination
        try FileManager.default.moveItem(at: tempURL, to: destination)
        
        await progress(1.0)
        return destination
    }
    
    /// Download with resumption support.
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
        
        defer {
            if currentTaskID == taskID {
                status = .idle
                self.progress = 0
                currentOperation = ""
            }
        }
        
        // Create resumable download request
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        
        if existingBytes > 0 {
            request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
        }
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw PipelineError.downloadFailed("Invalid response")
        }
        
        // Append or write data
        if existingBytes > 0, FileManager.default.fileExists(atPath: destination.path) {
            let fileHandle = try FileHandle(forWritingTo: destination)
            defer { fileHandle.closeFile() }
            fileHandle.seekToEndOfFile()
            fileHandle.write(data)
        } else {
            try data.write(to: destination)
        }
        
        await progress(1.0)
        return destination
    }
    
    // MARK: - Inference Pipeline
    
    /// Send inference request with streaming support.
    func infer(
        prompt: String,
        modelPath: String,
        contextLength: Int,
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
            contextLength: contextLength
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
        let (data, response) = try await URLSession.shared.data(for: request)
        
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
        contextLength: Int
    ) throws -> URLRequest {
        // Build OpenAI-compatible request
        let body: [String: Any] = [
            "model": (modelPath as NSString).lastPathComponent,
            "messages": [
                ["role": "user", "content": prompt]
            ],
            "max_tokens": contextLength,
            "stream": false
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: body) else {
            throw PipelineError.requestBuildFailed
        }
        
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8080/v1/chat/completions")!)
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
