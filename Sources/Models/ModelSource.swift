// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Unified protocol for different model sources.
/// Inspired by cc-haha's model-picker and opencode's model management.
protocol ModelSource: Sendable {
    /// Unique identifier for this source.
    var id: String { get }
    
    /// Human-readable name for this source.
    var name: String { get }
    
    /// Description of this source.
    var description: String { get }
    
    /// Whether this source is currently available.
    var isAvailable: Bool { get async }
    
    /// List available models from this source.
    func list() async throws -> [ModelSourceItem]
    
    /// Search for models matching a query.
    func search(query: String) async throws -> [ModelSourceItem]
    
    /// Download a model to local storage.
    func download(item: ModelSourceItem, progress: @escaping (Double) -> Void) async throws -> URL
    
    /// Get metadata for a specific model.
    func metadata(for item: ModelSourceItem) async throws -> ModelSourceMetadata
}

// MARK: - Model Source Item

/// Represents a model from any source.
struct ModelSourceItem: Identifiable, Sendable {
    let id: String
    let name: String
    let description: String
    let source: String
    let downloadURL: URL?
    let sizeBytes: Int64?
    let quantization: String?
    let architecture: String?
    let parameters: String?
    let license: String?
    let tags: [String]
    
    /// Human-readable size.
    var sizeDescription: String {
        guard let sizeBytes else { return "Unknown" }
        let gb = Double(sizeBytes) / 1_073_741_824
        if gb >= 1 {
            return String(format: "%.1f GB", gb)
        } else {
            let mb = Double(sizeBytes) / 1_048_576
            return String(format: "%.0f MB", mb)
        }
    }
}

// MARK: - Model Source Metadata

/// Extended metadata for a model.
struct ModelSourceMetadata: Sendable {
    let item: ModelSourceItem
    let readme: String?
    let samplePrompts: [String]
    let parentModel: String?
    let finetune: String?
    let quantizationNote: String?
    let usageGuide: String?
}

// MARK: - Local GGUF Source

/// Source for locally stored GGUF files.
struct LocalGGUFSource: ModelSource {
    let id = "local-gguf"
    let name = "Local Models"
    let description = "Models downloaded to your local models folder"
    
    var isAvailable: Bool { true }
    
    func list() async throws -> [ModelSourceItem] {
        let models = LocalModel.scan(in: ServerSettings.modelsDirectory)
        return models.map { model in
            ModelSourceItem(
                id: model.url.path,
                name: model.url.lastPathComponent,
                description: "Local GGUF file",
                source: id,
                downloadURL: nil,
                sizeBytes: model.sizeBytes,
                quantization: nil,
                architecture: nil,
                parameters: nil,
                license: nil,
                tags: ["local"]
            )
        }
    }
    
    func search(query: String) async throws -> [ModelSourceItem] {
        let allModels = try await list()
        return allModels.filter { model in
            model.name.localizedCaseInsensitiveContains(query) ||
            model.description.localizedCaseInsensitiveContains(query) ||
            model.tags.contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }
    
    func download(item: ModelSourceItem, progress: @escaping (Double) -> Void) async throws -> URL {
        // Local files are already downloaded - return the local path
        let localURL = URL(fileURLWithPath: item.id)
        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw ModelSourceError.fileNotFound
        }
        progress(1.0)
        return localURL
    }
    
    func metadata(for item: ModelSourceItem) async throws -> ModelSourceMetadata {
        ModelSourceMetadata(
            item: item,
            readme: nil,
            samplePrompts: [],
            parentModel: nil,
            finetune: nil,
            quantizationNote: nil,
            usageGuide: nil
        )
    }
}

// MARK: - Hugging Face Source

/// Source for Hugging Face models.
struct HuggingFaceSource: ModelSource {
    let id = "huggingface"
    let name = "Hugging Face"
    let description = "Browse and download models from Hugging Face"
    
    var isAvailable: Bool { true }
    
    func list() async throws -> [ModelSourceItem] {
        // Placeholder: In production, this would call the Hugging Face API
        // For now, return empty list
        return []
    }
    
    func search(query: String) async throws -> [ModelSourceItem] {
        // Placeholder: In production, this would search Hugging Face
        // For now, return empty list
        return []
    }
    
    func download(item: ModelSourceItem, progress: @escaping (Double) -> Void) async throws -> URL {
        guard let downloadURL = item.downloadURL else {
            throw ModelSourceError.downloadFailed
        }
        
        let modelsDir = ServerSettings.modelsDirectory
        let destination = modelsDir.appendingPathComponent(item.name)
        
        // Use async download API to avoid race condition
        let (tempURL, _) = try await URLSession.shared.download(from: downloadURL)
        
        // Copy temp file immediately (before system cleans it up)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: tempURL, to: destination)
        
        progress(1.0)
        return destination
    }
    
    func metadata(for item: ModelSourceItem) async throws -> ModelSourceMetadata {
        ModelSourceMetadata(
            item: item,
            readme: nil,
            samplePrompts: [],
            parentModel: nil,
            finetune: nil,
            quantizationNote: nil,
            usageGuide: nil
        )
    }
}

// MARK: - Ollama Source

/// Source for Ollama API integration.
struct OllamaSource: ModelSource {
    let id = "ollama"
    let name = "Ollama"
    let description = "Connect to a running Ollama instance"
    
    let baseURL: URL
    
    var isAvailable: Bool {
        get async {
            // Check if Ollama is running
            guard let url = URL(string: "\(baseURL)/api/tags") else { return false }
            var request = URLRequest(url: url)
            request.timeoutInterval = 2
            do {
                let (_, response) = try await URLSession.shared.data(for: request)
                return (response as? HTTPURLResponse)?.statusCode == 200
            } catch {
                return false
            }
        }
    }
    
    func list() async throws -> [ModelSourceItem] {
        guard let url = URL(string: "\(baseURL)/api/tags") else {
            throw ModelSourceError.invalidURL
        }
        
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try JSONDecoder().decode(OllamaTagsResponse.self, from: data)
        
        return response.models.map { model in
            ModelSourceItem(
                id: "ollama:\(model.name)",
                name: model.name,
                description: "Ollama model",
                source: id,
                downloadURL: nil,
                sizeBytes: model.size,
                quantization: nil,
                architecture: nil,
                parameters: nil,
                license: nil,
                tags: ["ollama"]
            )
        }
    }
    
    func search(query: String) async throws -> [ModelSourceItem] {
        let allModels = try await list()
        return allModels.filter { $0.name.localizedCaseInsensitiveContains(query) }
    }
    
    func download(item: ModelSourceItem, progress: @escaping (Double) -> Void) async throws -> URL {
        guard let url = URL(string: "\(baseURL)/api/pull") else {
            throw ModelSourceError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: ["name": item.name])
        
        let (_, _) = try await URLSession.shared.data(for: request)
        progress(1.0)
        
        // Ollama downloads to its own storage, return a placeholder URL
        return URL(fileURLWithPath: "")
    }
    
    func metadata(for item: ModelSourceItem) async throws -> ModelSourceMetadata {
        ModelSourceMetadata(
            item: item,
            readme: nil,
            samplePrompts: [],
            parentModel: nil,
            finetune: nil,
            quantizationNote: nil,
            usageGuide: nil
        )
    }
}

// MARK: - Ollama Response Types

private struct OllamaTagsResponse: Codable {
    let models: [OllamaModel]
}

private struct OllamaModel: Codable {
    let name: String
    let size: Int64
}

// MARK: - Model Source Error

enum ModelSourceError: LocalizedError {
    case fileNotFound
    case downloadFailed
    case invalidURL
    case sourceUnavailable
    case decodingFailed
    
    var errorDescription: String? {
        switch self {
        case .fileNotFound: return "Model file not found"
        case .downloadFailed: return "Download failed"
        case .invalidURL: return "Invalid URL"
        case .sourceUnavailable: return "Model source unavailable"
        case .decodingFailed: return "Failed to decode response"
        }
    }
}

// MARK: - Model Source Manager

/// Manages all available model sources.
@MainActor
final class ModelSourceManager: ObservableObject {
    static let shared = ModelSourceManager()
    
    @Published var sources: [ModelSource] = []
    @Published var activeSourceID: String = "local-gguf"
    
    private init() {
        // Initialize with default sources
        sources = [
            LocalGGUFSource(),
            HuggingFaceSource(),
            OllamaSource(baseURL: URL(string: "http://localhost:11434")!)
        ]
    }
    
    /// Get the currently active source.
    var activeSource: ModelSource? {
        sources.first { $0.id == activeSourceID }
    }
    
    /// Set the active source.
    func setActiveSource(_ id: String) {
        guard sources.contains(where: { $0.id == id }) else { return }
        activeSourceID = id
    }
    
    /// List models from all available sources.
    func listAllModels() async throws -> [ModelSourceItem] {
        var allModels: [ModelSourceItem] = []
        
        for source in sources where await source.isAvailable {
            let models = try await source.list()
            allModels.append(contentsOf: models)
        }
        
        return allModels
    }
    
    /// Search across all available sources.
    func searchAll(query: String) async throws -> [ModelSourceItem] {
        var results: [ModelSourceItem] = []
        
        for source in sources where await source.isAvailable {
            let models = try await source.search(query: query)
            results.append(contentsOf: models)
        }
        
        return results
    }
}
