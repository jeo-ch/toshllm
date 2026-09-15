// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Manages plugin loading, lifecycle, and event dispatching.
@MainActor
final class PluginManager: ObservableObject, PluginManagerProtocol {
    static let shared = PluginManager()
    
    /// Plugin state.
    enum PluginState: String, Sendable {
        case registered = "registered"
        case loading = "loading"
        case loaded = "loaded"
        case active = "active"
        case error = "error"
        case disabled = "disabled"
    }
    
    /// Plugin entry with metadata.
    struct PluginEntry: Identifiable, Sendable {
        let id: String
        let plugin: any Plugin
        var state: PluginState
        var loadedAt: Date?
        var error: String?
        
        var name: String { plugin.name }
        var version: String { plugin.version }
        var capabilities: [PluginCapability] { plugin.capabilities }
    }
    
    // MARK: - Properties
    
    @Published var plugins: [PluginEntry] = []
    @Published var isLoaded = false
    
    /// Registered services for dependency injection.
    private var services: [String: Any] = [:]
    
    /// Event handlers by plugin ID.
    private var eventHandlers: [String: [PluginEvent]] = [:]
    
    // MARK: - Initialization
    
    private init() {}
    
    // MARK: - Public API
    
    /// Register a plugin.
    func register(_ plugin: any Plugin) {
        guard !plugins.contains(where: { $0.id == plugin.id }) else {
            return
        }
        
        let entry = PluginEntry(
            id: plugin.id,
            plugin: plugin,
            state: .registered
        )
        
        plugins.append(entry)
    }
    
    /// Load all registered plugins.
    func loadAll() async {
        for index in plugins.indices {
            guard plugins[index].state == .registered else { continue }
            
            do {
                await loadPlugin(at: index)
            } catch {
                plugins[index].state = .error
                plugins[index].error = error.localizedDescription
            }
        }
        
        isLoaded = true
    }
    
    /// Load a specific plugin.
    private func loadPlugin(at index: Int) async {
        let plugin = plugins[index].plugin
        plugins[index].state = .loading
        
        let context = PluginContext(
            pluginManager: self,
            settings: PluginSettings(pluginID: plugin.id, settings: [:]),
            logger: PluginLogger(pluginID: plugin.id)
        )
        
        do {
            try await plugin.initialize(context: context)
            plugins[index].state = .loaded
            plugins[index].loadedAt = Date()
        } catch {
            plugins[index].state = .error
            plugins[index].error = error.localizedDescription
        }
    }
    
    /// Activate a plugin.
    func activate(id: String) async {
        guard let index = plugins.firstIndex(where: { $0.id == id }) else { return }
        guard plugins[index].state == .loaded else { return }
        
        plugins[index].state = .active
    }
    
    /// Deactivate a plugin.
    func deactivate(id: String) {
        guard let index = plugins.firstIndex(where: { $0.id == id }) else { return }
        guard plugins[index].state == .active else { return }
        
        plugins[index].state = .loaded
    }
    
    /// Unload a plugin.
    func unload(id: String) async {
        guard let index = plugins.firstIndex(where: { $0.id == id }) else { return }
        
        await plugins[index].plugin.cleanup()
        plugins[index].state = .registered
        plugins[index].loadedAt = nil
    }
    
    /// Remove a plugin.
    func remove(id: String) async {
        await unload(id: id)
        plugins.removeAll { $0.id == id }
    }
    
    /// Get a plugin by ID.
    func getPlugin(id: String) -> (any Plugin)? {
        plugins.first(where: { $0.id == id })?.plugin
    }
    
    /// Get all active plugins.
    func activePlugins() -> [PluginEntry] {
        plugins.filter { $0.state == .active }
    }
    
    /// Get plugins by capability.
    func plugins(with capability: PluginCapability) -> [PluginEntry] {
        plugins.filter { $0.capabilities.contains(capability) && $0.state == .active }
    }
    
    /// Dispatch an event to all active plugins.
    func dispatchEvent(_ event: PluginEvent) async {
        for plugin in activePlugins() {
            do {
                try await plugin.plugin.handleEvent(event)
            } catch {
                let logger = PluginLogger(pluginID: plugin.id)
                logger.error("Failed to handle event: \(error.localizedDescription)")
            }
        }
    }
    
    // MARK: - Service Registration
    
    /// Register a service for dependency injection.
    func registerService<T>(_ service: T, as type: T.Type) {
        let key = String(describing: type)
        services[key] = service
    }
    
    /// Get a service from the registry.
    func getService<T>(_ type: T.Type) -> T? {
        let key = String(describing: type)
        return services[key] as? T
    }
    
    // MARK: - Plugin Discovery
    
    /// Discover plugins from a directory.
    func discoverPlugins(from directory: URL) throws -> [any Plugin] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return []
        }
        
        var plugins: [any Plugin] = []
        
        for file in files where file.pathExtension == "toshplugin" {
            // In production, this would load the plugin bundle
            // For now, return empty
        }
        
        return plugins
    }
    
    /// Get plugin statistics.
    func statistics() -> PluginStatistics {
        PluginStatistics(
            total: plugins.count,
            active: activePlugins().count,
            loaded: plugins.filter { $0.state == .loaded }.count,
            error: plugins.filter { $0.state == .error }.count
        )
    }
}

// MARK: - Plugin Statistics

struct PluginStatistics: Sendable {
    let total: Int
    let active: Int
    let loaded: Int
    let error: Int
    
    var summary: String {
        "\(total) plugins, \(active) active, \(error) errors"
    }
}

// MARK: - Built-in Plugins

extension PluginManager {
    /// Register built-in plugins.
    func registerBuiltIns() {
        // Register example plugins
        register(InferenceBackendPlugin())
        register(ModelSourcePlugin())
    }
}

// MARK: - Example Inference Backend Plugin

/// Example plugin demonstrating inference backend integration.
private class InferenceBackendPlugin: BasePlugin {
    init() {
        super.init(
            id: "com.toshllm.inference.llamacpp",
            name: "llama.cpp Backend",
            version: "1.0.0",
            description: "llama.cpp inference backend for local model execution",
            capabilities: [.inferenceBackend]
        )
    }
    
    override func handleEvent(_ event: PluginEvent) async throws {
        switch event {
        case .serverStarted(let port):
            logger.info("Server started on port \(port)")
        case .serverStopped:
            logger.info("Server stopped")
        default:
            break
        }
    }
}

// MARK: - Example Model Source Plugin

/// Example plugin demonstrating model source integration.
private class ModelSourcePlugin: BasePlugin {
    init() {
        super.init(
            id: "com.toshllm.modelsource.local",
            name: "Local Model Source",
            version: "1.0.0",
            description: "Provides access to locally stored GGUF models",
            capabilities: [.modelSource]
        )
    }
    
    override func handleEvent(_ event: PluginEvent) async throws {
        switch event {
        case .modelLoaded(let path):
            logger.info("Model loaded: \(path)")
        case .modelUnloaded:
            logger.info("Model unloaded")
        default:
            break
        }
    }
}
