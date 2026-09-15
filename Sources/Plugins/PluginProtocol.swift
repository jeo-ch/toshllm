// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Plugin protocol defining the interface for all ToshLLM plugins.
/// Inspired by deepseek-harness Cordis and openclaw plugin SDK.
protocol Plugin: AnyObject, Sendable {
    
    /// Unique identifier for this plugin.
    var id: String { get }
    
    /// Human-readable name.
    var name: String { get }
    
    /// Plugin version.
    var version: String { get }
    
    /// Plugin description.
    var description: String { get }
    
    /// Plugin author.
    var author: String { get }
    
    /// Plugin dependencies (other plugin IDs).
    var dependencies: [String] { get }
    
    /// Plugin capabilities.
    var capabilities: [PluginCapability] { get }
    
    /// Initialize the plugin.
    func initialize(context: PluginContext) async throws
    
    /// Cleanup when plugin is unloaded.
    func cleanup() async
    
    /// Handle a plugin event.
    func handleEvent(_ event: PluginEvent) async throws
}

// MARK: - Plugin Capability

/// Defines what a plugin can do.
enum PluginCapability: String, Codable, Sendable {
    case inferenceBackend = "inference_backend"
    case modelSource = "model_source"
    case tool = "tool"
    case uiPanel = "ui_panel"
    case storage = "storage"
    case network = "network"
}

// MARK: - Plugin Context

/// Context provided to plugins during initialization.
struct PluginContext: Sendable {
    let pluginManager: PluginManagerProtocol
    let settings: PluginSettings
    let logger: PluginLogger
    
    /// Get a service from the plugin manager.
    func getService<T>(_ type: T.Type) -> T? {
        pluginManager.getService(type)
    }
}

// MARK: - Plugin Settings

/// Plugin-specific settings.
struct PluginSettings: Codable, Sendable {
    let pluginID: String
    var settings: [String: AnyCodable]
    
    /// Get a setting value.
    func get<T>(_ key: String, as type: T.Type) -> T? {
        settings[key]?.value as? T
    }
    
    /// Set a setting value.
    mutating func set<T>(_ key: String, value: T) {
        settings[key] = AnyCodable(value)
    }
}

// MARK: - AnyCodable Helper

/// Type-erased Codable value.
struct AnyCodable: Codable, Sendable {
    let value: Any
    
    init(_ value: Any) {
        self.value = value
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) {
            value = intVal
        } else if let doubleVal = try? container.decode(Double.self) {
            value = doubleVal
        } else if let boolVal = try? container.decode(Bool.self) {
            value = boolVal
        } else if let stringVal = try? container.decode(String.self) {
            value = stringVal
        } else {
            value = NSNull()
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let intVal = value as? Int {
            try container.encode(intVal)
        } else if let doubleVal = value as? Double {
            try container.encode(doubleVal)
        } else if let boolVal = value as? Bool {
            try container.encode(boolVal)
        } else if let stringVal = value as? String {
            try container.encode(stringVal)
        }
    }
}

// MARK: - Plugin Event

/// Events that plugins can handle.
enum PluginEvent: Sendable {
    case appLaunched
    case appTerminating
    case serverStarted(port: Int)
    case serverStopped
    case modelLoaded(path: String)
    case modelUnloaded
    case messageReceived(content: String)
    case messageSent(content: String)
    case toolCalled(name: String, arguments: [String: Any])
    case custom(name: String, data: [String: Any])
}

// MARK: - Plugin Logger

/// Logger for plugin output.
struct PluginLogger: Sendable {
    let pluginID: String
    
    func info(_ message: String) {
        print("[\(pluginID)] INFO: \(message)")
    }
    
    func warning(_ message: String) {
        print("[\(pluginID)] WARNING: \(message)")
    }
    
    func error(_ message: String) {
        print("[\(pluginID)] ERROR: \(message)")
    }
    
    func debug(_ message: String) {
        #if DEBUG
        print("[\(pluginID)] DEBUG: \(message)")
        #endif
    }
}

// MARK: - Plugin Manager Protocol

/// Protocol for plugin manager services.
protocol PluginManagerProtocol: Sendable {
    func getService<T>(_ type: T.Type) -> T?
    func registerService<T>(_ service: T, as type: T.Type)
}

// MARK: - Default Plugin Implementation

/// Base class for plugins with default implementations.
class BasePlugin: Plugin {
    let id: String
    let name: String
    let version: String
    let description: String
    let author: String
    let dependencies: [String]
    let capabilities: [PluginCapability]
    
    var context: PluginContext?
    var logger: PluginLogger { PluginLogger(pluginID: id) }
    
    init(
        id: String,
        name: String,
        version: String,
        description: String,
        author: String = "ToshLLM",
        dependencies: [String] = [],
        capabilities: [PluginCapability] = []
    ) {
        self.id = id
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies
        self.capabilities = capabilities
    }
    
    func initialize(context: PluginContext) async throws {
        self.context = context
        logger.info("Plugin initialized")
    }
    
    func cleanup() async {
        logger.info("Plugin cleaned up")
    }
    
    func handleEvent(_ event: PluginEvent) async throws {
        // Default: ignore events
    }
}
