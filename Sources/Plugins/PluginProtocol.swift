// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import os

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
/// `@MainActor`: it hands out the main-actor-isolated plugin manager.
@MainActor
struct PluginContext {
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

/// Type-erased Codable value using enum for type safety.
enum AnyCodable: Codable, Sendable {
    case int(Int)
    case double(Double)
    case bool(Bool)
    case string(String)
    case null
    
    init(_ value: Any) {
        if let intVal = value as? Int { self = .int(intVal) }
        else if let doubleVal = value as? Double { self = .double(doubleVal) }
        else if let boolVal = value as? Bool { self = .bool(boolVal) }
        else if let stringVal = value as? String { self = .string(stringVal) }
        else { self = .null }
    }
    
    var value: Any {
        switch self {
        case .int(let v): return v
        case .double(let v): return v
        case .bool(let v): return v
        case .string(let v): return v
        case .null: return NSNull()
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
    case toolCalled(name: String, arguments: [String: AnyCodable])
    case custom(name: String, data: [String: AnyCodable])
}

// MARK: - Plugin Logger

/// Logger for plugin output; routes into the unified log so plugin messages
/// show up alongside the rest of the app in Console and the session file.
struct PluginLogger: Sendable {
    let pluginID: String
    
    func info(_ message: String) {
        AppLog.plugins.info("[\(self.pluginID, privacy: .public)] INFO: \(message, privacy: .public)")
    }
    
    func warning(_ message: String) {
        AppLog.plugins.warning("[\(self.pluginID, privacy: .public)] WARNING: \(message, privacy: .public)")
    }
    
    func error(_ message: String) {
        AppLog.plugins.error("[\(self.pluginID, privacy: .public)] ERROR: \(message, privacy: .public)")
    }
    
    func debug(_ message: String) {
        #if DEBUG
        AppLog.plugins.debug("[\(self.pluginID, privacy: .public)] DEBUG: \(message, privacy: .public)")
        #endif
    }
}

// MARK: - Plugin Manager Protocol

/// Protocol for plugin manager services.
/// `@MainActor`: the only implementation is `PluginManager`, whose registry is
/// main-actor state. Declaring the isolation here keeps conformances from
/// crossing actor boundaries (an error under the Swift 6 language mode).
@MainActor
protocol PluginManagerProtocol {
    func getService<T>(_ type: T.Type) -> T?
    func registerService<T>(_ service: T, as type: T.Type)
}

// MARK: - Default Plugin Implementation

/// Base class for plugins with default implementations.
/// `@unchecked Sendable`: every stored property is immutable except `context`,
/// which is behind a lock — a non-final class can't get this checked.
class BasePlugin: Plugin, @unchecked Sendable {
    let id: String
    let name: String
    let version: String
    let description: String
    let author: String
    let dependencies: [String]
    let capabilities: [PluginCapability]
    
    private let contextLock = NSLock()
    private var _context: PluginContext?
    var context: PluginContext? {
        get { contextLock.lock(); defer { contextLock.unlock() }; return _context }
        set { contextLock.lock(); _context = newValue; contextLock.unlock() }
    }
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
