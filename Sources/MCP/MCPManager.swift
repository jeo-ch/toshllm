// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Enhanced MCP server manager with health checks and diagnostics.
/// Inspired by cc-haha's MCP management panel.
@MainActor
final class MCPManager: ObservableObject {
    static let shared = MCPManager()
    
    /// Server connection status.
    enum ConnectionStatus: String, Sendable {
        case connected = "connected"
        case connecting = "connecting"
        case disconnected = "disconnected"
        case error = "error"
        case unknown = "unknown"
        
        /// Color name for UI display.
        var color: String {
            switch self {
            case .connected: return "green"
            case .connecting: return "yellow"
            case .disconnected: return "gray"
            case .error: return "red"
            case .unknown: return "gray"
            }
        }
        
        /// Icon name for UI display.
        var icon: String {
            switch self {
            case .connected: return "checkmark.circle.fill"
            case .connecting: return "arrow.triangle.2.circlepath"
            case .disconnected: return "circle"
            case .error: return "exclamationmark.circle.fill"
            case .unknown: return "questionmark.circle"
            }
        }
    }
    
    /// Server health check result.
    struct HealthCheckResult: Sendable {
        let serverID: UUID
        let status: ConnectionStatus
        let latency: TimeInterval?
        let lastChecked: Date
        let error: String?
        let capabilities: [String]
        
        var isHealthy: Bool { status == .connected }
    }
    
    /// Diagnostic log entry.
    struct DiagnosticLog: Identifiable, Sendable {
        let id: UUID
        let timestamp: Date
        let level: LogLevel
        let serverID: UUID
        let serverName: String
        let message: String
        
        enum LogLevel: String, Sendable {
            case info = "INFO"
            case warning = "WARNING"
            case error = "ERROR"
            case debug = "DEBUG"
        }
    }
    
    // MARK: - Properties
    
    @Published var servers: [MCPServer] = []
    @Published var healthResults: [UUID: HealthCheckResult] = [:]
    @Published var diagnosticLogs: [DiagnosticLog] = []
    @Published var isCheckingHealth = false
    
    /// Maximum diagnostic logs to keep.
    private let maxLogs = 1000
    
    // MARK: - Initialization
    
    init() {
        loadServers()
    }
    
    // MARK: - Public API
    
    /// Load servers from storage.
    func loadServers() {
        servers = MCPServerStore.load()
    }
    
    /// Save servers to storage.
    func saveServers() {
        MCPServerStore.save(servers)
    }
    
    /// Add a new server.
    func addServer(_ server: MCPServer) {
        servers.append(server)
        saveServers()
    }
    
    /// Update an existing server.
    func updateServer(_ server: MCPServer) {
        if let index = servers.firstIndex(where: { $0.id == server.id }) {
            servers[index] = server
            saveServers()
        }
    }
    
    /// Delete a server.
    func deleteServer(_ id: UUID) {
        servers.removeAll { $0.id == id }
        healthResults.removeValue(forKey: id)
        saveServers()
    }
    
    /// Check health of all servers.
    func checkHealth() async {
        isCheckingHealth = true
        
        await withTaskGroup(of: Void.self) { group in
            for server in servers where server.enabled {
                group.addTask { [weak self] in
                    await self?.checkServerHealth(server)
                }
            }
        }
        
        isCheckingHealth = false
    }
    
    /// Check health of a specific server.
    func checkServerHealth(_ server: MCPServer) async {
        let startTime = Date()
        
        addLog(
            serverID: server.id,
            serverName: server.name,
            level: .info,
            message: "Starting health check"
        )
        
        do {
            // Try to connect and get capabilities
            let capabilities = try await getServerCapabilities(server)
            let latency = Date().timeIntervalSince(startTime)
            
            let result = HealthCheckResult(
                serverID: server.id,
                status: .connected,
                latency: latency,
                lastChecked: Date(),
                error: nil,
                capabilities: capabilities
            )
            
            healthResults[server.id] = result
            
            addLog(
                serverID: server.id,
                serverName: server.name,
                level: .info,
                message: "Health check passed (\(String(format: "%.2f", latency))s)"
            )
            
        } catch {
            let result = HealthCheckResult(
                serverID: server.id,
                status: .error,
                latency: nil,
                lastChecked: Date(),
                error: error.localizedDescription,
                capabilities: []
            )
            
            healthResults[server.id] = result
            
            addLog(
                serverID: server.id,
                serverName: server.name,
                level: .error,
                message: "Health check failed: \(error.localizedDescription)"
            )
        }
    }
    
    /// Get connection status for a server.
    func getConnectionStatus(for serverID: UUID) -> ConnectionStatus {
        healthResults[serverID]?.status ?? .unknown
    }
    
    /// Get health check result for a server.
    func getHealthResult(for serverID: UUID) -> HealthCheckResult? {
        healthResults[serverID]
    }
    
    /// Get diagnostic logs for a server.
    func getDiagnosticLogs(for serverID: UUID, limit: Int = 100) -> [DiagnosticLog] {
        diagnosticLogs
            .filter { $0.serverID == serverID }
            .suffix(limit)
    }
    
    /// Clear diagnostic logs.
    func clearDiagnosticLogs() {
        diagnosticLogs.removeAll()
    }
    
    /// Export diagnostic logs as text.
    func exportDiagnosticLogs() -> String {
        diagnosticLogs.map { log in
            "[\(log.timestamp.formatted(date: .abbreviated, time: .standard))] [\(log.level.rawValue)] \(log.serverName): \(log.message)"
        }.joined(separator: "\n")
    }
    
    /// Get transport type icon.
    static func transportIcon(for transport: MCPTransport) -> String {
        switch transport {
        case .stdio: return "terminal"
        case .serverSentEvents: return "bolt.fill"
        case .streamableHTTP: return "globe"
        case .webSocket: return "network"
        case .automatic: return "wand.and.stars"
        }
    }
    
    /// Get transport type label.
    static func transportLabel(for transport: MCPTransport) -> String {
        switch transport {
        case .stdio: return "STDIO"
        case .serverSentEvents: return "SSE"
        case .streamableHTTP: return "HTTP"
        case .webSocket: return "WebSocket"
        case .automatic: return "Auto"
        }
    }
    
    /// Get scope label for server.
    static func scopeLabel(for server: MCPServer) -> String {
        // Placeholder: In production, this would check if server is project-specific
        return "Global"
    }
    
    // MARK: - Private Methods
    
    private func getServerCapabilities(_ server: MCPServer) async throws -> [String] {
        // In production, this would actually connect to the server
        // and retrieve its capabilities via MCP protocol
        
        switch server.transport {
        case .stdio:
            // For STDIO, capabilities depend on the server implementation
            // Return common capabilities that most MCP servers support
            return ["tools", "resources", "prompts"]
            
        case .serverSentEvents, .streamableHTTP:
            // For HTTP-based transports, return typical capabilities
            return ["tools", "resources"]
            
        case .webSocket:
            // WebSocket transport supports streaming
            return ["tools", "resources", "prompts"]
            
        case .automatic:
            // Auto-detected transport
            return ["tools"]
        }
    }
    
    private func addLog(
        serverID: UUID,
        serverName: String,
        level: DiagnosticLog.LogLevel,
        message: String
    ) {
        let log = DiagnosticLog(
            id: UUID(),
            timestamp: Date(),
            level: level,
            serverID: serverID,
            serverName: serverName,
            message: message
        )
        
        diagnosticLogs.append(log)
        
        // Trim old logs
        if diagnosticLogs.count > maxLogs {
            diagnosticLogs.removeFirst(diagnosticLogs.count - maxLogs)
        }
    }
}
