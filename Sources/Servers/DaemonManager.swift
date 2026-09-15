// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Manages a background daemon process for the inference engine.
/// Inspired by jcode's single daemon architecture with Unix socket communication.
@MainActor
final class DaemonManager: ObservableObject {
    static let shared = DaemonManager()
    
    /// Daemon status.
    enum Status: String, Sendable {
        case stopped = "stopped"
        case starting = "starting"
        case running = "running"
        case error = "error"
        
        var icon: String {
            switch self {
            case .stopped: return "stop.circle"
            case .starting: return "arrow.triangle.2.circlepath"
            case .running: return "checkmark.circle.fill"
            case .error: return "exclamationmark.circle.fill"
            }
        }
        
        var color: String {
            switch self {
            case .stopped: return "gray"
            case .starting: return "yellow"
            case .running: return "green"
            case .error: return "red"
            }
        }
    }
    
    /// Daemon configuration.
    struct Configuration: Codable, Sendable {
        let port: Int
        let socketPath: String
        let logPath: String
        let pidPath: String
        let maxRestarts: Int
        let healthCheckInterval: TimeInterval
        
        static let `default` = Configuration(
            port: 8080,
            socketPath: "/tmp/toshllm.sock",
            logPath: "/tmp/toshllm-daemon.log",
            pidPath: "/tmp/toshllm.pid",
            maxRestarts: 3,
            healthCheckInterval: 5.0
        )
    }
    
    /// Daemon statistics.
    struct Statistics: Sendable {
        let uptime: TimeInterval
        let requestCount: Int
        let errorCount: Int
        let memoryUsage: UInt64
        let lastHealthCheck: Date?
        
        var uptimeFormatted: String {
            let hours = Int(uptime) / 3600
            let minutes = (Int(uptime) % 3600) / 60
            return "\(hours)h \(minutes)m"
        }
    }
    
    // MARK: - Properties
    
    @Published var status: Status = .stopped
    @Published var statistics: Statistics?
    @Published var lastError: String?
    
    let configuration: Configuration
    
    private var process: Process?
    private var healthCheckTask: Task<Void, Never>?
    private var restartCount = 0
    private var startedAt: Date?
    private var isRestarting = false
    
    // MARK: - Initialization
    
    init(configuration: Configuration = .default) {
        self.configuration = configuration
    }
    
    // MARK: - Public API
    
    /// Start the daemon process.
    func start() async throws {
        guard status == .stopped || status == .error else {
            return
        }
        
        status = .starting
        lastError = nil
        
        do {
            // Check if daemon is already running
            if isDaemonRunning() {
                throw DaemonError.alreadyRunning
            }
            
            // Launch daemon process
            try launchDaemon()
            
            // Wait for daemon to be ready
            try await waitForDaemonReady()
            
            // Start health check
            startHealthCheck()
            
            status = .running
            startedAt = Date()
            restartCount = 0
            
        } catch {
            status = .error
            lastError = error.localizedDescription
            throw error
        }
    }
    
    /// Stop the daemon process.
    func stop() {
        healthCheckTask?.cancel()
        healthCheckTask = nil
        
        if let process = process {
            // Send SIGTERM for graceful shutdown
            process.terminate()
            
            // Wait for process to exit (max 5 seconds) using non-blocking approach
            let deadline = Date().addingTimeInterval(5)
            while process.isRunning && Date() < deadline {
                // Use RunLoop to avoid blocking main thread
                RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0.1))
            }
            
            // Force kill if still running
            if process.isRunning {
                process.interrupt()
            }
        }
        
        process = nil
        status = .stopped
        startedAt = nil
        statistics = nil
        
        // Clean up PID file
        try? FileManager.default.removeItem(atPath: configuration.pidPath)
    }
    
    /// Restart the daemon process.
    func restart() async throws {
        stop()
        try await Task.sleep(for: .seconds(1))
        try await start()
    }
    
    /// Get daemon status.
    func getStatus() -> DaemonStatus {
        let uptime = startedAt.map { -($0.timeIntervalSinceNow) } ?? 0
        return DaemonStatus(
            isRunning: status == .running,
            pid: process?.processIdentifier,
            uptime: uptime,
            port: configuration.port,
            socketPath: configuration.socketPath
        )
    }
    
    /// Send a command to the daemon via Unix socket.
    func sendCommand(_ command: String) async throws -> String {
        guard status == .running else {
            throw DaemonError.notRunning
        }
        
        // Connect to Unix socket
        let socketPath = configuration.socketPath
        guard FileManager.default.fileExists(atPath: socketPath) else {
            throw DaemonError.socketNotFound
        }
        
        // Create socket connection
        let socket = try Socket.create(socketPath: socketPath)
        defer { try? socket.closeSocket() }
        
        // Send command
        try socket.write(command.data(using: .utf8)!)
        
        // Read response
        let response = try socket.read(maxLength: 4096)
        return String(data: response, encoding: .utf8) ?? ""
    }
    
    /// Reload daemon configuration.
    func reload() async throws {
        guard status == .running else {
            throw DaemonError.notRunning
        }
        
        _ = try await sendCommand("reload")
    }
    
    // MARK: - Private Methods
    
    private func isDaemonRunning() -> Bool {
        // Check PID file
        guard let pidData = try? Data(contentsOf: URL(fileURLWithPath: configuration.pidPath)),
              let pidString = String(data: pidData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
              let pid = Int32(pidString) else {
            return false
        }
        
        // Check if process is running
        return kill(pid, 0) == 0
    }
    
    private func launchDaemon() throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "swift", "run", "toshllm-daemon",
            "--port", String(configuration.port),
            "--socket", configuration.socketPath,
            "--log", configuration.logPath,
            "--pid", configuration.pidPath
        ]
        
        // Set up output pipes
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        
        // Set up termination handler
        process.terminationHandler = { [weak self] process in
            Task { @MainActor [weak self] in
                guard let self else { return }
                
                // Prevent recursive restart
                guard !self.isRestarting else { return }
                
                if process.terminationStatus != 0 {
                    self.status = .error
                    self.lastError = "Daemon exited with status \(process.terminationStatus)"
                    
                    // Attempt restart if within limits
                    if self.restartCount < self.configuration.maxRestarts {
                        self.isRestarting = true
                        self.restartCount += 1
                        try? await Task.sleep(for: .seconds(2))
                        self.isRestarting = false
                        try? await self.start()
                    }
                }
            }
        }
        
        try process.run()
        self.process = process
        
        // Write PID file
        let pidString = String(process.processIdentifier)
        try pidString.data(using: .utf8)?.write(to: URL(fileURLWithPath: configuration.pidPath))
    }
    
    private func waitForDaemonReady() async throws {
        let deadline = Date().addingTimeInterval(30) // 30 second timeout
        
        while Date() < deadline {
            // Check if daemon is listening on socket
            if FileManager.default.fileExists(atPath: configuration.socketPath) {
                // Try to connect
                if let socket = try? Socket.create(socketPath: configuration.socketPath) {
                    try? socket.closeSocket()
                    return
                }
            }
            
            // Check if process is still running
            guard let process, process.isRunning else {
                throw DaemonError.failedToStart
            }
            
            try await Task.sleep(for: .milliseconds(500))
        }
        
        throw DaemonError.startupTimeout
    }
    
    private func startHealthCheck() {
        healthCheckTask?.cancel()
        healthCheckTask = Task { [weak self] in
            guard let self else { return }
            
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(self.configuration.healthCheckInterval))
                
                guard !Task.isCancelled else { break }
                
                await self.performHealthCheck()
            }
        }
    }
    
    private func performHealthCheck() async {
        guard status == .running else { return }
        
        do {
            let response = try await sendCommand("health")
            if response.contains("ok") {
                // Update statistics
                let uptime = startedAt.map { -($0.timeIntervalSinceNow) } ?? 0
                statistics = Statistics(
                    uptime: uptime,
                    requestCount: 0,
                    errorCount: 0,
                    memoryUsage: 0,
                    lastHealthCheck: Date()
                )
            }
        } catch {
            // Health check failed
            status = .error
            lastError = "Health check failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Daemon Status

struct DaemonStatus: Sendable {
    let isRunning: Bool
    let pid: Int32?
    let uptime: TimeInterval
    let port: Int
    let socketPath: String
    
    var description: String {
        if isRunning {
            return "Running (PID: \(pid ?? 0), Port: \(port))"
        } else {
            return "Stopped"
        }
    }
}

// MARK: - Daemon Error

enum DaemonError: LocalizedError {
    case alreadyRunning
    case notRunning
    case failedToStart
    case startupTimeout
    case socketNotFound
    case commandFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .alreadyRunning: return "Daemon is already running"
        case .notRunning: return "Daemon is not running"
        case .failedToStart: return "Failed to start daemon"
        case .startupTimeout: return "Daemon startup timed out"
        case .socketNotFound: return "Socket file not found"
        case .commandFailed(let reason): return "Command failed: \(reason)"
        }
    }
}

// MARK: - Socket Helper

/// Simple Unix socket client for daemon communication.
private final class Socket {
    let fileDescriptor: Int32
    private var isClosed = false
    
    private init(fileDescriptor: Int32) {
        self.fileDescriptor = fileDescriptor
    }
    
    static func create(socketPath: String) throws -> Socket {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw DaemonError.commandFailed("Failed to create socket")
        }
        
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        
        guard socketPath.count < MemoryLayout.size(ofValue: addr.sun_path) else {
            close(fd)
            throw DaemonError.commandFailed("Socket path too long")
        }
        
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            socketPath.withCString { cstr in
                _ = strcpy(ptr, cstr)
            }
        }
        
        let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        guard withUnsafePointer(to: &addr, { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                connect(fd, sockPtr, addrLen) == 0
            }
        }) else {
            close(fd)
            throw DaemonError.commandFailed("Failed to connect to socket")
        }
        
        return Socket(fileDescriptor: fd)
    }
    
    func write(_ data: Data) throws {
        let result = data.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else {
                return -1
            }
            return send(fileDescriptor, baseAddress, data.count, 0)
        }
        guard result >= 0 else {
            throw DaemonError.commandFailed("Failed to write to socket")
        }
    }
    
    func read(maxLength: Int) throws -> Data {
        var buffer = [UInt8](repeating: 0, count: maxLength)
        let bytesRead = recv(fileDescriptor, &buffer, maxLength, 0)
        guard bytesRead >= 0 else {
            throw DaemonError.commandFailed("Failed to read from socket")
        }
        return Data(buffer.prefix(bytesRead))
    }
    
    func closeSocket() throws {
        guard !isClosed else { return }
        guard Darwin.close(fileDescriptor) >= 0 else {
            throw DaemonError.commandFailed("Failed to close socket")
        }
        isClosed = true
    }
    
    deinit {
        try? closeSocket()
    }
}
