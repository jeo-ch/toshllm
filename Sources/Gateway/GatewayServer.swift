// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Gateway server for multi-channel access to ToshLLM.
/// Inspired by openclaw Gateway architecture.
@MainActor
final class GatewayServer: ObservableObject {
    static let shared = GatewayServer()
    
    /// Gateway status.
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
    }
    
    /// Channel types supported by the gateway.
    enum ChannelType: String, Codable, CaseIterable, Sendable {
        case openAI = "openai"           // OpenAI-compatible API
        case webSocket = "websocket"     // WebSocket streaming
        case discord = "discord"         // Discord bot
        case slack = "slack"             // Slack bot
        case telegram = "telegram"       // Telegram bot
        case vscode = "vscode"           // VS Code extension
        case cline = "cline"             // Cline integration
        
        var displayName: String {
            switch self {
            case .openAI: return "OpenAI API"
            case .webSocket: return "WebSocket"
            case .discord: return "Discord"
            case .slack: return "Slack"
            case .telegram: return "Telegram"
            case .vscode: return "VS Code"
            case .cline: return "Cline"
            }
        }
        
        var icon: String {
            switch self {
            case .openAI: return "network"
            case .webSocket: return "bolt.fill"
            case .discord: return "gamecontroller"
            case .slack: return "bubble.left.and.bubble.right"
            case .telegram: return "paperplane"
            case .vscode: return "desktopcomputer"
            case .cline: return "terminal"
            }
        }
    }
    
    /// Channel configuration.
    struct ChannelConfig: Codable, Sendable {
        let type: ChannelType
        let enabled: Bool
        let port: Int?
        let apiKey: String?
        let webhookURL: String?
        let settings: [String: String]
        
        static let `default` = ChannelConfig(
            type: .openAI,
            enabled: true,
            port: 8080,
            apiKey: nil,
            webhookURL: nil,
            settings: [:]
        )
    }
    
    /// Gateway statistics.
    struct Statistics: Sendable {
        let activeChannels: Int
        let totalRequests: Int
        let requestsPerMinute: Double
        let averageLatency: TimeInterval
        let errorRate: Double
        
        var summary: String {
            "\(activeChannels) channels, \(totalRequests) requests, \(String(format: "%.1f", requestsPerMinute)) req/min"
        }
    }
    
    // MARK: - Properties
    
    @Published var status: Status = .stopped
    @Published var channels: [ChannelConfig] = []
    @Published var statistics: Statistics?
    
    let port: Int
    private var serverTask: Task<Void, Never>?
    
    // MARK: - Initialization
    
    init(port: Int = 9090) {
        self.port = port
        self.channels = ChannelType.allCases.map { ChannelConfig(type: $0, enabled: false, port: nil, apiKey: nil, webhookURL: nil, settings: [:]) }
    }
    
    // MARK: - Public API
    
    /// Start the gateway server.
    func start() async throws {
        guard status == .stopped else { return }
        
        status = .starting
        
        do {
            // Start HTTP server for OpenAI-compatible API
            try await startHTTPServer()
            
            // Start enabled channels
            for channel in channels where channel.enabled {
                try await startChannel(channel)
            }
            
            status = .running
            
        } catch {
            status = .error
            throw error
        }
    }
    
    /// Stop the gateway server.
    func stop() {
        serverTask?.cancel()
        serverTask = nil
        status = .stopped
        statistics = nil
    }
    
    /// Enable a channel.
    func enableChannel(_ type: ChannelType) async throws {
        guard let index = channels.firstIndex(where: { $0.type == type }) else { return }
        
        let config = channels[index]
        channels[index] = ChannelConfig(
            type: config.type,
            enabled: true,
            port: config.port,
            apiKey: config.apiKey,
            webhookURL: config.webhookURL,
            settings: config.settings
        )
        
        if status == .running {
            try await startChannel(channels[index])
        }
    }
    
    /// Disable a channel.
    func disableChannel(_ type: ChannelType) {
        guard let index = channels.firstIndex(where: { $0.type == type }) else { return }
        
        let config = channels[index]
        channels[index] = ChannelConfig(
            type: config.type,
            enabled: false,
            port: config.port,
            apiKey: config.apiKey,
            webhookURL: config.webhookURL,
            settings: config.settings
        )
    }
    
    /// Update channel configuration.
    func updateChannel(_ config: ChannelConfig) {
        guard let index = channels.firstIndex(where: { $0.type == config.type }) else { return }
        channels[index] = config
    }
    
    /// Process an incoming request.
    func processRequest(
        channel: ChannelType,
        messages: [[String: String]],
        model: String?,
        stream: Bool,
        apiKey: String? = nil
    ) async throws -> GatewayResponse {
        // Validate API key
        if let channelConfig = channels.first(where: { $0.type == channel }),
           let expectedKey = channelConfig.apiKey {
            guard let apiKey, apiKey == expectedKey else {
                throw GatewayError.authenticationFailed
            }
        }
        
        // Forward to local inference engine
        let inferenceURL = URL(string: "http://127.0.0.1:8080/v1/chat/completions")!
        
        var body: [String: Any] = [
            "messages": messages,
            "stream": false  // TODO: Implement SSE streaming support
        ]
        
        if let model {
            body["model"] = model
        }
        
        let jsonData = try JSONSerialization.data(withJSONObject: body)
        
        var request = URLRequest(url: inferenceURL)
        request.httpMethod = "POST"
        request.httpBody = jsonData
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let (data, response) = try await NetworkManager.session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw GatewayError.inferenceFailed
        }
        
        guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GatewayError.inferenceFailed
        }
        
        // Parse choices
        var choices: [GatewayResponse.Choice] = []
        if let choicesArray = result["choices"] as? [[String: Any]] {
            for (index, choiceDict) in choicesArray.enumerated() {
                if let messageDict = choiceDict["message"] as? [String: String],
                   let role = messageDict["role"],
                   let content = messageDict["content"] {
                    choices.append(GatewayResponse.Choice(
                        index: index,
                        message: .init(role: role, content: content),
                        finishReason: choiceDict["finish_reason"] as? String
                    ))
                }
            }
        }
        
        // Parse usage
        var usage: GatewayResponse.Usage?
        if let usageDict = result["usage"] as? [String: Int] {
            usage = GatewayResponse.Usage(
                promptTokens: usageDict["prompt_tokens"] ?? 0,
                completionTokens: usageDict["completion_tokens"] ?? 0,
                totalTokens: usageDict["total_tokens"] ?? 0
            )
        }
        
        return GatewayResponse(
            id: result["id"] as? String ?? UUID().uuidString,
            model: result["model"] as? String ?? model ?? "default",
            choices: choices,
            usage: usage
        )
    }
    
    // MARK: - Private Methods
    
    private func startHTTPServer() async throws {
        // Placeholder: In production, this would start a real HTTP server
        // using NIO or similar framework
    }
    
    private func startChannel(_ config: ChannelConfig) async throws {
        // Placeholder: In production, this would start the channel-specific server
        switch config.type {
        case .discord:
            // Start Discord bot
            break
        case .slack:
            // Start Slack bot
            break
        case .telegram:
            // Start Telegram bot
            break
        case .vscode:
            // Start VS Code extension server
            break
        case .cline:
            // Start Cline integration
            break
        default:
            break
        }
    }
}

// MARK: - Gateway Response

struct GatewayResponse: Codable, Sendable {
    let id: String
    let model: String
    let choices: [Choice]
    let usage: Usage?
    
    struct Choice: Codable, Sendable {
        let index: Int
        let message: Message
        let finishReason: String?
        
        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
        
        struct Message: Codable, Sendable {
            let role: String
            let content: String
        }
    }
    
    struct Usage: Codable, Sendable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int
        
        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

// MARK: - Gateway Error

enum GatewayError: LocalizedError {
    case inferenceFailed
    case channelNotFound
    case authenticationFailed
    case rateLimited
    
    var errorDescription: String? {
        switch self {
        case .inferenceFailed: return "Inference request failed"
        case .channelNotFound: return "Channel not found"
        case .authenticationFailed: return "Authentication failed"
        case .rateLimited: return "Rate limited"
        }
    }
}

// MARK: - Channel Adapter Protocol

/// Protocol for channel-specific adapters.
protocol ChannelAdapter: Sendable {
    var type: GatewayServer.ChannelType { get }
    
    func start(config: GatewayServer.ChannelConfig) async throws
    func stop() async
    
    func handleMessage(_ message: String, from user: String) async throws -> String
}

// MARK: - Discord Adapter Example

/// Example Discord channel adapter.
struct DiscordAdapter: ChannelAdapter {
    let type = GatewayServer.ChannelType.discord
    
    func start(config: GatewayServer.ChannelConfig) async throws {
        // Placeholder: Start Discord bot
    }
    
    func stop() async {
        // Placeholder: Stop Discord bot
    }
    
    func handleMessage(_ message: String, from user: String) async throws -> String {
        // Placeholder: Process Discord message
        return "Response to \(user): \(message)"
    }
}
