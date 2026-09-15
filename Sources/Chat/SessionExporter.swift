// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Enhanced session persistence with semantic summaries and cross-device support.
/// Inspired by jcode's semantic memory and cross-device recovery.
struct SessionExporter {
    
    /// Export format options.
    enum Format: String, CaseIterable, Sendable {
        case json = "json"
        case markdown = "markdown"
        case text = "plain text"
        
        var fileExtension: String {
            switch self {
            case .json: return "json"
            case .markdown: return "md"
            case .text: return "txt"
            }
        }
        
        var mimeType: String {
            switch self {
            case .json: return "application/json"
            case .markdown: return "text/markdown"
            case .text: return "text/plain"
            }
        }
    }
    
    /// Enhanced session metadata.
    struct SessionMetadata: Codable, Sendable {
        let id: UUID
        let name: String
        let createdAt: Date
        let updatedAt: Date
        let messageCount: Int
        let summary: String?
        let tags: [String]
        let modelPath: String?
        let contextLength: Int?
        let totalTokens: Int?
        let language: String?
        
        var description: String {
            "\(name) (\(messageCount) messages)"
        }
    }
    
    /// Exported session data.
    struct ExportedSession: Codable, Sendable {
        let metadata: SessionMetadata
        let messages: [ExportedMessage]
        let kvCache: Data?
        let toolCalls: [ExportedToolCall]?
        
        var jsonRepresentation: Data? {
            try? JSONEncoder.prettyEncoder.encode(self)
        }
        
        var markdownRepresentation: String {
            var md = "# \(metadata.name)\n\n"
            md += "Created: \(metadata.createdAt.formatted())\n"
            md += "Updated: \(metadata.updatedAt.formatted())\n"
            md += "Messages: \(metadata.messageCount)\n\n"
            
            if let summary = metadata.summary {
                md += "## Summary\n\(summary)\n\n"
            }
            
            md += "## Conversation\n\n"
            for message in messages {
                md += "**\(message.role):** \(message.content)\n\n"
            }
            
            return md
        }
        
        var textRepresentation: String {
            var text = "\(metadata.name)\n"
            text += "Created: \(metadata.createdAt.formatted())\n"
            text += "Messages: \(metadata.messageCount)\n\n"
            
            for message in messages {
                text += "[\(message.role)]: \(message.content)\n\n"
            }
            
            return text
        }
    }
    
    /// Exported message.
    struct ExportedMessage: Codable, Sendable {
        let role: String
        let content: String
        let timestamp: Date
        let tokenCount: Int?
    }
    
    /// Exported tool call.
    struct ExportedToolCall: Codable, Sendable {
        let id: String
        let name: String
        let arguments: String
        let result: String?
        let timestamp: Date
    }
    
    // MARK: - Export Methods
    
    /// Export a conversation to data.
    static func export(
        conversation: Conversation,
        format: Format,
        includeKVCache: Bool = false
    ) throws -> Data {
        let metadata = SessionMetadata(
            id: conversation.id,
            name: conversation.title,
            createdAt: conversation.created,
            updatedAt: conversation.updated,
            messageCount: conversation.messages.count,
            summary: conversation.summary,
            tags: [],
            modelPath: nil,
            contextLength: nil,
            totalTokens: nil,
            language: nil
        )
        
        let messages = conversation.messages.map { msg in
            ExportedMessage(
                role: msg.role,
                content: msg.content,
                timestamp: msg.date,
                tokenCount: msg.estimatedTokens
            )
        }
        
        let kvCache: Data? = nil  // KV cache persistence placeholder
        
        let exported = ExportedSession(
            metadata: metadata,
            messages: messages,
            kvCache: kvCache,
            toolCalls: nil
        )
        
        switch format {
        case .json:
            return try JSONEncoder.prettyEncoder.encode(exported)
        case .markdown:
            return exported.markdownRepresentation.data(using: String.Encoding.utf8) ?? Data()
        case .text:
            return exported.textRepresentation.data(using: String.Encoding.utf8) ?? Data()
        }
    }
    
    /// Import a conversation from data.
    static func `import`(
        from data: Data,
        format: Format
    ) throws -> (metadata: SessionMetadata, messages: [ExportedMessage]) {
        switch format {
        case .json:
            let session = try JSONDecoder().decode(ExportedSession.self, from: data)
            return (session.metadata, session.messages)
            
        case .markdown, .text:
            // Parse text format (simplified)
            guard let text = String(data: data, encoding: .utf8) else {
                throw SessionError.invalidFormat
            }
            return try parseTextFormat(text)
        }
    }
    
    /// Import from file URL.
    static func importFromFile(
        at url: URL,
        format: Format
    ) throws -> (metadata: SessionMetadata, messages: [ExportedMessage]) {
        let data = try Data(contentsOf: url)
        return try `import`(from: data, format: format)
    }
    
    // MARK: - KV Cache Serialization
    
    /// Serialize KV cache data for persistence.
    static func serializeKVCache(_ cache: Data) -> Data? {
        guard !cache.isEmpty else { return nil }
        
        // Compress if larger than 1MB
        if cache.count > 1_048_576 {
            return try? (cache as NSData).compressed(using: .zlib) as Data
        }
        
        return cache
    }
    
    /// Deserialize KV cache data.
    static func deserializeKVCache(_ data: Data?) -> Data? {
        guard let data, !data.isEmpty else { return nil }
        
        // Decompress if needed
        if let decompressed = try? (data as NSData).decompressed(using: .zlib) as Data {
            return decompressed
        }
        
        return data
    }
    
    // MARK: - Backup and Restore
    
    /// Create a full backup of all sessions.
    static func createBackup(
        sessions: [ExportedSession],
        includeKVCache: Bool = false
    ) throws -> Data {
        let backup: [String: Any] = [
            "version": "1.0",
            "createdAt": Date(),
            "sessions": sessions.compactMap { try? JSONSerialization.jsonObject(with: JSONEncoder().encode($0)) }
        ]
        
        return try JSONSerialization.data(withJSONObject: backup, options: .prettyPrinted)
    }
    
    /// Restore sessions from backup.
    static func restoreFromBackup(
        _ data: Data
    ) throws -> [ExportedSession] {
        guard let backup = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sessionsArray = backup["sessions"] as? [[String: Any]] else {
            throw SessionError.invalidBackup
        }
        
        return try sessionsArray.map { sessionDict in
            let sessionData = try JSONSerialization.data(withJSONObject: sessionDict)
            return try JSONDecoder().decode(ExportedSession.self, from: sessionData)
        }
    }
    
    // MARK: - Private Helpers
    
    private static func parseTextFormat(_ text: String) throws -> (metadata: SessionMetadata, messages: [ExportedMessage]) {
        let lines = text.components(separatedBy: .newlines)
        var messages: [ExportedMessage] = []
        var currentRole = ""
        var currentContent = ""
        
        for line in lines {
            if line.hasPrefix("[") && line.contains("]:") {
                // Save previous message
                if !currentRole.isEmpty {
                    messages.append(ExportedMessage(
                        role: currentRole,
                        content: currentContent.trimmingCharacters(in: .whitespacesAndNewlines),
                        timestamp: Date(),
                        tokenCount: nil
                    ))
                }
                
                // Parse new role
                if let closeBracket = line.firstIndex(of: "]") {
                    currentRole = String(line[line.index(after: line.startIndex)..<closeBracket])
                    // Safely get content after "]: " (at least 2 chars after close bracket)
                    let contentStartIndex = line.index(closeBracket, offsetBy: 1, limitedBy: line.endIndex)
                    if let contentStart = contentStartIndex {
                        let trimmed = line[contentStart...].trimmingCharacters(in: .whitespaces)
                        currentContent = trimmed.hasPrefix(": ") ? String(trimmed.dropFirst(2)) : String(trimmed)
                    }
                }
            } else {
                currentContent += "\n" + line
            }
        }
        
        // Add last message
        if !currentRole.isEmpty {
            messages.append(ExportedMessage(
                role: currentRole,
                content: currentContent.trimmingCharacters(in: .whitespacesAndNewlines),
                timestamp: Date(),
                tokenCount: nil
            ))
        }
        
        let metadata = SessionMetadata(
            id: UUID(),
            name: "Imported Session",
            createdAt: Date(),
            updatedAt: Date(),
            messageCount: messages.count,
            summary: nil,
            tags: [],
            modelPath: nil,
            contextLength: nil,
            totalTokens: nil,
            language: nil
        )
        
        return (metadata, messages)
    }
}

// MARK: - Session Errors

enum SessionError: LocalizedError {
    case invalidFormat
    case invalidBackup
    case exportFailed
    case importFailed
    
    var errorDescription: String? {
        switch self {
        case .invalidFormat: return "Invalid session format"
        case .invalidBackup: return "Invalid backup file"
        case .exportFailed: return "Failed to export session"
        case .importFailed: return "Failed to import session"
        }
    }
}

// MARK: - JSON Encoder Extension

private extension JSONEncoder {
    static let prettyEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
}

// MARK: - Conversation Extension

extension Conversation {
    // Note: summary and other properties already exist in the Conversation struct
}
