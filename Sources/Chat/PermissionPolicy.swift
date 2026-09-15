// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Permission policy for tool calls and dangerous operations.
/// Inspired by cc-haha and opencode's permission systems.
struct PermissionPolicy: Codable, Sendable {
    
    /// Permission levels.
    enum Level: String, Codable, CaseIterable, Sendable {
        case ask = "ask"           // Always ask for permission
        case allow = "allow"       // Allow operations
        case deny = "deny"         // Deny operations
        case yolo = "yolo"         // Allow everything (no questions)
        
        /// Human-readable description.
        var description: String {
            switch self {
            case .ask: return "Always ask"
            case .allow: return "Allow"
            case .deny: return "Deny"
            case .yolo: return "Allow all (YOLO)"
            }
        }
        
        /// Icon for UI display.
        var icon: String {
            switch self {
            case .ask: return "questionmark.circle"
            case .allow: return "checkmark.circle"
            case .deny: return "xmark.circle"
            case .yolo: return "bolt.circle"
            }
        }
        
        /// Color for UI display.
        var color: String {
            switch self {
            case .ask: return "yellow"
            case .allow: return "green"
            case .deny: return "red"
            case .yolo: return "orange"
            }
        }
    }
    
    /// Operation categories.
    enum Category: String, Codable, CaseIterable, Sendable {
        case fileRead = "file_read"
        case fileWrite = "file_write"
        case fileDelete = "file_delete"
        case commandExecution = "command_execution"
        case networkRequest = "network_request"
        case networkUpload = "network_upload"
        case systemConfig = "system_config"
        case credentials = "credentials"
        
        /// Human-readable description.
        var description: String {
            switch self {
            case .fileRead: return "Read files"
            case .fileWrite: return "Write files"
            case .fileDelete: return "Delete files"
            case .commandExecution: return "Execute commands"
            case .networkRequest: return "Network requests"
            case .networkUpload: return "Network uploads"
            case .systemConfig: return "System configuration"
            case .credentials: return "Access credentials"
            }
        }
        
        /// Risk level (1-5, higher is more dangerous).
        var riskLevel: Int {
            switch self {
            case .fileRead: return 1
            case .networkRequest: return 2
            case .fileWrite: return 3
            case .commandExecution: return 4
            case .fileDelete: return 4
            case .networkUpload: return 4
            case .systemConfig: return 5
            case .credentials: return 5
            }
        }
        
        /// Icon for UI display.
        var icon: String {
            switch self {
            case .fileRead: return "doc"
            case .fileWrite: return "doc.badge.plus"
            case .fileDelete: return "doc.badge.minus"
            case .commandExecution: return "terminal"
            case .networkRequest: return "network"
            case .networkUpload: return "arrow.up.circle"
            case .systemConfig: return "gearshape"
            case .credentials: return "key"
            }
        }
    }
    
    /// Permission rule.
    struct Rule: Codable, Sendable {
        let category: Category
        let level: Level
        let patterns: [String]  // Glob patterns for allowed/denied paths/commands
        let reason: String?
        let createdAt: Date
        
        /// Check if a specific operation matches this rule.
        func matches(operation: String) -> Bool {
            patterns.isEmpty || patterns.contains { pattern in
                // Use NSPredicate for proper Glob matching
                let predicate = NSPredicate(format: "SELF LIKE %@", pattern)
                return predicate.evaluate(with: operation)
            }
        }
    }
    
    // MARK: - Properties
    
    /// Global permission level (default for all categories).
    var globalLevel: Level = .ask
    
    /// Category-specific rules.
    var rules: [Rule] = []
    
    /// Whitelist patterns (always allowed regardless of level).
    var allowPatterns: [String] = []
    
    /// Blacklist patterns (always denied regardless of level).
    var denyPatterns: [String] = []
    
    /// Whether to log all permission checks.
    var logPermissionChecks: Bool = true
    
    /// Whether to show notifications for denied operations.
    var showDeniedNotifications: Bool = true
    
    // MARK: - Public API
    
    /// Check if an operation is allowed.
    func checkPermission(category: Category, operation: String) -> PermissionResult {
        // Check blacklist first (highest priority)
        if denyPatterns.contains(where: { pattern in
            NSPredicate(format: "SELF LIKE %@", pattern).evaluate(with: operation)
        }) {
            return .denied(reason: "Operation matches deny pattern")
        }
        
        // Check whitelist (second priority)
        if allowPatterns.contains(where: { pattern in
            NSPredicate(format: "SELF LIKE %@", pattern).evaluate(with: operation)
        }) {
            return .allowed(reason: "Operation matches allow pattern")
        }
        
        // Check category-specific rules
        if let rule = rules.first(where: { $0.category == category && $0.matches(operation: operation) }) {
            switch rule.level {
            case .allow:
                return .allowed(reason: rule.reason ?? "Allowed by rule")
            case .deny:
                return .denied(reason: rule.reason ?? "Denied by rule")
            case .ask, .yolo:
                // Continue to global level
                break
            }
        }
        
        // Apply global level
        switch globalLevel {
        case .allow:
            return .allowed(reason: "Global policy allows")
        case .deny:
            return .denied(reason: "Global policy denies")
        case .ask:
            return .needsApproval(reason: "Requires user approval")
        case .yolo:
            return .allowed(reason: "YOLO mode - all operations allowed")
        }
    }
    
    /// Add a rule for a category.
    mutating func addRule(category: Category, level: Level, patterns: [String] = [], reason: String? = nil) {
        let rule = Rule(
            category: category,
            level: level,
            patterns: patterns,
            reason: reason,
            createdAt: Date()
        )
        rules.append(rule)
    }
    
    /// Remove rules for a category.
    mutating func removeRules(for category: Category) {
        rules.removeAll { $0.category == category }
    }
    
    /// Clear all rules.
    mutating func clearRules() {
        rules.removeAll()
        allowPatterns.removeAll()
        denyPatterns.removeAll()
    }
    
    /// Get risk summary for current policy.
    func riskSummary() -> RiskSummary {
        var categoryRisks: [Category: RiskSummary.RiskLevel] = [:]
        
        for category in Category.allCases {
            let risk: RiskSummary.RiskLevel
            switch globalLevel {
            case .yolo:
                risk = .high
            case .allow:
                risk = .medium
            case .deny:
                risk = .low
            case .ask:
                risk = category.riskLevel >= 4 ? .medium : .low
            }
            categoryRisks[category] = risk
        }
        
        let overallRisk: RiskSummary.RiskLevel
        if globalLevel == .yolo {
            overallRisk = .high
        } else if globalLevel == .deny {
            overallRisk = .low
        } else {
            overallRisk = .medium
        }
        
        return RiskSummary(overall: overallRisk, categoryRisks: categoryRisks)
    }
    
    // MARK: - Default Policies
    
    /// Default policy (ask for dangerous operations).
    static let `default` = PermissionPolicy(
        globalLevel: .ask,
        rules: [
            Rule(category: .fileRead, level: .allow, patterns: [], reason: "Reading files is safe", createdAt: Date()),
            Rule(category: .networkRequest, level: .allow, patterns: [], reason: "Network requests are needed for model downloads", createdAt: Date()),
        ]
    )
    
    /// Permissive policy (allow most operations).
    static let permissive = PermissionPolicy(
        globalLevel: .allow,
        rules: [
            Rule(category: .fileDelete, level: .ask, patterns: [], reason: "Deletion requires confirmation", createdAt: Date()),
            Rule(category: .systemConfig, level: .ask, patterns: [], reason: "System changes require confirmation", createdAt: Date()),
            Rule(category: .credentials, level: .deny, patterns: [], reason: "Credential access is prohibited", createdAt: Date()),
        ]
    )
    
    /// Restrictive policy (deny most operations).
    static let restrictive = PermissionPolicy(
        globalLevel: .deny,
        rules: [
            Rule(category: .fileRead, level: .allow, patterns: [], reason: "Reading is safe", createdAt: Date()),
            Rule(category: .fileWrite, level: .ask, patterns: [], reason: "Writing requires approval", createdAt: Date()),
        ]
    )
}

// MARK: - Permission Result

/// Result of a permission check.
enum PermissionResult: Sendable {
    case allowed(reason: String)
    case denied(reason: String)
    case needsApproval(reason: String)
    
    var isAllowed: Bool {
        if case .allowed = self { return true }
        return false
    }
    
    var isDenied: Bool {
        if case .denied = self { return true }
        return false
    }
    
    var needsUserApproval: Bool {
        if case .needsApproval = self { return true }
        return false
    }
}

// MARK: - Risk Summary

/// Summary of policy risk levels.
struct RiskSummary: Sendable {
    let overall: RiskLevel
    let categoryRisks: [PermissionPolicy.Category: RiskLevel]
    
    enum RiskLevel: String, Sendable {
        case low = "low"
        case medium = "medium"
        case high = "high"
        case critical = "critical"
        
        var color: String {
            switch self {
            case .low: return "green"
            case .medium: return "yellow"
            case .high: return "orange"
            case .critical: return "red"
            }
        }
        
        var description: String {
            switch self {
            case .low: return "Low risk"
            case .medium: return "Medium risk"
            case .high: return "High risk"
            case .critical: return "Critical risk"
            }
        }
    }
}

// MARK: - Permission Policy Manager

/// Manages permission policies for the application.
@MainActor
final class PermissionPolicyManager: ObservableObject {
    static let shared = PermissionPolicyManager()
    
    @Published var currentPolicy: PermissionPolicy = .default
    @Published var permissionLog: [PermissionLogEntry] = []
    
    private let policyKey = "permissionPolicy"
    
    struct PermissionLogEntry: Identifiable, Sendable {
        let id: UUID
        let timestamp: Date
        let category: PermissionPolicy.Category
        let operation: String
        let result: PermissionResult
        let userApproved: Bool?
    }
    
    private init() {
        loadPolicy()
    }
    
    /// Check permission for an operation.
    func checkPermission(
        category: PermissionPolicy.Category,
        operation: String
    ) -> PermissionResult {
        let result = currentPolicy.checkPermission(category: category, operation: operation)
        
        // Log the check
        let entry = PermissionLogEntry(
            id: UUID(),
            timestamp: Date(),
            category: category,
            operation: operation,
            result: result,
            userApproved: nil
        )
        permissionLog.append(entry)
        
        // Trim old logs
        if permissionLog.count > 1000 {
            permissionLog.removeFirst(permissionLog.count - 1000)
        }
        
        return result
    }
    
    /// Update policy.
    func updatePolicy(_ policy: PermissionPolicy) {
        currentPolicy = policy
        savePolicy()
    }
    
    /// Load policy from storage.
    private func loadPolicy() {
        if let data = UserDefaults.standard.data(forKey: policyKey),
           let policy = try? JSONDecoder().decode(PermissionPolicy.self, from: data) {
            currentPolicy = policy
        }
    }
    
    /// Save policy to storage.
    private func savePolicy() {
        if let data = try? JSONEncoder().encode(currentPolicy) {
            UserDefaults.standard.set(data, forKey: policyKey)
        }
    }
    
    /// Get risk summary.
    func riskSummary() -> RiskSummary {
        currentPolicy.riskSummary()
    }
    
    /// Clear permission log.
    func clearLog() {
        permissionLog.removeAll()
    }
}
