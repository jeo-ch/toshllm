// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Standardized prefix cache key system for efficient KV cache reuse.
/// Inspired by vLLM's prefix caching and llama.cpp's slot management.
struct PrefixCache {
    
    // MARK: - Cache Key Components
    
    /// Components that uniquely identify a prefix cache entry.
    struct CacheKey: Hashable, Codable {
        /// SHA256 hash of the system prompt.
        let systemPromptHash: String
        
        /// SHA256 hash of tool definitions (if any).
        let toolDefsHash: String
        
        /// Number of tokens in the common prefix (for partial matches).
        let prefixTokenCount: Int
        
        /// Model identifier (path hash) to prevent cross-model cache hits.
        let modelHash: String
        
        /// Quantization tier to prevent cross-quantization cache hits.
        let quantizationTier: String
        
        /// Context length used (different ctx sizes produce different KV layouts).
        let contextLength: Int
        
        /// Whether flash attention is enabled (affects KV layout).
        let flashAttention: Bool
        
        /// KV cache type (f16, q8_0, q4_0).
        let cacheType: String
        
        /// Whether dynamic MoE is enabled (affects KV layout).
        let dynamicMoE: Bool
        
        /// Number of MoE slots (affects KV layout).
        let moeSlots: Int
        
        /// Human-readable description for debugging.
        var description: String {
            "system=\(systemPromptHash.prefix(8)) tools=\(toolDefsHash.prefix(8)) prefix=\(prefixTokenCount) model=\(modelHash.prefix(8)) quant=\(quantizationTier) ctx=\(contextLength) fa=\(flashAttention) cache=\(cacheType) moe=\(dynamicMoE)x\(moeSlots)"
        }
        
        /// LRU eviction timestamp (for cache management).
        var lastAccessed: Date = Date()
        
        /// Size estimate in bytes (for memory budget enforcement).
        var estimatedSizeBytes: Int {
            // Rough estimate: context_length * 2 (K+V) * 128 bytes per token for f16
            let bytesPerToken = cacheType == "f16" ? 256 : (cacheType == "q8_0" ? 128 : 64)
            return contextLength * 2 * bytesPerToken
        }
    }
    
    // MARK: - Cache Statistics
    
    /// Statistics for monitoring cache performance.
    struct Stats {
        var hits: Int = 0
        var misses: Int = 0
        var evictions: Int = 0
        var totalSizeBytes: Int = 0
        var entryCount: Int = 0
        
        var hitRate: Double {
            let total = hits + misses
            return total > 0 ? Double(hits) / Double(total) : 0
        }
        
        var description: String {
            "Cache: \(entryCount) entries, \(totalSizeBytes / 1024 / 1024)MB, hit rate \(Int(hitRate * 100))%"
        }
    }
    
    // MARK: - Properties
    
    /// Maximum cache size in bytes (default 512MB).
    private let maxSizeBytes: Int
    
    /// LRU cache storage.
    private var cache: [CacheKey: CacheEntry] = [:]
    
    /// Cache statistics.
    private(set) var stats = Stats()
    
    /// Lock for thread safety.
    private let lock = NSLock()
    
    // MARK: - Cache Entry
    
    /// Internal cache entry with metadata.
    private struct CacheEntry {
        let key: CacheKey
        let slotFile: String
        var lastAccessed: Date
        let sizeBytes: Int
        
        init(key: CacheKey, slotFile: String) {
            self.key = key
            self.slotFile = slotFile
            self.lastAccessed = Date()
            self.sizeBytes = key.estimatedSizeBytes
        }
    }
    
    // MARK: - Initialization
    
    /// Initialize with optional memory budget.
    /// - Parameter maxSizeBytes: Maximum cache size in bytes (default 512MB).
    init(maxSizeBytes: Int = 512 * 1024 * 1024) {
        self.maxSizeBytes = maxSizeBytes
    }
    
    // MARK: - Public API
    
    /// Generate a cache key for the current conversation state.
    static func makeKey(
        systemPrompt: String,
        toolDefs: [[String: Any]]?,
        prefixTokens: Int,
        modelPath: String,
        quantizationTier: String,
        contextLength: Int,
        flashAttention: Bool,
        cacheType: String,
        dynamicMoE: Bool,
        moeSlots: Int
    ) -> CacheKey {
        CacheKey(
            systemPromptHash: SHA256.hash(systemPrompt),
            toolDefsHash: SHA256.hash(toolDefs.map { String(describing: $0) } ?? ""),
            prefixTokenCount: prefixTokens,
            modelHash: SHA256.hash(modelPath),
            quantizationTier: quantizationTier,
            contextLength: contextLength,
            flashAttention: flashAttention,
            cacheType: cacheType,
            dynamicMoE: dynamicMoE,
            moeSlots: moeSlots
        )
    }
    
    /// Look up a cache entry by key.
    /// Returns the slot file path if found, nil otherwise.
    mutating func lookup(key: CacheKey) -> String? {
        lock.lock()
        defer { lock.unlock() }
        
        if var entry = cache[key] {
            entry.lastAccessed = Date()
            cache[key] = entry
            stats.hits += 1
            return entry.slotFile
        }
        
        stats.misses += 1
        return nil
    }
    
    /// Insert a new cache entry.
    mutating func insert(key: CacheKey, slotFile: String) {
        lock.lock()
        defer { lock.unlock() }
        
        // Evict if necessary
        while stats.totalSizeBytes + key.estimatedSizeBytes > maxSizeBytes, !cache.isEmpty {
            evictLRU()
        }
        
        let entry = CacheEntry(key: key, slotFile: slotFile)
        cache[key] = entry
        stats.totalSizeBytes += entry.sizeBytes
        stats.entryCount = cache.count
    }
    
    /// Remove a specific cache entry.
    mutating func remove(key: CacheKey) {
        lock.lock()
        defer { lock.unlock() }
        
        if let entry = cache.removeValue(forKey: key) {
            stats.totalSizeBytes -= entry.sizeBytes
            stats.entryCount = cache.count
            stats.evictions += 1
        }
    }
    
    /// Clear all cache entries.
    mutating func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        let count = cache.count
        cache.removeAll()
        stats.totalSizeBytes = 0
        stats.entryCount = 0
        stats.evictions += count
    }
    
    /// Get all cache keys sorted by last access time.
    func allKeys() -> [CacheKey] {
        lock.lock()
        defer { lock.unlock() }
        
        return cache.values
            .sorted { $0.lastAccessed > $1.lastAccessed }
            .map { $0.key }
    }
    
    // MARK: - Private Helpers
    
/// Estimate current cache usage in bytes.
    func currentUsageBytes() -> Int {
        // Return the tracked total size
        return max(0, stats.totalSizeBytes)
    }
    
    /// Evict the least recently used entry.
    private mutating func evictLRU() {
        guard let oldestKey = cache.min(by: { $0.value.lastAccessed < $1.value.lastAccessed })?.key else {
            return
        }
        
        if let entry = cache.removeValue(forKey: oldestKey) {
            stats.totalSizeBytes -= entry.sizeBytes
            stats.evictions += 1
        }
        stats.entryCount = cache.count
    }
}

// MARK: - SHA256 Helper

private enum SHA256 {
    static func hash(_ input: String) -> String {
        let data = Data(input.utf8)
        var digest = [UInt8](repeating: 0, count: 32)
        data.withUnsafeBytes {
            _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &digest)
        }
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

import CommonCrypto

// MARK: - Cache Manager

/// Manages prefix cache for the application.
@MainActor
final class PrefixCacheManager: ObservableObject {
    static let shared = PrefixCacheManager()
    
    @Published var stats: PrefixCache.Stats = .init()
    
    private var cache = PrefixCache()
    
    private init() {}
    
    /// Look up a cache entry and return the slot file path.
    func lookup(key: PrefixCache.CacheKey) -> String? {
        let result = cache.lookup(key: key)
        stats = cache.stats
        return result
    }
    
    /// Insert a new cache entry.
    func insert(key: PrefixCache.CacheKey, slotFile: String) {
        cache.insert(key: key, slotFile: slotFile)
        stats = cache.stats
    }
    
    /// Remove a specific cache entry.
    func remove(key: PrefixCache.CacheKey) {
        cache.remove(key: key)
        stats = cache.stats
    }
    
    /// Clear all cache entries.
    func clear() {
        cache.clear()
        stats = cache.stats
    }
    
    /// Get current cache statistics.
    func currentStats() -> PrefixCache.Stats {
        cache.stats
    }
}
