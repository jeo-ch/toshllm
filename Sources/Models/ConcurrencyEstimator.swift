// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Estimates concurrent session capacity for a given hardware and model configuration.
/// Inspired by llmfit's concurrency command.
struct ConcurrencyEstimator {
    
    /// Concurrency estimation result.
    struct Result: Sendable {
        /// Maximum number of concurrent sessions.
        let maxSessions: Int
        
        /// VRAM available for KV caches (GB).
        let availableForKV: Double
        
        /// KV cache size per session (GB).
        let kvPerSession: Double
        
        /// Estimated tokens per second per session.
        let tokensPerSecond: Double
        
        /// Memory efficiency (0-1).
        let memoryEfficiency: Double
        
        /// Whether estimation is reliable.
        let isReliable: Bool
        
        /// Human-readable summary.
        var summary: String {
            "\(maxSessions) sessions, \(String(format: "%.1f", kvPerSession)) GB/session, \(String(format: "%.0f", tokensPerSecond)) t/s each"
        }
    }
    
    // MARK: - Estimation Methods
    
    /// Estimate concurrent sessions for a model on given hardware.
    static func estimate(
        spec: ModelSpec,
        hw: HardwareInfo,
        ctx: Int = 16384,
        kvScale: Double = 1.0,
        reservedGB: Double = 2.0
    ) -> Result {
        let vramGB = hw.vramGB
        let bandwidthGBs = Estimator.bandwidthGBs(of: hw.bestGPU)
        let decodeEfficiency = Estimator.decodeEfficiency(of: hw.bestGPU)
        let effectiveBW = bandwidthGBs * decodeEfficiency
        
        // Calculate VRAM needed for model weights
        let weightsGB = spec.fileGB * 1.03  // 3% overhead
        
        // Calculate compute overhead
        let computeGB = 0.9 + spec.paramsB * 0.012
        
        // Calculate KV cache size per session
        let kvBytesPerToken = spec.kvBytesPerToken > 0 ? spec.kvBytesPerToken : 256.0  // Default to f16
        let kvPerSession = (Double(ctx) * kvBytesPerToken * Double(2)) / 1_073_741_824 * kvScale
        
        // Available VRAM for KV caches
        let availableForKV = max(0, vramGB - weightsGB - computeGB - reservedGB)
        
        // Calculate maximum sessions
        let maxSessions = kvPerSession > 0 ? Int(availableForKV / kvPerSession) : 0
        
        // Estimate tokens per second per session
        let tokensPerSecond: Double
        if maxSessions > 0 {
            // Bandwidth is shared across sessions
            let sharedBW = effectiveBW / Double(maxSessions)
            
            if spec.isMoE {
                let active = spec.activeParamsB > 0 ? spec.activeParamsB : spec.paramsB * 0.11
                let bytesPerParam = spec.fileGB / max(1, spec.paramsB)
                let activeGB = active * bytesPerParam
                tokensPerSecond = sharedBW / max(0.5, activeGB)
            } else {
                tokensPerSecond = sharedBW / max(0.5, weightsGB)
            }
        } else {
            tokensPerSecond = 0
        }
        
        // Memory efficiency
        let memoryEfficiency = availableForKV > 0 ? min(1.0, (availableForKV + weightsGB + computeGB) / vramGB) : 0
        
        // Reliability check
        let isReliable = maxSessions > 0 && kvPerSession > 0 && vramGB > 4
        
        return Result(
            maxSessions: maxSessions,
            availableForKV: availableForKV,
            kvPerSession: kvPerSession,
            tokensPerSecond: tokensPerSecond,
            memoryEfficiency: memoryEfficiency,
            isReliable: isReliable
        )
    }
    
    /// Quick estimate for UI display.
    static func quickEstimate(
        vramGB: Double,
        modelGB: Double,
        ctx: Int = 16384
    ) -> Int {
        let weightsGB = modelGB * 1.03
        let computeGB = 0.9
        let reservedGB = 2.0
        let availableForKV = max(0, vramGB - weightsGB - computeGB - reservedGB)
        
        // Rough estimate: 256 bytes per token * 2 (K+V) * ctx tokens
        let kvPerSession = (Double(ctx) * 512) / 1_073_741_824
        
        return kvPerSession > 0 ? Int(availableForKV / kvPerSession) : 0
    }
    
    /// Estimate for multiple models (router mode).
    static func estimateRouter(
        models: [(spec: ModelSpec, count: Int)],
        hw: HardwareInfo,
        ctx: Int = 16384
    ) -> [String: Result] {
        var results: [String: Result] = [:]
        
        for (spec, count) in models {
            let result = estimate(spec: spec, hw: hw, ctx: ctx)
            results["\(spec.paramsB)B x\(count)"] = result
        }
        
        return results
    }
}

// MARK: - Concurrency Display Helper

/// Helper for displaying concurrency information in UI.
struct ConcurrencyDisplay {
    let result: ConcurrencyEstimator.Result
    
    /// Color name based on session count.
    var color: String {
        switch result.maxSessions {
        case 0: return "red"
        case 1: return "yellow"
        case 2...4: return "green"
        default: return "blue"
        }
    }
    
    /// Icon name based on session count.
    var icon: String {
        switch result.maxSessions {
        case 0: return "xmark.circle"
        case 1: return "person.circle"
        case 2...4: return "person.2.circle"
        default: return "person.3.circle"
        }
    }
    
    /// Recommendation text.
    var recommendation: String {
        if result.maxSessions == 0 {
            return "Model too large for this hardware"
        } else if result.maxSessions == 1 {
            return "Single session only"
        } else if result.maxSessions <= 3 {
            return "Good for light multitasking"
        } else {
            return "Excellent for concurrent sessions"
        }
    }
    
    /// Performance warning if applicable.
    var performanceWarning: String? {
        if result.tokensPerSecond < 10 {
            return "Slow inference expected (\(String(format: "%.0f", result.tokensPerSecond)) t/s)"
        }
        if result.memoryEfficiency > 0.9 {
            return "Memory near capacity"
        }
        return nil
    }
}

// MARK: - Dashboard Integration

extension ConcurrencyEstimator {
    /// Get concurrency info for dashboard display.
    static func dashboardInfo(
        hw: HardwareInfo,
        activeModel: LocalModel?
    ) -> (sessions: Int, warning: String?) {
        guard let model = activeModel else {
            return (0, "No model selected")
        }
        
        let spec = ModelSpec.estimated(
            fileBytes: model.sizeBytes,
            isMoE: model.isMoE,
            name: model.name,
            path: model.url.path
        )
        
        let result = estimate(spec: spec, hw: hw)
        
        let warning: String?
        if result.maxSessions == 0 {
            warning = "Model doesn't fit in VRAM"
        } else if result.tokensPerSecond < 10 {
            warning = "Slow inference expected"
        } else {
            warning = nil
        }
        
        return (result.maxSessions, warning)
    }
}
