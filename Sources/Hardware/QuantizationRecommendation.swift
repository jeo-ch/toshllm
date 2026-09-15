// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

// MARK: - Quantization Tiers

/// Quantization quality tiers, from smallest/fastest to largest/best quality.
enum QuantizationTier: String, CaseIterable, Identifiable {
    case q4_0        // 4-bit, smallest, fastest
    case q4_k_m      // 4-bit with k-quant, good balance
    case q5_k_m      // 5-bit with k-quant, better quality
    case q6_k        // 6-bit with k-quant, near-lossless
    case q8_0        // 8-bit, lossless for most use cases
    case f16         // 16-bit, full precision
    
    var id: String { rawValue }
    
    /// Compare by quality score for sorting.
    static func < (lhs: QuantizationTier, rhs: QuantizationTier) -> Bool {
        lhs.qualityScore < rhs.qualityScore
    }
    
    /// Display name for UI.
    var displayName: String {
        switch self {
        case .q4_0: return "Q4_0"
        case .q4_k_m: return "Q4_K_M"
        case .q5_k_m: return "Q5_K_M"
        case .q6_k: return "Q6_K"
        case .q8_0: return "Q8_0"
        case .f16: return "F16"
        }
    }
    
    /// Relative quality score (0.0 to 1.0, higher is better).
    var qualityScore: Double {
        switch self {
        case .q4_0: return 0.60
        case .q4_k_m: return 0.75
        case .q5_k_m: return 0.85
        case .q6_k: return 0.92
        case .q8_0: return 0.98
        case .f16: return 1.00
        }
    }
    
    /// Relative size multiplier (Q4_0 = 1.0 baseline).
    var sizeMultiplier: Double {
        switch self {
        case .q4_0: return 1.0
        case .q4_k_m: return 1.15
        case .q5_k_m: return 1.35
        case .q6_k: return 1.55
        case .q8_0: return 2.0
        case .f16: return 4.0
        }
    }
    
    /// GB per billion parameters.
    var gbPerBillionParams: Double {
        sizeMultiplier * 0.57  // Q4_0 baseline: ~0.57 GB/B
    }
    
    /// Speed multiplier relative to Q4_0 (lower quant = faster decode).
    var speedMultiplier: Double {
        switch self {
        case .q4_0: return 1.0
        case .q4_k_m: return 0.95
        case .q5_k_m: return 0.88
        case .q6_k: return 0.80
        case .q8_0: return 0.65
        case .f16: return 0.40
        }
    }
    
    /// Human-readable description.
    var description: String {
        switch self {
        case .q4_0: return "Smallest and fastest, noticeable quality loss"
        case .q4_k_m: return "Good balance of size and quality"
        case .q5_k_m: return "Better quality, slightly larger"
        case .q6_k: return "Near-lossless quality"
        case .q8_0: return "Lossless for most use cases"
        case .f16: return "Full precision, largest file"
        }
    }
}

// MARK: - Quantization Recommendation

/// Recommends optimal quantization tier based on hardware capabilities.
struct QuantizationRecommendation {
    
    /// Recommendation reason for UI display.
    enum Reason: String {
        case vramTooSmall = "VRAM too small for higher quantization"
        case vramOptimal = "Best quality within VRAM budget"
        case speedOptimal = "Best speed for this hardware"
        case qualityOptimal = "Maximum quality supported"
        case noRecommendation = "Model not found"
    }
    
    /// The recommended quantization tier.
    let tier: QuantizationTier
    
    /// Reason for this recommendation.
    let reason: Reason
    
    /// Estimated file size in GB.
    let estimatedSizeGB: Double
    
    /// Estimated tokens/second for this quantization on this hardware.
    let estimatedSpeed: Double
    
    /// Whether this recommendation fits entirely in VRAM.
    let fitsInVRAM: Bool
    
    /// Alternative quantizations available (sorted by quality).
    let alternatives: [QuantizationTier]
    
    /// Calculate recommendation for a model on given hardware.
    static func recommend(
        modelParamsB: Double,
        modelLayers: Int,
        isMoE: Bool,
        hardware: HardwareInfo,
        ctx: Int = 16384,
        kvScale: Double = 1.0
    ) -> QuantizationRecommendation {
        let vramGB = hardware.vramGB
        let bandwidthGBs = Estimator.bandwidthGBs(of: hardware.bestGPU)
        let decodeEfficiency = Estimator.decodeEfficiency(of: hardware.bestGPU)
        let effectiveBW = bandwidthGBs * decodeEfficiency
        
        // VRAM budget: total VRAM minus system/driver overhead (1 GB)
        let vramBudget = max(0, vramGB - 1.0)
        
        // KV cache overhead (rough estimate)
        let kvGB = 0.05 * Double(ctx) / 1024 * kvScale
        
        // Compute overhead (attention + buffers)
        let computeGB = 0.9 + modelParamsB * 0.012
        
        // Fixed overhead for all quantizations
        let fixedOverhead = kvGB + computeGB
        
        // Find best fitting quantization
        var bestTier: QuantizationTier = .q4_0
        var bestReason: Reason = .vramTooSmall
        var bestSpeed: Double = 0
        var bestFits = false
        
        for tier in QuantizationTier.allCases.sorted(by: { $0.qualityScore > $1.qualityScore }) {
            let fileSizeGB = modelParamsB * tier.gbPerBillionParams
            let totalNeed = fileSizeGB * 1.03 + fixedOverhead  // 3% overhead
            
            if totalNeed <= vramBudget {
                // This quantization fits in VRAM
                let speed = (effectiveBW / max(0.5, fileSizeGB)) * tier.speedMultiplier
                
                if speed > bestSpeed {
                    bestSpeed = speed
                    bestTier = tier
                    bestReason = .vramOptimal
                    bestFits = true
                }
            } else if fileSizeGB * 0.5 <= vramBudget {
                // Partially fits, but slower
                let onGPU = max(0, vramBudget - fixedOverhead)
                let fracRAM = max(0, min(1, (totalNeed - onGPU) / max(0.5, totalNeed)))
                let perToken = fileSizeGB * ((1 - fracRAM) / effectiveBW + fracRAM / 48.0)  // 48 GB/s RAM
                let speed = 1 / max(0.0001, perToken) * tier.speedMultiplier
                
                if speed > bestSpeed {
                    bestSpeed = speed
                    bestTier = tier
                    bestReason = .speedOptimal
                    bestFits = false
                }
            }
        }
        
        // If nothing fits, fall back to Q4_0
        if bestSpeed == 0 {
            bestTier = .q4_0
            bestReason = .vramTooSmall
            bestSpeed = (effectiveBW / max(0.5, modelParamsB * 0.57)) * 1.0
            bestFits = false
        }
        
        // Calculate alternatives
        let alternatives = QuantizationTier.allCases
            .filter { $0 != bestTier }
            .sorted { $0.qualityScore > $1.qualityScore }
        
        return QuantizationRecommendation(
            tier: bestTier,
            reason: bestReason,
            estimatedSizeGB: modelParamsB * bestTier.gbPerBillionParams,
            estimatedSpeed: bestSpeed,
            fitsInVRAM: bestFits,
            alternatives: alternatives
        )
    }
    
    /// Quick recommendation for UI display.
    static func quickRecommend(
        fileSizeGB: Double,
        hardware: HardwareInfo
    ) -> QuantizationTier {
        let vramGB = hardware.vramGB
        
        // Simple heuristic based on VRAM
        if vramGB >= 24 {
            return .q8_0  // Plenty of VRAM, use high quality
        } else if vramGB >= 16 {
            return .q6_k  // Good VRAM, near-lossless
        } else if vramGB >= 12 {
            return .q5_k_m  // Decent VRAM, good quality
        } else if vramGB >= 8 {
            return .q4_k_m  // Limited VRAM, balanced
        } else {
            return .q4_0  // Very limited VRAM, smallest possible
        }
    }
}

// MARK: - QuantizationBadge View Helper

/// Provides a badge color and label for quantization tiers.
struct QuantizationBadge {
    let tier: QuantizationTier
    let isRecommended: Bool
    
    var label: String {
        if isRecommended {
            return "★ \(tier.displayName)"
        }
        return tier.displayName
    }
    
    var color: String {
        switch tier {
        case .q4_0: return "orange"
        case .q4_k_m: return "yellow"
        case .q5_k_m: return "green"
        case .q6_k: return "blue"
        case .q8_0: return "purple"
        case .f16: return "red"
        }
    }
}
