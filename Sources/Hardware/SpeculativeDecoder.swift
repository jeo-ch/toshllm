// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

// MARK: - Speculative Decoding Protocol

/// Unified interface for all speculative decoding methods.
/// Concrete implementations: MTPDecoder, DFlashDecoder.
protocol SpeculativeDecoder {
    /// Unique identifier for this decoder type.
    var id: String { get }
    
    /// Human-readable name (e.g., "MTP", "DFlash").
    var displayName: String { get }
    
    /// Whether this decoder is available for the given model.
    func isAvailable(forModel path: String) -> Bool
    
    /// Whether this decoder is currently enabled for the given model.
    func isEnabled(forModel path: String) -> Bool
    
    /// Enable or disable this decoder for the given model.
    func setEnabled(_ enabled: Bool, forModel path: String)
    
    /// The draft model path for this decoder, if any.
    func draftPath(forModel path: String) -> String?
    
    /// Whether this decoder requires a separate draft model file.
    var requiresDraft: Bool { get }
    
    /// Description of when this decoder is beneficial.
    var description: String { get }
    
    /// System image name for UI display.
    var systemImage: String { get }
}

// MARK: - Speculative Mode

/// Unified speculative decoding mode, applicable to any decoder type.
enum SpeculativeMode: String, CaseIterable, Identifiable {
    case off
    case auto
    case forced
    
    var id: String { rawValue }
}

// MARK: - MTP Decoder

/// Multi-Token Prediction speculative decoder.
/// Uses either an embedded MTP head in the model or an external draft model.
struct MTPDecoder: SpeculativeDecoder {
    let id = "mtp"
    let displayName = "MTP"
    let requiresDraft = false
    let systemImage = "hare.fill"
    
    var description: String {
        "Multi-Token Prediction: uses embedded MTP head or external draft for faster generation."
    }
    
    func isAvailable(forModel path: String) -> Bool {
        ServerSettings.modelUsesMTP(at: path)
    }
    
    func isEnabled(forModel path: String) -> Bool {
        ServerSettings.mtpEnabled(forModel: path)
    }
    
    func setEnabled(_ enabled: Bool, forModel path: String) {
        ServerSettings.setMTPEnabled(enabled, forModel: path)
    }
    
    func draftPath(forModel path: String) -> String? {
        ServerSettings.mtpDraftPath(forModel: path)
    }
}

// MARK: - DFlash Decoder

/// DFlash speculative decoder for MoE models with CPU-offloaded experts.
struct DFlashDecoder: SpeculativeDecoder {
    let id = "dflash"
    let displayName = "DFlash"
    let requiresDraft = true
    let systemImage = "bolt.fill"
    
    var description: String {
        "DFlash: speeds up MoE models with CPU-offloaded experts via speculative decoding."
    }
    
    func isAvailable(forModel path: String) -> Bool {
        ServerSettings.dflashDraftPath(forModel: path) != nil
    }
    
    func isEnabled(forModel path: String) -> Bool {
        ServerSettings.dflashEnabled(forModel: path)
    }
    
    func setEnabled(_ enabled: Bool, forModel path: String) {
        ServerSettings.setDflashEnabled(enabled, forModel: path)
    }
    
    func draftPath(forModel path: String) -> String? {
        ServerSettings.activeDflashDraft(forModel: path)
    }
    
    /// Get the DFlash mode for the given model.
    func mode(forModel path: String) -> DflashMode {
        ServerSettings.dflashMode(forModel: path)
    }
    
    /// Set the DFlash mode for the given model.
    func setMode(_ mode: DflashMode, forModel path: String) {
        ServerSettings.setDflashMode(mode, forModel: path)
    }
}

// MARK: - Speculative Decoder Manager

/// Central manager for all speculative decoding methods.
/// Provides a unified interface to query and configure decoders.
struct SpeculativeDecoderManager {
    /// All available decoder types.
    static let allDecoders: [SpeculativeDecoder] = [MTPDecoder(), DFlashDecoder()]
    
    /// Get all decoders available for a given model.
    static func availableDecoders(forModel path: String) -> [SpeculativeDecoder] {
        allDecoders.filter { $0.isAvailable(forModel: path) }
    }
    
    /// Get all enabled decoders for a given model.
    static func enabledDecoders(forModel path: String) -> [SpeculativeDecoder] {
        allDecoders.filter { $0.isEnabled(forModel: path) }
    }
    
    /// Check if any speculative decoding is enabled for a model.
    static func anyEnabled(forModel path: String) -> Bool {
        enabledDecoders(forModel: path).count > 0
    }
    
    /// Get the first enabled draft path for a model (prioritizes DFlash for MoE, MTP otherwise).
    static func activeDraftPath(forModel path: String) -> String? {
        // DFlash is preferred for MoE models
        if let dflashPath = DFlashDecoder().draftPath(forModel: path) {
            return dflashPath
        }
        // Fall back to MTP
        return MTPDecoder().draftPath(forModel: path)
    }
    
    /// Get the preferred decoder for a model (DFlash for MoE, MTP for dense).
    static func preferredDecoder(forModel path: String) -> SpeculativeDecoder? {
        if ServerSettings.modelIsMoE(at: path) {
            let dflash = DFlashDecoder()
            if dflash.isAvailable(forModel: path) { return dflash }
        }
        let mtp = MTPDecoder()
        if mtp.isAvailable(forModel: path) { return mtp }
        return nil
    }
}

// MARK: - Speculative Configuration

/// Unified configuration for speculative decoding.
struct SpeculativeConfiguration {
    let decoder: SpeculativeDecoder
    let mode: SpeculativeMode
    let draftPath: String?
    
    /// Create configuration from current settings.
    static func current(forModel path: String) -> SpeculativeConfiguration? {
        guard let decoder = SpeculativeDecoderManager.preferredDecoder(forModel: path) else {
            return nil
        }
        
        let mode: SpeculativeMode
        if let dflashDecoder = decoder as? DFlashDecoder {
            let dflashMode = dflashDecoder.mode(forModel: path)
            switch dflashMode {
            case .off: mode = .off
            case .auto: mode = .auto
            case .forced: mode = .forced
            }
        } else {
            mode = decoder.isEnabled(forModel: path) ? .auto : .off
        }
        
        return SpeculativeConfiguration(
            decoder: decoder,
            mode: mode,
            draftPath: decoder.draftPath(forModel: path)
        )
    }
}
