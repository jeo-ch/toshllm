// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Hardware profile for storing and sharing hardware configurations.
/// Inspired by llmfit's hardware detection and system profiling.
struct HardwareProfile: Codable, Identifiable {
    /// Unique identifier for this profile.
    let id: UUID
    
    /// Human-readable name for this profile.
    var name: String
    
    /// CPU information.
    let cpu: CPUInfo
    
    /// RAM in gigabytes.
    let ramGB: Double
    
    /// List of GPUs in this profile.
    let gpus: [GPUProfile]
    
    /// Whether this system has unified memory (Apple Silicon).
    let hasUnifiedMemory: Bool
    
    /// Memory bandwidth in GB/s (estimated or measured).
    let bandwidthGBs: Double
    
    /// FP16 TFLOPS (for Apple Silicon) or 0 for discrete GPUs.
    let fp16TFLOPS: Double
    
    /// Operating system version.
    let osVersion: String
    
    /// When this profile was created.
    let createdAt: Date
    
    /// When this profile was last updated.
    var updatedAt: Date
    
    /// Human-readable description of this profile.
    var description: String {
        var parts = [name]
        parts.append("\(cpu.brand) (\(cpu.physicalCores) cores)")
        parts.append("\(Int(ramGB)) GB RAM")
        if !gpus.isEmpty {
            parts.append(gpus.map { $0.name }.joined(separator: ", "))
        }
        if hasUnifiedMemory { parts.append("Unified Memory") }
        return parts.joined(separator: " · ")
    }
    
    // MARK: - Nested Types
    
    struct CPUInfo: Codable {
        let brand: String
        let physicalCores: Int
        let logicalCores: Int
        let arch: String
    }
    
    struct GPUProfile: Codable {
        let name: String
        let vramMB: Int
        let vendor: String  // "AMD", "Apple", "NVIDIA"
        let isIntegrated: Bool
        let isExternal: Bool
        let peerGroupID: UInt64
        let peerCount: Int
        let supportsBF16: Bool
    }
}

// MARK: - Preset Profiles

extension HardwareProfile {
    /// Pre-configured profiles for common hardware configurations.
    static let presets: [HardwareProfile] = [
        // AMD Discrete GPUs
        HardwareProfile(
            id: UUID(),
            name: "RX 6700 XT",
            cpu: CPUInfo(brand: "Intel Core i7-12700K", physicalCores: 12, logicalCores: 20, arch: "x86_64"),
            ramGB: 32,
            gpus: [GPUProfile(name: "AMD Radeon RX 6700 XT", vramMB: 12288, vendor: "AMD", isIntegrated: false, isExternal: false, peerGroupID: 0, peerCount: 1, supportsBF16: false)],
            hasUnifiedMemory: false,
            bandwidthGBs: 384,
            fp16TFLOPS: 0,
            osVersion: "macOS 15.0+",
            createdAt: Date(),
            updatedAt: Date()
        ),
        HardwareProfile(
            id: UUID(),
            name: "RX 7900 XT",
            cpu: CPUInfo(brand: "AMD Ryzen 9 7950X", physicalCores: 16, logicalCores: 32, arch: "x86_64"),
            ramGB: 64,
            gpus: [GPUProfile(name: "AMD Radeon RX 7900 XT", vramMB: 20480, vendor: "AMD", isIntegrated: false, isExternal: false, peerGroupID: 0, peerCount: 1, supportsBF16: true)],
            hasUnifiedMemory: false,
            bandwidthGBs: 800,
            fp16TFLOPS: 0,
            osVersion: "macOS 15.0+",
            createdAt: Date(),
            updatedAt: Date()
        ),
        HardwareProfile(
            id: UUID(),
            name: "Vega II Duo",
            cpu: CPUInfo(brand: "Intel Core i9-10900X", physicalCores: 10, logicalCores: 20, arch: "x86_64"),
            ramGB: 128,
            gpus: [GPUProfile(name: "AMD Radeon VII", vramMB: 16384, vendor: "AMD", isIntegrated: false, isExternal: false, peerGroupID: 1, peerCount: 2, supportsBF16: false)],
            hasUnifiedMemory: false,
            bandwidthGBs: 1024,
            fp16TFLOPS: 0,
            osVersion: "macOS 15.0+",
            createdAt: Date(),
            updatedAt: Date()
        ),
        
        // Apple Silicon
        HardwareProfile(
            id: UUID(),
            name: "M1 Max",
            cpu: CPUInfo(brand: "Apple M1 Max", physicalCores: 10, logicalCores: 10, arch: "arm64"),
            ramGB: 64,
            gpus: [GPUProfile(name: "Apple M1 Max", vramMB: 0, vendor: "Apple", isIntegrated: true, isExternal: false, peerGroupID: 0, peerCount: 1, supportsBF16: true)],
            hasUnifiedMemory: true,
            bandwidthGBs: 400,
            fp16TFLOPS: 10.4,
            osVersion: "macOS 15.0+",
            createdAt: Date(),
            updatedAt: Date()
        ),
        HardwareProfile(
            id: UUID(),
            name: "M2 Ultra",
            cpu: CPUInfo(brand: "Apple M2 Ultra", physicalCores: 24, logicalCores: 24, arch: "arm64"),
            ramGB: 192,
            gpus: [GPUProfile(name: "Apple M2 Ultra", vramMB: 0, vendor: "Apple", isIntegrated: true, isExternal: false, peerGroupID: 0, peerCount: 1, supportsBF16: true)],
            hasUnifiedMemory: true,
            bandwidthGBs: 800,
            fp16TFLOPS: 27.2,
            osVersion: "macOS 15.0+",
            createdAt: Date(),
            updatedAt: Date()
        ),
        HardwareProfile(
            id: UUID(),
            name: "M4 Max",
            cpu: CPUInfo(brand: "Apple M4 Max", physicalCores: 16, logicalCores: 16, arch: "arm64"),
            ramGB: 128,
            gpus: [GPUProfile(name: "Apple M4 Max", vramMB: 0, vendor: "Apple", isIntegrated: true, isExternal: false, peerGroupID: 0, peerCount: 1, supportsBF16: true)],
            hasUnifiedMemory: true,
            bandwidthGBs: 546,
            fp16TFLOPS: 18.0,
            osVersion: "macOS 15.0+",
            createdAt: Date(),
            updatedAt: Date()
        ),
    ]
    
    /// Detect the current hardware and create a profile.
    static func detect() -> HardwareProfile {
        let hw = HardwareInfo.detect()
        
        return HardwareProfile(
            id: UUID(),
            name: "Current System",
            cpu: CPUInfo(
                brand: hw.cpuBrand,
                physicalCores: hw.physicalCores,
                logicalCores: hw.logicalCores,
                arch: hw.arch
            ),
            ramGB: hw.ramGB,
            gpus: hw.gpus.map { gpu in
                GPUProfile(
                    name: gpu.name,
                    vramMB: gpu.vramMB,
                    vendor: gpu.isIntegrated ? "Apple" : "AMD",
                    isIntegrated: gpu.isIntegrated,
                    isExternal: gpu.isExternal,
                    peerGroupID: gpu.peerGroupID,
                    peerCount: gpu.peerCount,
                    supportsBF16: gpu.supportsBF16
                )
            },
            hasUnifiedMemory: hw.arch == "arm64",
            bandwidthGBs: Estimator.bandwidthGBs(of: hw.bestGPU),
            fp16TFLOPS: hw.arch == "arm64" ? 10.0 : 0,  // Conservative estimate
            osVersion: hw.osVersion,
            createdAt: Date(),
            updatedAt: Date()
        )
    }
}

// MARK: - Hardware Profile Store

/// Manages storage and retrieval of hardware profiles.
@MainActor
final class HardwareProfileStore: ObservableObject {
    static let shared = HardwareProfileStore()
    
    @Published var profiles: [HardwareProfile] = []
    @Published var currentProfile: HardwareProfile?
    
    private let storageKey = "hardwareProfiles"
    private let currentProfileKey = "currentHardwareProfile"
    
    private init() {
        loadProfiles()
        detectCurrentHardware()
    }
    
    /// Load saved profiles from UserDefaults.
    private func loadProfiles() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let saved = try? JSONDecoder().decode([HardwareProfile].self, from: data) {
            profiles = saved
        }
        
        // Add presets if none exist
        if profiles.isEmpty {
            profiles = HardwareProfile.presets
            saveProfiles()
        }
        
        // Load current profile
        if let data = UserDefaults.standard.data(forKey: currentProfileKey),
           let saved = try? JSONDecoder().decode(HardwareProfile.self, from: data) {
            currentProfile = saved
        }
    }
    
    /// Save profiles to UserDefaults.
    private func saveProfiles() {
        if let data = try? JSONEncoder().encode(profiles) {
            UserDefaults.standard.set(data, forKey: storageKey)
        }
    }
    
    /// Save current profile to UserDefaults.
    private func saveCurrentProfile() {
        if let profile = currentProfile,
           let data = try? JSONEncoder().encode(profile) {
            UserDefaults.standard.set(data, forKey: currentProfileKey)
        }
    }
    
    /// Detect current hardware and update the current profile.
    func detectCurrentHardware() {
        currentProfile = HardwareProfile.detect()
        saveCurrentProfile()
    }
    
    /// Add a new profile.
    func addProfile(_ profile: HardwareProfile) {
        profiles.append(profile)
        saveProfiles()
    }
    
    /// Update an existing profile.
    func updateProfile(_ profile: HardwareProfile) {
        if let index = profiles.firstIndex(where: { $0.id == profile.id }) {
            var updated = profile
            updated.updatedAt = Date()
            profiles[index] = updated
            saveProfiles()
        }
    }
    
    /// Delete a profile.
    func deleteProfile(_ id: UUID) {
        profiles.removeAll { $0.id == id }
        saveProfiles()
    }
    
    /// Export a profile as JSON data.
    func exportProfile(_ profile: HardwareProfile) -> Data? {
        try? JSONEncoder().encode(profile)
    }
    
    /// Import a profile from JSON data.
    func importProfile(from data: Data) -> HardwareProfile? {
        guard var profile = try? JSONDecoder().decode(HardwareProfile.self, from: data) else {
            return nil
        }
        // Assign a new ID to prevent duplicates
        profile = HardwareProfile(
            id: UUID(),
            name: profile.name,
            cpu: profile.cpu,
            ramGB: profile.ramGB,
            gpus: profile.gpus,
            hasUnifiedMemory: profile.hasUnifiedMemory,
            bandwidthGBs: profile.bandwidthGBs,
            fp16TFLOPS: profile.fp16TFLOPS,
            osVersion: profile.osVersion,
            createdAt: Date(),
            updatedAt: Date()
        )
        addProfile(profile)
        return profile
    }
    
    /// Find the best matching preset for the current hardware.
    func findBestMatchingPreset() -> HardwareProfile? {
        guard let current = currentProfile else { return nil }
        
        // Simple matching based on GPU name and RAM
        return profiles.first { profile in
            profile.gpus.first?.name == current.gpus.first?.name &&
            abs(profile.ramGB - current.ramGB) < 16
        }
    }
}
