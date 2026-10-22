// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Input for the hardware-aware starting point. It is deliberately free of UI
/// and UserDefaults so the recommendation can be unit-tested offline.
struct HardwareTuningInput: Equatable {
    var gpuName: String
    var vramMB: Int
    var ramGB: Double
    var physicalCores: Int
    var logicalCores: Int
    var reserveMB: Int
    var modelFileGB: Double
    var modelLayers: Int
    var isMoE: Bool
    /// Routed expert bytes from the GGUF tensor table; 0 means unknown.
    var expertGB: Double
    /// Exact f16 KV bytes per token; 0 means the geometry was unreadable.
    var kvBytesPerToken: Double
    var trainedContext: Int?
    var requestedCtx: Int
    var requestedUbatch: Int
    var currentNgl: Int
    var currentNcmoe: Int
    var currentCacheK: String
    var currentCacheV: String
    var currentThreads: Int
    var dynamicMoeActive: Bool
    var manualFullGPU: Bool
    var routerMode: Bool
    var gpuCount: Int = 1
}

enum HardwareTuningWarning: Equatable {
    case ubatchAboveSafeLimit(current: Int, safe: Int)
    case quantizedKVOnWave64(k: String, v: String)
    case expertsAboveGPULayers(ncmoe: Int, ngl: Int)
    case contextAboveEstimatedFit(requested: Int, safe: Int)
    case cpuOnly
    case dynamicMoeActive
    case fullGPUForced
    case routerMayShrinkContext
    case multiGPUSplit
}

enum HardwareTuningReason: Equatable {
    case measuredWave64Profile
    case fullGPU
    case partialMoE
    case partialDense
    case wave64KVF16
    case ubatchClipped(from: Int, to: Int)
    case ubatchForExpertUploads
    case trainedContextCap(Int)
    case physicalCores
    case reserve(Int)
}

/// A safe, explainable starting point for the manual offload knobs.
///
/// The allocator is conservative: it leaves the engine's fixed buffers alone,
/// counts only the weights that `-ngl` actually moves to the GPU, and splits the
/// KV cache across those layers. The one hard-coded profile is the machine the
/// optimization report measured; everywhere else the geometry decides.
struct HardwareTuningAdvice: Equatable {
    let gpuName: String
    let vramMB: Int
    let usableVRAMGB: Double
    let architecture: String?
    let ramGB: Double
    let physicalCores: Int
    let logicalCores: Int
    let gpuCount: Int

    let ngl: Int
    let ncmoe: Int
    let ctx: Int
    let ubatch: Int
    let cacheTypeK: String
    let cacheTypeV: String
    let threads: Int
    let estimatedVRAMGB: Double
    let fullGPU: Bool
    let measuredProfile: Bool
    let modelLayers: Int
    let warnings: [HardwareTuningWarning]
    let reasons: [HardwareTuningReason]

    var isWave64: Bool { architecture == "GCN / Vega" }

    /// Whether the server already runs exactly this configuration. The values
    /// that Dynamic MoE owns are never "applied", because the engine replaces
    /// them at launch.
    func isApplied(to settings: ServerSettings) -> Bool {
        guard !settings.usesAutoPlan, !settings.manualFullGPU else { return false }
        let nglMatches = ngl == 99
            ? settings.ngl >= modelLayers || settings.ngl >= 99
            : settings.ngl == ngl
        return nglMatches
            && settings.ncmoe == ncmoe
            && settings.ctx == ctx
            && !settings.contextAutomatic
            && settings.ubatch == ubatch
            && settings.cacheTypeK == cacheTypeK
            && settings.cacheTypeV == cacheTypeV
            && settings.threads == threads
    }
}

enum HardwareTuningAdvisor {
    /// Measured on the report's machine: iMac20,1, 4 GB Radeon Pro 575X
    /// (`gfx803`, wave64), Qwen3-Coder-30B-A3B Q4_K_M (17 GB, 48 layers).
    /// `-ngl 10 --n-cpu-moe 6 -c 16384 -ub 1024 -ctk f16 -ctv f16 -t 6`
    /// was the best combination for both prompt and generation: pp512 47.9,
    /// tg128 11.8. The profile is scaled by layer count for other models.
    private static let measuredVRAMCeilingMB = 5 * 1024
    private static let measuredFileFloorGB = 12.0
    private static let measuredLayers = 48
    private static let measuredNgl = 10
    private static let measuredNcmoe = 6
    private static let measuredUbatch = 1024

    /// 4 GB gfx803 + 17 GB MoE: the prefill compute buffer grows ~2.26 MiB per
    /// ubatch token. `--ubatch-size 4096` allocated 9,699,329,568 B (9.03 GiB)
    /// and failed; 2048 and 4096 failed at context creation, 1024 and 512
    /// loaded. The ceiling of 2048 is the report's calibration for 12 GB cards,
    /// so larger cards keep their current behavior instead of being regressed.
    static let ubatchMiBPerToken = 2.26
    static let ubatchStep = 512
    static let ubatchCeiling = 2048

    /// The engine's fixed compute/KV buffer measured at ~0.9 GiB in 5.7. It is
    /// intentionally not scaled by parameter count: the report's measured
    /// configuration ran with 2.9 GiB total on a 4 GB card.
    private static let computeGB = 0.9
    /// Small headroom for allocator granularity and the driver's own padding.
    private static let safetyGB = 0.05
    private static let defaultCtx = 16_384
    private static let minimumCtx = 4_096

    static func safeUbatch(vramMB: Int, reserveMB: Int) -> Int {
        let reserve = reserveMB > 0
            ? reserveMB
            : ServerSettings.effectiveVramReserveMB(0, vramMB: vramMB)
        let usableMiB = Double(max(0, vramMB - reserve))
        let tokens = Int(usableMiB / ubatchMiBPerToken)
        let rounded = (tokens / ubatchStep) * ubatchStep
        return max(ubatchStep, min(ubatchCeiling, rounded))
    }

    static func advise(_ input: HardwareTuningInput) -> HardwareTuningAdvice? {
        guard input.vramMB > 0, input.modelFileGB > 0, input.modelLayers > 0 else { return nil }

        let reserveMB = input.reserveMB > 0
            ? input.reserveMB
            : ServerSettings.effectiveVramReserveMB(0, vramMB: input.vramMB)
        let usableGB = Double(max(0, input.vramMB - reserveMB)) / 1024
        guard usableGB > 0.1 else { return nil }

        let architecture = GPUArchitectureClassifier.architecture(for: input.gpuName)
        let wave64 = architecture == "GCN / Vega"
        let layers = input.modelLayers
        let maxNgl = min(layers, 99)

        let weightsGB = max(0.1, input.modelFileGB)
        let expertGB = input.isMoE
            ? max(0, min(weightsGB, input.expertGB > 0 ? input.expertGB : weightsGB - 1.3))
            : 0
        let otherGB = max(0.05, weightsGB - expertGB)
        let expertPerLayer = expertGB / Double(layers)
        let otherPerLayer = otherGB / Double(layers)
        let densePerLayer = weightsGB / Double(layers)

        // GCN/Vega is wave64: the report measured f16 KV at pp512 47.9 vs 27.7
        // with quantized KV, even with the AMD attention kernel enabled. Leave
        // every other architecture on whatever the user selected.
        let recommendedK = wave64 ? "f16" : (input.currentCacheK.isEmpty ? "f16" : input.currentCacheK)
        let recommendedV = wave64 ? "f16" : (input.currentCacheV.isEmpty ? "f16" : input.currentCacheV)
        let kvScale = (Estimator.kvTypeScale(recommendedK) + Estimator.kvTypeScale(recommendedV)) / 2
        let kvPerToken = (input.kvBytesPerToken > 0
                          ? input.kvBytesPerToken
                          : fallbackKVBytesPerToken(fileGB: weightsGB)) * kvScale
        let kvPerLayerToken = max(0, kvPerToken / Double(layers))

        let targetCtx = contextTarget(trained: input.trainedContext)
        func kvGB(ngl: Int, ctx: Int) -> Double {
            Double(max(0, ngl)) * kvPerLayerToken * Double(max(0, ctx)) / 1_073_741_824
        }
        func residentWeightsGB(ngl: Int, expertLayers: Int) -> Double {
            Double(max(0, ngl)) * otherPerLayer + Double(max(0, expertLayers)) * expertPerLayer
        }

        var ngl = 0
        var ncmoe = 0
        var ctx = targetCtx
        var fullGPU = false
        var measuredProfile = false
        var reasons: [HardwareTuningReason] = []

        let fullNeed = weightsGB * 1.03 + computeGB + safetyGB + kvGB(ngl: layers, ctx: targetCtx)
        if fullNeed <= usableGB, layers <= 99 {
            fullGPU = true
            ngl = 99
            reasons.append(.fullGPU)
        } else if input.isMoE {
            let measured = wave64
                && input.vramMB <= measuredVRAMCeilingMB
                && weightsGB >= measuredFileFloorGB
                && layers >= 32
            if measured {
                let baselineNgl = max(1, Int((Double(measuredNgl) * Double(layers) / Double(measuredLayers)).rounded()))
                let baselineNcmoe = max(0, Int((Double(measuredNcmoe) * Double(layers) / Double(measuredLayers)).rounded()))
                ngl = baselineNgl
                ncmoe = min(baselineNcmoe, ngl)
                // If the file is denser than the measured 17 GB model, pull the
                // profile back until it fits instead of claiming a safe fit.
                while ngl > 1,
                      computeGB + safetyGB + residentWeightsGB(ngl: ngl, expertLayers: ngl - ncmoe)
                        + kvGB(ngl: ngl, ctx: ctx) > usableGB {
                    ngl -= 1
                    ncmoe = min(ncmoe, ngl)
                }
                if ngl == 1,
                   computeGB + safetyGB + residentWeightsGB(ngl: ngl, expertLayers: ngl)
                    + kvGB(ngl: ngl, ctx: ctx) > usableGB {
                    ngl = 0
                    ncmoe = 0
                } else if ngl == baselineNgl && ncmoe == baselineNcmoe && layers == measuredLayers {
                    measuredProfile = true
                    reasons.append(.measuredWave64Profile)
                } else {
                    reasons.append(.partialMoE)
                }
            }
            if !measuredProfile {
                // Max experts on the GPU, then max attention layers that still
                // fit, with ncmoe = ngl - expertLayers. This is the same shape
                // as the measured optimum and avoids the report's ngl16/ncmoe24
                // cliff (where experts were starved).
                var best: (ngl: Int, ncmoe: Int, experts: Int)?
                if expertPerLayer > 0 {
                    for candidate in 1...maxNgl {
                        let room = usableGB - computeGB - safetyGB
                            - kvGB(ngl: candidate, ctx: targetCtx)
                            - Double(candidate) * otherPerLayer
                        guard room >= 0 else { continue }
                        let experts = min(candidate, Int(floor(room / expertPerLayer)))
                        guard experts > 0 else { continue }
                        if let b = best {
                            if experts > b.experts || (experts == b.experts && candidate > b.ngl) {
                                best = (candidate, candidate - experts, experts)
                            }
                        } else {
                            best = (candidate, candidate - experts, experts)
                        }
                    }
                }
                if let best {
                    ngl = best.ngl
                    ncmoe = best.ncmoe
                    reasons.append(.partialMoE)
                } else {
                    // Not even one expert bank fits. Keep the attention/shared
                    // layers on the GPU and leave every expert in system RAM.
                    let perLayer = otherPerLayer + kvPerLayerToken * Double(targetCtx) / 1_073_741_824
                    ngl = perLayer > 0
                        ? max(0, min(maxNgl, Int(floor((usableGB - computeGB - safetyGB) / perLayer))))
                        : 0
                    ncmoe = ngl
                    if ngl > 0 { reasons.append(.partialMoE) }
                }
            }
        } else {
            let perLayer = densePerLayer + kvPerLayerToken * Double(targetCtx) / 1_073_741_824
            ngl = perLayer > 0
                ? max(0, min(maxNgl, Int(floor((usableGB - computeGB - safetyGB) / perLayer))))
                : 0
            ncmoe = 0
            if ngl > 0 { reasons.append(.partialDense) }
        }

        // KV is allocated only for the layers that live on the GPU. If the
        // target context still overcommits, lower it to what the chosen offload
        // can hold instead of reporting a fit that will fail at load time.
        if ngl > 0 {
            let resolvedNgl = min(ngl, layers)
            let resolvedExperts = max(0, min(ngl, layers) - ncmoe)
            let used = computeGB + safetyGB + residentWeightsGB(ngl: resolvedNgl, expertLayers: resolvedExperts)
            let room = usableGB - used
            if kvPerLayerToken > 0 {
                let maxCtx = Int(floor(room * 1_073_741_824 / (Double(resolvedNgl) * kvPerLayerToken)))
                let rounded = max(minimumCtx, (maxCtx / 2048) * 2048)
                if rounded < ctx { ctx = rounded }
            }
        } else {
            ncmoe = 0
        }

        let safeUbatch = safeUbatch(vramMB: input.vramMB, reserveMB: reserveMB)
        var ubatch = input.requestedUbatch
        if ubatch > safeUbatch {
            // The picker offers 0/1024/2048/4096. When the measured cap is 512,
            // 0 means the engine default (also 512) and keeps a selectable value.
            ubatch = safeUbatch >= 1024 ? safeUbatch : 0
            reasons.append(.ubatchClipped(from: input.requestedUbatch, to: safeUbatch))
        }
        if !fullGPU, ngl > 0, ncmoe > 0, ubatch == 0, safeUbatch >= 1024 {
            // Each prefill micro-batch pays one expert upload over the bus.
            // The measured profile used 1024; keep it if the card can hold it.
            ubatch = min(measuredUbatch, safeUbatch)
            reasons.append(.ubatchForExpertUploads)
        }

        let threads = max(1, min(input.logicalCores > 0 ? input.logicalCores : input.physicalCores,
                                 input.physicalCores > 0 ? input.physicalCores : 1))
        reasons.append(.physicalCores)

        if wave64 {
            reasons.append(.wave64KVF16)
        }
        if let trained = input.trainedContext, trained > 0, trained < defaultCtx {
            reasons.append(.trainedContextCap(trained))
        }
        reasons.append(.reserve(reserveMB))

        var warnings: [HardwareTuningWarning] = []
        if input.requestedUbatch > safeUbatch {
            warnings.append(.ubatchAboveSafeLimit(current: input.requestedUbatch, safe: safeUbatch))
        }
        if wave64, input.currentCacheK != "f16" || input.currentCacheV != "f16" {
            warnings.append(.quantizedKVOnWave64(k: input.currentCacheK, v: input.currentCacheV))
        }
        if input.isMoE, input.currentNcmoe > input.currentNgl {
            warnings.append(.expertsAboveGPULayers(ncmoe: input.currentNcmoe, ngl: input.currentNgl))
        }
        let currentWeights = residentWeightsGB(
            ngl: min(input.currentNgl, layers),
            expertLayers: max(0, min(input.currentNgl, layers) - input.currentNcmoe))
        let currentNeed = computeGB + safetyGB + currentWeights
            + kvGB(ngl: min(input.currentNgl, layers), ctx: input.requestedCtx)
        if input.requestedCtx > ctx, currentNeed > usableGB {
            warnings.append(.contextAboveEstimatedFit(requested: input.requestedCtx, safe: ctx))
        }
        if ngl == 0 {
            warnings.append(.cpuOnly)
        }
        if input.dynamicMoeActive {
            warnings.append(.dynamicMoeActive)
        }
        if input.manualFullGPU {
            warnings.append(.fullGPUForced)
        }
        if input.routerMode {
            warnings.append(.routerMayShrinkContext)
        }
        if input.gpuCount > 1 {
            warnings.append(.multiGPUSplit)
        }

        let estimated = fullGPU
            ? weightsGB * 1.03 + computeGB + safetyGB + kvGB(ngl: layers, ctx: ctx)
            : computeGB + safetyGB + residentWeightsGB(ngl: ngl, expertLayers: max(0, ngl - ncmoe))
                + kvGB(ngl: ngl, ctx: ctx)

        return HardwareTuningAdvice(
            gpuName: input.gpuName,
            vramMB: input.vramMB,
            usableVRAMGB: usableGB,
            architecture: architecture,
            ramGB: input.ramGB,
            physicalCores: input.physicalCores,
            logicalCores: input.logicalCores,
            gpuCount: input.gpuCount,
            ngl: ngl,
            ncmoe: ncmoe,
            ctx: ctx,
            ubatch: ubatch,
            cacheTypeK: recommendedK,
            cacheTypeV: recommendedV,
            threads: threads,
            estimatedVRAMGB: estimated,
            fullGPU: fullGPU,
            measuredProfile: measuredProfile,
            modelLayers: layers,
            warnings: warnings,
            reasons: reasons)
    }

    /// Model geometry for a path, reusing catalog metadata and falling back to
    /// the file size plus the GGUF header.
    static func spec(forModelPath path: String, models: [LocalModel]) -> ModelSpec? {
        guard !path.isEmpty else { return nil }
        if let local = models.first(where: { $0.url.path == path }) {
            return Catalog.spec(forLocal: local)
        }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let bytes = (attributes[.size] as? NSNumber)?.int64Value, bytes > 0 else { return nil }
        let isMoE = ServerSettings.modelIsMoE(at: path)
        var estimated = ModelSpec.estimated(
            fileBytes: bytes, isMoE: isMoE,
            name: ModelName.forPath(path).display, path: path)
        if let blocks = GGUFMetadataCache.metadata(at: path)?
            .uint32(forSuffix: "block_count"), blocks > 0, Int(blocks) != estimated.layers {
            estimated = ModelSpec(
                fileGB: estimated.fileGB, paramsB: estimated.paramsB,
                layers: Int(blocks), isMoE: estimated.isMoE,
                activeParamsB: estimated.activeParamsB,
                kvBytesPerToken: estimated.kvBytesPerToken,
                expertGB: estimated.expertGB, lazyGB: estimated.lazyGB)
        }
        return estimated
    }

    /// Reads the selected model and the current settings, so the card and its
    /// unit tests share the same path.
    static func advise(hardware: HardwareInfo, spec: ModelSpec,
                       settings: ServerSettings) -> HardwareTuningAdvice? {
        guard !settings.modelPath.isEmpty else { return nil }
        let gpus = hardware.splitEligibleGPUs
        let selected = gpus.filter { settings.selectedGPUIndices.contains($0.index) }
        let active = selected.isEmpty ? gpus : selected
        let best = active.max(by: { $0.vramMB < $1.vramMB }) ?? hardware.bestGPU
        guard let best, best.vramMB > 0 else { return nil }
        let vramMB = settings.isSplitting
            ? active.reduce(0) { $0 + $1.vramMB }
            : best.vramMB
        let trained = GGUFMetadataCache.metadata(at: settings.modelPath)?.trainedContext
        return advise(HardwareTuningInput(
            gpuName: best.name,
            vramMB: vramMB,
            ramGB: hardware.ramGB,
            physicalCores: hardware.physicalCores,
            logicalCores: hardware.logicalCores,
            reserveMB: settings.vramReserveMB,
            modelFileGB: spec.fileGB,
            modelLayers: spec.layers,
            isMoE: spec.isMoE,
            expertGB: spec.expertGB,
            kvBytesPerToken: spec.kvBytesPerToken,
            trainedContext: trained,
            requestedCtx: settings.ctx,
            requestedUbatch: settings.ubatch,
            currentNgl: settings.ngl,
            currentNcmoe: settings.ncmoe,
            currentCacheK: settings.cacheTypeK,
            currentCacheV: settings.cacheTypeV,
            currentThreads: settings.threads,
            dynamicMoeActive: settings.usesAutoPlan,
            manualFullGPU: settings.manualFullGPU,
            routerMode: settings.routerMode,
            gpuCount: settings.isSplitting ? active.count : 1))
    }

    private static func contextTarget(trained: Int?) -> Int {
        let cap = trained.flatMap { $0 > 0 ? $0 : nil } ?? defaultCtx
        let target = min(defaultCtx, cap)
        let floor = min(minimumCtx, cap)
        return max(floor, (target / 2048) * 2048)
    }

    /// Same GQA heuristic as `Estimator.kvCache`, expressed per token.
    private static func fallbackKVBytesPerToken(fileGB: Double) -> Double {
        let params = max(0.1, fileGB) / 0.57
        let perK = params >= 25 ? 0.10 : params >= 12 ? 0.08 : 0.05
        return perK * 1_073_741_824 / 1024
    }
}
