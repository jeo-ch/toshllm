// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import Metal

extension Notification.Name {
    /// Posted when a fresh engine process has launched (KV slots are empty).
    static let engineDidStart = Notification.Name("toshEngineDidStart")
}

struct GPUDevice: Identifiable, Hashable {
    let index: Int
    let name: String
    let vramMB: Int
    var isExternal: Bool = false   // eGPU (MTLDeviceLocation.external)
    var isIntegrated: Bool = false // iGPU (MTLDevice.isLowPower); never auto-selected
    /// Metal peer group: 0 when the card has no Infinity Fabric link, shared with
    /// the other members when it has one.
    var peerGroupID: UInt64 = 0
    var peerCount: Int = 0
    /// Metal exposes bf16 only from the Metal 3 family up; older cards abort on a bf16 weight.
    var supportsBF16: Bool = true
    var id: Int { index }
    /// Rounded, since Metal reports a working set a little off the nominal size.
    var vramGB: Int { Int((Double(vramMB) / 1024).rounded()) }
}

enum DynamicMoeAutoRoute: Equatable {
    case cache
    case normalDense
    case normalFitsVRAM
    case normalInsufficientRAM
    case normalUnsupportedGPU
    case normalMissingModel
    case normalSplitOrRouter
    case normalMissingMetadata
    case normalInsufficientVRAM
    case normalNoCacheBenefit
    case normalOversizedHostBank
}

struct DynamicMoeModelInfo: Equatable {
    let layerCount: Int
    let expertCount: Int
    let activeExpertCount: Int
}

struct DynamicMoeSlotPlan: Equatable {
    let model: DynamicMoeModelInfo
    /// Smallest cache that can hold every expert selected by one decoded token.
    let minimumSlots: Int
    /// Largest valid K for this architecture, regardless of memory pressure.
    let maximumSlots: Int
    /// Largest K estimated to fit after fixed weights, runtime and transfer buffers.
    let recommendedMaximumSlots: Int
    let automaticSlots: Int
    let estimatedBytesPerSlot: UInt64
    let estimatedFixedVRAMBytes: UInt64

    func clamped(_ slots: Int) -> Int {
        min(max(slots, minimumSlots), maximumSlots)
    }

    func estimatedVRAMBytes(slots: Int) -> UInt64 {
        estimatedFixedVRAMBytes + UInt64(clamped(slots)) * estimatedBytesPerSlot
    }
}

struct ServerSettings {
    var serverBinary: String
    var modelPath: String
    var port: Int
    var ngl: Int
    var ncmoe: Int
    var ctx: Int
    var contextAutomatic: Bool = false
    var threads: Int
    var flashAttn: String      // auto | on | off
    var noMmap: Bool
    var jinja: Bool
    var vramReserveMB: Int
    var gpuIndex: Int          // -1 = system default
    /// Explicit set of physical GPUs to split across (2+ entries). Overrides
    /// `gpuIndex` and the all/N `multiGPU` split.
    var gpuList: [Int] = []
    var extraArgs: String
    /// Serve /v1/embeddings (--embeddings). llama-server restricts the process
    /// to embedding use, so it's meant for a dedicated embedding-model server.
    var embeddings: Bool = false
    var agentToolsEnabled: Bool = false
    var uiMcpProxy: Bool = false
    /// `--tools-runtime` target (`docker:image`, `podman:image`, `ssh:host`...). Empty
    /// runs the tools in the app's own environment, which is the engine default.
    var toolsRuntime: String = ""
    var cacheTypeK: String     // f16 | q8_0 | q5_x | q4_x | iq4_nl
    var cacheTypeV: String
    var mlock: Bool
    /// Host-RAM prompt cache cap in MiB (0 disables). The engine's own 8192 default
    /// grows with use and pushes a big model into swap, degrading speed per session.
    var cacheRAM: Int = 2048
    /// Emit reasoning inline in `content` (<think>…) instead of the separate
    /// `reasoning_content` field, for external clients that ignore the latter.
    var reasoningInline: Bool = false
    /// Server slots (0 = engine auto). With 1, requests queue instead of competing
    /// for the GPU, and a prefill aborted by a client timeout stays in the slot so
    /// the retry resumes where it left off.
    var parallelSlots: Int = 1
    var apiKeyEnabled: Bool = false
    /// Serve beyond loopback and advertise with Bonjour. Off by default: it opens
    /// the API to the whole local network, so it wants `apiKeyEnabled` alongside.
    var localNetworkDiscovery: Bool = false
    /// Legacy profile field kept for decoding older saved settings. MTP is now
    /// selected automatically from model capability and expert offload.
    var specMTP: Bool = false
    /// Route attention through the AMD Metal Flash-Attention kernel (TOSH_FA_AMD).
    var faAmd: Bool = true
    /// MoE-offload prefill boost: overlap expert uploads with compute
    /// (GGML_SCHED_PREFETCH_EXPERTS) and keep CPU experts unpacked so their
    /// matmuls can offload (GGML_CPU_NO_REPACK).
    var prefetchExperts: Bool = true
    /// Prefill micro-batch (--ubatch-size); 0 leaves the engine's 512. Only worth
    /// raising with experts on CPU: each micro-batch pays one expert upload over
    /// the bus, so a long prompt pays it once per batch instead of once per token.
    /// Costs VRAM linearly (~0.5 GiB per 512 on a 35B), so it is not a default.
    var ubatch: Int = 0
    /// Experimental bounded-VRAM expert cache. Compiled into the bundled engine,
    /// but completely inert unless this persisted user choice is enabled.
    var dynamicMoe: Bool = false
    var dynamicMoeSlots: Int = 8
    var dynamicMoePrefetch: Int = 4
    var dynamicMoePolicy: String = "cache" // cache | auto
    /// Router mode (`--models-preset`): one process auto-loads/unloads whichever
    /// model a request's "model" field names, instead of the fixed `modelPath`.
    var routerMode: Bool = false
    /// Models the router keeps loaded at once (LRU); 1 is safest on a single GPU.
    var routerModelsMax: Int = 1
    /// Persist each conversation's KV cache to disk (`--slot-save-path`). Needs the
    /// AMD FA kernel: without it the V cache is transposed and save/restore copies
    /// it row-by-row, unusably slow.
    var persistCache: Bool = false
    /// EXPERIMENTAL: split one model across all detected GPUs. Unvalidated on
    /// AMD/Metal: cross-GPU copies take a different path than the staging the
    /// patch covers and could corrupt or hang.
    var multiGPU: Bool = false
    /// How many GPUs to split across when `multiGPU` is on. 0 = all detected.
    /// Fewer GPUs trade prompt speed for generation speed (less cross-card sync).
    var multiGPUCount: Int = 0
    /// How the split divides the model (`--split-mode`): `layer` gives each GPU
    /// whole layers, `tensor` splits every matmul so both work on the same token,
    /// which costs one allreduce per layer and only pays off on a big model.
    var splitMode: String = "layer"
    /// GPUs per tensor-parallel group when splitting by tensors. 0 or 1 means one
    /// group with every card, which is the plain tensor split. A smaller group
    /// splits tensors inside it and layers between groups, which keeps most of the
    /// prompt speed without paying the per-layer wait across every card.
    var splitGroupSize: Int = 0
    /// Hand off activations between GPUs with shared Metal events instead of
    /// draining both queues on every copy (TOSH_MGPU_EVENTS). Inert on a layer
    /// split; on a tensor split it is most of the generation speed.
    var mgpuEvents: Bool = true
    /// When two split GPUs share a Metal peer group (Infinity Fabric Link, e.g. a
    /// W6800X/Vega II Duo), copy activations die-to-die instead of via host
    /// (TOSH_MGPU_PEER). Worth 16% of the prefill, and only safe alongside mgpuEvents.
    var mgpuPeer: Bool = true
    /// Force VRAM-resident (private) Metal buffers. The backend forces shared ones
    /// for external GPUs, which streams weights over Thunderbolt every op; this
    /// covers the default-GPU case, where the app can't tell macOS picked an eGPU.
    var forcePrivateBuffers: Bool = false
    /// Reuse shifted KV chunks across mid-prompt divergences (agent edits, a
    /// stripped <think>). Fast but approximate, hence user-toggleable.
    var cacheReuse: Bool = true
    /// Load the multimodal projector (mmproj) for vision-capable models. Off skips it
    /// so a vision model runs text-only and frees the VRAM the image encoder would use.
    var loadVision: Bool = true
    /// Cap on the tokens one image may take. 0 keeps the model's own limit; lowering it
    /// shrinks the vision encoder's attention quadratically, which is what rescues a
    /// card that cannot allocate that buffer.
    var imageMaxTokens: Int = 0
    /// llama-bench workload sizes: prompt tokens (-p → ppN), generated tokens
    /// (-n → tgN) and context depth (-d, tokens already in the KV cache before
    /// measuring). Benchmark-only; llama-server ignores them.
    var benchPP: Int = 512
    var benchTG: Int = 128
    var benchDepth: Int = 0

    /// One model served across several GPUs, either by the all/N toggle or by an
    /// explicit selection of at least two cards.
    var isSplitting: Bool { multiGPU || gpuList.count >= 2 }
    /// Guards against a stale or hand-edited value: llama.cpp only takes these two,
    /// and a tensor split the model cannot take falls back instead of aborting.
    var effectiveSplitMode: String {
        (splitMode == "tensor" && tensorSplitLimit.map { splitDeviceCount <= $0 } != false) ? "tensor" : "layer"
    }

    /// The group size to pass, or nil when it would be a no-op or an invalid split.
    var effectiveSplitGroupSize: Int? {
        let n = splitDeviceCount
        guard splitGroupSize > 1, splitGroupSize < n, n % splitGroupSize == 0 else { return nil }
        return splitGroupSize
    }

    /// GPUs the split will span, matching how the launch environment picks them.
    var splitDeviceCount: Int {
        if gpuList.count >= 2 { return gpuList.count }
        guard multiGPU else { return 1 }
        let gpus = ServerController.availableGPUs()
        let discrete = gpus.filter { !$0.isIntegrated }.count
        let limit = discrete > 0 ? discrete : gpus.count
        return max(2, multiGPUCount > 0 ? min(multiGPUCount, limit) : limit)
    }

    /// Layer counts per GPU that even out the bytes rather than the layer count.
    /// `--n-cpu-moe` empties the experts of the first N layers, and llama.cpp still hands
    /// out layers by index, so without this the last GPUs take every expert and overflow.
    var layerBalancedTensorSplit: [Int]? {
        guard isSplitting, effectiveSplitMode == "layer", ncmoe > 0 else { return nil }
        let devices = splitDeviceCount
        guard devices >= 2 else { return nil }
        let flags = GGUFMetadataCache.tensorFlags(at: modelPath)
        let expert = Double(flags.expertBytesPerLayer)
        let other = Double(flags.otherBytesPerLayer)
        guard expert > 0, other > 0,
              let blocks = Self.ggufUInt32("block_count", at: modelPath).map(Int.init),
              blocks >= devices else { return nil }

        let weights = (0..<blocks).map { $0 < ncmoe ? other : other + expert }
        let target = weights.reduce(0, +) / Double(devices)
        var counts = [Int](repeating: 0, count: devices)
        var dev = 0
        var acc = 0.0
        for (il, w) in weights.enumerated() {
            // leave at least one layer for each device still waiting
            let mustAdvance = blocks - il == devices - dev
            if dev < devices - 1 && (mustAdvance || (acc + w/2 > target && counts[dev] > 0)) {
                dev += 1
                acc = 0
            }
            counts[dev] += 1
            acc += w
        }
        return counts.allSatisfy { $0 > 0 } ? counts : nil
    }

    /// GPUs this model's experts can be divided among, nil when nothing limits it.
    var tensorSplitLimit: Int? { Self.tensorSplitLimit(forModel: modelPath) }

    /// The user asked for a tensor split the model cannot take; the run uses layers.
    var tensorSplitDowngraded: Bool {
        splitMode == "tensor" && isSplitting && effectiveSplitMode == "layer"
    }

    var isMultimodal: Bool { Self.mmprojPath(forModel: modelPath) != nil }
    /// Vision actually loaded (projector available AND the eye is on); slot
    /// persistence gates on this, not on mere availability.
    var visionLoaded: Bool { loadVision && isMultimodal }

    /// `--no-mmap` and `--mlock` both write the engine's load mode and the last one wins, so
    /// passing the pair let mlock decide whether mmap stayed on. Nil keeps the engine default.
    static func loadMode(noMmap: Bool, mlock: Bool) -> String? {
        switch (noMmap, mlock) {
        case (true, true):   return "mlock"
        case (true, false):  return "none"
        case (false, true):  return "mmap+mlock"
        case (false, false): return nil
        }
    }

    /// Directory where one server's per-conversation KV slot files live. Namespaced
    /// by port so independent servers don't overwrite each other's slot 0 / prefix
    /// files in a shared folder.
    static func slotCacheDir(port: Int) -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ToshLLM/slots/\(port)/\(slotKVSignature())")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A slot only restores into an identical KV layout; separate folders per type
    /// turn a quant change into a cold prefill instead of a failed restore.
    private static func slotKVSignature() -> String {
        let d = UserDefaults.standard
        // a slot holding media only restores with a projector loaded, so vision state splits it too
        let vision = d.object(forKey: SettingsKeys.loadVision) as? Bool ?? true
        return "\(sanitizedKV(d.string(forKey: SettingsKeys.cacheTypeK), default: "f16"))"
             + "-\(sanitizedKV(d.string(forKey: SettingsKeys.cacheTypeV), default: "f16"))"
             + (vision ? "-v" : "")
    }

    /// Slot directory of the primary server (the one the native chat talks to).
    static var primarySlotCacheDir: URL { slotCacheDir(port: fromDefaults().port) }

    /// Drops a KV type left over from the retired turbo engine.
    static func sanitizedKV(_ value: String?, default def: String) -> String {
        guard let v = value, kvCacheTypes.contains(v) else { return def }
        return v
    }

    static let defaultFaAmd = true
    static let dynamicMoeTensorOverride = #"\.ffn_(up|down|gate|gate_up)_(ch|)exps=MTL0"#
    static let kvCacheTypes = ["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "turbo4", "turbo3"]

    var usesTurboKV: Bool {
        cacheTypeK.hasPrefix("turbo") || cacheTypeV.hasPrefix("turbo")
    }

    var usesTurboValuesWithoutKeys: Bool {
        cacheTypeV.hasPrefix("turbo") && !cacheTypeK.hasPrefix("turbo")
    }

    var usesUnsupportedTurboQ4Mix: Bool {
        (cacheTypeK.hasPrefix("turbo") && cacheTypeV == "q4_0") ||
        (cacheTypeV.hasPrefix("turbo") && cacheTypeK == "q4_0")
    }

    var arguments: [String] {
        if routerMode { return routerArguments }
        // Quantized KV requires FA, so it stays forced. Elsewhere the AMD kernel rides
        // on "auto": an explicit "1" would drop uncovered models to the CPU fallback.
        let faValue = kvNeedsFlashAttention ? "1" : (effectiveFaAmd ? "auto" : flashAttn)
        var args = [
            "-m", modelPath,
            "-ngl", String(ngl),
            "-c", String(ctx),
            "-t", String(threads),
            "-fa", faValue,
            "--host", localNetworkDiscovery ? "0.0.0.0" : "127.0.0.1",
            "--port", String(port),
        ]
        if !effectiveDynamicMoe && ncmoe > 0 { args += ["--n-cpu-moe", String(ncmoe)] }
        if let ub = effectiveUbatch { args += ["--ubatch-size", String(ub), "--batch-size", String(ub)] }
        let mode = effectiveDynamicMoe ? "mlock" : Self.loadMode(noMmap: noMmap, mlock: mlock)
        if let mode { args += ["--load-mode", mode] }
        if effectiveDynamicMoe {
            args += ["-ot", Self.dynamicMoeTensorOverride]
        }
        // A sibling projector lets the model read images, and needs --jinja.
        let mmproj = loadVision ? Self.mmprojPath(forModel: modelPath) : nil
        if let mmproj {
            args += ["--mmproj", mmproj]
            if Self.projectorNeedsCPU(mmproj) { args.append("--no-mmproj-offload") }
            if imageMaxTokens > 0 { args += ["--image-max-tokens", String(imageMaxTokens)] }
        }
        if jinja || mmproj != nil { args.append("--jinja") }
        if cacheTypeK != "f16" { args += ["-ctk", cacheTypeK] }
        if cacheTypeV != "f16" { args += ["-ctv", cacheTypeV] }
        // localhost-only endpoint; feeds the speculative-decoding readout in Diagnostics
        args.append("--metrics")
        args += ["--cache-ram", String(cacheRAM)]
        if cacheReuse && mmproj == nil {
            args += ["--cache-reuse", "256"]
        }
        if parallelSlots > 0 { args += ["--parallel", String(parallelSlots)] }
        // An explicit --parallel N splits the context pool across slots instead of
        // sharing it (only "auto" unifies), so ask for one pool ourselves: the chat
        // keeps its full window and API requests don't multiply KV memory.
        if parallelSlots > 1 { args.append("--kv-unified") }
        // Which devices to split across is decided by the env vars below.
        if isSplitting { args += ["--split-mode", effectiveSplitMode] }
        if let counts = layerBalancedTensorSplit {
            args += ["--tensor-split", counts.map(String.init).joined(separator: ",")]
        }
        if embeddings { args.append("--embeddings") }
        if agentToolsEnabled {
            if !args.contains("--jinja") { args.append("--jinja") }
            args += ["--tools", "all"]
            if !toolsRuntime.isEmpty { args += ["--tools-runtime", toolsRuntime] }
        }
        if uiMcpProxy { args.append("--ui-mcp-proxy") }
        if persistCache && effectiveFaAmd {
            args += ["--slot-save-path", Self.slotCacheDir(port: port).path]
        }
        if reasoningInline { args += ["--reasoning-format", "none"] }
        if apiKeyEnabled { args += ["--api-key", Keychain.apiKey()] }
        // A compatible downloaded DFlash draft takes precedence over embedded MTP.
        // Speculation decodes several tokens at once and the expert cache only has slots
        // for one token's experts, so the two cannot run together.
        if !effectiveDynamicMoe {
            if let selection = dflashSelection(modelPath: modelPath, ncmoe: ncmoe) {
                // Quantize the draft's KV cache: it doubles KV pressure at high ctx, and
                // q8_0 halves that footprint at no measurable quality cost for a draft.
                args += ["-md", selection.draft, "--spec-type", "draft-dflash",
                         "-ngld", String(selection.ngld),
                         "-ctkd", "q8_0", "-ctvd", "q8_0"]
            } else if Self.mtpEnabled(forModel: modelPath), let draft = Self.mtpDraftPath(forModel: modelPath) {
                args += ["-md", draft, "--spec-type", "draft-mtp"]
            } else if Self.mtpEnabled(forModel: modelPath), Self.modelHasMTP(at: modelPath) {
                args += ["--spec-type", "draft-mtp"]
            }
        }
        if let ui = Self.chatUIPath { args += ["--path", ui] }
        args += extraArgTokens.cli
        return args
    }

    /// Router writes one entry per model with the same context, but a multi-head model
    /// caches several times more per token than a GQA one and can overflow VRAM on
    /// load, which hangs the driver. Shrink the context for the models that don't fit.
    nonisolated static func routerCtx(forModel path: String, requested: Int, ncmoe: Int,
                                      kvScale: Double, reserveMB: Int) -> Int {
        let kvPerToken = ModelSpec.kvBytesPerToken(atPath: path) * kvScale
        guard kvPerToken > 0, requested > 2048 else { return requested }
        let vramGB = Double(ServerController.availableGPUs().map(\.vramMB).max() ?? 0) / 1024
        guard vramGB > 0 else { return requested }
        let budget = vramGB - Double(reserveMB) / 1024
        let size = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int64) ?? 0
        let weightsGB = Double(size) / 1_073_741_824 * (ncmoe > 0 ? 0.45 : 1.0)
        let computeGB = 0.9
        var value = requested
        while value > 2048,
              weightsGB + computeGB + kvPerToken * Double(value) / 1_073_741_824 > budget {
            value /= 2
        }
        return value
    }

    /// Router-mode CLI args: no `-m`, the preset file lists every model. Per-model
    /// flags (mmproj, ncmoe, MTP...) live in that INI instead, see `routerPresetINI`.
    private var routerArguments: [String] {
        var args = [
            "--models-preset", Self.routerPresetPath(port: port).path,
            "--models-max", String(routerModelsMax),
            "--models-autoload",
            "--host", localNetworkDiscovery ? "0.0.0.0" : "127.0.0.1",
            "--port", String(port),
        ]
        args.append("--metrics")
        if agentToolsEnabled {
            args += ["--jinja", "--tools", "all"]
            if !toolsRuntime.isEmpty { args += ["--tools-runtime", toolsRuntime] }
        }
        if uiMcpProxy { args.append("--ui-mcp-proxy") }
        if apiKeyEnabled { args += ["--api-key", Keychain.apiKey()] }
        if let ui = Self.chatUIPath { args += ["--path", ui] }
        args += extraArgTokens.cli
        return args
    }

    /// Same folder `ModelStore` scans (`~/models` or the custom override),
    /// resolved independently since `ServerController` has no `ModelStore`.
    static var modelsDirectory: URL {
        let custom = UserDefaults.standard.string(forKey: SettingsKeys.modelsDir) ?? ""
        return custom.isEmpty ? ModelStore.defaultDirectory : URL(fileURLWithPath: custom, isDirectory: true)
    }

    static func routerPresetPath(port: Int) -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ToshLLM/router")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("preset-\(port).ini")
    }

    /// Stable, INI/URL-safe id derived from a model's filename: the router
    /// preset's section name and the value clients send as `"model"`.
    static func routerAlias(for modelPath: String) -> String {
        let base = URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
        var slug = String(base.lowercased().map { $0.isLetter || $0.isNumber || $0 == "-" ? $0 : "-" })
        while slug.contains("--") { slug = slug.replacingOccurrences(of: "--", with: "-") }
        slug = slug.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return slug.isEmpty ? "model" : slug
    }

    /// Builds the router's `--models-preset` INI: one `[alias]` section per model,
    /// shared engine config plus per-path ncmoe/mmproj/MTP. `extraArgs` (free-form
    /// CLI tokens) isn't representable generically here, so it's skipped.
    func routerPresetINI(modelPaths: [String], ncmoeByPath: [String: Int]) -> String {
        // Same FA policy as `arguments`: force only for quantized KV.
        let faValue = kvNeedsFlashAttention ? "on" : (effectiveFaAmd ? "auto" : flashAttn)
        var seenAliases = Set<String>()
        var sections: [String] = []
        for path in modelPaths.sorted() {
            var alias = Self.routerAlias(for: path)
            if seenAliases.contains(alias) { alias += "-\(abs(path.hashValue) % 1000)" }
            seenAliases.insert(alias)

            let modelCtx = Self.routerCtx(forModel: path, requested: ctx, ncmoe: ncmoeByPath[path] ?? 0,
                                          kvScale: Estimator.kvTypeScale(cacheTypeK),
                                          reserveMB: vramReserveMB)
            var lines = ["[\(alias)]", "model = \(path)", "n-gpu-layers = \(ngl)",
                         "ctx-size = \(modelCtx)", "threads = \(threads)", "flash-attn = \(faValue)"]
            if !effectiveDynamicMoe, let ncmoe = ncmoeByPath[path], ncmoe > 0 { lines.append("n-cpu-moe = \(ncmoe)") }
            if let mode = Self.loadMode(noMmap: noMmap, mlock: mlock) { lines.append("load-mode = \(mode)") }
            let mmproj = loadVision ? Self.mmprojPath(forModel: path) : nil
            if let mmproj {
                lines.append("mmproj = \(mmproj)")
                if Self.projectorNeedsCPU(mmproj) { lines.append("mmproj-offload = false") }
                if imageMaxTokens > 0 { lines.append("image-max-tokens = \(imageMaxTokens)") }
            }
            if jinja || mmproj != nil { lines.append("jinja = true") }
            if cacheTypeK != "f16" { lines.append("cache-type-k = \(cacheTypeK)") }
            if cacheTypeV != "f16" { lines.append("cache-type-v = \(cacheTypeV)") }
            lines.append("cache-ram = \(cacheRAM)")
            if cacheReuse && mmproj == nil { lines.append("cache-reuse = 256") }
            if parallelSlots > 0 { lines.append("parallel = \(parallelSlots)") }
            if parallelSlots > 1 { lines.append("kv-unified = true") }
            if isSplitting { lines.append("split-mode = \(effectiveSplitMode)") }
            if persistCache && effectiveFaAmd {
                // the engine rejects a slot path that isn't an existing directory
                let slotDir = Self.slotCacheDir(port: port).appendingPathComponent(alias)
                try? FileManager.default.createDirectory(at: slotDir, withIntermediateDirectories: true)
                lines.append("slot-save-path = \(slotDir.path)")
            }
            if reasoningInline { lines.append("reasoning-format = none") }
            if effectiveDynamicMoe {
                // see the speculation note in arguments(): it does not mix with the cache
            } else if let selection = dflashSelection(modelPath: path, ncmoe: ncmoeByPath[path] ?? 0) {
                lines.append("model-draft = \(selection.draft)")
                lines.append("spec-type = draft-dflash")
                lines.append("gpu-layers-draft = \(selection.ngld)")
                lines.append("cache-type-k-draft = q8_0")
                lines.append("cache-type-v-draft = q8_0")
            } else if Self.mtpEnabled(forModel: path), let draft = Self.mtpDraftPath(forModel: path) {
                lines.append("model-draft = \(draft)")
                lines.append("spec-type = draft-mtp")
            } else if Self.mtpEnabled(forModel: path), Self.modelHasMTP(at: path) {
                lines.append("spec-type = draft-mtp")
            }
            sections.append(lines.joined(separator: "\n"))
        }
        return sections.joined(separator: "\n\n") + "\n"
    }

    /// Splits the Extra arguments field: `KEY=VALUE` with an UPPERCASE name becomes an
    /// env var, everything else a CLI argument. The uppercase rule is what keeps
    /// lowercase flag values containing `=` (`--override-kv key=str:foo`) out of the env.
    var extraArgTokens: (env: [String: String], cli: [String]) {
        func isEnvName(_ s: Substring) -> Bool {
            guard let first = s.first, first == "_" || (first.isLetter && first.isUppercase) else { return false }
            return s.allSatisfy { $0 == "_" || $0.isNumber || ($0.isLetter && $0.isUppercase) }
        }
        var env: [String: String] = [:]
        var cli: [String] = []
        // The model's own arguments come last so they win over the shared field.
        let combined = extraArgs + " " + Self.extraArgs(forModel: modelPath)
        for tok in ShellWords.split(combined) {
            if let eq = tok.firstIndex(of: "="), eq != tok.startIndex, isEnvName(tok[..<eq]) {
                env[String(tok[..<eq])] = String(tok[tok.index(after: eq)...])
            } else {
                cli.append(tok)
            }
        }
        return (env, cli)
    }

    /// Arguments for `llama-bench`: separate from the server's because server-only
    /// flags are invalid here, but every option affecting speed must carry over.
    var benchmarkArguments: [String] {
        // Same load mode as the server, or the numbers are not the ones the app
        // delivers: with experts on the CPU, locking the model is worth most of
        // the prompt speed.
        var args = ["-m", modelPath, "-ngl", String(ngl), "-r", "2",
                    "-p", String(benchPPClamped), "-n", String(benchTGClamped)]
        let mode = effectiveDynamicMoe ? "mlock" : Self.loadMode(noMmap: noMmap, mlock: mlock)
        if let mode { args += ["--load-mode", mode] }
        if benchDepthClamped > 0 { args += ["-d", String(benchDepthClamped)] }
        if !effectiveDynamicMoe && ncmoe > 0 { args += ["-ncmoe", String(ncmoe)] }
        if let ub = effectiveUbatch { args += ["-ub", String(ub), "-b", String(ub)] }
        if effectiveDynamicMoe { args += ["-ot", Self.dynamicMoeTensorOverride] }
        if cacheTypeK != "f16" { args += ["-ctk", cacheTypeK] }
        if cacheTypeV != "f16" { args += ["-ctv", cacheTypeV] }
        if kvNeedsFlashAttention || flashAttn == "on" {
            args += ["-fa", "1"]
        } else if effectiveFaAmd {
            args += ["-fa", "auto"]
        }
        if isSplitting { args += ["--split-mode", effectiveSplitMode] }
        if let counts = layerBalancedTensorSplit {
            args += ["--tensor-split", counts.map(String.init).joined(separator: ",")]
        }
        return args
    }

    /// Workload sizes kept within what llama-bench accepts and a Mac can finish.
    var benchPPClamped: Int { min(max(benchPP, 16), 32768) }
    var benchTGClamped: Int { min(max(benchTG, 16), 8192) }
    var benchDepthClamped: Int { min(max(benchDepth, 0), 32768) }

    /// Human-readable name of the GPU a run actually used, for the benchmark
    /// record. Resolves the macOS-picked default to its real device name.
    var gpuLabel: String {
        let gpus = ServerController.availableGPUs()
        // Only the non-default mode is named, so old records stay comparable.
        let mode = effectiveSplitMode == "tensor" ? " · tensor" : ""
        if gpuList.count >= 2 { return "Split · \(gpuList.count) GPUs\(mode)" }
        if multiGPU {
            let discrete = gpus.filter { !$0.isIntegrated }.count
            let limit = discrete > 0 ? discrete : gpus.count
            let n = multiGPUCount > 0 ? min(multiGPUCount, limit) : limit
            return "Split · \(max(2, n)) GPUs\(mode)"
        }
        if gpuIndex >= 0 { return gpus.first { $0.index == gpuIndex }?.name ?? "GPU \(gpuIndex)" }
        return MTLCreateSystemDefaultDevice()?.name ?? "default"
    }

    /// Web chat UI bundled with the app (served via llama-server --path). The rebranded
    /// llama.cpp UI wins; the basic console stays as fallback.
    static var chatUIPath: String? {
        guard let res = Bundle.main.resourceURL else { return nil }
        for name in ["web-ui", "test-ui"] {
            let dir = res.appendingPathComponent(name).path
            if FileManager.default.fileExists(atPath: dir + "/index.html") { return dir }
        }
        return nil
    }

    var environment: [String: String] {
        var env = ProcessInfo.processInfo.environment
        // Finder-launched apps inherit a minimal PATH. Video input is decoded by
        // llama.cpp through ffmpeg/ffprobe, so expose their standard macOS locations.
        env["PATH"] = VideoRuntimeAvailability.augmentedPath(env["PATH"])
        // The engine logs this, so a log pasted from a bug report identifies the
        // app build even when the engine binary is older than the app.
        env["TOSH_APP_VERSION"] = AppInfo.version
        env["GGML_METAL_VRAM_RESERVE_MB"] = String(vramReserveMB)
        // Physical GPU selection (consumed by the patched Metal backend, which maps
        // these to MTLCopyAllDevices() — the same order as the app's GPU picker).
        let gpus = ServerController.availableGPUs()
        if gpuList.count >= 2 {
            // Split across exactly these GPUs; slot i maps to the i-th listed index.
            env["GGML_METAL_DEVICE_LIST"] = gpuList.map(String.init).joined(separator: ",")
        } else if multiGPU {
            // Split across N GPUs. Fewer than all lets the user trade prompt speed
            // (more GPUs) for generation speed (fewer, less cross-card sync). 0 = all.
            // Integrated iGPUs don't count: the engine skips them when mapping slots.
            let discrete = gpus.filter { !$0.isIntegrated }.count
            let limit = discrete > 0 ? discrete : gpus.count
            let n = multiGPUCount > 0 ? min(multiGPUCount, limit) : limit
            env["GGML_METAL_DEVICES"] = String(max(2, n))
        } else if gpuIndex >= 0 {
            // Pin the engine to one physical GPU by index.
            env["GGML_METAL_DEVICE_INDEX"] = String(gpuIndex)
        }
        // The backend forces shared buffers for external GPUs, which streams weights
        // over Thunderbolt every op. Auto-override when a selected card is external;
        // the manual toggle covers the case where macOS picked the GPU.
        let splittingList = gpuList.count >= 2
        let selectedExternal = !multiGPU && !splittingList && gpuIndex >= 0
            && gpus.first { $0.index == gpuIndex }?.isExternal == true
        let anyExternalInSplit = (multiGPU && !splittingList && gpus.contains { $0.isExternal })
            || (splittingList && gpus.contains { gpuList.contains($0.index) && $0.isExternal })
        if forcePrivateBuffers || selectedExternal || anyExternalInSplit {
            env["GGML_METAL_SHARED_BUFFERS_DISABLE"] = "1"
        }
        if effectiveFaAmd { env["TOSH_FA_AMD"] = "1" }
        // Only a tensor split reduces across cards, which is the copy the bridge speeds up.
        // A layer split has nothing for it to carry.
        if mgpuPeer && isSplitting && effectiveSplitMode == "tensor" { env["TOSH_MGPU_PEER"] = "1" }
        if mgpuEvents && isSplitting { env["TOSH_MGPU_EVENTS"] = "1" }
        // Groups only mean anything inside a tensor split, and only when they are
        // smaller than the split itself and divide it evenly.
        if effectiveSplitMode == "tensor", let g = effectiveSplitGroupSize {
            env["TOSH_MGPU_TENSOR_GROUP"] = String(g)
        }
        // A DFlash draft runs its selector over the target's logits, so a head split by
        // vocabulary leaves no card holding a whole row and the engine aborts. Keep the head
        // on every card for that pairing; it costs its size per card and nothing otherwise.
        if isSplitting, effectiveSplitMode == "tensor", !effectiveDynamicMoe,
           dflashSelection(modelPath: modelPath, ncmoe: ncmoe) != nil {
            env["TOSH_MIRROR_OUTPUT_HEAD"] = "1"
        }
        // Router mode has no single ncmoe (it's per-model, in the INI); the envs are
        // no-ops for dense models anyway.
        if effectiveDynamicMoe {
            env["TOSH_MOE_MODE"] = "cache"
            if dynamicMoePolicy == "auto" { env["TOSH_MOE_AUTO"] = "1" }
            env["TOSH_MOE_SLOTS"] = String(effectiveDynamicMoeSlots)
            env["TOSH_MOE_CPU_BANK"] = "1"
            env["GGML_SCHED_PREFETCH_EXPERTS"] = String(effectiveDynamicMoePrefetch)
            env["GGML_METAL_NCB"] = "8"
            if dynamicMoeExecutionRoute == .split {
                let profile = dynamicMoeOptimizationProfile
                env["TOSH_MOE_SPLIT_BANK"] = "1"
                env["TOSH_MOE_SLOTS"] = String(dynamicMoeSlotSplit.fixed)
                env["TOSH_MOE_SPLIT_RING"] = String(dynamicMoeSlotSplit.ring)
                env["TOSH_MOE_BOUNDED_STAGE"] = "1"
                env["TOSH_MOE_BOUNDED_STAGE_FORCE"] = "1"
                env["TOSH_MOE_DOUBLE_BUFFER"] = "1"
                if let mapPath = profile?.hotMapPath,
                   FileManager.default.fileExists(atPath: mapPath) {
                    env["TOSH_MOE_HOT_MAP"] = mapPath
                    env["TOSH_MOE_HOT_MAP_OUT"] = mapPath
                    if let experts = dynamicMoeModelInfo?.expertCount {
                        env["TOSH_MOE_HOT_MAP_K"] = String(experts)
                    }
                }
            }
        } else if prefetchExperts && (ncmoe > 0 || routerMode) {
            // At/above the measured cliff the prefetch overlap collapses and stalls the
            // GPU, so stay below it. Router mode has no single ncmoe to compare.
            let cliff = Self.recalledPrefetchCliff(forModel: modelPath)
            if routerMode || cliff == nil || ncmoe < cliff! {
                env["GGML_SCHED_PREFETCH_EXPERTS"] = "1"
                env["GGML_CPU_NO_REPACK"] = "1"
            }
        }
        // KEY=VALUE tokens from Extra arguments become env vars (e.g. the GCN/Vega
        // wave64 safe-mode flag). Applied last so the user can override the above.
        for (k, v) in extraArgTokens.env { env[k] = v }
        // K is structural: a value larger than the GGUF's expert dimension makes
        // the backend skip cache creation. Keep the model-derived/clamped value even
        // when an old Extra arguments recipe still contains TOSH_MOE_SLOTS.
        if effectiveDynamicMoe {
            env["TOSH_MOE_SLOTS"] = String(dynamicMoeExecutionRoute == .split
                ? dynamicMoeSlotSplit.fixed : effectiveDynamicMoeSlots)
        }
        return env.compactMapValues { $0 }
    }

    /// Metal concurrency is stable and faster on Apple Silicon, but it corrupts
    /// output on discrete AMD GPUs (Intel Macs / Hackintosh), where it must stay off.
    static let isAppleSilicon: Bool = {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        sysctlbyname("hw.optional.arm64", &value, &size, nil, 0)
        return value == 1
    }()



    /// Default engine: the one bundled with the app (portable); falls back to the dev checkout.
    static var defaultBinary: String {
        // Cached: view bodies reach this through fromDefaults(), and hitting the
        // filesystem on every render shows up as lag.
        if let resolved = cachedDefaultBinary { return resolved }
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("bin/llama-server").path,
           FileManager.default.fileExists(atPath: bundled) {
            cachedDefaultBinary = bundled
            return bundled
        }
        // patched master build: supports recent architectures (qwen35moe / Qwen 3.6)
        let fallback = NSString(string: "~/dev/repositorios/llama.cpp/build/bin/llama-server").expandingTildeInPath
        cachedDefaultBinary = fallback
        return fallback
    }

    nonisolated(unsafe) private static var cachedDefaultBinary: String?

    /// The engine to launch. "bundled" resolves against the running bundle, so two
    /// installs sharing this defaults domain each use their own binary.
    static var engineKind: String {
        let d = UserDefaults.standard
        if let kind = d.string(forKey: SettingsKeys.engineKind) { return kind }
        // Migration: the kind used to be implied by a stored absolute path. A path
        // into any bundle (including the retired turbo engine) becomes "bundled".
        let legacy = d.string(forKey: SettingsKeys.serverBinary) ?? ""
        let kind = legacy.isEmpty || legacy.contains("/Contents/Resources/bin") ? "bundled" : "custom"
        d.set(kind, forKey: SettingsKeys.engineKind)
        if kind == "bundled" { d.removeObject(forKey: SettingsKeys.serverBinary) }
        return kind
    }

    /// Path of the engine to launch, resolved from the kind at read time.
    static func resolvedBinary() -> String {
        if engineKind == "custom" {
            let custom = UserDefaults.standard.string(forKey: SettingsKeys.serverBinary) ?? ""
            if !custom.isEmpty { return custom }
        }
        return defaultBinary
    }

    /// gpuList is persisted as a comma-separated string so @AppStorage can bind it.
    static func gpuList(fromCSV csv: String?) -> [Int] {
        (csv ?? "").split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    /// Reads persisted settings (same keys as the views' @AppStorage).
    static func fromDefaults() -> ServerSettings {
        let d = UserDefaults.standard
        func int(_ key: String, _ def: Int) -> Int { d.object(forKey: key) == nil ? def : d.integer(forKey: key) }
        func bool(_ key: String, _ def: Bool) -> Bool { d.object(forKey: key) == nil ? def : d.bool(forKey: key) }
        return ServerSettings(
            serverBinary: resolvedBinary(),
            modelPath: d.string(forKey: SettingsKeys.modelPath) ?? "",
            port: int(SettingsKeys.port, 8080),
            ngl: int(SettingsKeys.ngl, 99),
            ncmoe: int(SettingsKeys.ncmoe, 0),
            ctx: int(SettingsKeys.ctx, 16384),
            contextAutomatic: bool(SettingsKeys.contextAutomatic, false),
            threads: int(SettingsKeys.threads, 6),
            flashAttn: d.string(forKey: SettingsKeys.flashAttn) ?? "auto",
            noMmap: bool(SettingsKeys.noMmap, true),
            jinja: bool(SettingsKeys.jinja, true),
            vramReserveMB: int(SettingsKeys.vramReserve, 1024),
            gpuIndex: int(SettingsKeys.gpuIndex, -1),
            gpuList: gpuList(fromCSV: d.string(forKey: SettingsKeys.gpuList)),
            extraArgs: d.string(forKey: SettingsKeys.extraArgs) ?? "",
            embeddings: bool(SettingsKeys.embeddings, false),
            agentToolsEnabled: bool(SettingsKeys.agentToolsEnabled, false),
            uiMcpProxy: bool(SettingsKeys.uiMcpProxy, false),
            toolsRuntime: (d.string(forKey: SettingsKeys.toolsRuntime) ?? "")
                .trimmingCharacters(in: .whitespaces),
            cacheTypeK: Self.sanitizedKV(d.string(forKey: SettingsKeys.cacheTypeK), default: "f16"),
            cacheTypeV: Self.sanitizedKV(d.string(forKey: SettingsKeys.cacheTypeV), default: "f16"),
            mlock: bool(SettingsKeys.mlock, false),
            cacheRAM: int(SettingsKeys.cacheRAM, 2048),
            reasoningInline: bool(SettingsKeys.reasoningInline, false),
            parallelSlots: int(SettingsKeys.parallelSlots, 1),
            apiKeyEnabled: bool(SettingsKeys.apiKeyEnabled, false),
            localNetworkDiscovery: bool(SettingsKeys.localNetworkDiscovery, false),
            specMTP: bool(SettingsKeys.specMTP, false),
            faAmd: bool(SettingsKeys.faAmd, defaultFaAmd),
            prefetchExperts: bool(SettingsKeys.prefetchExperts, true),
            ubatch: int(SettingsKeys.ubatch, 0),
            dynamicMoe: bool(SettingsKeys.dynamicMoe, false),
            dynamicMoeSlots: int(SettingsKeys.dynamicMoeSlots, 8),
            dynamicMoePrefetch: int(SettingsKeys.dynamicMoePrefetch, 4),
            dynamicMoePolicy: d.string(forKey: SettingsKeys.dynamicMoePolicy) ?? "cache",
            routerMode: bool(SettingsKeys.routerMode, false),
            routerModelsMax: int(SettingsKeys.routerModelsMax, 1),
            persistCache: bool(SettingsKeys.persistCache, false),
            multiGPU: bool(SettingsKeys.multiGPU, false),
            multiGPUCount: int(SettingsKeys.multiGPUCount, 0),
            splitMode: d.string(forKey: SettingsKeys.splitMode) ?? "layer",
            splitGroupSize: d.integer(forKey: SettingsKeys.splitGroupSize),
            mgpuEvents: bool(SettingsKeys.mgpuEvents, true),
            mgpuPeer: bool(SettingsKeys.mgpuPeer, true),
            forcePrivateBuffers: bool(SettingsKeys.forcePrivateBuffers, false),
            cacheReuse: bool(SettingsKeys.cacheReuse, true),
            loadVision: bool(SettingsKeys.loadVision, true),
            imageMaxTokens: int(SettingsKeys.imageMaxTokens, 0),
            benchPP: int(SettingsKeys.benchPP, 512),
            benchTG: int(SettingsKeys.benchTG, 128),
            benchDepth: int(SettingsKeys.benchDepth, 0))
    }

    /// True when the model's attention head dim exceeds 256 (Gemma 4's global layers
    /// use key_length 512).
    nonisolated static func modelHasBigHeadDim(at path: String) -> Bool {
        (GGUFMetadataCache.metadata(at: path)?.uint32(forSuffix: "attention.key_length") ?? 0) > 256
    }

    nonisolated static func modelSupportsTurboKV(at path: String) -> Bool {
        guard let metadata = GGUFMetadataCache.metadata(at: path) else { return false }
        let keyLength = metadata.uint32(forSuffix: "attention.key_length") ??
            metadata.uint32(forSuffix: "attention.key_length_mla") ?? {
            guard let embedding = metadata.uint32(forSuffix: "embedding_length"),
                  let heads = metadata.uint32(forSuffix: "attention.head_count"), heads > 0,
                  embedding % heads == 0 else { return nil }
            return embedding / heads
        }()
        let valueLength = metadata.uint32(forSuffix: "attention.value_length") ??
            metadata.uint32(forSuffix: "attention.value_length_mla") ?? keyLength

        guard let keyLength, let valueLength, keyLength > 0, valueLength > 0 else { return false }
        let sharedPadded = ((max(keyLength, valueLength) + 127) / 128) * 128
        return [128, 256, 384, 512, 640].contains(sharedPadded)
    }

    nonisolated static func modelUsesMLA(at path: String) -> Bool {
        guard let metadata = GGUFMetadataCache.metadata(at: path) else { return false }
        return metadata.uint32(forSuffix: "attention.key_length_mla") != nil ||
            metadata.uint32(forSuffix: "attention.value_length_mla") != nil
    }

    var kvNeedsFlashAttention: Bool {
        cacheTypeK != "f16" || cacheTypeV != "f16"
    }

    /// The user's AMD Flash-Attention choice. Quantized KV may still force
    /// normal Flash Attention when this is off.
    var effectiveFaAmd: Bool {
        faAmd
    }

    /// Router presets load models independently and cannot carry the per-tensor
    /// override required by this prototype, so keep the experiment fixed-model only.
    var dynamicMoeUIUnlocked: Bool { extraArgTokens.env["TOSH_MOE_UI"] == "1" }
    var dynamicMoeGPU: GPUDevice? {
        let selected = ServerController.availableGPUs().filter { selectedGPUIndices.contains($0.index) }
        return selected.max { $0.vramMB < $1.vramMB }
    }
    var dynamicMoeOptimizationProfile: DynamicMoeOptimizationProfile? {
        DynamicMoeProfileStore.load(modelPath: modelPath, gpu: dynamicMoeGPU)
    }
    var dynamicMoeExecutionRoute: DynamicMoeExecutionRoute {
        if dynamicMoePolicy == "auto", let profile = dynamicMoeOptimizationProfile {
            return profile.route
        }
        guard let size = GGUFFile.totalSize(at: modelPath), let gpu = dynamicMoeGPU else {
            return .direct
        }
        return Self.dynamicMoeHostBankFitsDirectMetal(
            modelBytes: size, gpuVRAMMB: gpu.vramMB,
            physicalRAMBytes: ProcessInfo.processInfo.physicalMemory) ? .direct : .split
    }
    var dynamicMoeAutoRoute: DynamicMoeAutoRoute {
        guard !modelPath.isEmpty, let size = GGUFFile.totalSize(at: modelPath) else {
            return .normalMissingModel
        }
        let gpu = dynamicMoeGPU
        if dynamicMoeOptimizationProfile != nil, !isSplitting, !routerMode {
            return .cache
        }
        let base = Self.resolveDynamicMoeAuto(
            isMoE: Self.modelIsMoE(at: modelPath),
            modelBytes: size,
            gpuVRAMMB: gpu?.vramMB ?? 0,
            reserveMB: vramReserveMB,
            physicalRAMBytes: ProcessInfo.processInfo.physicalMemory,
            hasDiscreteGPU: gpu?.isIntegrated == false,
            splitOrRouter: isSplitting || routerMode)
        guard base == .cache else { return base }
        guard let info = dynamicMoeModelInfo else { return .normalMissingMetadata }
        guard info.activeExpertCount < info.expertCount else { return .normalNoCacheBenefit }
        guard dynamicMoeSlotPlan(prefetch: 4) != nil else { return .normalInsufficientVRAM }
        return .cache
    }
    /// MoE only: measured up to twice the prefill with experts on CPU (one expert
    /// upload per batch instead of per 512 tokens) and ~10% with the model whole on
    /// the card. A dense model measures slower with it, so it stays gated.
    /// Dynamic MoE gains the most: on a 35B A3B its prefill went 317 to 1088 t/s from
    /// 512 to 2048, against 455 to 862 for the same model on plain expert offload.
    var effectiveUbatch: Int? {
        guard ubatch > 0 else { return nil }
        guard routerMode || Self.modelIsMoE(at: modelPath) else { return nil }
        return ubatch
    }
    var effectiveDynamicMoe: Bool {
        guard dynamicMoe && dynamicMoeUIUnlocked,
              Self.modelIsMoE(at: modelPath), dynamicMoeModelInfo != nil else { return false }
        if dynamicMoePolicy == "auto" { return dynamicMoeAutoRoute == .cache }
        return !routerMode && !isSplitting
    }
    var effectiveDynamicMoeSlots: Int {
        if dynamicMoePolicy == "auto" {
            if let profile = dynamicMoeOptimizationProfile { return profile.slots }
            return dynamicMoeSlotPlan(prefetch: 4)?.automaticSlots ?? 0
        }
        if let info = dynamicMoeModelInfo {
            let minimum = min(max(info.activeExpertCount, 1), info.expertCount)
            return min(max(dynamicMoeSlots, minimum), info.expertCount)
        }
        return min(max(dynamicMoeSlots, 1), 256)
    }
    /// Splits the affordable slot budget between the permanently resident experts and the
    /// LRU ring. This is a trade, not a free win: measured on a 35B-A3B at the same VRAM,
    /// moving the budget into the ring is worth about +64% of generation and costs close to
    /// half the prefill, so it suits chat rather than long prompts. It flattens out once only
    /// the active experts stay fixed.
    var dynamicMoeSlotSplit: (fixed: Int, ring: Int) {
        let budget = effectiveDynamicMoeSlots
        guard let info = dynamicMoeModelInfo, budget > 0 else { return (budget, 0) }
        if let ring = dynamicMoeOptimizationProfile?.ringSlots {
            return (max(1, budget - ring), ring)
        }
        // The experts left out of the slots are wrapped as one host resource and read by the
        // fetch kernel, so they have to stay under what the card can keep resident: past it the
        // reservation is refused outright, and just below it generation collapses.
        var floor = 1
        if let size = GGUFFile.totalSize(at: modelPath), let gpu = dynamicMoeGPU {
            let mib = UInt64(1024 * 1024)
            let shared = min(size, UInt64(Double(UInt64(1024) * mib) * 1.3))
            let expertBytes = size > shared ? size - shared : 0
            let ceiling = UInt64(max(0, gpu.vramMB)) * mib * 3 / 4
            if expertBytes > ceiling {
                floor = Int((Double(info.expertCount) *
                    (1.0 - Double(ceiling)/Double(expertBytes))).rounded(.up))
            }
        }
        var fixed = min(max(max(info.activeExpertCount, floor), 1), budget)
        // A batch is served on the device only when its routing fits the ring; past it the whole
        // logical bank is reassembled layer by layer. The unit is ids, not experts, so a decode
        // step of a few tokens already needs several times the active count. Measured on a 35B
        // A3B: a ring of 8 dropped generation to 17.6 t/s against 49.7, at the same prefill.
        // The cold bank ceiling still wins, since `floor` is what makes the model start at all.
        let ringWanted = info.activeExpertCount * 8
        if budget - fixed < ringWanted {
            fixed = max(min(fixed, budget - ringWanted), max(floor, 1))
        }
        return (fixed, max(0, budget - fixed))
    }

    var effectiveDynamicMoePrefetch: Int {
        if dynamicMoePolicy == "auto", let profile = dynamicMoeOptimizationProfile {
            return profile.prefetch
        }
        return dynamicMoePolicy == "auto" ? 4 : min(max(dynamicMoePrefetch, 0), 16)
    }

    var dynamicMoeModelInfo: DynamicMoeModelInfo? {
        let layers = Int(Self.ggufUInt32("block_count", at: modelPath) ?? 0)
        let experts = Int(Self.ggufUInt32("expert_count", at: modelPath) ?? 0)
        let active = Int(Self.ggufUInt32("expert_used_count", at: modelPath) ?? 0)
        guard layers > 0, experts > 0, active > 0, active <= experts else { return nil }
        return DynamicMoeModelInfo(layerCount: layers, expertCount: experts,
                                   activeExpertCount: active)
    }

    func dynamicMoeSlotPlan(prefetch: Int? = nil) -> DynamicMoeSlotPlan? {
        guard let info = dynamicMoeModelInfo,
              let size = GGUFFile.totalSize(at: modelPath) else {
            return nil
        }
        let selected = ServerController.availableGPUs().filter { selectedGPUIndices.contains($0.index) }
        guard let gpu = selected.max(by: { $0.vramMB < $1.vramMB }), !gpu.isIntegrated else { return nil }
        return Self.resolveDynamicMoeSlots(
            modelBytes: size, model: info, gpuVRAMMB: gpu.vramMB,
            reserveMB: vramReserveMB, prefetch: prefetch ?? effectiveDynamicMoePrefetch,
            ubatch: effectiveUbatch ?? 512)
    }

    static func resolveDynamicMoeSlots(
        modelBytes: UInt64,
        model: DynamicMoeModelInfo,
        gpuVRAMMB: Int,
        reserveMB: Int,
        prefetch: Int,
        ubatch: Int = 512
    ) -> DynamicMoeSlotPlan? {
        guard modelBytes > 0, model.layerCount > 0, model.expertCount > 0,
              model.activeExpertCount > 0, model.activeExpertCount <= model.expertCount,
              gpuVRAMMB > reserveMB else { return nil }

        let mib = UInt64(1024 * 1024)
        let gib = UInt64(1024) * mib
        // The 1.3 GiB shared/non-MoE estimate is the same conservative split used by
        // ToshLLM's ncmoe planner. The remainder is the quantized expert pool.
        let sharedBytes = min(modelBytes, UInt64(Double(gib) * 1.3))
        let expertBytes = modelBytes - sharedBytes
        guard expertBytes > 0 else { return nil }
        let bytesPerSlot = max(UInt64(1),
            UInt64(ceil(Double(expertBytes) / Double(model.expertCount))))

        // One full widest bank is staging; each prefetch slot can hold another. A
        // gate_up bank is approximately 2/3 of one layer's three expert matrices.
        let widestBankBytes = UInt64(ceil(
            Double(expertBytes) / Double(model.layerCount) * (2.0 / 3.0)))
        let transferBuffers = widestBankBytes * UInt64(max(0, min(prefetch, 16)) + 1)
        // A wider prefill micro-batch buys a lot of prompt speed but its compute buffers
        // come out of the same VRAM as the slots: measured about 640 MiB per extra 512
        // tokens on a 35B A3B.
        let extraUbatch = max(0, ubatch - 512)/512
        let runtimeBytes = UInt64(512) * mib + UInt64(extraUbatch) * UInt64(640) * mib
        let fixedBytes = sharedBytes + runtimeBytes + transferBuffers
        let availableBytes = UInt64(max(0, gpuVRAMMB - reserveMB)) * mib
        let budgetSlots = availableBytes > fixedBytes
            ? Int((availableBytes - fixedBytes) / bytesPerSlot) : 0
        let recommendedMaximum = min(model.expertCount, max(0, budgetSlots))
        guard recommendedMaximum >= model.activeExpertCount else { return nil }

        return DynamicMoeSlotPlan(
            model: model,
            minimumSlots: model.activeExpertCount,
            maximumSlots: model.expertCount,
            recommendedMaximumSlots: recommendedMaximum,
            automaticSlots: model.activeExpertCount,
            estimatedBytesPerSlot: bytesPerSlot,
            estimatedFixedVRAMBytes: fixedBytes)
    }

    static func resolveDynamicMoeAuto(
        isMoE: Bool,
        modelBytes: UInt64,
        gpuVRAMMB: Int,
        reserveMB: Int,
        physicalRAMBytes: UInt64,
        hasDiscreteGPU: Bool,
        splitOrRouter: Bool
    ) -> DynamicMoeAutoRoute {
        guard !splitOrRouter else { return .normalSplitOrRouter }
        guard isMoE else { return .normalDense }
        guard modelBytes > 0 else { return .normalMissingModel }
        guard hasDiscreteGPU, gpuVRAMMB > 0 else { return .normalUnsupportedGPU }

        // Besides the user's reserve, leave 512 MiB for compute/KV allocations. A GGUF
        // that fits below this line gains nothing from duplicating its expert bank in RAM.
        let mib = UInt64(1024 * 1024)
        let usableVRAM = UInt64(max(0, gpuVRAMMB - reserveMB - 512)) * mib
        guard modelBytes > usableVRAM else { return .normalFitsVRAM }

        // Dynamic MoE pins the quantized expert bank. Keep the model plus 25% and 4 GiB
        // for the OS/app/KV; if total RAM cannot provide that, normal ncmoe is safer.
        let ramHeadroom = max(modelBytes / 4, UInt64(4) * 1024 * 1024 * 1024)
        guard physicalRAMBytes >= modelBytes + ramHeadroom else { return .normalInsufficientRAM }
        return .cache
    }

    /// The current decode kernel directly binds the complete host expert pool to Metal. On a
    /// discrete GPU, measured banks larger than the device working set can stall the driver even
    /// when system RAM and swap are healthy. Auto stays on the validated side of that boundary;
    /// private Manual mode remains available for developing a bounded staging implementation.
    /// Direct mode wraps the whole expert bank as host memory, and Metal wires those pages, so
    /// what bounds it is system RAM rather than the card: the slots the graph reads are what
    /// lives in VRAM. Measured on a 32 GiB machine, a 9.6 GiB bank runs fine while 12.7 and
    /// 16.9 GiB starve the compositor, because wired pages cannot be evicted and swap is
    /// capped. A third of physical RAM is the line those three points draw.
    static func dynamicMoeHostBankFitsDirectMetal(modelBytes: UInt64, gpuVRAMMB: Int,
                                                  physicalRAMBytes: UInt64) -> Bool {
        guard modelBytes > 0, gpuVRAMMB > 0, physicalRAMBytes > 0 else { return false }
        let mib = UInt64(1024 * 1024)
        let sharedBytes = min(modelBytes, UInt64(Double(UInt64(1024) * mib) * 1.3))
        let estimatedExpertBytes = modelBytes - sharedBytes
        return estimatedExpertBytes <= physicalRAMBytes / 3
    }

    /// Resolves DFlash against the same physical GPU selection and memory reserve
    /// that will be passed to the engine. A nil plan means metadata or hardware is
    /// unavailable; a plan with nil layers means DFlash cannot fit safely.
    var selectedGPUIndices: Set<Int> {
        let gpus = ServerController.availableGPUs()
        if gpuList.count >= 2 { return Set(gpuList) }
        if multiGPU {
            let eligible = gpus.filter { !$0.isIntegrated }
            let pool = eligible.isEmpty ? gpus : eligible
            let count = multiGPUCount > 0 ? min(multiGPUCount, pool.count) : pool.count
            return Set(pool.prefix(count).map(\.index))
        }
        if gpuIndex >= 0 { return [gpuIndex] }
        if let gpu = gpus.filter({ !$0.isIntegrated }).max(by: { $0.vramMB < $1.vramMB })
            ?? gpus.max(by: { $0.vramMB < $1.vramMB }) {
            return [gpu.index]
        }
        return []
    }

    func dflashMemoryPlan(modelPath: String, draftPath: String, ncmoe: Int,
                          honorReserve: Bool = true) -> DflashMemoryPlan? {
        let gpus = ServerController.availableGPUs()
        let indices = selectedGPUIndices
        let selected = gpus.filter { indices.contains($0.index) }
        guard !selected.isEmpty else { return nil }

        guard let baseBytes = (try? FileManager.default.attributesOfItem(atPath: modelPath)[.size] as? NSNumber)?.doubleValue,
              let draftBytes = (try? FileManager.default.attributesOfItem(atPath: draftPath)[.size] as? NSNumber)?.doubleValue else {
            return nil
        }
        let baseLayers = Int(Self.ggufUInt32("block_count", at: modelPath) ?? 0)
        let repeatingDraftLayers = Int(Self.ggufUInt32("block_count", at: draftPath) ?? 0)
        guard baseLayers > 0, repeatingDraftLayers > 0 else { return nil }

        let headsKV = Double(Self.ggufUInt32("attention.head_count_kv", at: draftPath) ?? 0)
        let keyLength = Double(Self.ggufUInt32("attention.key_length", at: draftPath) ?? 0)
        let valueLength = Double(Self.ggufUInt32("attention.value_length", at: draftPath) ?? 0)
        guard headsKV > 0, keyLength > 0, valueLength > 0 else { return nil }
        // q8_0 stores 32 values plus a 16-bit scale in each 34-byte block.
        let q8BytesPerElement = 34.0 / 32.0
        let draftKVBytesPerToken = Double(repeatingDraftLayers) * headsKV
            * (keyLength + valueLength) * q8BytesPerElement
        let mainKVScale = (Estimator.kvTypeScale(cacheTypeK) + Estimator.kvTypeScale(cacheTypeV)) / 2
        let vramGB = Double(selected.reduce(0) { $0 + $1.vramMB }) / 1024
        let reserveGB = honorReserve ? Double(vramReserveMB * max(1, selected.count)) / 1024 : 0
        return DflashMemoryPlanner.plan(
            vramGB: vramGB, reserveGB: reserveGB,
            baseFileGB: baseBytes / 1_073_741_824, baseLayers: baseLayers, ncmoe: ncmoe,
            ctx: ctx, mainKVScale: mainKVScale,
            draftFileGB: draftBytes / 1_073_741_824, draftLayers: repeatingDraftLayers + 1,
            draftKVBytesPerToken: draftKVBytesPerToken)
    }

    private func dflashSelection(modelPath: String, ncmoe: Int) -> (draft: String, ngld: Int)? {
        guard let draft = Self.dflashDraftPath(forModel: modelPath) else { return nil }
        switch Self.dflashMode(forModel: modelPath) {
        case .off:
            return nil
        case .auto:
            guard DflashPolicy.autoEligible(isMoE: Self.modelIsMoE(at: modelPath), ncmoe: ncmoe),
                  let layers = dflashMemoryPlan(modelPath: modelPath, draftPath: draft,
                                                ncmoe: ncmoe)?.gpuLayers else { return nil }
            return (draft, layers)
        case .forced:
            let layers = dflashMemoryPlan(modelPath: modelPath, draftPath: draft,
                                          ncmoe: ncmoe, honorReserve: false)?.gpuLayers ?? 0
            return (draft, layers)
        }
    }

    var dflashPlanSummary: String {
        guard let draft = Self.dflashDraftPath(forModel: modelPath) else { return "not installed" }
        let mode = Self.dflashMode(forModel: modelPath)
        guard mode != .off else { return "off by user" }
        guard let plan = dflashMemoryPlan(modelPath: modelPath, draftPath: draft, ncmoe: ncmoe,
                                          honorReserve: mode == .auto) else {
            return "off (metadata or GPU capacity unavailable)"
        }
        let need = (plan.estimatedVRAMGB * 100).rounded() / 100
        let budget = (plan.budgetGB * 100).rounded() / 100
        guard let layers = plan.gpuLayers else {
            return "off by memory planner (need \(need) GiB, budget \(budget) GiB)"
        }
        return "\(mode.rawValue), ngld=\(layers) (estimated \(need) / \(budget) GiB budget)"
    }

    var benchmarkFlashAttentionRoute: String {
        if effectiveFaAmd { return "amd-gpu" }
        if flashAttn == "on" || kvNeedsFlashAttention { return "standard-cpu" }
        if flashAttn == "auto" { return "standard-auto" }
        return "off"
    }

    var benchmarkFlashAttentionLabel: String {
        switch benchmarkFlashAttentionRoute {
        case "amd-gpu": return "AMD Flash Attention (GPU)"
        case "standard-cpu": return "standard Flash Attention (CPU)"
        case "standard-auto": return "standard Flash Attention (auto)"
        default: return "off"
        }
    }

    /// Whether a GGUF ships the MTP (multi-token prediction) head. Quantizers often
    /// strip it but leave the key at 0, and the server aborts on the missing tensors,
    /// so trust the value and fall back to the tensor names.
    nonisolated static func modelHasMTP(at path: String) -> Bool {
        if let layers = ggufUInt32("nextn_predict_layers", at: path) {
            return layers >= 1
        }
        return GGUFMetadataCache.hasNextNTensor(at: path)
    }

    /// Finds a separate MTP assistant stored beside its base model.
    nonisolated static func mtpDraftPath(forModel modelPath: String) -> String? {
        guard !modelPath.isEmpty, !GGUFFile.isDraft(modelPath) else { return nil }
        let modelURL = URL(fileURLWithPath: modelPath)
        let modelStem = modelURL.deletingPathExtension().lastPathComponent.lowercased()
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: modelURL.deletingLastPathComponent(), includingPropertiesForKeys: nil
        ) else { return nil }

        let candidates = files.filter { file in
            guard file.pathExtension.lowercased() == "gguf", GGUFFile.isDraft(file.path) else { return false }
            let name = file.lastPathComponent.lowercased()
            let target: String
            if name.hasPrefix("mtp-") {
                target = String(name.dropFirst(4).dropLast(5))
            } else if name.hasSuffix(".mtp.gguf") {
                target = String(name.dropLast(9))
            } else {
                return false
            }
            guard modelStem == target || modelStem.hasPrefix(target + "-") || modelStem.hasPrefix(target + ".") else {
                return false
            }
            if let baseArch = ggufString("general.architecture", at: modelPath),
               let draftArch = ggufString("general.architecture", at: file.path) {
                return draftArch == baseArch + "_assistant"
            }
            return true
        }
        return candidates.sorted { $0.lastPathComponent.count > $1.lastPathComponent.count }.first?.path
    }

    nonisolated static func modelUsesMTP(at path: String) -> Bool {
        modelHasMTP(at: path) || mtpDraftPath(forModel: path) != nil
    }

    nonisolated static func mtpEnabled(forModel path: String) -> Bool {
        !(UserDefaults.standard.stringArray(forKey: SettingsKeys.mtpDisabledModels) ?? []).contains(path)
    }

    nonisolated static func setMTPEnabled(_ enabled: Bool, forModel path: String) {
        var disabled = Set(UserDefaults.standard.stringArray(forKey: SettingsKeys.mtpDisabledModels) ?? [])
        if enabled { disabled.remove(path) } else { disabled.insert(path) }
        UserDefaults.standard.set(Array(disabled), forKey: SettingsKeys.mtpDisabledModels)
    }

    /// True when the model's weights use a TurboQuant type (ggml_type 45/46). Read from
    /// the tensor types: these GGUFs carry no usable `general.file_type`. False on an
    /// unreadable header, so a model we can't parse is never blocked.
    nonisolated static func modelIsTurboQuantWeights(at path: String) -> Bool {
        GGUFMetadataCache.tensorFlags(at: path).hasTurboQuantTensor
    }

    /// First uint32 value for an exact GGUF metadata key or architecture suffix.
    nonisolated static func ggufUInt32(_ keySuffix: String, at path: String) -> UInt32? {
        GGUFMetadataCache.metadata(at: path)?.uint32(forSuffix: keySuffix)
    }

    /// String value for an exact GGUF metadata key.
    nonisolated static func ggufString(_ key: String, at path: String) -> String? {
        GGUFMetadataCache.metadata(at: path)?.string(for: key)
    }

    /// GPUs a tensor split can divide this model's experts among. Experts whose scales
    /// sit in their own tensor span two buffers, which the meta backend cannot divide.
    nonisolated static func tensorSplitLimit(forModel path: String) -> Int? {
        guard !path.isEmpty else { return nil }
        return GGUFMetadataCache.tensorFlags(at: path).hasExpertScaleTensor ? 1 : nil
    }

    /// True when the model is a Mixture-of-Experts (GGUF `<arch>.expert_count` > 0).
    /// Gates the `--n-cpu-moe` control, which a dense model ignores.
    nonisolated static func modelIsMoE(at path: String) -> Bool {
        GGUFMetadataCache.metadata(at: path)?.isMoE ?? false
    }

    /// Micro-batch sizes offered in the UI. 0 leaves the engine's own 512.
    nonisolated static let ubatchOptions = [0, 1024, 2048, 4096]

    nonisolated static func ubatchLabel(_ value: Int, loc: Localizer) -> String {
        value == 0 ? loc.t("512 (por defecto)", "512 (default)") : String(value)
    }

    /// Remembers the ncmoe the user settled on for a MoE model, so selecting
    /// that model again restores it instead of re-deriving the recommendation.
    nonisolated static func rememberNcmoe(_ value: Int, forModel path: String) {
        guard !path.isEmpty, modelIsMoE(at: path) else { return }
        var map = UserDefaults.standard.dictionary(forKey: SettingsKeys.ncmoeByModel) as? [String: Int] ?? [:]
        map[path] = value
        UserDefaults.standard.set(map, forKey: SettingsKeys.ncmoeByModel)
    }

    nonisolated static func recalledNcmoe(forModel path: String) -> Int? {
        (UserDefaults.standard.dictionary(forKey: SettingsKeys.ncmoeByModel) as? [String: Int])?[path]
    }

    /// Remembers the ncmoe at which the expert-prefetch overlap stalls the GPU; prefetch
    /// only runs below it. nil clears it, so a re-measuring sweep starts prefetch on.
    nonisolated static func rememberPrefetchCliff(_ value: Int?, forModel path: String) {
        guard !path.isEmpty else { return }
        var map = UserDefaults.standard.dictionary(forKey: SettingsKeys.prefetchCliffByModel) as? [String: Int] ?? [:]
        if let value { map[path] = value } else { map.removeValue(forKey: path) }
        UserDefaults.standard.set(map, forKey: SettingsKeys.prefetchCliffByModel)
    }

    nonisolated static func recalledPrefetchCliff(forModel path: String) -> Int? {
        (UserDefaults.standard.dictionary(forKey: SettingsKeys.prefetchCliffByModel) as? [String: Int])?[path]
    }

    /// Finds the multimodal projector (mmproj) paired with a model, if any.
    private static func mmprojPairKey(_ model: String, _ projector: String) -> String {
        model + "\u{1}" + projector
    }

    /// Records that `projector` failed to load with `model` (e.g. an unknown CLIP
    /// projector type), so `mmprojPath` won't auto-attach it again. Persistent.
    nonisolated static func recordIncompatibleMmproj(model: String, projector: String) {
        var list = UserDefaults.standard.stringArray(forKey: SettingsKeys.incompatibleMmproj) ?? []
        let key = mmprojPairKey(model, projector)
        if !list.contains(key) {
            list.append(key)
            UserDefaults.standard.set(list, forKey: SettingsKeys.incompatibleMmproj)
        }
    }

    nonisolated static func isIncompatibleMmproj(model: String, projector: String) -> Bool {
        (UserDefaults.standard.stringArray(forKey: SettingsKeys.incompatibleMmproj) ?? [])
            .contains(mmprojPairKey(model, projector))
    }

    nonisolated static func extraArgs(forModel path: String) -> String {
        (UserDefaults.standard.dictionary(forKey: SettingsKeys.extraArgsByModel) as? [String: String])?[path] ?? ""
    }

    nonisolated static func setExtraArgs(_ args: String, forModel path: String) {
        var map = UserDefaults.standard.dictionary(forKey: SettingsKeys.extraArgsByModel) as? [String: String] ?? [:]
        let trimmed = args.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { map.removeValue(forKey: path) } else { map[path] = trimmed }
        UserDefaults.standard.set(map, forKey: SettingsKeys.extraArgsByModel)
    }

    nonisolated static func mmprojOverride(forModel path: String) -> String? {
        (UserDefaults.standard.dictionary(forKey: SettingsKeys.mmprojOverride) as? [String: String])?[path]
    }

    nonisolated static func setMmprojOverride(_ projector: String?, forModel path: String) {
        var map = UserDefaults.standard.dictionary(forKey: SettingsKeys.mmprojOverride) as? [String: String] ?? [:]
        if let projector { map[path] = projector } else { map.removeValue(forKey: path) }
        UserDefaults.standard.set(map, forKey: SettingsKeys.mmprojOverride)
    }

    /// True only when both dims are readable and differ; unknown pairs pass.
    nonisolated static func mmprojIncompatible(model: String, projector: String) -> Bool {
        guard let embd = ggufUInt32("embedding_length", at: model),
              let proj = ggufUInt32("projection_dim", at: projector) else { return false }
        return proj != embd
    }

    /// Whether to offer the projector control for a model: it has a paired or
    /// pinned projector, an explicit choice (incl. off), or is a catalog vision model.
    nonisolated static func mightSupportVision(modelPath: String) -> Bool {
        if mmprojOverride(forModel: modelPath) != nil { return true }
        if mmprojPath(forModel: modelPath) != nil { return true }
        let name = URL(fileURLWithPath: modelPath).lastPathComponent
        return Catalog.models.contains { $0.fileName == name && $0.isVision }
    }

    /// The DFlash draft downloaded for a model, saved as `<model>.dflash.gguf`.
    nonisolated static func dflashDraftPath(forModel modelPath: String) -> String? {
        guard !modelPath.isEmpty else { return nil }
        let url = URL(fileURLWithPath: modelPath)
        let stem = url.deletingPathExtension().lastPathComponent
        let path = url.deletingLastPathComponent().appendingPathComponent("\(stem).dflash.gguf").path
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        return path
    }

    nonisolated static func dflashMode(forModel modelPath: String) -> DflashMode {
        let defaults = UserDefaults.standard
        if let raw = (defaults.dictionary(forKey: SettingsKeys.dflashModes) as? [String: String])?[modelPath],
           let mode = DflashMode(rawValue: raw) {
            return mode
        }
        let disabled = defaults.stringArray(forKey: SettingsKeys.dflashDisabled) ?? []
        return disabled.contains(modelPath) ? .off : .auto
    }

    nonisolated static func setDflashMode(_ mode: DflashMode, forModel modelPath: String) {
        var modes = UserDefaults.standard.dictionary(forKey: SettingsKeys.dflashModes) as? [String: String] ?? [:]
        modes[modelPath] = mode.rawValue
        UserDefaults.standard.set(modes, forKey: SettingsKeys.dflashModes)
    }

    nonisolated static func dflashEnabled(forModel modelPath: String) -> Bool {
        dflashMode(forModel: modelPath) != .off
    }

    nonisolated static func setDflashEnabled(_ on: Bool, forModel modelPath: String) {
        setDflashMode(on ? .auto : .off, forModel: modelPath)
    }

    /// The active DFlash draft: present on disk and not switched off for this model.
    nonisolated static func activeDflashDraft(forModel modelPath: String) -> String? {
        guard dflashEnabled(forModel: modelPath) else { return nil }
        return dflashDraftPath(forModel: modelPath)
    }

    /// A bf16 projector aborts the engine on a card without the Metal 3 family: the loader
    /// puts the weight in device memory and the scheduler then refuses to run it there.
    /// Keeping the tower on the CPU costs image-encode time and keeps vision working.
    nonisolated static func projectorNeedsCPU(_ mmproj: String) -> Bool {
        guard GGUFMetadataCache.tensorFlags(at: mmproj).hasBF16Tensor else { return false }
        return ServerController.availableGPUs().contains { !$0.isIntegrated && !$0.supportsBF16 }
    }

    nonisolated static func mmprojPath(forModel modelPath: String) -> String? {
        guard !modelPath.isEmpty else { return nil }
        let url = URL(fileURLWithPath: modelPath)
        let name = url.deletingPathExtension().lastPathComponent
        if name.lowercased().contains("mmproj") { return nil }  // the model itself is not a projector

        // A manual override wins: a path pins that projector, "" disables vision;
        // a stale path (file moved) falls through to auto-pairing.
        if let override = mmprojOverride(forModel: modelPath) {
            if override.isEmpty { return nil }
            if FileManager.default.fileExists(atPath: override) { return override }
        }

        let dir = url.deletingLastPathComponent()
        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return nil }
        var projectors = files.filter {
            $0.pathExtension.lowercased() == "gguf" && $0.lastPathComponent.lowercased().contains("mmproj")
        }
        guard !projectors.isEmpty else { return nil }

        // Keep only projectors whose projection_dim matches the model's
        // embedding_length; unreadable ones are kept rather than punished.
        if let modelEmbd = ggufUInt32("embedding_length", at: modelPath) {
            let compatible = projectors.filter { p in
                guard let proj = ggufUInt32("projection_dim", at: p.path) else { return true }
                return proj == modelEmbd
            }
            if compatible.isEmpty { return nil }
            projectors = compatible
        }

        // Skip projectors recorded as incompatible with this model; a different
        // one for the same model still gets picked up.
        projectors = projectors.filter { !isIncompatibleMmproj(model: modelPath, projector: $0.path) }
        guard !projectors.isEmpty else { return nil }

        // Managed downloads use an exact model-specific stem: highest confidence,
        // so try it before the legacy names below.
        func core(_ s: String) -> String {
            String(s.lowercased().replacingOccurrences(of: "mmproj", with: "").filter { $0.isLetter || $0.isNumber })
        }
        let mn = core(name)
        guard !mn.isEmpty else { return nil }
        if let exact = projectors.first(where: {
            core($0.deletingPathExtension().lastPathComponent) == mn
        }) {
            return exact.path
        }

        // Legacy/manual names often omit the quant or model size. Fall back only
        // when the GGUF dimensions match and exactly one same-family projector
        // remains; ambiguity deliberately returns nil instead of guessing.
        guard let modelEmbd = ggufUInt32("embedding_length", at: modelPath) else { return nil }
        func family(_ value: String) -> String {
            ModelName(value).title.lowercased().filter { $0.isLetter || $0.isNumber }
        }
        func sameFamily(_ lhs: String, _ rhs: String) -> Bool {
            guard lhs.count >= 5, rhs.count >= 5 else { return false }
            return lhs == rhs || lhs.hasPrefix(rhs) || rhs.hasPrefix(lhs)
        }
        let modelFamily = family(name)
        let matches = projectors.filter { projector in
            guard ggufUInt32("projection_dim", at: projector.path) == modelEmbd else { return false }
            return sameFamily(modelFamily, family(projector.lastPathComponent))
        }
        return matches.count == 1 ? matches[0].path : nil
    }

    /// The API key the chat must send, when protection is enabled in Settings.
    static func activeAPIKey() -> String? {
        UserDefaults.standard.bool(forKey: SettingsKeys.apiKeyEnabled) ? Keychain.apiKey() : nil
    }

    /// The model alias the native chat should send as `"model"`, or nil when
    /// the primary server isn't in router mode (single-model requests omit it).
    static func activeRouterModel() -> String? {
        let d = UserDefaults.standard
        guard d.bool(forKey: SettingsKeys.routerMode) else { return nil }
        let alias = d.string(forKey: SettingsKeys.chatSelectedModel) ?? ""
        if !alias.isEmpty { return alias }
        // Default to the first model when the chat hasn't picked one yet (its
        // picker's default-selection task lives in a lazily-built popover).
        return LocalModel.scan(in: modelsDirectory).first.map { routerAlias(for: $0.url.path) }
    }
}
