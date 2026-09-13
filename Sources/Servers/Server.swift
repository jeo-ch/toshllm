// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import Metal

/// Owns the running engine instances.
@MainActor
final class ServerManager: ObservableObject {
    static let shared = ServerManager()

    @Published var servers: [ServerController]
    /// The instance the chat and benchmark act on.
    @Published var activeID: UUID

    private static let storeKey = "multiServerProfiles"
    private var pendingPersist: Task<Void, Never>?

    private init() {
        // Server 1 is the default: nil profile → driven by the global settings.
        let first = ServerController()
        var list = [first]
        // Recreate any extra servers the user added, each with its own config.
        if let data = UserDefaults.standard.data(forKey: Self.storeKey),
           let profiles = try? JSONDecoder().decode([Profile].self, from: data) {
            for var p in profiles {
                // Freeze the currently inherited model once, without changing the
                // model users were seeing before instance selection was restored.
                if let pins = p.pinned, !pins.contains(Profile.Pin.model) {
                    let current = ServerSettings.fromDefaults()
                    p.selectInstanceModel(path: current.modelPath,
                                          ncmoe: pins.contains(Profile.Pin.moe) ? p.ncmoe : current.ncmoe)
                }
                let c = ServerController()
                c.name = p.name
                c.profile = p
                list.append(c)
            }
        }
        servers = list
        activeID = first.id
        schedulePersist()
    }

    var active: ServerController { servers.first { $0.id == activeID } ?? servers[0] }

    func setActive(_ id: UUID) {
        if servers.contains(where: { $0.id == id }) { activeID = id }
    }

    /// Lowest port not already taken by a server, starting at the default.
    func freePort() -> Int {
        let used = Set(servers.map { $0.profile?.port ?? ServerSettings.fromDefaults().port })
        var p = 8080
        while used.contains(p) { p += 1 }
        return p
    }

    /// Adds a server from a base profile (or the current config), on a free port.
    @discardableResult
    func addServer(name: String, from base: Profile?) -> ServerController {
        var p = base ?? ServerSettings.fromDefaults().makeProfile(name: name)
        p.name = name
        p.port = freePort()
        // Each instance owns its model from creation. Other settings still inherit.
        if base == nil { p.pinned = [Profile.Pin.model] }
        p.selectInstanceModel(path: p.modelPath, ncmoe: p.ncmoe)
        let c = ServerController()
        c.name = name
        c.profile = p
        servers.append(c)
        activeID = c.id
        schedulePersist()
        return c
    }

    /// Removes an added server (never the default). Stops it first.
    func removeServer(_ id: UUID) {
        guard let i = servers.firstIndex(where: { $0.id == id }), i != 0 else { return }
        servers[i].stop()
        let wasActive = servers[i].id == activeID
        servers.remove(at: i)
        if wasActive { activeID = servers[0].id }
        schedulePersist()
    }

    func stopAll() { servers.forEach { $0.stop() } }

    func stopAllImmediately() { servers.forEach { $0.stopImmediately() } }

    /// Persists only the added servers (those with their own profile).
    private func persist() {
        pendingPersist?.cancel()
        let profiles = servers.compactMap { $0.profile }
        if let data = try? JSONEncoder().encode(profiles) {
            UserDefaults.standard.set(data, forKey: Self.storeKey)
        }
    }

    func schedulePersist() {
        pendingPersist?.cancel()
        pendingPersist = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(180))
            guard !Task.isCancelled else { return }
            self?.persist()
        }
    }
}

/// Engine output updates much faster than dashboard state. Keeping it in its own
/// observable prevents every log line from invalidating all server cards.
@MainActor
final class ServerLogBuffer: ObservableObject {
    private(set) var text = ""
    private var notificationPending = false

    func set(_ value: String) {
        text = value
        scheduleNotification()
    }

    func append(_ value: String, limit: Int = 120_000, retained: Int = 80_000) {
        text += value
        if text.count > limit { text = String(text.suffix(retained)) }
        scheduleNotification()
    }

    private func scheduleNotification() {
        guard !notificationPending else { return }
        notificationPending = true
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(0.10))
            guard let self else { return }
            self.notificationPending = false
            self.objectWillChange.send()
        }
    }
}

@MainActor
final class ServerController: ObservableObject {
    let id = UUID()
    @Published var name: String = "Servidor 1"
    /// Per-server config. nil = the default server, which follows the global settings.
    @Published var profile: Profile?

    /// Config this server launches with: the global defaults plus its pinned
    /// fields. A nil pinned list is a pre-0.83 server: full snapshot, as before.
    func effectiveSettings() -> ServerSettings {
        guard let profile else { return .fromDefaults() }
        var s = ServerSettings.fromDefaults()
        if let pinned = profile.pinned {
            s.applyPinned(profile, Set(pinned))
        } else {
            s.apply(profile)
        }
        return s
    }

    enum State: Equatable { case stopped, starting, running, failed(String) }

    @Published var state: State = .stopped
    let logBuffer = ServerLogBuffer()
    var log: String {
        get { logBuffer.text }
        set { logBuffer.set(newValue) }
    }
    @Published var promptSpeed: Double?
    @Published var genSpeed: Double?
    @Published var genHistory: [Double] = []
    @Published var requestCount = 0
    @Published private(set) var startedAt: Date?
    @Published var dflashWarning: DflashRuntimeWarning?
    /// Model whose running engine actually has DFlash engaged, nil otherwise.
    @Published private(set) var activeDflashModelPath: String?
    @Published var dflashAcceptance: Double?

    private var process: Process?
    private var healthTask: Task<Void, Never>?
    private var dflashMemoryTask: Task<Void, Never>?
    private var launchedSettings: ServerSettings?
    private var lastStoppedPID: Int32?
    /// After a projector load failure, makes the next launch drop `--mmproj`
    /// (text-only). Reset on every fresh `start()`.
    private var retryWithoutMmproj = false
    private var currentPort = 8080
    private var discoveryService: NetService?
    private var discoveryEnabled = false
    private let fileLog = RotatingFileLog(name: "server.log")
    /// Pre-warm slot 0 across restarts for external clients (VS Code/Cline resend a
    /// fixed 15-19k-token prefix every request). Off for MTP models: the extra KV
    /// breaks slot restore.
    private var prewarmActive = false
    /// Single fixed file for the external-client prefix (not a conversation UUID,
    /// so the chat's orphan-prune leaves it alone).
    static func externalSlotFile(port: Int) -> URL {
        ServerSettings.slotCacheDir(port: port).appendingPathComponent("external.bin")
    }

    var logFileURL: URL { fileLog.fileURL }
    /// Folder with every per-session log file (kept ~3 days), for sharing past runs.
    var logsDirectory: URL { fileLog.directory }

    var serverURL: URL { URL(string: "http://127.0.0.1:\(currentPort)/")! }

    /// Web chat URL carrying the app's language and the real device names, so the
    /// bundled console reports what's in use instead of guessing.
    var webChatURL: URL {
        let lang = UserDefaults.standard.string(forKey: SettingsKeys.language) ?? "en"
        var comps = URLComponents(string: "http://127.0.0.1:\(currentPort)/")!
        var items = [URLQueryItem(name: "lang", value: lang)]
        if let gpu = ServerController.availableGPUs().max(by: { $0.vramMB < $1.vramMB })?.name {
            items.append(URLQueryItem(name: "gpu", value: gpu))
        }
        // Real inference backend, read from the engine's startup log (a custom
        // external build may use Vulkan instead of the bundled Metal engine).
        let backend = log.range(of: "vulkan", options: .caseInsensitive) != nil ? "Vulkan" : "Metal"
        items.append(URLQueryItem(name: "backend", value: backend))
        comps.queryItems = items
        return comps.url!
    }

    /// Cached: the UI reads this from view bodies, and MTLCopyAllDevices() is far
    /// too expensive to run on every render.
    nonisolated static func availableGPUs(rescan: Bool = false) -> [GPUDevice] {
        gpuCacheLock.lock()
        defer { gpuCacheLock.unlock() }
        if !rescan, let cached = cachedGPUs, let ts = gpuCacheTimestamp, Date().timeIntervalSince(ts) < 30 {
            return cached
        }
        let devices = MTLCopyAllDevices().enumerated().map { i, dev in
            GPUDevice(index: i, name: dev.name,
                      vramMB: Int(dev.recommendedMaxWorkingSetSize / 1_048_576),
                      isExternal: dev.location == .external,
                      isIntegrated: dev.isLowPower,
                      peerGroupID: dev.peerGroupID,
                      peerCount: Int(dev.peerCount),
                      supportsBF16: dev.supportsFamily(.metal3))
        }
        cachedGPUs = devices
        gpuCacheTimestamp = Date()
        return devices
    }

    nonisolated(unsafe) private static var cachedGPUs: [GPUDevice]?
    nonisolated(unsafe) private static var gpuCacheTimestamp: Date?
    private nonisolated static let gpuCacheLock = NSLock()

    /// Whether any detected GPU is an external eGPU. Used to surface the
    /// VRAM-resident-weights option, which fixes eGPU slowness over Thunderbolt.
    nonisolated static func hasExternalGPU() -> Bool {
        availableGPUs().contains { $0.isExternal }
    }

    func start(_ settings: ServerSettings) {
        guard state == .stopped || isFailed else { return }
        guard FileManager.default.fileExists(atPath: settings.serverBinary) else {
            state = .failed("No existe el binario llama-server en la ruta configurada")
            return
        }
        if settings.routerMode {
            let models = LocalModel.scan(in: ServerSettings.modelsDirectory)
            guard !models.isEmpty else {
                let lang = UserDefaults.standard.string(forKey: SettingsKeys.language) ?? "en"
                state = .failed(lang == "es"
                    ? "No hay modelos descargados en la carpeta de modelos"
                    : "No models downloaded in the models folder")
                return
            }
            if settings.usesTurboKV,
               let incompatible = models.first(where: { !ServerSettings.modelSupportsTurboKV(at: $0.url.path) }) {
                failTurboKV(model: incompatible.url.lastPathComponent)
                return
            }
            if settings.usesTurboValuesWithoutKeys,
               let mla = models.first(where: { ServerSettings.modelUsesMLA(at: $0.url.path) }) {
                failTurboKV(model: mla.url.lastPathComponent)
                return
            }
            if settings.usesUnsupportedTurboQ4Mix {
                failTurboKV(model: "router")
                return
            }
        } else {
            guard FileManager.default.fileExists(atPath: settings.modelPath) else {
                state = .failed("Selecciona un modelo en la pestaña Modelos")
                return
            }
            // TurboQuant weight quants (tq3_1s/tq4_1s) decode to garbage, so refuse
            // rather than serve it.
            if ServerSettings.modelIsTurboQuantWeights(at: settings.modelPath) {
                let lang = UserDefaults.standard.string(forKey: SettingsKeys.language) ?? "en"
                state = .failed(lang == "es"
                    ? "Modelo TurboQuant no soportado: la cuantización de pesos TurboQuant (tq3_1s/tq4_1s) produce salida incorrecta en este motor, tanto en modelos densos como MoE. Usa un modelo en cuantización estándar (Q4_K, Q5_K, Q6_K, Q8_0…)."
                    : "TurboQuant model not supported: TurboQuant weight quantization (tq3_1s/tq4_1s) produces incorrect output on this engine, for both dense and MoE models. Use a standard-quant model (Q4_K, Q5_K, Q6_K, Q8_0…).")
                return
            }
            if settings.usesTurboKV &&
               (ServerSettings.isAppleSilicon || !ServerSettings.modelSupportsTurboKV(at: settings.modelPath)) {
                failTurboKV(model: URL(fileURLWithPath: settings.modelPath).lastPathComponent)
                return
            }
            if settings.usesTurboValuesWithoutKeys && ServerSettings.modelUsesMLA(at: settings.modelPath) {
                failTurboKV(model: URL(fileURLWithPath: settings.modelPath).lastPathComponent)
                return
            }
            if settings.usesUnsupportedTurboQ4Mix {
                failTurboKV(model: URL(fileURLWithPath: settings.modelPath).lastPathComponent)
                return
            }
        }

        log = ""
        retryWithoutMmproj = false
        promptSpeed = nil
        genSpeed = nil
        genHistory = []
        requestCount = 0
        currentPort = settings.port
        discoveryEnabled = settings.localNetworkDiscovery
        stopDiscovery()
        state = .starting
        startedAt = nil

        // A stopped engine can take seconds to die (SIGTERM mid-generation) and still
        // holds the port meanwhile, so wait for the previous PID before binding.
        let previousPID = lastStoppedPID
        lastStoppedPID = nil
        Task { [weak self] in
            if let pid = previousPID {
                for _ in 0..<24 where kill(pid, 0) == 0 {
                    try? await Task.sleep(for: .milliseconds(250))
                }
            }
            self?.launch(settings)
        }
    }

    private func failTurboKV(model: String) {
        let lang = UserDefaults.standard.string(forKey: SettingsKeys.language) ?? "en"
        state = .failed(lang == "es"
            ? "TurboQuant KV no es compatible con \(model) o con la combinación elegida: requiere Metal AMD y cabezas con padding 128, 256, 384, 512 o 640; en MLA, Turbo en valores también requiere Turbo en claves; q4_0 no se puede mezclar con Turbo."
            : "TurboQuant KV is not compatible with \(model) or the selected combination: it requires AMD Metal and heads padded to 128, 256, 384, 512 or 640; on MLA, Turbo values also require Turbo keys; q4_0 cannot be mixed with Turbo.")
    }

    /// Header at the top of the server log: version, engine, model, GPUs and the
    /// resolved settings, so a pasted log is debuggable without round-trips.
    nonisolated static func startupBanner(settings: ServerSettings) -> String {
        func redact(_ items: [String]) -> [String] {
            var out = items
            if let i = out.firstIndex(of: "--api-key"), i + 1 < out.count { out[i + 1] = "***" }
            return out
        }
        let engine: String
        engine = settings.serverBinary == ServerSettings.defaultBinary ? "bundled (official)" : "external"
        // Device order changes between boots, so an index alone does not identify a
        // card in a pasted log; the peer group does identify who it is linked to.
        let gpus = availableGPUs().map {
            let peer = $0.peerGroupID == 0
                ? " · no peer group"
                : " · peer group \($0.peerGroupID) (\($0.peerCount) GPUs)"
            return "    [\($0.index)] \($0.name) · \($0.vramGB) GB\(peer)\($0.isExternal ? " · EXTERNAL/eGPU" : "")\($0.isIntegrated ? " · iGPU (not auto-selected)" : "")"
        }.joined(separator: "\n")
        let envKeys = ["GGML_METAL_VRAM_RESERVE_MB",
                       "GGML_METAL_DEVICE_INDEX", "GGML_METAL_DEVICES", "GGML_METAL_DEVICE_LIST",
                       "GGML_METAL_SHARED_BUFFERS_DISABLE", "TOSH_FA_AMD",
                       "GGML_SCHED_PREFETCH_EXPERTS", "GGML_CPU_NO_REPACK",
                       "TOSH_MOE_UI", "TOSH_MOE_MODE", "TOSH_MOE_SLOTS", "TOSH_MOE_CPU_BANK",
                       "TOSH_MOE_SPLIT_BANK", "TOSH_MOE_SPLIT_RING", "TOSH_MOE_BOUNDED_STAGE",
                       "TOSH_MOE_BOUNDED_STAGE_FORCE", "TOSH_MOE_DOUBLE_BUFFER", "TOSH_MOE_HOT_MAP",
                       "TOSH_MOE_HOT_MAP_OUT", "TOSH_MOE_HOT_MAP_K",
                       "GGML_METAL_NCB",
                       "TOSH_MGPU_PEER", "TOSH_MGPU_PEER_DISABLE", "TOSH_MGPU_EVENTS"]
        let env = settings.environment
        // Include user-provided environment variables in diagnostic logs.
        let userKeys = settings.extraArgTokens.env.keys.filter { !envKeys.contains($0) }.sorted()
        let envLine = (envKeys + userKeys)
            .compactMap { k in env[k].map { "\(k)=\($0)" } }
            .joined(separator: " ")
        let moeLine = settings.effectiveDynamicMoe
            ? "ncmoe=0 dynamic-moe=K\(settings.effectiveDynamicMoeSlots)"
            : "ncmoe=\(settings.ncmoe)"
        var gpuSel = settings.multiGPU ? "split-all" : (settings.gpuIndex >= 0 ? "index \(settings.gpuIndex)" : "default (macOS picks)")
        if settings.isSplitting { gpuSel += " · split-mode \(settings.effectiveSplitMode)" }
        if settings.tensorSplitDowngraded {
            gpuSel += " (tensor needs \(settings.splitDeviceCount) way split of experts this model only divides \(settings.tensorSplitLimit ?? 1) ways)"
        }
        return """
        ========================================================
         ToshLLM \(AppInfo.version) — server start (\(ServerSettings.isAppleSilicon ? "arm64" : "x86_64")\(AppInfo.isNoAVX2 ? " · no-AVX2 build" : ""))
         engine : \(engine)
         model  : \(settings.routerMode ? "router (autoload, max \(settings.routerModelsMax) loaded)" : (settings.modelPath as NSString).lastPathComponent)
         GPUs detected:
        \(gpus.isEmpty ? "    (none)" : gpus)
         GPU select: \(gpuSel) | force-VRAM-buffers: \(env["GGML_METAL_SHARED_BUFFERS_DISABLE"] == "1" ? "yes" : "no")
         settings: ngl=\(settings.ngl) \(moeLine) ctx=\(settings.ctx) fa=\(settings.flashAttn) ctk=\(settings.cacheTypeK) ctv=\(settings.cacheTypeV) cacheRAM=\(settings.cacheRAM)
         dflash : \(settings.routerMode ? "per-model router plan" : settings.dflashPlanSummary)
         env: \(envLine)
         args: \(redact(settings.arguments).joined(separator: " "))
        ========================================================

        """
    }

    private func launch(_ settings: ServerSettings) {
        guard state == .starting else { return }   // user hit Stop meanwhile

        if settings.routerMode {
            let models = LocalModel.scan(in: ServerSettings.modelsDirectory)
            let paths = models.map(\.url.path)
            let ncmoeByPath = Dictionary(uniqueKeysWithValues: paths.map {
                ($0, Estimator.ncmoeForSelection(path: $0, models: models))
            })
            let ini = settings.routerPresetINI(modelPaths: paths, ncmoeByPath: ncmoeByPath)
            try? ini.write(to: ServerSettings.routerPresetPath(port: settings.port), atomically: true, encoding: .utf8)
        }

        prewarmActive = !settings.routerMode && settings.persistCache && settings.effectiveFaAmd
            && !settings.visionLoaded && !ServerSettings.modelUsesMTP(at: settings.modelPath)

        let p = Process()
        p.executableURL = URL(fileURLWithPath: settings.serverBinary)
        var args = settings.arguments
        if retryWithoutMmproj, let i = args.firstIndex(of: "--mmproj") {
            args.removeSubrange(i ..< min(i + 2, args.count))   // drop "--mmproj <path>"
        }
        p.arguments = args
        p.environment = settings.environment
        launchedSettings = settings
        activeDflashModelPath = args.contains("draft-dflash") ? settings.modelPath : nil
        dflashAcceptance = nil

        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.consume(text) }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in
                guard let self else { return }
                // A process we already replaced (stop → start) must not touch
                // the new engine's state, health watch or PID lockfile.
                guard self.process === proc else { return }
                self.healthTask?.cancel()
                self.stopDiscovery()
                EngineLock.remove(pid: proc.processIdentifier)
                if case .failed = self.state { return }
                if proc.terminationStatus == 0 || proc.terminationStatus == 15 {
                    self.state = .stopped
                } else {
                    // A projector that won't load fails the whole launch; retry
                    // once without it so the model still runs text-only.
                    let tail = self.log.suffix(6000).lowercased()
                    let clipFailed = tail.contains("failed to load clip")
                        || tail.contains("unknown projector type")
                        || tail.contains("failed to load multimodal model")
                    if clipFailed && !self.retryWithoutMmproj && args.contains("--mmproj") {
                        // Don't auto-attach this projector again for this model.
                        if let i = args.firstIndex(of: "--mmproj"), i + 1 < args.count {
                            ServerSettings.recordIncompatibleMmproj(model: settings.modelPath, projector: args[i + 1])
                        }
                        self.retryWithoutMmproj = true
                        EngineLock.remove(pid: proc.processIdentifier)
                        self.consume("\n[ToshLLM] el proyector (mmproj) no se pudo cargar — reintentando solo-texto (visión desactivada) / projector failed to load — retrying text-only (vision disabled)\n")
                        self.state = .starting
                        self.launch(settings)
                        return
                    }
                    AppLog.server.error("engine exited with status \(proc.terminationStatus)")
                    self.state = .failed(Self.diagnose(self.log, exitCode: proc.terminationStatus))
                }
            }
        }

        fileLog.startSession()   // new timestamped per-session file, prunes old ones
        consume(Self.startupBanner(settings: settings))
        do {
            try p.run()
            process = p
            EngineLock.add(pid: p.processIdentifier)
            // A fresh engine starts with empty KV slots; tell the chat so it
            // re-restores the active conversation's persisted cache on next turn.
            NotificationCenter.default.post(name: .engineDidStart, object: nil)
            watchHealth(port: settings.port)
        } catch {
            state = .failed("No se pudo lanzar: \(error.localizedDescription)")
        }
    }

    /// Maps known engine failure patterns to actionable, bilingual messages.
    static func diagnose(_ log: String, exitCode: Int32) -> String {
        let tail = log.suffix(6000).lowercased()
        if tail.contains("unknown model architecture") || tail.contains("unknown architecture") {
            return "Arquitectura no soportada por este motor / model architecture not supported by this engine"
        }
        if tail.contains("split_mode_tensor not implemented for architecture") {
            return "Esta arquitectura no admite el reparto por tensores: cámbialo a por capas en Ajustes / this architecture has no tensor split: switch to by layers in Settings"
        }
        if tail.contains("address already in use") || tail.contains("couldn't bind") {
            return "Puerto ocupado: cambia el puerto en Ajustes / port busy: change it in Settings"
        }
        // A refused block does not necessarily mean total memory is exhausted.
        if tail.contains("failed to allocate buffer, size =") {
            // Graph tensors cannot be split by the weight-buffer cap.
            if tail.contains("ggml_gallocr_reserve") {
                return "La tarjeta rechazó un bloque de memoria del grafo: reduce el contexto, y si el modelo tiene visión baja el tope de tokens por imagen / the card refused a graph memory block: reduce context, and lower the image token cap if the model has vision"
            }
            return "La tarjeta rechazó un bloque de memoria demasiado grande: reduce el contexto, o arranca con TOSH_METAL_MAX_BUFFER_MB=1024 para repartirlo / the card refused a single oversized memory block: reduce context, or start with TOSH_METAL_MAX_BUFFER_MB=1024 to split it"
        }
        if tail.contains("out of memory") || tail.contains("failed to allocate")
            || tail.contains("insufficient memory") || tail.contains("kiogpucommandbuffercallbackerroroutofmemory") {
            return "Memoria insuficiente: sube 'Expertos MoE en CPU' o reduce el contexto / out of memory: raise 'MoE experts on CPU' or reduce context"
        }
        if tail.contains("quantized v cache") {
            return "El KV cuantizado requiere Flash Attention: activa el kernel AMD o usa FA estándar / quantized KV requires Flash Attention: enable the AMD kernel or use standard FA"
        }
        // The tensor named in the abort says whether it is the projector or the model itself.
        if tail.contains("pre-allocated tensor") && tail.contains("cannot run the operation") {
            return tail.contains("pre-allocated tensor (v.")
                ? "El proyector de visión usa un formato que esta tarjeta no ejecuta (normalmente BF16): descarga el mmproj en F16 / the vision projector uses a format this card cannot run (usually BF16): download the F16 mmproj"
                : "Un tensor del modelo usa un formato que esta tarjeta no ejecuta (normalmente BF16): usa un GGUF en F16 o cuantizado / a model tensor uses a format this card cannot run (usually BF16): use an F16 or quantized GGUF"
        }
        // The engine's own wording. Matching a bare "mtp" also matched every file named -MTP-.
        if tail.contains("doesn't contain mtp layers") || tail.contains("failed to create mtp context") {
            return "Este modelo no trae cabezal MTP: descarga la variante -MTP- / model has no MTP head: download the -MTP- variant"
        }
        if tail.contains("invalid ggml type") || tail.contains("should be in [0,") {
            return "Cuantización no soportada por el motor (formato de un fork, p. ej. Prism ML): usa un GGUF con quant estándar (Q4_K_M, Q8_0, Q2_0_g64…) / quantization not supported by the engine (a fork's format, e.g. Prism ML): use a GGUF with a standard quant (Q4_K_M, Q8_0, Q2_0_g64…)"
        }
        if tail.contains("invalid magic") || tail.contains("failed to load model")
            || tail.contains("error loading model") {
            return "Modelo dañado o incompleto: vuelve a descargarlo / model file damaged or incomplete: re-download it"
        }
        if exitCode == SIGILL {
            return AppInfo.isNoAVX2
                ? "Instrucción ilegal (SIGILL): este CPU no soporta el motor — reporta tu modelo de CPU en GitHub / illegal instruction (SIGILL): this CPU can't run the engine — report your CPU model on GitHub"
                : "Instrucción ilegal (SIGILL): este CPU no soporta AVX2 — instala la versión no-AVX2 del release / illegal instruction (SIGILL): this CPU lacks AVX2 — install the no-AVX2 build from the release"
        }
        return "El motor terminó con código \(exitCode) — revisa el registro en Ajustes / engine exited with code \(exitCode) — see the log in Settings"
    }

    /// Router mode spawns one child engine per loaded model. They hold the weights
    /// (mlock'd), so a survivor keeps the RAM until it is killed by hand.
    nonisolated static func reapChildren(of parent: pid_t) {
        let pgrep = Process()
        pgrep.executableURL = URL(fileURLWithPath: "/usr/bin/pgrep")
        pgrep.arguments = ["-P", String(parent)]
        let pipe = Pipe()
        pgrep.standardOutput = pipe
        pgrep.standardError = FileHandle.nullDevice
        guard (try? pgrep.run()) != nil else { return }
        let out = pipe.fileHandleForReading.readDataToEndOfFile()
        pgrep.waitUntilExit()
        let kids = String(decoding: out, as: UTF8.self)
            .split(whereSeparator: \.isNewline).compactMap { pid_t($0.trimmingCharacters(in: .whitespaces)) }
        guard !kids.isEmpty else { return }
        for kid in kids { kill(kid, SIGTERM) }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 5) {
            for kid in kids where kill(kid, 0) == 0 { kill(kid, SIGKILL) }
        }
    }

    func stop() {
        AudioStudioController.shared.shutdown()
        SpeechDictationController.shared.shutdown()
        AppleSpeechDictationController.shared.shutdown()
        healthTask?.cancel()
        dflashMemoryTask?.cancel()
        activeDflashModelPath = nil
        dflashAcceptance = nil
        stopDiscovery()
        if let p = process {
            Self.reapChildren(of: p.processIdentifier)
            let pid = p.processIdentifier
            lastStoppedPID = pid
            // Drop this engine's PID now: once process is niled below, the termination
            // handler's guard skips its own removal.
            EngineLock.remove(pid: pid)
            let prewarm = prewarmActive
            let port = currentPort
            // SIGKILL fallback in case the deferred terminate stalls: the engine's
            // working set has to be freed.
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 6) {
                if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
            }
            if prewarm {
                // Snapshot slot 0 to disk before killing the engine, so the next
                // launch can restore the fixed external-client prefix. Best-effort,
                // bounded; terminate runs even if the save fails or times out.
                Task.detached {
                    await ServerController.slotAction("save", port: port,
                                                      file: ServerController.externalSlotFile(port: port).lastPathComponent)
                    p.terminate()
                }
            } else {
                p.terminate()
            }
        } else if let pid = lastStoppedPID {
            EngineLock.remove(pid: pid)
        }
        process = nil
        state = .stopped
        startedAt = nil
    }

    /// Quitting: the engine must be signalled inline, since a detached task does
    /// not outlive the process.  Called from synchronous app-termination handlers
    /// (SIGTERM, applicationWillTerminate) so this cannot use async/await — the
    /// brief usleep is the trade-off for guaranteed cleanup before exit.
    func stopImmediately() {
        AudioStudioController.shared.shutdown()
        SpeechDictationController.shared.shutdown()
        AppleSpeechDictationController.shared.shutdown()
        healthTask?.cancel()
        dflashMemoryTask?.cancel()
        activeDflashModelPath = nil
        dflashAcceptance = nil
        stopDiscovery()
        defer { process = nil; state = .stopped }
        guard let p = process else {
            if let pid = lastStoppedPID { EngineLock.remove(pid: pid) }
            return
        }
        let pid = p.processIdentifier
        lastStoppedPID = pid
        Self.reapChildren(of: pid)
        EngineLock.remove(pid: pid)
        if prewarmActive {
            Self.slotSaveBlocking(port: currentPort,
                                  file: Self.externalSlotFile(port: currentPort).lastPathComponent,
                                  timeout: 10)
        }
        p.terminate()
        // Poll for graceful exit (50ms intervals, max 1s).  SIGKILL is the
        // fallback if the process refuses to die — same as stop()'s 6s timer.
        var waited = 0
        while p.isRunning && waited < 20 {
            usleep(50_000)
            waited += 1
        }
        if p.isRunning { kill(pid, SIGKILL) }
    }

    /// Restart with new settings, if currently up. Waits (bounded) for the old
    /// engine to exit so the relaunch doesn't race it for the port.
    func restart(_ settings: ServerSettings) {
        guard state == .running || state == .starting else { return }
        stop()
        Task { @MainActor in
            for _ in 0..<40 {
                guard let pid = lastStoppedPID, kill(pid, 0) == 0 else { break }
                try? await Task.sleep(for: .milliseconds(100))
            }
            start(settings)
        }
    }

    /// POST /slots/0?action=save|restore (best-effort, short timeout). Used to
    /// pre-warm the external-client prefix across engine restarts.
    nonisolated static func slotAction(_ action: String, port: Int, file: String) async {
        guard var comps = URLComponents(string: "http://127.0.0.1:\(port)/slots/0") else { return }
        comps.queryItems = [URLQueryItem(name: "action", value: action)]
        guard let url = comps.url else { return }
        var req = URLRequest(url: url, timeoutInterval: 30)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = ServerSettings.activeAPIKey() { req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["filename": file])
        _ = try? await NetworkManager.session.data(for: req)
    }

    /// Same request on the way out, where an await would never resume. Bounded so
    /// a busy engine cannot hold the quit open.
    nonisolated static func slotSaveBlocking(port: Int, file: String, timeout: TimeInterval) {
        guard var comps = URLComponents(string: "http://127.0.0.1:\(port)/slots/0") else { return }
        comps.queryItems = [URLQueryItem(name: "action", value: "save")]
        guard let url = comps.url else { return }
        var req = URLRequest(url: url, timeoutInterval: timeout)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = ServerSettings.activeAPIKey() { req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        req.httpBody = try? JSONSerialization.data(withJSONObject: ["filename": file])
        let done = DispatchSemaphore(value: 0)
        let task = NetworkManager.session.dataTask(with: req) { _, _, _ in done.signal() }
        task.resume()
        if done.wait(timeout: .now() + timeout) == .timedOut { task.cancel() }
    }

    /// Restore the saved external-client prefix into slot 0 (only if a file exists).
    private func restoreExternalSlot(port: Int) async {
        guard prewarmActive,
              FileManager.default.fileExists(atPath: Self.externalSlotFile(port: port).path) else { return }
        await Self.slotAction("restore", port: port, file: Self.externalSlotFile(port: port).lastPathComponent)
    }

    private var isFailed: Bool { if case .failed = state { return true }; return false }

    private func watchHealth(port: Int) {
        healthTask?.cancel()
        healthTask = Task { [weak self] in
            let url = URL(string: "http://127.0.0.1:\(port)/health")!
            for _ in 0..<150 {   // up to ~5 min for large models
                if Task.isCancelled { return }
                if let (data, _) = try? await NetworkManager.session.data(from: url),
                   String(data: data, encoding: .utf8)?.contains("ok") == true {
                    await MainActor.run {
                        self?.state = .running
                        self?.startedAt = Date()
                        self?.startDiscoveryIfNeeded(port: port)
                        self?.startDflashMemoryCheck()
                    }
                    // Pre-warm slot 0 with the last session's prefix so an external
                    // client's first request skips the multi-minute cold prefill.
                    await self?.restoreExternalSlot(port: port)
                    return
                }
                try? await Task.sleep(for: .seconds(2))
            }
            await MainActor.run {
                self?.state = .failed("El servidor no respondió al health check")
                self?.stopDiscovery()
                self?.process?.terminate()
            }
        }
    }

    private func startDflashMemoryCheck() {
        dflashMemoryTask?.cancel()
        guard let settings = launchedSettings,
              settings.arguments.contains("draft-dflash") else { return }
        let monitoredGPUIndices = settings.selectedGPUIndices
        dflashMemoryTask = Task { [weak self] in
            var peak: GPUStat?
            var fractions: [Double] = []
            for _ in 0..<8 {
                if Task.isCancelled { return }
                let allStats = await Task.detached(priority: .utility) { VRAMMonitor.snapshot() }.value
                let selectedStats = allStats.filter { monitoredGPUIndices.contains($0.id) }
                let stats = selectedStats.isEmpty ? allStats : selectedStats
                if let sample = stats.max(by: { $0.fraction < $1.fraction }) {
                    fractions.append(sample.fraction)
                    if let current = peak {
                        if sample.fraction > current.fraction { peak = sample }
                    } else {
                        peak = sample
                    }
                }
                try? await Task.sleep(for: .seconds(1))
            }
            guard let self, let peak else { return }
            if peak.fraction >= 0.90 {
                consume("\n[ToshLLM] DFlash runtime VRAM peak: \(Int(peak.fraction * 100))% (\(Int(peak.usedMB)) / \(Int(peak.totalMB)) MiB)\n")
            }
            guard DflashPolicy.shouldWarn(fractions: fractions) else { return }
            let signature = dflashWarningSignature(settings: settings)
            let acknowledged = UserDefaults.standard.stringArray(
                forKey: SettingsKeys.dflashWarningAcknowledged) ?? []
            guard !acknowledged.contains(signature) else { return }
            dflashWarning = DflashRuntimeWarning(
                modelPath: settings.modelPath,
                usedGB: peak.usedMB / 1024,
                totalGB: peak.totalMB / 1024,
                fraction: peak.fraction)
        }
    }

    private func dflashWarningSignature(settings: ServerSettings) -> String {
        let ngld = settings.arguments.firstIndex(of: "-ngld").flatMap {
            settings.arguments.indices.contains($0 + 1) ? settings.arguments[$0 + 1] : nil
        } ?? "none"
        let gpus = settings.selectedGPUIndices.sorted().map(String.init).joined(separator: ",")
        return "\(settings.modelPath)|\(settings.ctx)|\(settings.ncmoe)|\(settings.cacheTypeK)|\(settings.cacheTypeV)|\(ngld)|gpus=\(gpus)"
    }

    func acknowledgeDflashWarning() {
        guard let settings = launchedSettings else { dflashWarning = nil; return }
        let signature = dflashWarningSignature(settings: settings)
        var acknowledged = UserDefaults.standard.stringArray(
            forKey: SettingsKeys.dflashWarningAcknowledged) ?? []
        if !acknowledged.contains(signature) { acknowledged.append(signature) }
        UserDefaults.standard.set(acknowledged, forKey: SettingsKeys.dflashWarningAcknowledged)
        dflashWarning = nil
    }

    func useAutomaticDflashAndRestart() {
        guard let settings = launchedSettings else { dflashWarning = nil; return }
        ServerSettings.setDflashMode(.auto, forModel: settings.modelPath)
        dflashWarning = nil
        restart(effectiveSettings())
    }

    func disableDflashAndRestart() {
        guard let settings = launchedSettings else { dflashWarning = nil; return }
        ServerSettings.setDflashMode(.off, forModel: settings.modelPath)
        dflashWarning = nil
        restart(effectiveSettings())
    }

    private func startDiscoveryIfNeeded(port: Int) {
        guard discoveryEnabled else { return }
        stopDiscovery()
        let service = NetService(domain: "local.", type: "_http._tcp.", name: "ToshLLM API", port: Int32(port))
        let txt: [String: Data] = [
            "path": Data("/v1".utf8),
            "protocol": Data("openai-compatible".utf8),
            "auth": Data((UserDefaults.standard.bool(forKey: SettingsKeys.apiKeyEnabled) ? "bearer" : "none").utf8),
        ]
        service.setTXTRecord(NetService.data(fromTXTRecord: txt))
        service.publish()
        discoveryService = service
    }

    private func stopDiscovery() {
        discoveryService?.stop()
        discoveryService = nil
    }

    private func consume(_ text: String) {
        logBuffer.append(text)
        fileLog.append(text)

        for line in text.split(separator: "\n") {
            if line.contains("draft acceptance ="),
               let m = line.range(of: #"draft acceptance = ([0-9]+\.[0-9]+)"#, options: .regularExpression) {
                dflashAcceptance = Double(line[m].split(separator: "=")[1].trimmingCharacters(in: .whitespaces))
            }
            guard line.contains("tokens per second"), line.contains("eval time") else { continue }
            guard let match = line.range(of: #"([0-9]+\.[0-9]+) tokens per second"#, options: .regularExpression) else { continue }
            let value = Double(line[match].split(separator: " ")[0]) ?? 0
            if line.contains("prompt eval") {
                promptSpeed = value
            } else {
                genSpeed = value
                genHistory.append(value)
                if genHistory.count > 60 { genHistory.removeFirst() }
                requestCount += 1
            }
        }
    }
}
