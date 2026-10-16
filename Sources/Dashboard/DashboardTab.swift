// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

// MARK: - Home

struct DashboardView: View {
    @EnvironmentObject var server: ServerController
    @EnvironmentObject var control: ControlPanelState
    @State private var showHardware = false
    @EnvironmentObject var manager: ServerManager
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var profileStore: ProfileStore
    @AppStorage(SettingsKeys.modelPath) private var modelPath = ""
    @AppStorage(SettingsKeys.ncmoe) private var ncmoe = 0
    @AppStorage(SettingsKeys.port) private var port = 8080
    @AppStorage(SettingsKeys.ctx) private var ctx = 16384
    @AppStorage(SettingsKeys.ubatch) private var ubatch = 0
    @AppStorage(SettingsKeys.localNetworkDiscovery) private var localNetworkDiscovery = false
    @AppStorage(SettingsKeys.apiKeyEnabled) private var apiKeyEnabled = false
    @AppStorage(SettingsKeys.gpuIndex) private var gpuIndex = -1
    @AppStorage(SettingsKeys.gpuList) private var gpuListCSV = ""
    @AppStorage(SettingsKeys.embeddings) private var embeddings = false
    @AppStorage(SettingsKeys.uiMcpProxy) private var uiMcpProxy = false
    @AppStorage(SettingsKeys.extraArgs) private var extraArgs = ""
    @State private var showNotes = false
    @AppStorage(SettingsKeys.routerMode) private var routerMode = false
    @AppStorage(SettingsKeys.routerModelsMax) private var routerModelsMax = 1

    @EnvironmentObject var updates: UpdateChecker

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if control.serverAnchor == nil {
                    updateBanner
                    HStack {
                        Label(loc.t("Tu equipo", "Your machine"), systemImage: "desktopcomputer")
                            .font(.system(size: 17, weight: .semibold))
                        Spacer()
                        Button { showHardware.toggle() } label: {
                            Label(loc.t("Detalles del hardware", "Hardware details"), systemImage: "info.circle")
                        }
                        .buttonStyle(.plain).foregroundStyle(.secondary)
                        .popover(isPresented: $showHardware) {
                            ScrollView {
                                VStack(spacing: 14) {
                                    MachineCard()
                                    // The dashboard already shows the GPU card on a
                                    // multi-GPU machine; repeating it here reads as a bug.
                                    if hardware.splitEligibleGPUs.count <= 1 { GPUsCard() }
                                }
                                .padding(20)
                            }
                            .frame(width: 600, height: 480)
                        }
                    }
                    DashboardMetricsView()
                    DashboardMultiGPUOverview()
                    }
                    if let selectedID = control.serverAnchor,
                       let selected = manager.servers.first(where: { $0.id == selectedID }) {
                        ServerDetailView(server: selected)
                            .id(selected.id)
                    } else {
                        ForEach(manager.servers, id: \.id) { instance in
                            ServerOverviewView(server: instance, configure: {
                                control.focusServer(instance.id)
                            }, onDelete: instance.profile == nil ? nil : {
                                manager.removeServer(instance.id)
                            })
                            .id(instance.id)
                        }
                    }
                }
                .padding(24)
                .frame(maxWidth: .infinity)
                .frame(maxWidth: .infinity)
            }
            .background(WorkspaceStyle.canvas)
            .onChange(of: manager.servers.map(\.id)) { _, ids in
                if let selected = control.serverAnchor, !ids.contains(selected) { control.serverAnchor = nil }
            }
            .onChange(of: control.serverNavigationID) { _, _ in
                if let anchor = control.serverAnchor { proxy.scrollTo(anchor, anchor: .top) }
            }
            .onAppear {
                if let anchor = control.serverAnchor { proxy.scrollTo(anchor, anchor: .top) }
            }
        }
    }

    @ViewBuilder
    private var updateBanner: some View {
        if let version = updates.latestVersion {
            HStack {
                Label(loc.t("ToshLLM %@ está disponible", "ToshLLM %@ is available", "\(version)"),
                      systemImage: "arrow.down.app")
                    .fontWeight(.medium)
                Spacer()
                if let error = updates.installError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
                if updates.installing {
                    ProgressView().controlSize(.small)
                    Text(loc.t("Actualizando…", "Updating…")).font(.caption).foregroundStyle(.secondary)
                } else {
                    Button(loc.t("Notas", "Notes")) {
                        showNotes = true
                    }
                    .popover(isPresented: $showNotes, arrowEdge: .bottom) { ReleaseNotesPopover() }
                    .help(loc.t("Muestra las novedades desde tu versión hasta la más reciente.",
                                "Shows what changed from your version up to the latest one."))
                    Button(loc.t("Descargar e instalar", "Download and install")) {
                        Task { await updates.downloadAndInstall() }
                    }
                    .buttonStyle(.borderedProminent)
                    .help(loc.t("Descarga el DMG, verifica su checksum, instala la nueva versión en Aplicaciones y reinicia la app.",
                                "Downloads the DMG, verifies its checksum, installs the new version into Applications and relaunches the app."))
                }
            }
            .padding(12)
            .background(Color.appAccent.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
        }
    }


    // Networking is a launch flag, so restart the running primary to apply it now.
    private func setDiscoverable(_ on: Bool) {
        localNetworkDiscovery = on
        if server.state == .running || server.state == .starting {
            server.restart(.fromDefaults())
        }
    }

    private var profileMenu: some View {
        Menu {
            if profileStore.activeProfileName != nil {
                Button {
                    profileStore.clearActive()
                    if server.state == .running { server.stop() }
                } label: {
                    Label(loc.t("Predeterminado (sin perfil)", "Default (no profile)"),
                          systemImage: "arrow.uturn.backward")
                }
                Divider()
            }
            ForEach(profileStore.profiles) { p in
                Button {
                    profileStore.apply(p)
                    if server.state == .running { server.stop() }
                } label: {
                    Label(p.name, systemImage: profileStore.activeProfileName == p.name ? "checkmark" : "person.2")
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "person.2").font(.system(size: 10, weight: .semibold))
                Text(profileStore.activeProfileName ?? loc.t("Perfil", "Profile"))
                    .font(.caption.weight(.medium)).lineLimit(1)
                    .frame(maxWidth: 150)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold))
            }
            .padding(.horizontal, 9).padding(.vertical, 4)
            .background(Color.appAccent.opacity(0.15), in: Capsule())
            .foregroundStyle(Color.appAccent)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(loc.t("Carga un perfil guardado. Si el servidor está activo, se detiene para aplicar.",
                    "Loads a saved profile. If the server is running, it stops so the profile applies."))
    }

    private func row(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon).frame(width: 18).foregroundStyle(.secondary)
            Text(text).font(.callout).lineLimit(1)
            Spacer(minLength: 0)
        }
    }
}

private struct DashboardMultiGPUOverview: View {
    @EnvironmentObject private var vram: VRAMMonitor

    var body: some View {
        if vram.gpus.count > 1 || hardware.splitEligibleGPUs.count > 1 {
            GPUsCard()
        }
    }
}

/// Its own view so the 3s device poll it watches re-renders only this card. The GPU
/// list can grow after launch: macOS wakes the cards one by one.
struct MachineCard: View {
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var vram: VRAMMonitor
    @State private var machine = hardware
    @State private var showGPUDetail = false

    var body: some View {
        Card(title: loc.t("Tu equipo", "Your machine"), icon: "desktopcomputer", fill: true) {
            row("cpu", machine.cpuBrand
                .replacingOccurrences(of: "(R)", with: "")
                .replacingOccurrences(of: "(TM)", with: ""))
            row("square.grid.3x3",
                loc.t("%@ núcleos / %@ hilos", "%@ cores / %@ threads", "\(machine.physicalCores)", "\(machine.logicalCores)"))
            row("memorychip", String(format: "%.0f GB RAM", machine.ramGB))
            if machine.splitEligibleGPUs.count > 1 {
                HStack(spacing: 8) {
                    Image(systemName: "rectangle.on.rectangle")
                        .frame(width: 18).foregroundStyle(.secondary)
                    Text(gpuCountSummary).font(.callout).lineLimit(1)
                    Spacer(minLength: 8)
                    Button { showGPUDetail.toggle() } label: {
                        HStack(spacing: 3) {
                            Text(loc.t("Detalle", "Details"))
                            Image(systemName: "chevron.right").font(.system(size: 8, weight: .bold))
                        }
                    }
                        .buttonStyle(.plain).font(.caption.weight(.semibold))
                        .foregroundStyle(Color.accentColor)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Color.accentColor.opacity(0.12), in: Capsule())
                        .popover(isPresented: $showGPUDetail, arrowEdge: .bottom) { gpuDetailPopover }
                }
                .help(gpuListTooltip)
                row("link", peerLinkSummary)
                    .help(loc.t("Las GPUs unidas por un enlace Infinity Fabric comparten un grupo de pares en Metal, y solo así pueden copiarse activaciones directamente en vez de pasar por la RAM. La memoria del grupo es la que un modelo repartido recorre sin pasar por el sistema. macOS no lo muestra en ningún sitio: si el puente no está o no funciona, cada tarjeta aparece en su propio grupo.",
                            "GPUs joined by an Infinity Fabric link share a Metal peer group, which is what lets them copy activations directly instead of going through system RAM. The memory of a group is what a split model reaches without going through the system. macOS shows this nowhere: if the bridge is missing or not working, each card ends up in its own group."))
            } else if let gpu = machine.bestGPU {
                row("rectangle.on.rectangle", "\(gpu.name) · \(gpu.vramGB) GB VRAM")
            }
            if !machine.model.isEmpty { row("desktopcomputer", machine.model) }
            if !machine.osVersion.isEmpty { row("apple.logo", machine.osVersion) }
            row("bolt.fill", ServerSettings.isAppleSilicon
                ? loc.t("Backend: Metal (Apple Silicon)", "Backend: Metal (Apple Silicon)")
                : loc.t("Backend: Metal (build AMD parcheado)", "Backend: Metal (patched AMD build)"))
        }
        .onChange(of: vram.gpus.count) { _, _ in
            machine.gpus = ServerController.availableGPUs(rescan: true)
        }
    }

    private var gpuModelRows: [String] {
        HardwareInfo.modelGroups(of: machine.splitEligibleGPUs).map { g in
            let head = g.cards == 1 ? g.name : "\(g.cards) × \(g.name)"
            // A Duo is one card with two GPUs, so the card count alone hides half of them.
            let dies = g.gpus == g.cards ? "" : " · \(g.gpus) GPUs"
            if g.cards == 1 && g.gpus == 1 { return "\(head) · \(g.vramGB) GB" }
            return loc.t("%@%@ · %@ GB c/u", "%@%@ · %@ GB each", "\(head)", "\(dies)", "\(g.vramGB)")
        }
    }

    /// Summed from the rounded per-card figures so the panel adds up on screen.
    private var gpuCountSummary: String {
        let gpus = machine.splitEligibleGPUs
        let total = gpus.reduce(0) { $0 + $1.vramGB }
        return loc.t("%@ GPUs · %@ GB de VRAM", "%@ GPUs · %@ GB VRAM", "\(gpus.count)", "\(total)")
    }

    private var gpuDetailPopover: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(loc.t("GPUs detectadas", "Detected GPUs")).font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                ForEach(gpuModelRows, id: \.self) { Text($0).font(.callout) }
            }
            if !machine.peerGroups.isEmpty {
                Divider()
                Text(loc.t("Enlaces Infinity Fabric", "Infinity Fabric links")).font(.headline)
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(peerLinkRows, id: \.self) { Text($0).font(.callout) }
                }
            }
            Divider()
            Text(loc.t("Memoria según la reporta Metal, que no tiene por qué ser la nominal de la caja. Solo se suma cuando repartes un modelo entre varias GPUs; uno que no reparta debe caber en una sola.",
                       "Memory as Metal reports it, which need not match the number on the box. It only adds up when you split a model across several GPUs; one that is not split has to fit in a single card."))
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(width: 380, alignment: .leading)
    }

    private var gpuListTooltip: String {
        let list = machine.gpus
            .map { "\($0.name) · \($0.vramGB) GB\($0.isExternal ? " · eGPU" : "")" }
            .joined(separator: "\n")
        return loc.t("VRAM que reporta Metal, no la nominal de la caja. Solo se suma cuando repartes un modelo entre varias GPUs; un modelo que no reparta debe caber en una sola.\n\n%@", "VRAM as Metal reports it, not the number on the box. It only adds up when you split a model across several GPUs; a model that is not split has to fit in one.\n\n%@", "\(list)")
    }

    /// The memory behind a link is what a split model moves without touching RAM.
    private var peerLinkSummary: String {
        let groups = machine.peerGroups
        guard !groups.isEmpty else {
            return loc.t("Infinity Fabric: sin enlace entre tarjetas",
                         "Infinity Fabric: no link between cards")
        }
        if let only = groups.first, groups.count == 1 {
            let total = only.reduce(0) { $0 + $1.vramGB }
            return loc.t("Infinity Fabric: %@ GPUs enlazadas · %@ GB", "Infinity Fabric: %@ GPUs linked · %@ GB", "\(only.count)", "\(total)")
        }
        let totals = groups.map { String($0.reduce(0) { $0 + $1.vramGB }) }.joined(separator: " + ")
        return loc.t("Infinity Fabric: %@ enlaces · %@ GB", "Infinity Fabric: %@ links · %@ GB", "\(groups.count)", "\(totals)")
    }

    private var peerLinkRows: [String] {
        let groups = machine.peerGroups
        guard !groups.isEmpty else {
            return [loc.t("Infinity Fabric: sin enlace entre tarjetas",
                          "Infinity Fabric: no link between cards")]
        }
        let labels = GPUPeerTopology.labels(machine.gpus.map { ($0.index, $0.peerGroupID) })
        return groups.map { members in
            let total = members.reduce(0) { $0 + $1.vramGB }
            let prefix = groups.count == 1
                ? "Infinity Fabric"
                : "Fabric " + (labels[members[0].peerGroupID] ?? "?")
            let names = Set(members.map(\.name))
            let what = names.count == 1
                ? "\(members.count) × \(members[0].name)"
                : loc.t("%@ GPUs", "%@ GPUs", "\(members.count)")
            return "\(prefix): \(what) · \(total) GB"
        }
    }


    private func row(_ icon: String, _ text: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon).frame(width: 18).foregroundStyle(.secondary)
            Text(text).font(.callout).lineLimit(1)
            Spacer(minLength: 0)
        }
    }
}

/// Per-GPU VRAM bars. Owns the 3s-polling monitor so its ticks re-render only this card.
struct GPUsCard: View {
    // -1 until the user decides, so a machine with many cards starts folded.
    @AppStorage(SettingsKeys.gpusCardCollapsed) private var collapsedRaw = -1
    @AppStorage(SettingsKeys.mgpuPeer) private var mgpuPeer = true
    @AppStorage(SettingsKeys.multiGPU) private var multiGPU = false
    @AppStorage(SettingsKeys.gpuList) private var gpuListCSV = ""
    @AppStorage(SettingsKeys.splitMode) private var splitMode = "layer"
    @AppStorage(SettingsKeys.vramWarningThreshold) private var warningThreshold = 85.0
    @AppStorage(SettingsKeys.vramCriticalThreshold) private var criticalThreshold = 95.0
    @EnvironmentObject var vram: VRAMMonitor
    @EnvironmentObject var loc: Localizer

    var body: some View {
        Card(title: cardTitle, icon: "rectangle.on.rectangle", fill: true,
             collapsed: vram.gpus.count > 1 ? collapsedBinding : nil,
             trailing: { rescanButton }) {
            if vram.gpus.isEmpty {
                Label(loc.t("Sin datos de uso de GPU", "No GPU usage data"),
                      systemImage: "rectangle.on.rectangle")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                ForEach(vram.gpus) { gpuRow($0) }
            }
        }
    }

    private var rescanButton: some View {
        Button(loc.t("Volver a buscar", "Rescan"), systemImage: "arrow.clockwise") {
            vram.refreshDevices()
        }
        .buttonStyle(.plain).labelStyle(.iconOnly)
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(.secondary)
        .padding(5)
        .background(WorkspaceStyle.inset, in: Circle())
        .overlay(Circle().strokeBorder(WorkspaceStyle.border))
        .help(loc.t("Vuelve a preguntar a Metal por las tarjetas, por si has conectado o quitado una eGPU. La app ya no lo hace sola porque en algunos Mac Pro con varias tarjetas esa consulta llega a colgar el equipo.",
                    "Asks Metal for the cards again, in case you plugged or unplugged an eGPU. The app no longer does it on its own because on some multi-card Mac Pros that query can hang the machine."))
    }

    private var cardTitle: String {
        vram.gpus.count > 1
            ? loc.t("GPUs · %@", "GPUs · %@", "\(vram.gpus.count)")
            : loc.t("GPUs", "GPUs")
    }

    private var collapsedBinding: Binding<Bool> {
        Binding(get: { collapsedRaw < 0 ? vram.gpus.count > 4 : collapsedRaw == 1 },
                set: { collapsedRaw = $0 ? 1 : 0 })
    }

    private var peerLabels: [UInt64: String] {
        GPUPeerTopology.labels(vram.gpus.map { ($0.id, $0.peerGroupID) })
    }

    @ViewBuilder private func gpuRow(_ g: GPUStat) -> some View {
        let fractionPct = g.fraction * 100
        let isCritical = fractionPct >= criticalThreshold
        let isWarning = fractionPct >= warningThreshold
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                if vram.gpus.count > 1 {
                    Text("#\(g.id)")
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(.tertiary)
                }
                Text(g.name).font(.callout).lineLimit(1)
                peerBadge(for: g)
                if isCritical {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .font(.system(size: 10))
                        .help(loc.t("VRAM casi llena — cierra apps o reduce ctx",
                                    "VRAM nearly full — close apps or reduce ctx"))
                } else if isWarning {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .font(.system(size: 10))
                        .help(loc.t("Uso de VRAM alto",
                                    "High VRAM usage"))
                }
                Spacer(minLength: 8)
                Text(String(format: "%.1f / %.0f GB", g.usedMB / 1024, g.totalMB / 1024))
                    .font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
            }
            ProgressView(value: g.fraction)
                .tint(isCritical ? .red : isWarning ? .orange : .accentColor)
        }
        .padding(.leading, peerColors.isEmpty ? 0 : 11)
        // As an overlay the bar takes the height of the row; a Shape in the row
        // itself has no height of its own and stretches it instead.
        .overlay(alignment: .leading) {
            if let color = peerColors[g.peerGroupID] {
                RoundedRectangle(cornerRadius: 1.5).fill(color).frame(width: 3)
            }
        }
        .help(rowTooltip(g))
    }

    private var peerColors: [UInt64: Color] {
        let palette: [Color] = [.accentColor, .teal, .orange, .purple, .pink]
        let ids = GPUPeerTopology.groupIDs(vram.gpus.map { ($0.id, $0.peerGroupID) })
        return ids.enumerated().reduce(into: [:]) { out, pair in
            out[pair.element] = palette[pair.offset % palette.count]
        }
    }

    /// The bridge only carries anything when the split is by tensors and the toggle is on:
    /// a layer split never reduces across cards, so the link sits idle.
    private var peerInUse: Bool {
        let splitting = multiGPU || ServerSettings.gpuList(fromCSV: gpuListCSV).count >= 2
        return splitting && mgpuPeer && splitMode == "tensor"
    }

    @ViewBuilder private func peerBadge(for g: GPUStat) -> some View {
        let labels = peerLabels
        // Nothing on unlinked cards: the row tooltip spells that out.
        if let label = labels[g.peerGroupID] {
            let name = labels.count > 1 ? "Fabric \(label)" : "Fabric"
            capsule(peerInUse ? name : loc.t("%@ · sin usar", "%@ · unused", "\(name)"),
                    icon: peerInUse ? "link" : "link.badge.plus",
                    tint: peerInUse ? (peerColors[g.peerGroupID] ?? .accentColor) : .secondary)
        }
    }

    private func capsule(_ text: String, icon: String, tint: Color) -> some View {
        HStack(spacing: 3) {
            Image(systemName: icon).font(.system(size: 8, weight: .semibold))
            Text(text).font(.system(size: 10, weight: .medium))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 6).padding(.vertical, 2)
        .background(tint.opacity(0.14), in: Capsule())
    }

    private func rowTooltip(_ g: GPUStat) -> String {
        let vramUse = loc.t("VRAM en uso de %@.", "VRAM in use on %@.", "\(g.name)")
        let labels = peerLabels
        guard !labels.isEmpty else { return vramUse }
        guard let label = labels[g.peerGroupID] else {
            return vramUse + loc.t(" Sin enlace Infinity Fabric: sus copias pasan por la RAM del sistema.",
                                   " No Infinity Fabric link: its copies go through system RAM.")
        }
        let peers = vram.gpus.filter { $0.peerGroupID == g.peerGroupID }.count
        let linked = vramUse + loc.t(" Enlazada por Infinity Fabric con otras %@ GPUs (grupo %@).", " Linked by Infinity Fabric to %@ other GPUs (group %@).", "\(peers - 1)", "\(label)")
        if peerInUse {
            return linked + loc.t(" Se están copiando activaciones entre ellas sin pasar por la RAM.",
                                  " Activations are being copied between them without going through RAM.")
        }
        let splitting = multiGPU || ServerSettings.gpuList(fromCSV: gpuListCSV).count >= 2
        if !splitting {
            return linked + loc.t(" No se usa: el modelo no está repartido entre varias GPUs.",
                                  " Unused: the model is not split across several GPUs.")
        }
        if splitMode != "tensor" {
            return linked + loc.t(" No se usa: el reparto por capas no reduce entre tarjetas, así que no hay nada que copiar por el puente.",
                                  " Unused: a layer split does not reduce across cards, so there is nothing for the bridge to carry.")
        }
        return linked + loc.t(" No se usa: el puente está apagado en Ajustes.",
                              " Unused: the bridge is turned off in Settings.")
    }
}

/// Aggregate-VRAM badge for window toolbars, so it's visible outside the dashboard.
struct GPUUsageBadge: View {
    @EnvironmentObject var vram: VRAMMonitor
    @EnvironmentObject var loc: Localizer

    var body: some View {
        if vram.totalMB > 0 {
            HStack(spacing: 5) {
                Image(systemName: "memorychip").font(.caption).foregroundStyle(.secondary)
                ProgressView(value: min(vram.fraction, 1)).frame(width: 52)
                    .tint(vram.fraction > 0.9 ? .red : vram.fraction > 0.75 ? .orange : .accentColor)
                Text(String(format: "%.1f/%.0f", vram.usedMB / 1024, vram.totalMB / 1024))
                    .font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
            }
            .help(loc.t("VRAM en uso (todas las GPUs).", "VRAM in use (all GPUs)."))
        }
    }
}

/// Compact server state indicator, shared by the main and the added server cards.
struct ServerStateBadge: View {
    let state: ServerController.State
    var phase: ServerController.StartupPhase? = nil
    var since: Date? = nil
    @EnvironmentObject var loc: Localizer
    var body: some View {
        switch state {
        case .running:
            Label(loc.t("Activo", "Running"), systemImage: "circle.fill")
                .foregroundStyle(.green).font(.caption)
        case .starting:
            if let phase {
                Label(loc.t("Iniciando…", "Starting…"), systemImage: "circle.fill")
                    .foregroundStyle(.orange).font(.caption)
                    .help("\(phase.title(loc)): \(phase.help(loc))")
            } else {
                Label(loc.t("Iniciando…", "Starting…"), systemImage: "circle.fill")
                    .foregroundStyle(.orange).font(.caption)
            }
        case .failed(let msg):
            Label(loc.t("Error", "Error"), systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red).font(.caption)
                .help(loc.half(msg))
        case .stopped:
            Label(loc.t("Detenido", "Stopped"), systemImage: "circle")
                .foregroundStyle(.secondary).font(.caption)
        }
    }

}

extension ServerController.StartupPhase {
    /// Fits under the state in the server header, which must keep its width.
    func shortTitle(_ loc: Localizer) -> String {
        switch self {
        case .planning:       return loc.t("Planificando", "Planning")
        case .loadingWeights: return loc.t("Cargando pesos", "Loading weights")
        case .lockingMemory:  return loc.t("Reservando RAM", "Reserving RAM")
        case .fillingCache:   return loc.t("Preparando", "Preparing")
        }
    }

    func title(_ loc: Localizer) -> String {
        switch self {
        case .planning:       return loc.t("Planificando memoria", "Planning memory")
        case .loadingWeights: return loc.t("Cargando pesos", "Loading weights")
        case .lockingMemory:  return loc.t("Reservando RAM", "Reserving RAM")
        case .fillingCache:   return loc.t("Preparando expertos", "Preparing experts")
        }
    }

    func help(_ loc: Localizer) -> String {
        switch self {
        case .planning:
            return loc.t("El motor mide cuánta VRAM y RAM hay libres para decidir cómo repartir el modelo.",
                         "The engine measures free VRAM and RAM to decide how to split the model.")
        case .loadingWeights:
            return loc.t("Leyendo el modelo del disco a la GPU.", "Reading the model from disk into the GPU.")
        case .lockingMemory:
            return loc.t("Dynamic MoE reserva RAM para los expertos. Mientras tanto el Mac puede ir más lento.",
                         "Dynamic MoE is reserving RAM for the experts. The Mac may feel slower meanwhile.")
        case .fillingCache:
            return loc.t("Cargando en RAM los expertos más usados antes de la primera respuesta.",
                         "Loading the most used experts into RAM before the first answer.")
        }
    }
}

struct Card<Content: View, Trailing: View>: View {
    let title: String
    let icon: String
    var fill: Bool = false
    var collapsed: Binding<Bool>?
    @ViewBuilder let trailing: Trailing
    @ViewBuilder let content: Content

    init(title: String, icon: String, fill: Bool = false, collapsed: Binding<Bool>? = nil,
         @ViewBuilder trailing: () -> Trailing = { EmptyView() },
         @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.fill = fill
        self.collapsed = collapsed
        self.trailing = trailing()
        self.content = content()
    }

    private var isCollapsed: Bool { collapsed?.wrappedValue ?? false }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 8) {
                Label(title, systemImage: icon)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)
                if collapsed != nil {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isCollapsed ? 0 : 90))
                }
                if Trailing.self != EmptyView.self {
                    Spacer(minLength: 8)
                    trailing
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
            .onTapGesture {
                guard let collapsed else { return }
                withAnimation(.easeInOut(duration: 0.18)) { collapsed.wrappedValue.toggle() }
            }
            if !isCollapsed { content }
        }
        .padding(14)
        .frame(maxWidth: .infinity, maxHeight: fill && !isCollapsed ? .infinity : nil, alignment: .topLeading)
        .cardSurface()
    }
}
