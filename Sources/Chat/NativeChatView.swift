// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import AppKit
import PDFKit
import Vision
import UniformTypeIdentifiers
import AVFoundation

// MARK: - Main chat view

/// How far the conversation's end sits past the bottom of the view, in points.
private struct EndOffsetKey: PreferenceKey {
    static let defaultValue: CGFloat? = nil
    static func reduce(value: inout CGFloat?, nextValue: () -> CGFloat?) { value = nextValue() ?? value }
}

/// When the reader last turned the wheel. A reference type so wheel events do
/// not re-render the transcript, and not @Published for the same reason.
final class ScrollGestureClock {
    var last = Date.distantPast
    var isRecent: Bool { Date().timeIntervalSince(last) < 0.4 }
}

private struct ChatComposerTextEditor: NSViewRepresentable {
    @Binding var text: String
    var isFocused: FocusState<Bool>.Binding
    let fontSize: CGFloat
    let onSubmit: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true

        let textView = ComposerTextView()
        textView.delegate = context.coordinator
        textView.onSubmit = onSubmit
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.backgroundColor = .clear
        textView.textColor = .labelColor
        textView.insertionPointColor = .controlAccentColor
        textView.font = .systemFont(ofSize: fontSize)
        textView.textContainerInset = NSSize(width: 5, height: 7)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.autoresizingMask = [.width]
        textView.string = text
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? ComposerTextView else { return }
        context.coordinator.parent = self
        textView.onSubmit = onSubmit
        textView.font = .systemFont(ofSize: fontSize)
        textView.textColor = .labelColor
        textView.insertionPointColor = .controlAccentColor
        // Assigning the string ends an input method's composition, even with the same value.
        if !textView.hasMarkedText(), textView.string != text {
            textView.string = text
            textView.setSelectedRange(NSRange(location: (text as NSString).length, length: 0))
        }
        if isFocused.wrappedValue,
           textView.window?.firstResponder !== textView {
            Task { @MainActor [weak textView] in
                guard let textView, textView.window?.firstResponder !== textView else { return }
                textView.window?.makeFirstResponder(textView)
            }
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ChatComposerTextEditor

        init(parent: ChatComposerTextEditor) {
            self.parent = parent
        }

        func textDidBeginEditing(_ notification: Notification) {
            parent.isFocused.wrappedValue = true
        }

        func textDidEndEditing(_ notification: Notification) {
            parent.isFocused.wrappedValue = false
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }
    }

    final class ComposerTextView: NSTextView {
        var onSubmit: (() -> Void)?

        override func keyDown(with event: NSEvent) {
            let isReturn = event.keyCode == 36 || event.keyCode == 76
            guard isReturn, !hasMarkedText() else {
                super.keyDown(with: event)
                return
            }

            if event.modifierFlags.contains(.shift) || event.modifierFlags.contains(.option) {
                insertNewline(nil)
            } else {
                onSubmit?()
            }
        }
    }
}

struct NativeChatView: View {
    @Environment(\.openWindow) private var openWindow
    @Environment(\.chatFontScale) private var chatFontScale
    @EnvironmentObject var server: ServerController
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var chat: ChatStore
    @EnvironmentObject var models: ModelStore
    @EnvironmentObject var control: ControlPanelState
    @AppStorage(SettingsKeys.chatTemp) private var temperature = 0.7
    @AppStorage(SettingsKeys.chatMaxTokens) private var maxTokens = 2048
    @AppStorage(SettingsKeys.chatSystem) private var systemPrompt = ""
    @AppStorage(SettingsKeys.chatThinking) private var thinkingEnabled = true
    @AppStorage(SettingsKeys.chatReasoningEffort) private var reasoningEffort = "medium"
    @AppStorage(SettingsKeys.chatTopP) private var topP = 0.95
    @AppStorage(SettingsKeys.chatMinP) private var minP = 0.05
    @AppStorage(SettingsKeys.chatTopK) private var topK = 40
    @AppStorage(SettingsKeys.chatRepeatPenalty) private var repeatPenalty = 1.0
    @AppStorage(SettingsKeys.chatRepeatLastN) private var repeatLastN = 64
    @AppStorage(SettingsKeys.chatSeed) private var seed = -1
    @AppStorage(SettingsKeys.chatDynatempRange) private var dynatempRange = 0.0
    @AppStorage(SettingsKeys.chatDynatempExponent) private var dynatempExponent = 1.0
    @AppStorage(SettingsKeys.chatXTCProbability) private var xtcProbability = 0.0
    @AppStorage(SettingsKeys.chatXTCThreshold) private var xtcThreshold = 0.1
    @AppStorage(SettingsKeys.chatTypicalP) private var typicalP = 1.0
    @AppStorage(SettingsKeys.chatPresencePenalty) private var presencePenalty = 0.0
    @AppStorage(SettingsKeys.chatFrequencyPenalty) private var frequencyPenalty = 0.0
    @AppStorage(SettingsKeys.chatDryMultiplier) private var dryMultiplier = 0.0
    @AppStorage(SettingsKeys.chatDryBase) private var dryBase = 1.75
    @AppStorage(SettingsKeys.chatDryAllowedLength) private var dryAllowedLength = 2
    @AppStorage(SettingsKeys.chatDryPenaltyLastN) private var dryPenaltyLastN = 0
    @AppStorage(SettingsKeys.chatSamplers) private var samplers = ""
    @AppStorage(SettingsKeys.chatBackendSampling) private var backendSampling = false
    @AppStorage(SettingsKeys.chatCustomJSON) private var customJSON = ""
    @AppStorage(SettingsKeys.chatAgenticMaxTurns) private var agenticMaxTurns = 10
    @AppStorage(SettingsKeys.chatPasteLongTextLength) private var pasteLongTextLength = 2500
    @AppStorage(SettingsKeys.chatMaxImageMegapixels) private var maxImageMegapixels = 1.0
    @AppStorage(SettingsKeys.chatPDFAsImages) private var pdfAsImages = false
    @AppStorage(SettingsKeys.chatShowSystemMessage) private var showSystemMessage = true
    @AppStorage(SettingsKeys.port) private var port = 8080
    @AppStorage(SettingsKeys.ctx) private var contextLimit = 16384
    @AppStorage(SettingsKeys.routerMode) private var routerMode = false
    @AppStorage(SettingsKeys.chatSelectedModel) private var chatSelectedModel = ""
    @State private var draft = ""
    @State private var attachments: [ChatAttachment] = []
    @State private var images: [String] = []   // attached images as data URIs (vision models)
    @State private var attachError: String?
    @State private var ocrPending = 0
    @State private var transcriptionPending = 0
    @State private var transcriptionQueue: [PendingSpeechTranscription] = []
    @AppStorage(SettingsKeys.modelPath) private var modelPath = ""
    @AppStorage(SettingsKeys.gpuIndex) private var gpuIndex = -1
    @AppStorage(SettingsKeys.whisperModel) private var whisperModelID = WhisperModel.recommendedID
    @AppStorage(SettingsKeys.speechInputMethod) private var speechInputMethodRaw = ""
    @AppStorage(SettingsKeys.whisperLoadPolicy) private var whisperLoadPolicyRaw = WhisperLoadPolicy.onDemand.rawValue
    @State private var showSystem = false
    @State private var promptConversation: Conversation?
    @State private var promptProject: ChatProject?
    @State private var headerTitle = ""
    @AppStorage(SettingsKeys.appAccent) private var accentRaw = AppTheme.defaultKey
    @AppStorage(SettingsKeys.agentToolsEnabled) private var agentToolsEnabled = false
    // True while the conversation's end is on screen (inverted scroll rests here).
    @State private var atBottom = true
    // Set when the reader scrolls away mid-answer; the bubble renders it until
    // they come back, so the transcript stops moving under them.
    @State private var frozenStream: StreamSnapshot?
    @FocusState private var inputFocused: Bool
    @State private var pasteMonitor: Any?
    @State private var scrollMonitor: Any?
    @State private var scrollGesture = ScrollGestureClock()
    @State private var draftOwnerID: UUID?
    @State private var draftSaveTask: Task<Void, Never>?
    @State private var showMCPBrowser = false
    @State private var showAttachments = false
    @State private var showTools = false
    @State private var showVoiceOptions = false
    @StateObject private var audioRecorder = AudioRecorderController()
    @StateObject private var dictation = SpeechDictationController.shared
    @StateObject private var appleDictation = AppleSpeechDictationController.shared
    @State private var dictationBase = ""
    @State private var previewAttachment: ChatAttachment?
    @State private var availableTools: [BuiltinToolInfo] = []
    @State private var loadingTools = false
    @State private var forkMessage: ChatMessage?
    @State private var modelModalities: ModelModalities?
    @State private var loadingModalities = false
    @State private var capabilitiesAreComplete = false

    private var maxTokenOptions: [Int] {
        [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072].filter { $0 <= contextLimit }
    }

    private var whisperLoadPolicy: WhisperLoadPolicy {
        WhisperLoadPolicy(rawValue: whisperLoadPolicyRaw) ?? .onDemand
    }

    private var whisperResidencyTaskID: String {
        let model = WhisperModel.model(id: whisperModelID)
        return "\(whisperLoadPolicyRaw)-\(whisperModelID)-\(gpuIndex)-\(models.whisperModelInstalled(model))-\(String(describing: server.state))"
    }

    private var maxTokensIsLarge: Bool {
        contextLimit > 0 && Double(maxTokens) / Double(contextLimit) > 0.5
    }

    /// Levels the loaded model's template validates; the usual four when it
    /// validates nothing or the template could not be read.
    private var reasoningEffortLevels: [String] {
        let detected = modelModalities?.reasoning?.levels ?? []
        return detected.isEmpty ? ["low", "medium", "high", "max"] : detected
    }

    private func reasoningEffortIsSupported(_ effort: String) -> Bool {
        effort == "off" || effort == "default" || reasoningEffortLevels.contains(effort)
    }

    /// A level the model would reject never reaches the request: the template
    /// raises and the server answers HTTP 500.
    private func supportedReasoningEffort(_ effort: String) -> String {
        if reasoningEffortIsSupported(effort) { return effort }
        return ReasoningEffortDetector.closest(to: effort, in: reasoningEffortLevels) ?? "default"
    }

    private var effectiveReasoningEffort: String { supportedReasoningEffort(reasoningEffort) }

    private var modelDefaultEffortLabel: String {
        let label = loc.t("Predeterminado del modelo", "Model default")
        guard let value = modelModalities?.reasoning?.modelDefault, !value.isEmpty else { return label }
        return label + " · " + value
    }

    private func reasoningEffortLabel(_ level: String) -> String {
        switch level {
        case "none", "no_think": loc.t("Sin razonamiento", "No reasoning")
        case "minimal": loc.t("Mínimo · sin presupuesto", "Minimal · no budget")
        case "low": loc.t("Bajo · 512 tokens", "Low · 512 tokens")
        case "medium": loc.t("Medio · 2.048 tokens", "Medium · 2,048 tokens")
        case "high": loc.t("Alto · 8.192 tokens", "High · 8,192 tokens")
        case "xhigh": loc.t("Muy alto · sin presupuesto", "Very high · no budget")
        case "max": loc.t("Máximo · sin presupuesto", "Maximum · no budget")
        default: level
        }
    }

    private var samplingSettings: ChatSamplingSettings {
        ChatSamplingSettings(reasoningEffort: effectiveReasoningEffort,
                             topP: topP, minP: minP, topK: topK,
                             repeatPenalty: repeatPenalty, repeatLastN: repeatLastN, seed: seed,
                             dynatempRange: dynatempRange, dynatempExponent: dynatempExponent,
                             xtcProbability: xtcProbability, xtcThreshold: xtcThreshold,
                             typicalP: typicalP, presencePenalty: presencePenalty,
                             frequencyPenalty: frequencyPenalty, dryMultiplier: dryMultiplier,
                             dryBase: dryBase, dryAllowedLength: dryAllowedLength,
                             dryPenaltyLastN: dryPenaltyLastN, samplers: samplers,
                             backendSampling: backendSampling, customJSON: customJSON)
    }

    private var capabilityTaskID: String {
        "\(String(describing: server.state))-\(port)-\(routerMode ? chatSelectedModel : modelPath)-\(chat.generating)"
    }

    private var contextMaySlowGeneration: Bool {
        guard !ServerSettings.isAppleSilicon, let used = chat.contextUsed, contextLimit > 0 else {
            return false
        }
        return used >= 2560 || Double(used) / Double(contextLimit) >= 0.15
    }

    var body: some View {
        chatColumn
            .onAppear {
                inputFocused = true
                loadDraft(for: chat.currentID)
                pasteMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                    guard inputFocused,
                          event.modifierFlags.contains(.command),
                          event.charactersIgnoringModifiers?.lowercased() == "v",
                          clipboardHasAttachables else { return event }
                    pasteFromClipboard()
                    return nil
                }
                scrollMonitor = NSEvent.addLocalMonitorForEvents(
                    matching: [.scrollWheel, .leftMouseDragged]) { event in
                    scrollGesture.last = Date()
                    return event
                }
            }
            .onDisappear {
                saveDraftNow()
                draftSaveTask?.cancel()
                if let pasteMonitor { NSEvent.removeMonitor(pasteMonitor) }
                pasteMonitor = nil
                if let scrollMonitor { NSEvent.removeMonitor(scrollMonitor) }
                scrollMonitor = nil
            }
            .onChange(of: chat.currentID) { oldID, newID in
                saveDraftNow(for: oldID)
                loadDraft(for: newID)
            }
            .onChange(of: server.state) { _, state in
                switch state {
                case .stopped, .failed:
                    stopSpeechSubsystem()
                case .starting, .running:
                    break
                }
            }
            .task(id: "\(chat.currentID?.uuidString ?? "")-\(String(describing: server.state))") {
                await refreshAvailableTools()
            }
            .task(id: capabilityTaskID) {
                await refreshModelModalities()
            }
            .task(id: whisperResidencyTaskID) {
                configureWhisperResidency()
            }
            .sheet(item: $promptConversation) { c in
                PromptEditorSheet(
                    title: loc.t("Prompt de esta conversación", "This conversation's prompt"),
                    hint: loc.t("Sustituye al prompt del proyecto y al global solo en esta conversación. Vacío = heredar.",
                                "Overrides the project and global prompts for this conversation only. Empty = inherit."),
                    initial: c.systemPrompt ?? ""
                ) { chat.setConversationPrompt(c, $0) }
            }
            .sheet(item: $promptProject) { p in
                PromptEditorSheet(
                    title: loc.t("Prompt del proyecto \"%@\"", "Project prompt for \"%@\"", "\(p.name)"),
                    hint: loc.t("Lo heredan todas las conversaciones del proyecto que no tengan prompt propio.",
                                "Inherited by every conversation in the project without its own prompt."),
                    initial: p.systemPrompt
                ) { chat.setProjectPrompt(p, $0) }
            }
            .sheet(isPresented: $showMCPBrowser) {
                MCPBrowserView(
                    addAttachment: { attachment in
                        if !attachments.contains(where: { $0.name == attachment.name && $0.content == attachment.content }) {
                            attachments.append(attachment)
                        }
                    },
                    insertPrompt: { value in
                        draft = [draft, value].filter { !$0.isEmpty }.joined(separator: "\n\n")
                    })
                    .environmentObject(loc)
            }
            .sheet(item: $previewAttachment) { attachment in
                MediaAttachmentPreview(attachment: attachment)
            }
            .sheet(item: $forkMessage) { message in
                ForkConversationSheet(sourceTitle: chat.current.map(chat.displayTitle) ?? "") {
                    title, includeAttachments in
                    chat.forkConversation(at: message.id, title: title,
                                          includeAttachments: includeAttachments)
                }
                .environmentObject(loc)
            }
            .environment(\.openURL, OpenURLAction { url in
                let raw = url.scheme == nil ? "https://\(url.absoluteString)" : url.absoluteString
                guard let target = URL(string: raw), let scheme = target.scheme?.lowercased(),
                      scheme == "http" || scheme == "https" || scheme == "mailto",
                      target.host != nil || scheme == "mailto" else { return .discarded }
                NSWorkspace.shared.open(target)
                return .handled
            })
    }

    // MARK: messages column

    private var chatColumn: some View {
        VStack(spacing: 0) {
            conversationHeader
            messagesScroll
            inputArea
        }
    }

    private func directoryIsMissing(_ path: String?) -> Bool {
        guard let path, !path.isEmpty else { return false }
        var isDir: ObjCBool = false
        return !(FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue)
    }

    /// Shows the folder the tools are actually using, own or inherited, so it is
    /// readable without opening the chat parameters. Clicking it changes the chat's.
    @ViewBuilder private var workingDirectoryChip: some View {
        let own = chat.current?.workingDirectory
        let inherited = chat.current.flatMap { chat.project(id: $0.projectID)?.workingDirectory }
        let active = own ?? inherited
        let missing = directoryIsMissing(active)
        if agentToolsEnabled || active != nil {
            Menu {
                if let active {
                    Text(active)
                    if missing {
                        Text(loc.t("La carpeta ya no existe", "That folder no longer exists"))
                    }
                    Divider()
                }
                Button(loc.t("Elegir carpeta de este chat…", "Pick this chat's folder…")) {
                    guard let c = chat.current else { return }
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    panel.allowsMultipleSelection = false
                    if panel.runModal() == .OK, let path = panel.url?.path {
                        chat.setConversationWorkingDirectory(c, path)
                    }
                }
                if own != nil {
                    Button(inherited == nil
                           ? loc.t("Quitar carpeta", "Clear folder")
                           : loc.t("Heredar la del proyecto", "Inherit the project's")) {
                        if let c = chat.current { chat.setConversationWorkingDirectory(c, nil) }
                    }
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: missing ? "folder.badge.questionmark"
                          : active == nil ? "folder.badge.questionmark" : "folder.fill")
                        .font(.system(size: 10, weight: .medium))
                    Text(active.map { URL(fileURLWithPath: $0).lastPathComponent }
                         ?? loc.t("Sin carpeta", "No folder"))
                        .lineLimit(1)
                }
                .font(.caption.weight(.medium))
                .foregroundStyle(missing ? AnyShapeStyle(.red)
                                 : own == nil && active != nil ? AnyShapeStyle(.tertiary)
                                 : AnyShapeStyle(.secondary))
                .padding(.horizontal, 10).padding(.vertical, 4)
                .glassSurface(in: Capsule(), interactive: true)
                .overlay(Capsule().strokeBorder(.primary.opacity(0.07)))
                .contentShape(Capsule())
            }
            .menuStyle(.button).buttonStyle(.plain)
            .menuIndicator(.hidden).fixedSize()
            .tint(.secondary)
            .help(missing
                  ? loc.t("La carpeta ya no existe: las herramientas fallarán hasta que elijas otra.",
                          "That folder no longer exists: the tools will fail until you pick another.")
                  : own != nil
                  ? loc.t("Carpeta de esta conversación para las herramientas.",
                          "This conversation's folder for the tools.")
                  : active != nil
                    ? loc.t("Carpeta heredada del proyecto. Haz clic para fijar una propia.",
                            "Folder inherited from the project. Click to pin its own.")
                    : loc.t("Sin carpeta para las herramientas. Haz clic para elegir una.",
                            "No folder for the tools. Click to pick one."))
        }
    }

    private var conversationHeader: some View {
        HStack(spacing: 8) {
            if let p = chat.project(id: chat.current?.projectID) {
                Menu {
                    Button(loc.t("Quitar del proyecto", "Remove from project")) {
                        if let c = chat.current { chat.move(c, toProject: nil) }
                    }
                    Button(loc.t("Prompt del proyecto…", "Project prompt…")) { promptProject = p }
                    Button(p.workingDirectory == nil
                           ? loc.t("Carpeta del proyecto…", "Project folder…")
                           : loc.t("Cambiar carpeta del proyecto…", "Change project folder…")) {
                        chat.pickProjectWorkingDirectory(p)
                    }
                    if p.workingDirectory != nil {
                        Button(loc.t("Quitar carpeta del proyecto", "Clear project folder")) {
                            chat.setProjectWorkingDirectory(p, nil)
                        }
                    }
                } label: {
                    // Chrome lives inside the label so the menu's hit area is the whole chip.
                    HStack(spacing: 5) {
                        Image(systemName: "folder")
                            .font(.system(size: 10, weight: .medium))
                        Text(p.name).lineLimit(1)
                        Image(systemName: "chevron.down")
                            .font(.system(size: 8, weight: .semibold)).opacity(0.7)
                    }
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12).padding(.vertical, 5)
                    .glassSurface(in: Capsule(), interactive: true)
                    .overlay(Capsule().strokeBorder(.primary.opacity(0.07)))
                    .contentShape(Capsule())
                }
                .menuStyle(.button).buttonStyle(.plain)
                .menuIndicator(.hidden).fixedSize()
                .tint(.secondary)
                .help(loc.t("Proyecto de esta conversación.", "This conversation's project."))
            }
            TextField(loc.t("Título de la conversación", "Conversation title"),
                      text: $headerTitle)
                .textFieldStyle(.plain)
                .font(.callout.weight(.medium))
                .onSubmit {
                    if let c = chat.current { chat.rename(c, to: headerTitle) }
                }
                .task(id: chat.currentID) { syncHeaderTitle() }
                .onChange(of: chat.current?.title) { syncHeaderTitle() }
                .help(loc.t("Haz clic para renombrar la conversación (Enter guarda).",
                            "Click to rename the conversation (Enter saves)."))
            Spacer()
            workingDirectoryChip
            if let branches = chat.current?.branches, branches.count > 1,
               let position = chat.currentBranchPosition {
                Menu {
                    ForEach(Array(branches.enumerated()), id: \.element.id) { index, branch in
                        Button {
                            chat.switchBranch(branch.id)
                        } label: {
                            if branch.id == chat.current?.activeBranchID {
                                Label(loc.t("Rama %@", "Branch %@", "\(index + 1)"),
                                      systemImage: "checkmark")
                            } else {
                                Text(loc.t("Rama %@", "Branch %@", "\(index + 1)"))
                            }
                        }
                    }
                } label: {
                    Label("\(position.index)/\(position.count)", systemImage: "arrow.triangle.branch")
                        .font(.caption)
                }
                .menuStyle(.button)
                .help(loc.t("Cambiar entre respuestas y ediciones alternativas sin salir del chat.",
                            "Switch between alternate responses and edits without leaving the chat."))
            }
            if !(chat.current?.systemPrompt ?? "").isEmpty {
                Button {
                    promptConversation = chat.current
                } label: {
                    Image(systemName: "text.bubble.fill").font(.caption)
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .accessibilityLabel(loc.t("Prompt de esta conversación", "This conversation's prompt"))
                .help(loc.t("Esta conversación tiene prompt propio; clic para editarlo.",
                            "This conversation has its own prompt; click to edit it."))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
    }

    /// Header shows the stored title; a brand-new empty chat starts blank
    /// instead of the "…" placeholder so typing sets a real title.
    private func syncHeaderTitle() {
        guard let c = chat.current else { headerTitle = ""; return }
        headerTitle = c.title.isEmpty
            ? (c.messages.isEmpty ? "" : chat.displayTitle(c))
            : c.title
    }

    /// First message not covered by the compaction summary; the transcript
    /// shows a marker above it.
    private var compactionBoundaryID: UUID? {
        guard let c = chat.current, c.summary != nil,
              let n = c.summarizedCount, n > 0, n < c.messages.count else { return nil }
        return c.messages[n].id
    }

    private var messagesScroll: some View {
        // Tool results are shown inside the assistant's tool-call card, so the
        // separate tool-role message is display-only noise and is hidden here.
        let messages = (chat.current?.messages ?? []).filter { $0.role != "tool" }
        let newestID = messages.last?.id
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 0) {
                    endProbe
                    transcript(messages, newestID: newestID)
                }
            }
            .coordinateSpace(name: Self.scrollSpace)
            .flippedUpsideDown()
            .overlay {
                if messages.isEmpty { emptyChatState }
            }
            .overlay(alignment: .bottomTrailing) {
                if !atBottom {
                    jumpToBottomButton(writingBehind: frozenStream != nil) { followEnd(proxy) }
                }
            }
            // Only a real wheel gesture may change this: layout settling after a
            // turn ends also moves the probe, and that must not scroll anyone.
            .onPreferenceChange(EndOffsetKey.self) { offset in
                guard let offset, scrollGesture.isRecent else { return }
                if atBottom {
                    if offset < -Self.leaveSlack { atBottom = false; freezeStream() }
                } else if offset > -Self.returnSlack {
                    atBottom = true
                    followEnd(proxy, animated: false)
                }
            }
            .onChange(of: chat.currentID) { _, _ in
                followEnd(proxy, animated: false)
                inputFocused = true
            }
            // A new turn (send/regenerate) jumps to the bottom, unless the reader
            // stepped away: then it keeps writing off screen.
            .onChange(of: chat.current?.messages.last?.id) { _, _ in
                if atBottom { followEnd(proxy, animated: false) } else { freezeStream() }
            }
            .onChange(of: chat.generating) { _, generating in
                if generating { if !atBottom { freezeStream() } } else { frozenStream = nil }
            }
        }
    }

    private var endProbe: some View {
        Color.clear.frame(height: 1).id(Self.bottomID)
            .background {
                GeometryReader { g in
                    // Quantized: sub-pixel changes would fire on every frame.
                    let y = g.frame(in: .named(Self.scrollSpace)).minY
                    Color.clear.preference(key: EndOffsetKey.self, value: (y / 4).rounded() * 4)
                }
            }
    }

    @ViewBuilder
    private func transcript(_ messages: [ChatMessage], newestID: UUID?) -> some View {
        LazyVStack(spacing: 14) {
            if chat.pendingAgentContinuation?.conversationID == chat.currentID {
                AgentContinuationCard(
                    continueAction: { chat.respondToAgentContinuation(true) },
                    stopAction: { chat.respondToAgentContinuation(false) })
                    .environmentObject(loc)
                    .flippedUpsideDown()
            }
            if let request = chat.pendingToolPermission,
               request.conversationID == chat.currentID {
                ToolPermissionCard(request: request) { chat.respondToToolPermission($0) }
                    .environmentObject(loc)
                    .flippedUpsideDown()
            }
            if let err = chat.lastError {
                Label(err, systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.red)
                    .flippedUpsideDown()
            }
            ForEach(messages.reversed()) { msg in
                messageRow(msg, isNewest: msg.id == newestID)
                    .flippedUpsideDown()
                    .id(msg.id)
            }
            if showSystemMessage {
                let prompt = chat.effectiveSystemPrompt(global: systemPrompt)
                if !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    SystemPromptCard(prompt: prompt) { promptConversation = chat.current }
                        .flippedUpsideDown()
                }
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 16)
        .frame(maxWidth: 1120)
        .frame(maxWidth: .infinity)
    }

    /// Pins the transcript back to the end and resumes following the answer.
    private func followEnd(_ proxy: ScrollViewProxy, animated: Bool = true) {
        atBottom = true
        frozenStream = nil
        if animated {
            withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(Self.bottomID) }
        } else {
            proxy.scrollTo(Self.bottomID)
        }
    }

    private func freezeStream() {
        guard frozenStream == nil, chat.generating, chat.current?.id == chat.generatingConvID else { return }
        frozenStream = chat.live.snapshot
    }

    private static let bottomID = "convBottom"
    private static let scrollSpace = "chatScroll"
    /// Freezing only once the answer's last lines are well out of view: doing it
    /// while they are still on screen reads as the generation having stalled.
    private static let leaveSlack: CGFloat = 120
    /// Coming back, on the other hand, means actually reaching the end.
    private static let returnSlack: CGFloat = 12

    @ViewBuilder
    private func messageRow(_ msg: ChatMessage, isNewest: Bool) -> some View {
        let isLastUser = msg.role == "user"
            && msg.id == chat.current?.messages.last(where: { $0.role == "user" })?.id
        VStack(alignment: .leading, spacing: 14) {
            if msg.id == compactionBoundaryID {
                Label(loc.t("Mensajes anteriores resumidos para liberar contexto",
                            "Earlier messages summarized to free context"),
                      systemImage: "archivebox")
                    .font(.caption2).foregroundStyle(.secondary)
                    .help(loc.t("Lo anterior a esta marca se envía al modelo como un resumen automático; aquí sigue visible íntegro.",
                                "History above this mark is sent to the model as an automatic summary; it remains fully visible here."))
            }
            if chat.generating && chat.current?.id == chat.generatingConvID
                && isNewest && msg.role == "assistant" {
                StreamingBubble(live: chat.live, message: msg, frozen: frozenStream)
            } else {
                MessageBubble(
                    message: msg,
                    streaming: false,
                    liveSpeed: nil,
                    isLastAssistant: msg.role == "assistant" && isNewest,
                    isLastUser: isLastUser,
                    canRegenerate: !chat.generating,
                    onRegenerate: {
                        chat.regenerate(port: port, temperature: temperature,
                                        maxTokens: maxTokens,
                                        system: chat.effectiveSystemPrompt(global: systemPrompt),
                                        thinking: thinkingEnabled, sampling: samplingSettings,
                                        modalities: modelModalities)
                    },
                    onContinue: {
                        chat.continueResponse(port: port, temperature: temperature,
                                              maxTokens: maxTokens,
                                              system: chat.effectiveSystemPrompt(global: systemPrompt),
                                              thinking: thinkingEnabled, sampling: samplingSettings,
                                              modalities: modelModalities)
                    },
                    onEdit: {
                        if let m = chat.editMessage(msg.id) {
                            draft = m.content
                            attachments = m.attachments ?? []
                            images = m.imageURIs ?? []
                            inputFocused = true
                        }
                    },
                    onFork: {
                        forkMessage = msg
                    },
                    onDelete: {
                        chat.deleteMessageAndFollowing(msg.id)
                    })
                    .equatable()
            }
        }
    }

    private func jumpToBottomButton(writingBehind: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: "chevron.down")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(writingBehind ? Color.appAccent : .secondary)
                .frame(width: 34, height: 34)
                // Without this only the glyph takes the click, so presses landing
                // on the empty part of the circle do nothing.
                .contentShape(Circle())
                .glassSurface(in: Circle(), interactive: true)
                .overlay(alignment: .topTrailing) {
                    if writingBehind {
                        Circle().fill(Color.appAccent)
                            .frame(width: 8, height: 8)
                            .overlay(Circle().stroke(.background, lineWidth: 1.5))
                            .offset(x: 1, y: -1)
                    }
                }
        }
        .buttonStyle(.plain)
        .padding(14)
        .help(writingBehind
              ? loc.t("La respuesta sigue escribiéndose fuera de pantalla; ir al final y volver a seguirla.",
                      "The answer is still being written off screen; jump to the end and follow it again.")
              : loc.t("Ir al final de la conversación y seguir la respuesta.",
                      "Jump to the end of the conversation and follow the response."))
    }

    private var emptyChatState: some View {
        ChatEmptyState { action in
            switch action {
            case .ask:
                draft = loc.t("Ayúdame a explorar una idea.", "Help me explore an idea.")
                inputFocused = true
            case .code:
                draft = loc.t("Ayúdame a escribir y mejorar este código:", "Help me write and improve this code:")
                inputFocused = true
            case .files:
                showAttachments = true
            case .summarize:
                draft = loc.t("Resume este contenido y destaca las ideas principales:",
                              "Summarize this content and highlight the main ideas:")
                inputFocused = true
            case .explore:
                control.openSettings(.chat)
                openWindow(id: "control")
            }
        }
    }

    // MARK: input

    private var inputArea: some View {
        VStack(spacing: 8) {
            if let queued = chat.queuedMessage, queued.conversationID == chat.currentID {
                QueuedMessageBanner(message: queued, cancel: chat.cancelQueuedMessage)
                    .environmentObject(loc)
            }
            if !attachments.isEmpty { attachmentChips }
            if !images.isEmpty { imageChips }
            if ocrPending > 0 {
                HStack(spacing: 5) {
                    ProgressView().controlSize(.mini)
                    Text(loc.t("Procesando PDF en el dispositivo…",
                               "Processing PDF on device…"))
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            if let status = dictation.statusText {
                HStack(spacing: 5) {
                    ProgressView().controlSize(.mini)
                    Text(status)
                        .font(.caption).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            if let attachError {
                Label(attachError, systemImage: "exclamationmark.triangle")
                    .font(.caption2).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            composerShell
        }
        .padding(.horizontal, 18)
        .padding(.top, 8)
        .padding(.bottom, 16)
        // Files dropped anywhere on the composer become attachments.
        .dropDestination(for: URL.self) { urls, _ in
            addAttachments(urls: urls)
            return true
        }
        .onChange(of: attachments) { scheduleDraftSave() }
        .onChange(of: images) { scheduleDraftSave() }
            .alert(loc.t("Problema con la voz", "Voice problem"),
               isPresented: Binding(
                   get: { audioRecorder.error != nil || dictation.error != nil || appleDictation.error != nil },
                   set: {
                       if !$0 {
                           audioRecorder.error = nil
                           dictation.error = nil
                           appleDictation.error = nil
                       }
                   })) {
        } message: { Text(audioRecorder.error ?? dictation.error ?? appleDictation.error ?? "") }
    }

    private var composerShell: some View {
        VStack(spacing: 8) {
            if server.state != .running {
                serverAvailabilityBar
                    .background(WorkspaceStyle.surface,
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(WorkspaceStyle.border))
            }
            if server.state == .running || chat.generating || chat.compacting {
                statusStrip
                    .padding(.horizontal, 10)
            }
            composerRow
                .background(WorkspaceStyle.surface,
                            in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(WorkspaceStyle.border))
        }
    }

    private var serverAvailabilityBar: some View {
        HStack(spacing: 10) {
            switch server.state {
            case .starting:
                ProgressView().controlSize(.small)
                Text(loc.t("Iniciando el servidor… Puedes seguir consultando tus chats.",
                           "Starting the server… You can keep browsing your chats."))
                    .foregroundStyle(.secondary)
            case .failed(let error):
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(loc.half(error)).lineLimit(2)
                Spacer(minLength: 8)
                Button(loc.t("Reintentar", "Try again")) { server.start(.fromDefaults()) }
                    .buttonStyle(.borderedProminent)
                    .disabled(modelPath.isEmpty)
            case .stopped:
                Image(systemName: "circle")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(modelPath.isEmpty
                     ? loc.t("Elige un modelo para poder enviar mensajes.",
                             "Choose a model before sending messages.")
                     : loc.t("El servidor está detenido. Puedes consultar y organizar tus chats.",
                             "The server is stopped. You can browse and organize your chats."))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                if modelPath.isEmpty {
                    Button(loc.t("Elegir modelo", "Choose model")) {
                        control.section = .models
                        openWindow(id: "control")
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button(loc.t("Iniciar servidor", "Start server"), systemImage: "play.fill") {
                        server.start(.fromDefaults())
                    }
                    .buttonStyle(.borderedProminent)
                }
            case .running:
                EmptyView()
            }
        }
        .font(.callout)
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
    }

    private var composerRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            messageField
            HStack(alignment: .center, spacing: 10) {
                paramsButton
                attachButton
                if !availableTools.isEmpty { toolsButton }
                voiceButton
                Spacer(minLength: 8)
                sendControls
            }
        }
        .padding(14)
        .frame(minHeight: 104)
    }

    private var messageField: some View {
        ZStack(alignment: .topLeading) {
            if draft.isEmpty {
                Text(loc.t("Escribe tu mensaje…", "Type your message…"))
                    .chatFont(.body)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 7)
                    .allowsHitTesting(false)
            }
            ChatComposerTextEditor(
                text: $draft,
                isFocused: $inputFocused,
                fontSize: ChatFont.Base.body.points * chatFontScale,
                onSubmit: send)
        }
            .frame(height: 64)
            .onChange(of: draft) { _, value in
                absorbLargeDraft(value)
                scheduleDraftSave()
            }
            .onPasteCommand(of: [.image, .png, .tiff, .fileURL]) { _ in
                pasteFromClipboard()
            }
            .help(loc.t("Intro envía; Mayús+Intro inserta un salto de línea. Los textos pegados grandes se convierten en un adjunto; pegar una imagen (captura) la adjunta si el modelo tiene visión.",
                        "Return sends; Shift+Return inserts a line break. Large pasted text becomes an attachment; pasting an image (screenshot) attaches it if the model has vision."))
    }

    @ViewBuilder
    private var sendControls: some View {
        if chat.generating {
            Button(loc.t("Intervenir", "Steer"), systemImage: "arrow.up.circle.fill",
                   action: send)
                       .labelStyle(.iconOnly)
                .font(.system(size: 26))
                .symbolRenderingMode(.palette)
                .foregroundStyle(.white, canSend ? AppTheme.accent(accentRaw) : Color.secondary.opacity(0.45))
                .buttonStyle(.borderless)
                .padding(.bottom, 2)
                .disabled(!canSend)
                .help(loc.t("Enviar una intervención: se aplicará al terminar el turno actual del agente.",
                            "Send a steering message after the agent's current turn."))
            Button(loc.t("Detener", "Stop"), systemImage: "stop.circle.fill",
                   action: chat.stop)
                       .labelStyle(.iconOnly)
                .font(.system(size: 26))
                .foregroundStyle(.red)
                .buttonStyle(.borderless)
                .padding(.bottom, 2)
                .help(loc.t("Detener la generación.", "Stop generation."))
        } else {
            Button(loc.t("Enviar", "Send"), systemImage: "arrow.up.circle.fill", action: send)
                .labelStyle(.iconOnly)
                .font(.system(size: 26))
                .symbolRenderingMode(.palette)
                .foregroundStyle(.white, canSend ? AppTheme.accent(accentRaw) : Color.secondary.opacity(0.45))
                .buttonStyle(.borderless)
                .padding(.bottom, 2)
                .disabled(!canSend)
                .help(loc.t("Enviar mensaje (Intro).", "Send message (Return)."))
        }
    }

    private var canSend: Bool {
        server.state == .running
            && ocrPending == 0 && transcriptionPending == 0
            && (!draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !attachments.isEmpty || !images.isEmpty)
    }

    private func loadDraft(for conversationID: UUID?) {
        draftSaveTask?.cancel()
        draftOwnerID = conversationID
        let saved = chat.conversations.first(where: { $0.id == conversationID })?.draft
        draft = saved?.text ?? ""
        attachments = saved?.attachments ?? []
        images = saved?.imageURIs ?? []
    }

    private func scheduleDraftSave() {
        let owner = draftOwnerID
        let text = draft
        let files = attachments
        let imageValues = images
        draftSaveTask?.cancel()
        draftSaveTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled, let owner else { return }
            chat.updateDraft(conversationID: owner, text: text,
                             attachments: files, imageURIs: imageValues)
        }
    }

    private func saveDraftNow(for conversationID: UUID? = nil) {
        guard let owner = conversationID ?? draftOwnerID else { return }
        chat.updateDraft(conversationID: owner, text: draft,
                         attachments: attachments, imageURIs: images)
    }

    private var paramsButton: some View {
        Button {
            showSystem.toggle()
        } label: {
            ComposerCircleLabel(
                title: loc.t("Parámetros del chat", "Chat parameters"),
                systemImage: "slider.horizontal.3",
                active: showSystem)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showSystem, arrowEdge: .top) { paramsPopover }
        .help(loc.t("Parámetros del chat: razonamiento, creatividad, longitud de respuesta y prompt de sistema.",
                    "Chat parameters: reasoning, creativity, response length and system prompt."))
    }

    /// Folder the server tools run in. Shown here because it is per chat and the
    /// user has to see which one is active before sending.
    @ViewBuilder private var workingDirectoryRow: some View {
        let own = chat.current?.workingDirectory
        let inherited = chat.current.flatMap { chat.project(id: $0.projectID)?.workingDirectory }
        VStack(alignment: .leading, spacing: 6) {
            Label(loc.t("Directorio de trabajo", "Working directory"), systemImage: "folder")
                .font(.subheadline.weight(.medium))
                .infoTip(loc.t("Carpeta en la que se ejecutan las herramientas del servidor en esta conversación. Si no fijas una, hereda la del proyecto.",
                               "Folder the server tools run in for this conversation. With none set, it inherits the project's."))
            HStack(spacing: 8) {
                if directoryIsMissing(own ?? inherited) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 10)).foregroundStyle(.red)
                        .help(loc.t("La carpeta ya no existe.", "That folder no longer exists."))
                }
                Text(own ?? inherited ?? loc.t("Sin carpeta", "None"))
                    .font(.system(size: 11, design: .monospaced))
                    .lineLimit(1).truncationMode(.head)
                    .foregroundStyle(directoryIsMissing(own ?? inherited) ? AnyShapeStyle(.red)
                                     : own == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(.primary))
                Spacer()
                if own == nil, inherited != nil {
                    Text(loc.t("del proyecto", "from project"))
                        .font(.caption2).foregroundStyle(.tertiary)
                }
                Button(loc.t("Elegir…", "Choose…")) {
                    guard let c = chat.current else { return }
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    panel.allowsMultipleSelection = false
                    if panel.runModal() == .OK, let path = panel.url?.path {
                        chat.setConversationWorkingDirectory(c, path)
                    }
                }
                .help(loc.t("Elegir la carpeta de esta conversación.", "Pick this conversation's folder."))
                Button(loc.t("Quitar", "Clear")) {
                    if let c = chat.current { chat.setConversationWorkingDirectory(c, nil) }
                }
                .disabled(own == nil)
                .help(loc.t("Vuelve a heredar la carpeta del proyecto.", "Go back to inheriting the project's folder."))
            }
        }
    }

    private var paramsPopover: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(loc.t("Parámetros del chat", "Chat parameters")).font(.headline)

            if routerMode {
                Picker(selection: $chatSelectedModel) {
                    ForEach(ModelFamilyGroup.grouped(models.models)) { group in
                        Section(group.isOther ? loc.t("Otros", "Others") : group.family) {
                            ForEach(group.models) { m in
                                Text(ModelName.forPath(m.url.path).display
                                     + (ModelTraitsCache.cached(for: m.url.path)?.pickerSuffix(spanish: loc.isSpanish) ?? ""))
                                    .tag(ServerSettings.routerAlias(for: m.url.path))
                            }
                        }
                    }
                } label: {
                    Label(loc.t("Modelo", "Model"), systemImage: "shippingbox")
                }
                .infoTip(loc.t("Modelo para el próximo mensaje. El router lo carga solo (y descarga el anterior si hace falta), sin reiniciar el servidor.",
                            "Model for the next message. The router loads it on demand (unloading the previous one if needed), no server restart."))
                .task(id: models.models.map(\.url.path)) {
                    guard chatSelectedModel.isEmpty || !models.models.contains(where: {
                        ServerSettings.routerAlias(for: $0.url.path) == chatSelectedModel
                    }) else { return }
                    if let first = models.models.first {
                        chatSelectedModel = ServerSettings.routerAlias(for: first.url.path)
                    }
                }
            }

            modalityBadges

            Divider()

            workingDirectoryRow

            Divider()

            Picker(selection: $reasoningEffort) {
                Text(loc.t("Desactivado", "Off")).tag("off")
                Text(modelDefaultEffortLabel).tag("default")
                ForEach(reasoningEffortLevels, id: \.self) { level in
                    Text(reasoningEffortLabel(level)).tag(level)
                }
            } label: {
                Label(loc.t("Esfuerzo de razonamiento", "Reasoning effort"), systemImage: "brain")
            }
            .disabled(modelModalities?.thinking == false)
            .onChange(of: reasoningEffort) { _, value in thinkingEnabled = value != "off" }
            .onChange(of: modelModalities?.reasoning) { _, _ in
                let supported = supportedReasoningEffort(reasoningEffort)
                if supported != reasoningEffort { reasoningEffort = supported }
            }
            .infoTip(loc.t("Los modelos razonadores piensan antes de responder (esos tokens cuentan dentro del límite de respuesta). La lista muestra los niveles que acepta la plantilla del modelo cargado, y «Predeterminado del modelo» deja que la elija él. Al desactivarlo se envía enable_thinking:false y /no_think; algunos modelos entrenados solo para razonar (p. ej. R1) pueden seguir pensando de todos modos.",
                        "Reasoning models think before answering (those tokens count toward the response limit). The list shows the levels the loaded model's template accepts, and \"Model default\" lets the model choose. Turning it off sends enable_thinking:false and /no_think; some reasoning-only models (e.g. R1) may still think regardless."))

            Toggle(isOn: $showSystemMessage) {
                Label(loc.t("Mostrar prompt de sistema", "Show system prompt"),
                      systemImage: "text.bubble")
            }

            HStack(spacing: 8) {
                Label(loc.t("Creatividad", "Creativity"), systemImage: "dial.medium")
                Slider(value: $temperature, in: 0...1.5)
                Text(String(format: "%.2f", temperature))
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 34)
            }
            .infoTip(loc.t("Temperatura: 0 = más determinista; valores altos = respuestas más variadas.",
                        "Temperature: 0 = more deterministic; higher values = more varied responses."))

            Picker(selection: $maxTokens) {
                ForEach(maxTokenOptions, id: \.self) { Text($0.formatted()).tag($0) }
            } label: {
                Label(loc.t("Tokens de respuesta", "Response tokens"),
                      systemImage: "text.line.last.and.arrowtriangle.forward")
            }
            .infoTip(loc.t("Máximo de tokens que el modelo puede generar en este turno, incluyendo razonamiento y respuesta visible. No aumenta el contexto. Recomendado: 2.048–4.096.",
                        "Maximum tokens the model may generate this turn, including reasoning and visible answer. It does not increase context. Recommended: 2,048–4,096."))
            if maxTokensIsLarge {
                Label(loc.t("Este límite reserva más de la mitad del contexto para una sola respuesta.",
                            "This limit reserves more than half the context for a single response."),
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
            }

            if false { DisclosureGroup {
                VStack(alignment: .leading, spacing: 10) {
                    samplingSlider(loc.t("Top P", "Top P"), value: $topP, range: 0...1)
                    samplingSlider(loc.t("Min P", "Min P"), value: $minP, range: 0...1)
                    samplingSlider(loc.t("Typical P", "Typical P"), value: $typicalP, range: 0...1)
                    Stepper(value: $topK, in: 0...200) {
                        parameterValue(loc.t("Top K", "Top K"), value: topK.formatted())
                    }
                    samplingSlider(loc.t("Penalización de repetición", "Repeat penalty"),
                                   value: $repeatPenalty, range: 0.5...2)
                    samplingSlider(loc.t("Penalización de presencia", "Presence penalty"),
                                   value: $presencePenalty, range: -2...2)
                    samplingSlider(loc.t("Penalización de frecuencia", "Frequency penalty"),
                                   value: $frequencyPenalty, range: -2...2)
                    Stepper(value: $repeatLastN, in: 0...4096, step: 16) {
                        parameterValue(loc.t("Ventana de repetición", "Repeat window"),
                                       value: repeatLastN.formatted())
                    }
                    HStack {
                        Text(loc.t("Semilla", "Seed"))
                        Spacer()
                        DeferredNumberField("-1", value: $seed, width: 90)
                    }
                    GroupBox(loc.t("Temperatura dinámica y XTC", "Dynamic temperature and XTC")) {
                        VStack(alignment: .leading, spacing: 8) {
                            samplingSlider(loc.t("Rango dinámico", "Dynamic range"),
                                           value: $dynatempRange, range: 0...2)
                            samplingSlider(loc.t("Exponente dinámico", "Dynamic exponent"),
                                           value: $dynatempExponent, range: 0.1...4)
                            samplingSlider(loc.t("Probabilidad XTC", "XTC probability"),
                                           value: $xtcProbability, range: 0...1)
                            samplingSlider(loc.t("Umbral XTC", "XTC threshold"),
                                           value: $xtcThreshold, range: 0...1)
                        }
                    }
                    GroupBox("DRY") {
                        VStack(alignment: .leading, spacing: 8) {
                            samplingSlider(loc.t("Multiplicador", "Multiplier"),
                                           value: $dryMultiplier, range: 0...2)
                            samplingSlider(loc.t("Base", "Base"), value: $dryBase, range: 1...3)
                            Stepper(value: $dryAllowedLength, in: 0...32) {
                                parameterValue(loc.t("Longitud permitida", "Allowed length"),
                                               value: dryAllowedLength.formatted())
                            }
                            Stepper(value: $dryPenaltyLastN, in: 0...32768, step: 64) {
                                parameterValue(loc.t("Ventana DRY", "DRY window"),
                                               value: dryPenaltyLastN.formatted())
                            }
                        }
                    }
                    TextField(loc.t("Orden: top_k;typ_p;top_p;min_p;temperature",
                                    "Order: top_k;typ_p;top_p;min_p;temperature"),
                              text: $samplers)
                        .workspaceTextField()
                    Toggle(loc.t("Muestreo en backend", "Backend sampling"),
                           isOn: $backendSampling)
                    Stepper(value: $agenticMaxTurns, in: 1...100) {
                        parameterValue(loc.t("Turnos máximos del agente", "Maximum agent turns"),
                                       value: agenticMaxTurns.formatted())
                    }
                    Stepper(value: $pasteLongTextLength, in: 0...100_000, step: 500) {
                        parameterValue(loc.t("Texto pegado a archivo", "Paste text to file"),
                                       value: pasteLongTextLength == 0
                                           ? loc.t("Desactivado", "Off")
                                           : pasteLongTextLength.formatted())
                    }
                    HStack {
                        Text(loc.t("Máximo de imagen (MP)", "Maximum image size (MP)"))
                        Spacer()
                        DeferredNumberField("0", value: $maxImageMegapixels, width: 80)
                    }
                    Toggle(loc.t("PDF como imágenes para modelos con visión",
                                 "PDF as images for vision models"), isOn: $pdfAsImages)
                    DisclosureGroup(loc.t("JSON personalizado de la petición",
                                          "Custom request JSON")) {
                        TextEditor(text: $customJSON)
                            .font(.system(.caption, design: .monospaced))
                            .frame(minHeight: 80)
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                        Text(loc.t("Las claves válidas reemplazan los parámetros anteriores para esta petición.",
                                   "Valid keys override the parameters above for this request."))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Button(loc.t("Restaurar muestreo", "Reset sampling"),
                           systemImage: "arrow.counterclockwise") {
                        topP = 0.95
                        minP = 0.05
                        topK = 40
                        repeatPenalty = 1
                        repeatLastN = 64
                        seed = -1
                        dynatempRange = 0
                        dynatempExponent = 1
                        xtcProbability = 0
                        xtcThreshold = 0.1
                        typicalP = 1
                        presencePenalty = 0
                        frequencyPenalty = 0
                        dryMultiplier = 0
                        dryBase = 1.75
                        dryAllowedLength = 2
                        dryPenaltyLastN = 0
                        samplers = ""
                        backendSampling = false
                        customJSON = ""
                        agenticMaxTurns = 10
                        pasteLongTextLength = 2500
                        maxImageMegapixels = 1
                        pdfAsImages = false
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                }
                .padding(.top, 8)
            } label: {
                Label(loc.t("Muestreo avanzado", "Advanced sampling"),
                      systemImage: "slider.horizontal.2.square")
            }

            Divider()

            Label(loc.t("Prompt de sistema global", "Global system prompt"), systemImage: "gearshape")
                .font(.subheadline.weight(.medium))
                .infoTip(loc.t("Instrucciones permanentes para el modelo. Prioridad: prompt de la conversación, luego el del proyecto, luego este global.",
                               "Permanent instructions for the model. Priority: the conversation's prompt, then the project's, then this global one."))
            TextEditor(text: $systemPrompt)
                .font(.system(size: 12))
                .frame(height: 90)
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 7))
            }

            Button {
                showSystem = false
                control.openSettings(.chat)
                openWindow(id: "control")
            } label: {
                Label(loc.t("Abrir ajustes avanzados del chat…", "Open advanced chat settings…"),
                      systemImage: "gearshape.2")
            }
            .glassButton()

            HStack(spacing: 10) {
                Button {
                    showSystem = false
                    promptConversation = chat.current
                } label: {
                    Label(loc.t("De esta conversación…", "This conversation's…"),
                          systemImage: (chat.current?.systemPrompt ?? "").isEmpty ? "text.bubble" : "text.bubble.fill")
                }
                .buttonStyle(.link).font(.caption)
                .help(loc.t("Prompt propio de esta conversación; sustituye al del proyecto y al global.",
                            "This conversation's own prompt; overrides the project and global ones."))
                if let p = chat.project(id: chat.current?.projectID) {
                    Button {
                        showSystem = false
                        promptProject = p
                    } label: {
                        Label(loc.t("Del proyecto…", "Project's…"),
                              systemImage: p.systemPrompt.isEmpty ? "folder" : "folder.fill")
                    }
                    .buttonStyle(.link).font(.caption)
                    .help(loc.t("Prompt compartido por las conversaciones del proyecto \"%@\".",
                                "Prompt shared by the conversations in project \"%@\".", p.name))
                }
            }
            Text(activePromptCaption)
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(14)
        .frame(width: 380)
    }

    private func samplingSlider(_ title: String, value: Binding<Double>,
                                range: ClosedRange<Double>) -> some View {
        HStack(spacing: 8) {
            Text(title)
            Slider(value: value, in: range)
            Text(value.wrappedValue, format: .number.precision(.fractionLength(2)))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 38, alignment: .trailing)
        }
    }

    private func parameterValue(_ title: String, value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).font(.system(.caption, design: .monospaced))
        }
    }

    /// Which system prompt actually applies to the open conversation.
    private var activePromptCaption: String {
        if !(chat.current?.systemPrompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return loc.t("Activo: el prompt de esta conversación.", "Active: this conversation's prompt.")
        }
        if let p = chat.project(id: chat.current?.projectID),
           !p.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return loc.t("Activo: el prompt del proyecto \"%@\".", "Active: project \"%@\"'s prompt.", "\(p.name)")
        }
        return systemPrompt.isEmpty
            ? loc.t("Sin prompt de sistema.", "No system prompt.")
            : loc.t("Activo: el prompt global.", "Active: the global prompt.")
    }

    private var attachButton: some View {
        Button {
            showAttachments.toggle()
        } label: {
            ComposerCircleLabel(
                title: loc.t("Adjuntar", "Attach"),
                systemImage: "paperclip",
                active: showAttachments || !attachments.isEmpty || !images.isEmpty)
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showAttachments, arrowEdge: .top) { attachmentsPopover }
        .help(loc.t("Adjuntar archivos: texto, código y PDF (se extrae su texto; los PDF escaneados por OCR); de otros binarios se extraen las cadenas legibles. Imágenes solo si el modelo tiene visión (su mmproj). También puedes arrastrarlos al área de escritura.",
                    "Attach files: text, code and PDF (text is extracted; scanned PDFs via OCR); other binaries contribute their readable strings. Images only if the model has vision (its mmproj). You can also drag them onto the input area."))
    }

    private var attachmentsPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(loc.t("Adjuntar", "Attach")).font(.headline)
            Button(loc.t("Archivos del Mac…", "Files from Mac…"), systemImage: "doc.badge.plus") {
                showAttachments = false
                pickAttachments()
            }
            .buttonStyle(.plain)
            .padding(.vertical, 6)
            if MCPServerStore.load().contains(where: \.enabled) {
                Divider()
                Button(loc.t("Recursos MCP…", "MCP resources…"),
                       systemImage: "point.3.connected.trianglepath.dotted") {
                    showAttachments = false
                    showMCPBrowser = true
                }
                .buttonStyle(.plain)
                .padding(.vertical, 6)
            }
        }
        .padding(14)
        .frame(width: 240)
    }

    private var voiceButton: some View {
        Button(action: primaryVoiceAction) {
            ComposerCircleLabel(
                title: voiceButtonTitle,
                systemImage: dictation.isTranscribing ? "xmark"
                    : (dictation.isDictating || appleDictation.isDictating || audioRecorder.isRecording)
                    ? "stop.fill" : "mic.fill",
                active: dictation.isTranscribing || dictation.isDictating
                    || appleDictation.isDictating || audioRecorder.isRecording)
        }
        .buttonStyle(.plain)
        .fixedSize()
        .popover(isPresented: $showVoiceOptions, arrowEdge: .top) {
            VoiceInputOptionsView(
                methodRaw: $speechInputMethodRaw,
                loadPolicyRaw: $whisperLoadPolicyRaw,
                whisperModelID: $whisperModelID,
                dismiss: { showVoiceOptions = false })
                .environmentObject(loc)
                .environmentObject(models)
        }
        .contextMenu { voiceContextMenu }
        .help(voiceButtonHelp)
    }

    @ViewBuilder
    private var voiceContextMenu: some View {
        Picker(loc.t("Método del micrófono", "Microphone method"), selection: $speechInputMethodRaw) {
            Text(loc.t("Dictado de Apple", "Apple Dictation")).tag(SpeechInputMethod.apple.rawValue)
            Text("Whisper.cpp · GPU").tag(SpeechInputMethod.whisper.rawValue)
        }

        if speechInputMethodRaw == SpeechInputMethod.whisper.rawValue {
            let selected = WhisperModel.model(id: whisperModelID)
            Picker(loc.t("Carga de Whisper", "Whisper loading"), selection: $whisperLoadPolicyRaw) {
                Text(loc.t("Bajo demanda", "On demand")).tag(WhisperLoadPolicy.onDemand.rawValue)
                Text(loc.t("Siempre cargado", "Always loaded")).tag(WhisperLoadPolicy.alwaysLoaded.rawValue)
            }
            Menu(loc.t("Modelo Whisper", "Whisper model"), systemImage: "waveform.badge.magnifyingglass") {
                Picker(loc.t("Modelo Whisper", "Whisper model"), selection: $whisperModelID) {
                    ForEach(WhisperModel.catalog) { model in
                        Text("\(model.name) · \(model.sizeMB) MB").tag(model.id)
                    }
                }
            }
            if dictation.isModelKeptLoaded {
                Label(loc.t("Modelo listo en la GPU", "Model ready on the GPU"),
                      systemImage: "memorychip.fill")
                Button(loc.t("Detener y liberar Whisper", "Stop and unload Whisper"),
                       systemImage: "stop.circle") {
                    dictation.shutdown()
                }
            }
            if !models.whisperModelInstalled(selected) {
                if let download = models.whisperDownload(selected) {
                    if download.error != nil {
                        Button(loc.t("Reintentar descarga", "Retry download"),
                               systemImage: "arrow.clockwise") {
                            models.retryWhisperDownload(download)
                        }
                    } else if download.phase == .paused {
                        Button(loc.t("Reanudar descarga", "Resume download"),
                               systemImage: "arrow.down.circle", action: download.resume)
                    } else {
                        Label(loc.t("Descargando %@…", "Downloading %@…", "\(selected.name)"),
                              systemImage: "arrow.down.circle")
                    }
                } else {
                    Button(loc.t("Descargar %@", "Download %@", "\(selected.name)"),
                           systemImage: "arrow.down.circle") {
                        models.downloadWhisperModel(selected)
                    }
                }
            }
        }

        Divider()
        Button(loc.t("Mostrar opciones de voz…", "Show voice options…"),
               systemImage: "slider.horizontal.3") {
            showVoiceOptions = true
        }
        if audioAvailable {
            Button(loc.t("Grabar para el modelo", "Record for the model"),
                   systemImage: "waveform", action: startRecording)
        }
    }

    private var voiceButtonTitle: String {
        if dictation.isTranscribing { return loc.t("Cancelar transcripción", "Cancel transcription") }
        if dictation.isDictating { return loc.t("Transcribir", "Transcribe") }
        if appleDictation.isDictating { return loc.t("Detener dictado", "Stop dictation") }
        if audioRecorder.isRecording { return loc.t("Detener grabación", "Stop recording") }
        return loc.t("Voz", "Voice")
    }

    private var voiceButtonHelp: String {
        if dictation.isTranscribing {
            return loc.t("Whisper.cpp está transcribiendo en la GPU; clic para cancelar.",
                         "Whisper.cpp is transcribing on the GPU; click to cancel.")
        }
        if dictation.isDictating {
            return loc.t("Escuchando con Whisper… clic para transcribir.",
                         "Listening with Whisper… click to transcribe.")
        }
        if appleDictation.isDictating {
            return loc.t("Dictado de Apple activo; clic para detener.",
                         "Apple Dictation is active; click to stop.")
        }
        if speechInputMethodRaw.isEmpty {
            return loc.t("Configura la entrada de voz. Después: clic para dictar, clic derecho para cambiar.",
                         "Set up voice input. Then: click to dictate, right-click to change it.")
        }
        return loc.t("Clic para dictar; clic derecho para método, modelo y carga.",
                     "Click to dictate; right-click for method, model, and loading.")
    }

    private func primaryVoiceAction() {
        if dictation.isDictating || dictation.isTranscribing
            || appleDictation.isDictating || audioRecorder.isRecording {
            stopVoice()
            return
        }
        guard let method = SpeechInputMethod(rawValue: speechInputMethodRaw) else {
            showVoiceOptions = true
            return
        }
        switch method {
        case .apple: startAppleDictation()
        case .whisper: startWhisperDictation()
        }
    }

    private func prepareDictationBase() {
        attachError = nil
        let base = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        dictationBase = base.isEmpty ? "" : base + " "
    }

    private func startAppleDictation() {
        prepareDictationBase()
        appleDictation.toggle { draft = dictationBase + $0 }
    }

    private func startWhisperDictation() {
        prepareDictationBase()
        let model = WhisperModel.model(id: whisperModelID)
        guard models.whisperModelInstalled(model) else {
            showVoiceOptions = true
            return
        }
        dictation.toggle(modelURL: model.url(in: models.whisperDirectory),
                         gpuIndex: gpuIndex,
                         loadPolicy: whisperLoadPolicy) {
            draft = dictationBase + $0
        }
    }

    private func startRecording() {
        attachError = nil
        audioRecorder.toggle { attachments.append($0) }
    }

    private func stopVoice() {
        if dictation.isDictating {
            let model = WhisperModel.model(id: whisperModelID)
            dictation.toggle(modelURL: model.url(in: models.whisperDirectory),
                             gpuIndex: gpuIndex,
                             loadPolicy: whisperLoadPolicy) {
                draft = dictationBase + $0
            }
        } else if dictation.isTranscribing {
            dictation.cancel()
        }
        if appleDictation.isDictating { appleDictation.stop() }
        if audioRecorder.isRecording { audioRecorder.toggle { attachments.append($0) } }
    }

    private func configureWhisperResidency() {
        guard whisperLoadPolicy == .alwaysLoaded else {
            dictation.unloadPersistentModel()
            return
        }
        // Do not reload Whisper after an explicit engine stop.
        guard server.state == .starting || server.state == .running else {
            dictation.unloadPersistentModel()
            return
        }
        let model = WhisperModel.model(id: whisperModelID)
        guard models.whisperModelInstalled(model) else { return }
        dictation.preload(modelURL: model.url(in: models.whisperDirectory), gpuIndex: gpuIndex)
    }

    private func stopSpeechSubsystem() {
        let pendingAttachmentIDs = Set(transcriptionQueue.map(\.attachmentID))
        transcriptionQueue.removeAll()
        transcriptionPending = 0
        attachments.removeAll { pendingAttachmentIDs.contains($0.id) }
        dictation.shutdown()
        appleDictation.shutdown()
        if audioRecorder.isRecording {
            audioRecorder.cancel()
        }
    }

    private var toolsButton: some View {
        Button {
            showTools.toggle()
        } label: {
            ComposerCircleLabel(
                title: loc.t("Herramientas", "Tools"),
                systemImage: "wrench.and.screwdriver",
                active: showTools)
        }
        .buttonStyle(.plain)
        .disabled(chat.generating)
        .popover(isPresented: $showTools, arrowEdge: .top) { toolsPopover }
        .help(loc.t("Herramientas disponibles para esta conversación",
                    "Tools available to this conversation"))
    }

    private var toolsPopover: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(loc.t("Herramientas del chat", "Chat tools"),
                      systemImage: "wrench.and.screwdriver")
                    .font(.headline)
                Spacer()
                Button(loc.t("Activar todas", "Enable all"), systemImage: "checkmark.circle") {
                    chat.enableAllTools()
                }
                .buttonStyle(.borderless)
            }
            Divider()
            ForEach(availableTools) { tool in
                let enabled = chat.current?.enabledToolNames.map { $0.contains(tool.name) } ?? true
                Button {
                    chat.toggleTool(tool.name, allTools: availableTools)
                } label: {
                    HStack {
                        Image(systemName: enabled ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(enabled ? Color.appAccent : .secondary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(tool.displayName)
                            Text(tool.name).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .frame(width: 320)
    }

    private func refreshAvailableTools() async {
        loadingTools = true
        var tools: [BuiltinToolInfo] = []
        if UserDefaults.standard.bool(forKey: SettingsKeys.agentToolsEnabled),
           let builtins = try? await ChatToolsService.list(port: port) {
            tools += builtins
        }
        if UserDefaults.standard.bool(forKey: SettingsKeys.jsSandboxEnabled) {
            tools.append(JavaScriptSandboxService.tool)
        }
        if ChatMemoryService.isEnabled { tools += ChatMemoryService.tools }
        tools += await ToshMCPService.shared.discoverTools()
        var seen = Set<String>()
        availableTools = tools.filter { seen.insert($0.name).inserted }
        loadingTools = false
    }

    private func refreshModelModalities() async {
        guard server.state == .running else {
            modelModalities = nil
            loadingModalities = false
            return
        }
        loadingModalities = true
        let selected = routerMode && !chatSelectedModel.isEmpty ? chatSelectedModel : nil
        if let fetched = try? await ModelCapabilitiesService.fetch(port: port, model: selected) {
            modelModalities = fetched
            capabilitiesAreComplete = true
        } else if routerMode,
                  let local = models.models.first(where: {
                      ServerSettings.routerAlias(for: $0.url.path) == chatSelectedModel
                  }) {
            modelModalities = ModelModalities(
                vision: ServerSettings.mmprojPath(forModel: local.url.path) != nil,
                audio: false, video: false, thinking: nil)
            capabilitiesAreComplete = false
        } else {
            modelModalities = nil
            capabilitiesAreComplete = false
        }
        loadingModalities = false
    }

    @ViewBuilder
    private var modalityBadges: some View {
        HStack(spacing: 6) {
            if loadingModalities {
                ProgressView().controlSize(.small)
                Text(loc.t("Detectando capacidades…", "Detecting capabilities…"))
                    .font(.caption).foregroundStyle(.secondary)
            } else if let modelModalities {
                modalityBadge(loc.t("Texto", "Text"), icon: "text.alignleft", enabled: true)
                modalityBadge(loc.t("Visión", "Vision"), icon: "eye", enabled: modelModalities.vision)
                if capabilitiesAreComplete {
                    modalityBadge(loc.t("Audio", "Audio"), icon: "waveform", enabled: modelModalities.audio)
                    modalityBadge(loc.t("Video", "Video"), icon: "film", enabled: modelModalities.video)
                } else {
                    Label(loc.t("Se completan al cargar", "Complete after loading"),
                          systemImage: "clock.arrow.circlepath")
                        .font(.caption).foregroundStyle(.secondary)
                }
            } else {
                Label(loc.t("Capacidades no disponibles", "Capabilities unavailable"),
                      systemImage: "questionmark.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func modalityBadge(_ title: String, icon: String, enabled: Bool) -> some View {
        Label(title, systemImage: icon)
            .font(.caption2.weight(.medium))
            .foregroundStyle(enabled ? AnyShapeStyle(.primary) : AnyShapeStyle(.tertiary))
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background((enabled ? Color.accentColor : Color.secondary).opacity(enabled ? 0.14 : 0.07),
                        in: Capsule())
            .help(enabled
                  ? loc.t("Compatible", "Supported")
                  : loc.t("No compatible con el modelo seleccionado", "Not supported by the selected model"))
    }

    /// Decoded images are handed to the body on every keystroke while one sits in
    /// the composer or a message; decoding base64 + the bitmap each time is the
    /// single most expensive thing a render can do here.
    private static let imageCache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 32
        return cache
    }()

    static func nsImage(fromDataURI uri: String) -> NSImage? {
        let key = uri as NSString
        if let hit = imageCache.object(forKey: key) { return hit }
        guard let comma = uri.firstIndex(of: ","),
              let data = Data(base64Encoded: String(uri[uri.index(after: comma)...])),
              let image = NSImage(data: data) else { return nil }
        imageCache.setObject(image, forKey: key)
        return image
    }

    private var imageChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(images.enumerated()), id: \.offset) { idx, uri in
                    ZStack(alignment: .topTrailing) {
                        if let img = Self.nsImage(fromDataURI: uri) {
                            Image(nsImage: img).resizable().scaledToFill()
                                .frame(width: 54, height: 54)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        Button { images.remove(at: idx) } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.white, .black.opacity(0.55))
                        }
                        .buttonStyle(.plain).padding(2)
                        .iconHelp(loc.t("Quitar esta imagen", "Remove this image"))
                    }
                }
            }
            .padding(.vertical, 2)
        }
    }

    private var attachmentChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(attachments) { a in
                    HStack(spacing: 5) {
                        Image(systemName: a.mediaKind == "audio" ? "waveform"
                              : a.mediaKind == "video" ? "film" : "doc.text")
                        Text(a.name).lineLimit(1)
                        if a.mediaKind == nil {
                            Text("~\(a.estimatedTokens)t")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundStyle(.secondary)
                        } else {
                            Button(loc.t("Vista previa", "Preview"), systemImage: "play.circle") {
                                previewAttachment = a
                            }
                            .labelStyle(.iconOnly).buttonStyle(.borderless)
                        }
                        Button {
                            attachments.removeAll { $0.id == a.id }
                        } label: {
                            Image(systemName: "xmark.circle.fill").font(.system(size: 10))
                        }
                        .buttonStyle(.borderless)
                        .iconHelp(loc.t("Quitar este archivo.", "Remove this file."))
                    }
                    .font(.caption)
                    .padding(.horizontal, 9).padding(.vertical, 5)
                    .background(.quaternary.opacity(0.5), in: Capsule())
                    .help(loc.t("Se enviará al modelo junto con tu mensaje (~%@ tokens).",
                                "Sent to the model along with your message (~%@ tokens).",
                                String(a.estimatedTokens)))
                }
                if attachmentsExceedContext {
                    Label(loc.t("Adjuntos ~%@k tokens > contexto %@k: sube el contexto en Ajustes o quita archivos", "Attachments ~%@k tokens > context %@k: raise the context in Settings or remove files", "\(attachmentTokens / 1000)", "\(contextLimit / 1000)"),
                          systemImage: "exclamationmark.octagon.fill")
                        .font(.caption2).foregroundStyle(.red)
                        .help(loc.t("Lo adjunto no cabe en el contexto configurado, así que el envío fallará con 'contexto lleno'. Sube el contexto en Ajustes, quita archivos o inicia un chat nuevo.",
                                    "The attachments don't fit the configured context, so sending will fail with 'context full'. Raise the context in Settings, remove files or start a new chat."))
                } else if attachmentsTooLarge {
                    Label(loc.t("Adjuntos ~%@k tokens: pueden llenar el contexto (%@k)", "Attachments ~%@k tokens: may fill the context (%@k)", "\(attachmentTokens / 1000)", "\(contextLimit / 1000)"),
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption2).foregroundStyle(.orange)
                        .help(loc.t("El total estimado supera la mitad del contexto configurado en Ajustes; con el historial podría llenarlo.",
                                    "The estimated total exceeds half the context configured in Settings; with the history it could fill it."))
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var attachmentTokens: Int {
        attachments.reduce(0) { $0 + $1.estimatedTokens }
    }

    private var attachmentsTooLarge: Bool {
        contextLimit > 0 && attachmentTokens > contextLimit / 2
    }

    /// The attachments alone already exceed the configured context, so the send
    /// will fail with the server's "context full" error — warn before sending.
    private var attachmentsExceedContext: Bool {
        contextLimit > 0 && attachmentTokens >= contextLimit
    }

    private func absorbLargeDraft(_ value: String) {
        guard pasteLongTextLength > 0, value.count > pasteLongTextLength else { return }
        let base = loc.t("Texto pegado", "Pasted text")
        let existing = attachments.filter { $0.name.hasPrefix(base) }.count
        let name = existing == 0 ? base + ".txt" : "\(base) \(existing + 1).txt"
        attachments.append(ChatAttachment(name: name, content: value))
        draft = ""
    }

    /// Whether the clipboard holds something we attach instead of pasting as text.
    /// NSImage covers every readable image type (PNG, JPEG, TIFF, HEIC…).
    private var clipboardHasAttachables: Bool {
        let pb = NSPasteboard.general
        return (pb.types ?? []).contains(.fileURL) || NSImage.canInit(with: pb)
    }

    /// Cmd+V with an image (screenshot) or copied files on the clipboard attaches
    /// them like a drop; plain text keeps the field's normal paste.
    private func pasteFromClipboard() {
        let pb = NSPasteboard.general
        if let urls = pb.readObjects(forClasses: [NSURL.self],
                                     options: [.urlReadingFileURLsOnly: true]) as? [URL],
           !urls.isEmpty {
            addAttachments(urls: urls)
            return
        }
        guard let img = NSImage(pasteboard: pb), let data = img.tiffRepresentation else { return }
        guard visionAvailable else {
            attachError = loc.t("Imagen pegada: el modelo actual no admite imágenes (carga un modelo con visión y su mmproj)",
                                "Pasted image: the current model can't read images (load a vision model with its mmproj)")
            return
        }
        guard let uri = Self.imageDataURI(from: data, maxMegapixels: maxImageMegapixels) else {
            attachError = loc.t("No se pudo procesar la imagen pegada", "Couldn't process the pasted image")
            return
        }
        attachError = nil
        images.append(uri)
    }

    private func pickAttachments() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        if panel.runModal() == .OK { addAttachments(urls: panel.urls) }
    }

    private func loadVideoDuration(url: URL, attachmentID: UUID) {
        Task { @MainActor in
            let asset = AVURLAsset(url: url)
            guard let duration = try? await asset.load(.duration) else { return }
            let seconds = CMTimeGetSeconds(duration)
            guard seconds.isFinite, seconds > 0 else { return }
            var area: Int? = nil
            if let track = try? await asset.loadTracks(withMediaType: .video).first,
               let size = try? await track.load(.naturalSize) {
                area = Int(abs(size.width) * abs(size.height))
            }
            guard let idx = attachments.firstIndex(where: { $0.id == attachmentID }) else { return }
            attachments[idx].durationSeconds = seconds
            attachments[idx].videoFrameArea = area
        }
    }

    // Read cap (raw bytes) and extracted-text cap. Text beyond the latter is
    // truncated with a note; the proactive context warning catches huge totals.
    private static let maxAttachBytes = 40 * 1024 * 1024
    private static let maxAttachChars = 400_000

    private func addAttachments(urls: [URL]) {
        var errors: [String] = []
        for url in urls {
            let name = url.lastPathComponent
            let ext = (name as NSString).pathExtension.lowercased()

            if ["wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "mp4", "mov", "webm", "mkv"].contains(ext) {
                // Microphone dictation cannot advance the file queue.
                if (dictation.isDictating || dictation.isTranscribing || appleDictation.isDictating),
                   transcriptionPending == 0 {
                    errors.append(loc.t("%@: termina primero el dictado del micrófono", "%@: finish microphone dictation first", "\(name)"))
                    continue
                }
                let model = WhisperModel.model(id: whisperModelID)
                guard models.whisperModelInstalled(model) else {
                    models.downloadWhisperModel(model)
                    errors.append(loc.t("%@: descarga primero %@ para transcribirlo", "%@: download %@ first to transcribe it", "\(name)", "\(model.name)"))
                    continue
                }
                let pendingID = UUID()
                attachments.append(ChatAttachment(
                    id: pendingID, name: name,
                    content: loc.t("(transcribiendo localmente con Whisper.cpp…)",
                                   "(transcribing locally with Whisper.cpp…)")))
                transcriptionQueue.append(PendingSpeechTranscription(
                    id: UUID(), fileURL: url, modelURL: model.url(in: models.whisperDirectory),
                    attachmentID: pendingID, name: name))
                transcriptionPending += 1
                startNextFileTranscription()
                continue
            }

            guard let data = try? Data(contentsOf: url) else {
                errors.append(loc.t("%@: no se pudo leer", "%@: couldn't read it", "\(name)")); continue
            }
            guard data.count <= Self.maxAttachBytes else {
                errors.append(loc.t("%@: demasiado grande (máx 40 MB)", "%@: too large (max 40 MB)", "\(name)")); continue
            }

            // Images → vision (only if the loaded model has a multimodal projector).
            if ["png", "jpg", "jpeg", "heic", "heif", "gif", "bmp", "tiff", "tif", "webp"].contains(ext) {
                guard visionAvailable else {
                    errors.append(loc.t("%@: el modelo actual no admite imágenes (carga un modelo con visión y su mmproj)", "%@: the current model can't read images (load a vision model with its mmproj)", "\(name)")); continue
                }
                guard let uri = Self.imageDataURI(from: data, maxMegapixels: maxImageMegapixels) else {
                    errors.append(loc.t("%@: no se pudo procesar la imagen", "%@: couldn't process the image", "\(name)")); continue
                }
                images.append(uri); continue
            }

            var text: String
            if ext == "pdf" || data.prefix(5) == Data("%PDF-".utf8) {
                guard let pdf = PDFDocument(data: data) else {
                    errors.append(loc.t("%@: no se pudo abrir el PDF", "%@: couldn't open the PDF", "\(name)")); continue
                }
                if pdfAsImages && visionAvailable {
                    ocrPending += 1
                    Task { @MainActor in
                        let pages = Self.pdfImageURIs(pdf, maxPages: 20,
                                                      maxMegapixels: maxImageMegapixels)
                        images.append(contentsOf: pages)
                        ocrPending -= 1
                        if pages.isEmpty {
                            attachError = (attachError.map { $0 + "\n" } ?? "")
                                + loc.t("%@: no se pudieron renderizar sus páginas", "%@: its pages couldn't be rendered", "\(name)")
                        }
                    }
                    continue
                }
                if let s = pdf.string, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    text = s
                } else {
                    // No text layer (scanned PDF) → OCR the page images on-device
                    // (Vision framework), asynchronously so the UI doesn't block.
                    let pendingID = UUID()
                    attachments.append(ChatAttachment(id: pendingID, name: name,
                        content: loc.t("(extrayendo texto por OCR…)", "(extracting text via OCR…)")))
                    ocrPending += 1
                    Task { @MainActor in
                        let ocr = await Self.ocrPDF(data: data, maxPages: 20, maxChars: Self.maxAttachChars)
                        ocrPending -= 1
                        guard let idx = attachments.firstIndex(where: { $0.id == pendingID }) else { return }
                        if ocr.isEmpty {
                            attachments.remove(at: idx)
                            attachError = (attachError.map { $0 + "\n" } ?? "")
                                + loc.t("%@: el OCR no encontró texto", "%@: OCR found no text", "\(name)")
                        } else {
                            attachments[idx].content = loc.t("[Texto extraído por OCR — %@]\n\n", "[Text extracted via OCR — %@]\n\n", "\(name)") + ocr
                        }
                    }
                    continue
                }
            } else if let decoded = Self.decodeText(data) {
                text = decoded
            } else {
                // Binary: raw bytes are useless to a text model, so extract its
                // printable strings (symbols, embedded text) instead.
                let s = Self.printableStrings(from: data, limit: Self.maxAttachChars)
                guard !s.isEmpty else {
                    errors.append(loc.t("%@: binario sin texto legible", "%@: binary with no readable text", "\(name)")); continue
                }
                text = loc.t("[Cadenas extraídas de un binario — %@]\n\n", "[Strings extracted from a binary — %@]\n\n", "\(name)") + s
            }

            if text.count > Self.maxAttachChars {
                text = String(text.prefix(Self.maxAttachChars))
                    + loc.t("\n\n[…contenido truncado…]", "\n\n[…content truncated…]")
            }
            guard !attachments.contains(where: { $0.name == name && $0.content == text }) else { continue }
            attachments.append(ChatAttachment(name: name, content: text))
        }
        attachError = errors.isEmpty ? nil : errors.joined(separator: "\n")
    }

    private func startNextFileTranscription() {
        guard !dictation.isTranscribing, !dictation.isDictating, !appleDictation.isDictating,
              let pending = transcriptionQueue.first else { return }
        dictation.transcribe(fileURL: pending.fileURL,
                             modelURL: pending.modelURL,
                             gpuIndex: gpuIndex,
                             loadPolicy: whisperLoadPolicy) { transcript in
            completeFileTranscription(pending, transcript: transcript)
        } onFailure: {
            failFileTranscription(pending)
        }
    }

    private func completeFileTranscription(_ pending: PendingSpeechTranscription,
                                           transcript: String) {
        transcriptionQueue.removeAll { $0.id == pending.id }
        transcriptionPending = max(0, transcriptionPending - 1)
        if let idx = attachments.firstIndex(where: { $0.id == pending.attachmentID }) {
            let clipped = transcript.count > Self.maxAttachChars
                ? String(transcript.prefix(Self.maxAttachChars))
                    + loc.t("\n\n[…transcripción truncada…]", "\n\n[…transcript truncated…]")
                : transcript
            attachments[idx].content = loc.t("[Transcripción local — %@]\n\n", "[Local transcript — %@]\n\n", "\(pending.name)") + clipped
        }
        startNextFileTranscription()
    }

    private func failFileTranscription(_ pending: PendingSpeechTranscription) {
        transcriptionQueue.removeAll { $0.id == pending.id }
        transcriptionPending = max(0, transcriptionPending - 1)
        attachments.removeAll { $0.id == pending.attachmentID }
        startNextFileTranscription()
    }

    private static func decodeText(_ data: Data) -> String? {
        if data.starts(with: [0xFF, 0xFE]) || data.starts(with: [0xFE, 0xFF]) {
            return String(data: data, encoding: .utf16)
        }
        if let s = String(data: data, encoding: .utf8) { return s }
        let sample = data.prefix(8192)
        if !sample.contains(0) {
            let bad = sample.filter { $0 != 0x09 && $0 != 0x0A && $0 != 0x0D && ($0 < 0x20 || $0 == 0x7F) }.count
            if Double(bad) / Double(max(1, sample.count)) < 0.05 {
                return String(data: data, encoding: .windowsCP1252) ?? String(data: data, encoding: .isoLatin1)
            }
        }
        return nil
    }

    private var visionAvailable: Bool {
        modelModalities?.vision ?? (!routerMode && ServerSettings.mmprojPath(forModel: modelPath) != nil)
    }

    private var audioAvailable: Bool { modelModalities?.audio ?? false }
    private var videoAvailable: Bool { modelModalities?.video ?? false }

    private static func imageDataURI(from data: Data, maxMegapixels: Double) -> String? {
        guard let img = NSImage(data: data),
              let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        let w = CGFloat(cg.width), h = CGFloat(cg.height)
        let pixelLimit = maxMegapixels > 0 ? maxMegapixels * 1_000_000 : Double.greatestFiniteMagnitude
        let scale = min(1, sqrt(pixelLimit / Double(w * h)))
        let nw = max(1, Int(w * scale)), nh = max(1, Int(h * scale))
        guard let ctx = CGContext(data: nil, width: nw, height: nh, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        guard let out = ctx.makeImage() else { return nil }
        let rep = NSBitmapImageRep(cgImage: out)
        guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else { return nil }
        return "data:image/jpeg;base64," + jpeg.base64EncodedString()
    }

    @MainActor
    private static func pdfImageURIs(_ document: PDFDocument, maxPages: Int,
                                     maxMegapixels: Double) -> [String] {
        var output: [String] = []
        for index in 0..<min(document.pageCount, maxPages) {
            guard let page = document.page(at: index) else { continue }
            let bounds = page.bounds(for: .mediaBox)
            let aspect = max(0.1, bounds.width / max(1, bounds.height))
            let pixels = maxMegapixels > 0 ? maxMegapixels * 1_000_000 : 4_000_000
            let height = sqrt(pixels / Double(aspect))
            let size = CGSize(width: max(1, height * Double(aspect)), height: max(1, height))
            let thumbnail = page.thumbnail(of: size, for: .mediaBox)
            guard let data = thumbnail.tiffRepresentation,
                  let uri = imageDataURI(from: data, maxMegapixels: 0) else { continue }
            output.append(uri)
        }
        return output
    }

    private static func ocrPDF(data: Data, maxPages: Int, maxChars: Int) async -> String {
        await Task.detached(priority: .userInitiated) { () -> String in
            guard let doc = PDFDocument(data: data) else { return "" }
            var out = ""
            for i in 0..<min(doc.pageCount, maxPages) {
                guard let page = doc.page(at: i) else { continue }
                let rect = page.bounds(for: .mediaBox)
                let scale: CGFloat = 2
                let w = Int(rect.width * scale), h = Int(rect.height * scale)
                guard w > 0, h > 0,
                      let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { continue }
                ctx.setFillColor(CGColor(gray: 1, alpha: 1))
                ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
                ctx.scaleBy(x: scale, y: scale)
                ctx.translateBy(x: -rect.minX, y: -rect.minY)
                page.draw(with: .mediaBox, to: ctx)
                guard let cg = ctx.makeImage() else { continue }
                let req = VNRecognizeTextRequest()
                req.recognitionLevel = .accurate
                req.recognitionLanguages = ["es-ES", "en-US"]
                req.usesLanguageCorrection = true
                let handler = VNImageRequestHandler(cgImage: cg, options: [:])
                try? handler.perform([req])
                for obs in req.results ?? [] {
                    if let top = obs.topCandidates(1).first { out += top.string + "\n" }
                }
                if out.count >= maxChars { break }
            }
            return String(out.prefix(maxChars))
        }.value
    }

    /// `strings`-style extraction: runs of >= 4 printable ASCII chars, one per line.
    private static func printableStrings(from data: Data, limit: Int) -> String {
        var out = ""
        var run: [UInt8] = []
        for b in data {
            if b == 0x09 || (b >= 0x20 && b < 0x7F) {
                run.append(b)
            } else {
                if run.count >= 4 { out += String(decoding: run, as: UTF8.self) + "\n" }
                run.removeAll(keepingCapacity: true)
                if out.count >= limit { break }
            }
        }
        if run.count >= 4 && out.count < limit { out += String(decoding: run, as: UTF8.self) }
        return out
    }

    private var statusStrip: some View {
        HStack(spacing: 12) {
                if chat.compacting {
                    HStack(spacing: 5) {
                        ProgressView().controlSize(.mini)
                        Text(loc.t("Compactando…", "Compacting…"))
                            .foregroundStyle(.secondary)
                    }
                    .help(loc.t("Resumiendo los mensajes antiguos con el modelo para liberar contexto.",
                                "Summarizing older messages with the model to free context."))
                }

                if server.state == .running && contextLimit > 0 {
                    let used = chat.contextUsed ?? 0
                    let fraction = Double(used) / Double(contextLimit)
                    HStack(spacing: 5) {
                        Label(loc.t("Contexto", "Context"), systemImage: "memorychip")
                            .foregroundStyle(.secondary)
                        ProgressView(value: min(fraction, 1))
                            .frame(width: 76)
                            .tint(fraction > 0.9 ? .red : fraction > 0.8 ? .orange : .accentColor)
                        Text("\(used / 1000)k / \(contextLimit / 1000)k")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(fraction > 0.8 ? .orange : .secondary)
                    }
                    .help(loc.t("Tokens usados por el historial y la última respuesta. Al superar 70%, la app intenta resumir los turnos antiguos.",
                                "Tokens used by history and the latest response. Past 70%, the app attempts to summarize older turns."))
                }

                if server.state == .running && contextMaySlowGeneration && chat.canCompactCurrent {
                    Button {
                        chat.compactCurrent(port: port)
                    } label: {
                        Label(loc.t("Recuperar velocidad", "Recover speed"),
                              systemImage: "archivebox")
                            .foregroundStyle(.orange)
                    }
                    .buttonStyle(.borderless)
                    .help(loc.t("Resume el historial completado para que el próximo turno procese menos contexto. El chat completo sigue visible y guardado.",
                                "Summarizes completed history so the next turn processes less context. The full chat remains visible and saved."))
                }

                Spacer(minLength: 0)
                LiveSpeedBadge(live: chat.live)
        }
        .font(.caption)
        .frame(minHeight: 24)
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSend else { return }
        draft = ""
        let files = attachments
        let imgs = images
        attachments = []
        images = []
        attachError = nil
        atBottom = true
        frozenStream = nil
        if chat.agentFlowActive {
            chat.queueMessage(text: text, attachments: files, images: imgs)
            return
        }
        chat.send(text: text, attachments: files, images: imgs, port: port, temperature: temperature,
                  maxTokens: maxTokens, system: chat.effectiveSystemPrompt(global: systemPrompt),
                  thinking: thinkingEnabled, sampling: samplingSettings,
                  modalities: modelModalities)
    }
}