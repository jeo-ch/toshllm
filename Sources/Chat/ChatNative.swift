// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import AppKit

@MainActor
final class LiveStream: ObservableObject {
    @Published var visibleText = ""
    @Published var displayedReasoning = ""
    @Published var reasoningTail = ""
    @Published var hasReasoning = false
    @Published var reasoningExpanded = false
    @Published var speed: Double?
    /// Prompt-processing progress 0…1 before the first token; nil otherwise.
    @Published var prefillProgress: Double?

    private var latestReasoning = ""
    private var lastReasoningPublish = Date.distantPast
    private let reasoningPublishInterval: TimeInterval = 0.5
    private var lastTailPublish = Date.distantPast
    private let tailPublishInterval: TimeInterval = 0.3

    func reset() {
        visibleText = ""
        displayedReasoning = ""
        reasoningTail = ""
        hasReasoning = false
        reasoningExpanded = false
        latestReasoning = ""
        lastReasoningPublish = .distantPast
        lastTailPublish = .distantPast
        speed = nil
        prefillProgress = nil
    }

    var snapshot: StreamSnapshot {
        StreamSnapshot(visible: visibleText, reasoning: displayedReasoning, reasoningTail: reasoningTail)
    }

    func setPrefillProgress(_ p: Double?) {
        if prefillProgress != p { prefillProgress = p }
    }

    func update(reasoning: String, visible: String, speed: Double?, now: Date = Date()) {
        latestReasoning = reasoning
        if hasReasoning != !reasoning.isEmpty { hasReasoning = !reasoning.isEmpty }
        if reasoningExpanded,
           now.timeIntervalSince(lastReasoningPublish) >= reasoningPublishInterval,
           displayedReasoning != reasoning {
            displayedReasoning = reasoning
            lastReasoningPublish = now
        }
        // One-line live tail so collapsed "Thinking…" visibly progresses (cheap).
        if !reasoningExpanded, visible.isEmpty, !reasoning.isEmpty,
           now.timeIntervalSince(lastTailPublish) >= tailPublishInterval {
            let tail = Self.tailSnippet(reasoning)
            if reasoningTail != tail { reasoningTail = tail }
            lastTailPublish = now
        }
        if visibleText != visible { visibleText = visible }
        if let speed { self.speed = speed }
    }

    /// Tail of the reasoning as one flowing line, capped to recent chars at a
    /// word boundary, so the peek scrolls continuously instead of jumping lines.
    private static func tailSnippet(_ s: String, limit: Int = 200) -> String {
        let flat = s.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: "  ")
        if flat.count <= limit { return flat }
        var cut = String(flat.suffix(limit))
        if let space = cut.firstIndex(of: " ") { cut = String(cut[cut.index(after: space)...]) }
        return "… " + cut
    }

    func setReasoningExpanded(_ expanded: Bool, now: Date = Date()) {
        reasoningExpanded = expanded
        displayedReasoning = expanded ? latestReasoning : ""
        lastReasoningPublish = expanded ? now : .distantPast
    }
}

final class StreamBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var reasoning = ""
    private var visible = ""
    private var speed: Double?
    private var progress: Double?
    private var dirty = false
    private var done = false
    /// Push notification: when data arrives, signal the pump immediately
    /// instead of polling every 40ms.
    private var continuation: CheckedContinuation<Void, Never>?

    func write(reasoning: String, visible: String, speed: Double?) {
        lock.lock()
        self.reasoning = reasoning
        self.visible = visible
        if let speed { self.speed = speed }
        dirty = true
        let cont = continuation
        continuation = nil
        lock.unlock()
        cont?.resume()
    }

    func writeProgress(_ p: Double?) {
        lock.lock(); progress = p; dirty = true
        let cont = continuation
        continuation = nil
        lock.unlock()
        cont?.resume()
    }

    func finish() {
        lock.lock(); done = true; dirty = true
        let cont = continuation
        continuation = nil
        lock.unlock()
        cont?.resume()
    }

    /// The latest snapshot if it changed since the last take (or the stream
    /// ended); nil when there is nothing new to render.
    func take() -> (reasoning: String, visible: String, speed: Double?, progress: Double?, done: Bool)? {
        lock.lock(); defer { lock.unlock() }
        guard dirty else { return nil }
        dirty = false
        return (reasoning, visible, speed, progress, done)
    }

    /// Await until data arrives or finish() is called. Returns immediately if
    /// dirty data is already pending.
    func waitForData() async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            lock.lock()
            if dirty {
                lock.unlock()
                cont.resume()
            } else {
                continuation = cont
                lock.unlock()
            }
        }
    }
}

private struct AgentRunContext {
    let port: Int
    let temperature: Double
    let maxTokens: Int
    let system: String
    let thinking: Bool
    let sampling: ChatSamplingSettings
    let modalities: ModelModalities?
    var remainingTurns: Int
    var tools: [BuiltinToolInfo]
    var workingDirectory: String?
}

@MainActor
final class ChatStore: ObservableObject {
    nonisolated static func reasoningBudget(for effort: String) -> Int? {
        switch effort {
        case "minimal": 128
        case "low": 512
        case "medium": 2_048
        case "high": 8_192
        default: nil
        }
    }
    private static var configuredAgentTurnLimit: Int {
        max(1, UserDefaults.standard.object(forKey: SettingsKeys.chatAgenticMaxTurns) as? Int ?? 10)
    }

    @Published var conversations: [Conversation] = []
    @Published var projects: [ChatProject] = []
    @Published var currentID: UUID?
    @Published var generating = false
    /// Conversation currently streaming, so its live bubble only shows there.
    @Published var generatingConvID: UUID?
    @Published var compacting = false
    @Published var lastError: String?
    @Published var pendingToolPermission: PendingToolPermission?
    @Published var pendingAgentContinuation: PendingAgentContinuation?
    @Published var queuedMessage: QueuedChatMessage?
    let live = LiveStream()
    static let streamingSession: URLSession = {
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest  = 600   // idle between bytes (covers a slow first token)
        cfg.timeoutIntervalForResource = 3600
        cfg.waitsForConnectivity = true
        NetworkManager.applyProxy(to: cfg)
        return URLSession(configuration: cfg)
    }()
    /// Context consumed by the current conversation's last exchange; drives the
    /// usage bar. Stored per chat, so switching conversations follows the value.
    var contextUsed: Int? { current?.contextUsed }

    func setContextUsed(_ value: Int?, for id: UUID) {
        guard let i = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[i].contextUsed = value
    }

    private var task: Task<Void, Never>?
    private var watchdog: Task<Void, Never>?
    /// Where the running reply is streamed from, so Stop can cancel it on the engine too.
    private var activeStream: (port: Int, identity: String)?
    private var lastStreamActivity = Date()
    private var sawFirstToken = false
    private var slotConvID: UUID?
    private var agentContext: AgentRunContext?

    var agentFlowActive: Bool {
        generating || pendingToolPermission != nil || pendingAgentContinuation != nil || agentContext != nil
    }

    private var fileURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ToshLLM")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("conversations.json")
    }

    /// Projects live in their own file so conversations.json keeps its schema
    /// and older builds simply show the flat list.
    private var projectsURL: URL {
        fileURL.deletingLastPathComponent().appendingPathComponent("projects.json")
    }

    /// Backup files — written just before the primary, so they hold the
    /// previous (known-good) snapshot.  Load falls back to these when the
    /// primary is missing or corrupted.
    private var backupURL: URL {
        fileURL.deletingLastPathComponent().appendingPathComponent("conversations.json.backup")
    }

    private var backupProjectsURL: URL {
        fileURL.deletingLastPathComponent().appendingPathComponent("projects.json.backup")
    }

    init() {
        load()
        Self.live = self
        // Open ready to type a new message: reuse the most recent empty
        // conversation or start a fresh one. Earlier chats stay one click away.
        if let empty = conversations.first(where: { $0.messages.isEmpty }) {
            currentID = empty.id
        } else {
            newConversation()
        }
        pruneOrphanSlots()
        // A fresh engine has empty KV slots: forget which conversation slot 0
        // held, so the next turn restores the active one's persisted cache.
        NotificationCenter.default.addObserver(forName: .engineDidStart, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.slotConvID = nil }
        }
    }

    var currentIndex: Int? { conversations.firstIndex { $0.id == currentID } }
    var current: Conversation? { currentIndex.map { conversations[$0] } }

    func newConversation(in projectID: UUID? = nil) {
        // Reuse an existing empty conversation in the same scope instead of
        // piling up blanks when the button is clicked repeatedly.
        if let empty = conversations.first(where: { $0.messages.isEmpty && $0.projectID == projectID }) {
            currentID = empty.id
            lastError = nil
            return
        }
        let c = Conversation(title: "", projectID: projectID)
        conversations.insert(c, at: 0)
        currentID = c.id
        lastError = nil
    }

    func delete(_ c: Conversation) {
        if generating && c.id == currentID { stop() }
        if pendingToolPermission?.conversationID == c.id {
            pendingToolPermission = nil
            agentContext = nil
            task?.cancel()
        }
        if pendingAgentContinuation?.conversationID == c.id {
            pendingAgentContinuation = nil
            agentContext = nil
        }
        if queuedMessage?.conversationID == c.id { queuedMessage = nil }
        conversations.removeAll { $0.id == c.id }
        if currentID == c.id { currentID = conversations.first?.id }
        if conversations.isEmpty { newConversation() }
        // Drop its persisted KV slot file too, and forget it if it was loaded.
        try? FileManager.default.removeItem(
            at: ServerSettings.primarySlotCacheDir.appendingPathComponent(Self.slotFile(c.id)))
        if slotConvID == c.id { slotConvID = nil }
        save()
    }

    /// The live store, so the settings window (which has no access to the chat
    /// window's instance) can reach it.
    private(set) static weak var live: ChatStore?

    /// Erases the stored conversations without a store: the settings window can
    /// outlive the chat window, and then nothing holds them in memory.
    static func eraseStoredConversations() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ToshLLM")
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("conversations.json"))
        let slots = ServerSettings.primarySlotCacheDir
        guard let files = try? FileManager.default.contentsOfDirectory(at: slots,
                                                                      includingPropertiesForKeys: nil)
        else { return }
        for f in files where f.pathExtension == "bin"
            && UUID(uuidString: f.deletingPathExtension().lastPathComponent) != nil {
            try? FileManager.default.removeItem(at: f)
        }
    }

    /// Drops every conversation (projects and their prompts stay). Their
    /// persisted KV slots go with them, or they would outlive their chat.
    func deleteAll() {
        if generating { stop() }
        pendingToolPermission = nil
        pendingAgentContinuation = nil
        agentContext = nil
        queuedMessage = nil
        task?.cancel()
        for c in conversations {
            try? FileManager.default.removeItem(
                at: ServerSettings.primarySlotCacheDir.appendingPathComponent(Self.slotFile(c.id)))
        }
        conversations.removeAll()
        slotConvID = nil
        currentID = nil
        newConversation()
        save()
    }

    func rename(_ c: Conversation, to title: String) {
        guard let i = conversations.firstIndex(where: { $0.id == c.id }) else { return }
        conversations[i].title = title.trimmingCharacters(in: .whitespaces)
        save()
    }

    func togglePin(_ c: Conversation) {
        guard let i = conversations.firstIndex(where: { $0.id == c.id }) else { return }
        conversations[i].pinned = !(conversations[i].pinned ?? false)
        save()
    }

    func displayTitle(_ c: Conversation) -> String {
        if !c.title.isEmpty { return c.title }
        if let first = c.messages.first(where: { $0.role == "user" }) {
            let smart = Self.smartTitle(from: first.content)
            if !smart.isEmpty { return smart }
            if let name = first.attachments?.first?.name { return name }
        }
        return "…"
    }

    /// Sidebar title derived from a message: first meaningful line, markdown
    /// markers stripped, cut at a word boundary instead of mid-word.
    nonisolated static func smartTitle(from text: String, limit: Int = 48) -> String {
        let line = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty } ?? ""
        var clean = line.drop { "#>-*• \t".contains($0) }
            .replacingOccurrences(of: "`", with: "")
        clean = clean.trimmingCharacters(in: .whitespaces)
        guard clean.count > limit else { return clean }
        let cut = String(clean.prefix(limit))
        let word = cut.lastIndex(of: " ").map { String(cut[..<$0]) } ?? cut
        return (word.count >= limit / 2 ? word : cut) + "…"
    }

    // MARK: projects

    func project(id: UUID?) -> ChatProject? {
        guard let id else { return nil }
        return projects.first { $0.id == id }
    }

    @discardableResult
    func newProject(name: String) -> ChatProject {
        let p = ChatProject(name: name.trimmingCharacters(in: .whitespaces))
        projects.insert(p, at: 0)
        save()
        return p
    }

    func renameProject(_ p: ChatProject, to name: String) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        projects[i].name = name.trimmingCharacters(in: .whitespaces)
        save()
    }

    func togglePinProject(_ p: ChatProject) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        projects[i].pinned = !(projects[i].pinned ?? false)
        save()
    }

    func setProjectCollapsed(_ p: ChatProject, _ collapsed: Bool) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        projects[i].collapsed = collapsed
        save()
    }

    func setProjectPrompt(_ p: ChatProject, _ prompt: String) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        projects[i].systemPrompt = prompt
        save()
    }

    /// Removes the folder; its conversations survive as ungrouped.
    func deleteProject(_ p: ChatProject) {
        for i in conversations.indices where conversations[i].projectID == p.id {
            conversations[i].projectID = nil
        }
        projects.removeAll { $0.id == p.id }
        save()
    }

    func move(_ c: Conversation, toProject projectID: UUID?) {
        guard let i = conversations.firstIndex(where: { $0.id == c.id }) else { return }
        conversations[i].projectID = projectID
        save()
    }

    func setConversationPrompt(_ c: Conversation, _ prompt: String) {
        guard let i = conversations.firstIndex(where: { $0.id == c.id }) else { return }
        conversations[i].systemPrompt = prompt.isEmpty ? nil : prompt
        save()
    }

    func setConversationWorkingDirectory(_ c: Conversation, _ path: String?) {
        guard let i = conversations.firstIndex(where: { $0.id == c.id }) else { return }
        conversations[i].workingDirectory = (path?.isEmpty ?? true) ? nil : path
        save()
    }

    /// Folder picker for a project, so the menu entry is one line at each call site.
    func pickProjectWorkingDirectory(_ p: ChatProject) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let path = panel.url?.path {
            setProjectWorkingDirectory(p, path)
        }
    }

    func setProjectWorkingDirectory(_ p: ChatProject, _ path: String?) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        projects[i].workingDirectory = (path?.isEmpty ?? true) ? nil : path
        save()
    }

    /// System prompt actually sent: the chat's own, else its project's, else
    /// the global one. First non-empty wins.
    func effectiveSystemPrompt(global: String) -> String {
        guard let c = current else { return global }
        return Self.resolvePrompt(chat: c.systemPrompt,
                                  project: project(id: c.projectID)?.systemPrompt,
                                  global: global)
    }

    /// Working directory sent with tool calls: the chat's own, else its project's.
    func effectiveWorkingDirectory(for id: UUID?) -> String? {
        guard let c = conversations.first(where: { $0.id == (id ?? current?.id) }) else { return nil }
        return Self.meaningful(c.workingDirectory) ?? Self.meaningful(project(id: c.projectID)?.workingDirectory)
    }

    nonisolated static func resolvePrompt(chat: String?, project: String?, global: String) -> String {
        meaningful(chat) ?? meaningful(project) ?? global
    }

    private nonisolated static func meaningful(_ s: String?) -> String? {
        guard let s, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return s
    }

    // MARK: sending

    func send(text: String, attachments: [ChatAttachment] = [], images: [String] = [],
              port: Int, temperature: Double,
              maxTokens: Int, system: String, thinking: Bool,
              sampling: ChatSamplingSettings = ChatSamplingSettings(),
              modalities: ModelModalities? = nil) {
        guard !generating, let i = currentIndex else { return }
        lastError = nil
        pendingAgentContinuation = nil
        queuedMessage = nil
        agentContext = nil
        conversations[i].messages.append(ChatMessage(role: "user", content: text,
                                                     attachments: attachments.isEmpty ? nil : attachments,
                                                     imageURIs: images.isEmpty ? nil : images))
        if conversations[i].title.isEmpty {
            let smart = Self.smartTitle(from: text)
            conversations[i].title = smart.isEmpty ? (attachments.first?.name ?? "…") : smart
        }
        stream(into: i, port: port, temperature: temperature, maxTokens: maxTokens,
               system: system, thinking: thinking, sampling: sampling, modalities: modalities)
    }

    func regenerate(port: Int, temperature: Double, maxTokens: Int, system: String, thinking: Bool,
                    sampling: ChatSamplingSettings = ChatSamplingSettings(),
                    modalities: ModelModalities? = nil) {
        guard !generating, let i = currentIndex,
              conversations[i].messages.last?.role == "assistant" else { return }
        let path = Array(conversations[i].messages.dropLast())
        beginAlternativeBranch(conversationIndex: i, messages: path)
        stream(into: i, port: port, temperature: temperature, maxTokens: maxTokens,
               system: system, thinking: thinking, sampling: sampling, modalities: modalities)
    }

    func continueResponse(port: Int, temperature: Double, maxTokens: Int,
                          system: String, thinking: Bool,
                          sampling: ChatSamplingSettings = ChatSamplingSettings(),
                          modalities: ModelModalities? = nil) {
        guard !generating, let i = currentIndex,
              conversations[i].messages.last?.role == "assistant" else { return }
        stream(into: i, port: port, temperature: temperature, maxTokens: maxTokens,
               system: system, thinking: thinking,
               sampling: sampling, modalities: modalities,
               continuationInstruction: "Continue exactly where the previous response stopped. Do not repeat any text.")
    }

    private func stream(into i: Int, port: Int, temperature: Double, maxTokens: Int,
                        system: String, thinking: Bool,
                        sampling: ChatSamplingSettings = ChatSamplingSettings(),
                        modalities: ModelModalities? = nil,
                        continuationInstruction: String? = nil,
                        agentRun: AgentRunContext? = nil) {
        generating = true
        generatingConvID = conversations[i].id
        live.reset()
        startWatchdog(port: port)
        // The user can switch or delete conversations mid-stream; the result
        // must land in the one this request started from, found by id.
        let convID = conversations[i].id
        let toolCwd = effectiveWorkingDirectory(for: convID)
        activeStream = (port, ChatStreamIdentity.value(conversationID: convID,
                                                       model: ServerSettings.activeRouterModel()))
        let systemWithCwd = toolCwd.map {
            (system.isEmpty ? "" : system + "\n\n")
            + "File tools work inside \($0). Use paths relative to it, and never call them for text that only exists in this conversation."
        } ?? system
        let toolsEnabled = UserDefaults.standard.bool(forKey: SettingsKeys.agentToolsEnabled)
        let javaScriptEnabled = UserDefaults.standard.bool(forKey: SettingsKeys.jsSandboxEnabled)
        let memoryToolsEnabled = ChatMemoryService.isEnabled
        let agentTurnLimit = Self.configuredAgentTurnLimit
        let enabledToolNames = conversations[i].enabledToolNames

        var history = Self.requestHistory(system: systemWithCwd,
                                          summary: conversations[i].summary,
                                          messages: conversations[i].messages,
                                          from: conversations[i].summarizedCount ?? 0,
                                          archived: conversations[i].archived,
                                          modalities: modalities)
        if let continuationInstruction {
            history.append(["role": "user", "content": continuationInstruction])
        }

        // Reasoning off can come from the toggle or a typed /no_think; a typed
        // switch overrides the toggle for this turn. Not persisted to history.
        var reasoningOff = !thinking || sampling.reasoningEffort == "off"
        if let last = history.lastIndex(where: { ($0["role"] as? String) == "user" }) {
            let typed = Self.messageText(history[last]["content"])
            if typed.contains("/no_think") { reasoningOff = true }
            else if typed.contains("/think") { reasoningOff = false }

            if reasoningOff, !typed.contains("/no_think") {
                if let s = history[last]["content"] as? String {
                    history[last]["content"] = s + "\n/no_think"
                } else if var parts = history[last]["content"] as? [[String: Any]] {
                    if let ti = parts.firstIndex(where: { ($0["type"] as? String) == "text" }) {
                        parts[ti]["text"] = ((parts[ti]["text"] as? String) ?? "") + "\n/no_think"
                    } else {
                        parts.insert(["type": "text", "text": "/no_think"], at: 0)
                    }
                    history[last]["content"] = parts
                }
            }
        }

        conversations[i].messages.append(ChatMessage(role: "assistant", content: ""))

        let buffer = StreamBuffer()
        let pump = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                // Push-based: wake immediately when data arrives or on finish.
                await buffer.waitForData()
                guard let snap = buffer.take() else { continue }
                if snap.done { break }   // finish() publishes the final transcript
                self?.live.update(reasoning: snap.reasoning, visible: snap.visible, speed: snap.speed)
                let prefilling = snap.reasoning.isEmpty && snap.visible.isEmpty
                self?.live.setPrefillProgress(prefilling ? snap.progress : nil)
                self?.noteStreamActivity()
                // Adaptive throttle: longer sleep when output is long to avoid UI thrash.
                let n = snap.visible.count
                let ms = n > 12000 ? 600 : n > 6000 ? 350 : n > 2500 ? 180 : 80
                try? await Task.sleep(for: .milliseconds(ms))
            }
        }

        task = Task.detached(priority: .userInitiated) { [weak self, buffer, pump] in
            var nTokens = 0
            let tSent = Date()
            var tFirst: Date?
            // Token arrival times within the last seconds; drives the live
            // t/s as an instantaneous reading instead of a cumulative average.
            var stamps: [Date] = []
            var accumulator = ChatStreamAccumulator()
            var availableTools = agentRun?.tools ?? []
            var lastFlush = Date.distantPast
            var cancelled = false
            var reportedError = false
            var bytesReceived = 0

            func composed() -> String {
                guard !accumulator.reasoning.isEmpty else { return accumulator.visible }
                return "<think>" + accumulator.reasoning
                    + (accumulator.visible.isEmpty ? "" : "</think>" + accumulator.visible)
            }

            func flush() {
                let now = Date()
                let interval: TimeInterval = {
                    let n = accumulator.visible.count
                    return n > 12000 ? 0.6 : n > 6000 ? 0.35 : n > 2500 ? 0.18 : 0.08
                }()
                guard now.timeIntervalSince(lastFlush) > interval else { return }
                lastFlush = now
                if let cut = stamps.firstIndex(where: { now.timeIntervalSince($0) < 3 }) {
                    stamps.removeFirst(cut)
                } else {
                    stamps.removeAll()
                }
                var speed: Double?
                if let first = stamps.first, stamps.count > 4 {
                    let dt = now.timeIntervalSince(first)
                    if dt > 0.3 { speed = Double(stamps.count - 1) / dt }
                }
                buffer.write(reasoning: accumulator.reasoning,
                             visible: accumulator.visible, speed: speed)
            }

            func drain(_ bytes: URLSession.AsyncBytes) async throws -> Bool {
                for try await line in bytes.lines {
                    if Task.isCancelled { throw CancellationError() }
                    bytesReceived += line.utf8.count + 1
                    guard let event = try accumulator.consume(line) else { continue }
                    if let progress = event.progress { buffer.writeProgress(progress) }
                    if event.receivedContent {
                        let now = Date()
                        if tFirst == nil { tFirst = now }
                        nTokens += 1
                        stamps.append(now)
                        flush()
                    }
                    if event.completed { return true }
                }
                return false
            }

            do {
                // Restore this conversation's persisted KV (if any) so the slot
                // holds the unchanged history and only the new turn is prefilled.
                await self?.prepareSlot(convID: convID, port: port)

                if availableTools.isEmpty {
                    if toolsEnabled {
                        availableTools = try await ChatToolsService.list(port: port)
                        // without a folder these write wherever the engine happens to run,
                        // and the model invents paths for text that is not a file at all
                        if toolCwd == nil {
                            availableTools.removeAll { $0.usesCwd && $0.writesData }
                        }
                    }
                    if javaScriptEnabled { availableTools.append(JavaScriptSandboxService.tool) }
                    if memoryToolsEnabled { availableTools.append(contentsOf: ChatMemoryService.tools) }
                    availableTools += await ToshMCPService.shared.discoverTools()
                    if ToolSupport.isBlocked(ToolSupport.currentModelIdentity) { availableTools = [] }
                    if let enabledToolNames {
                        let selected = Set(enabledToolNames)
                        availableTools.removeAll { !selected.contains($0.name) }
                    }
                }

                guard let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else { return }
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.timeoutInterval = 600
                if let key = ServerSettings.activeAPIKey() {
                    req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
                }
                let activeModel = ServerSettings.activeRouterModel()
                let streamIdentity = ChatStreamIdentity.value(conversationID: convID, model: activeModel)
                req.setValue(streamIdentity, forHTTPHeaderField: "X-Conversation-Id")
                var body: [String: Any] = [
                    "messages": history,
                    "stream": true,
                    "temperature": temperature,
                    "top_p": sampling.topP,
                    "min_p": sampling.minP,
                    "top_k": sampling.topK,
                    "repeat_penalty": sampling.repeatPenalty,
                    "repeat_last_n": max(0, sampling.repeatLastN),
                    "dynatemp_range": sampling.dynatempRange,
                    "dynatemp_exponent": sampling.dynatempExponent,
                    "xtc_probability": sampling.xtcProbability,
                    "xtc_threshold": sampling.xtcThreshold,
                    "typ_p": sampling.typicalP,
                    "presence_penalty": sampling.presencePenalty,
                    "frequency_penalty": sampling.frequencyPenalty,
                    "dry_multiplier": sampling.dryMultiplier,
                    "dry_base": sampling.dryBase,
                    "dry_allowed_length": sampling.dryAllowedLength,
                    // settings saved before the engine rejected negatives still hold -1
                    "dry_penalty_last_n": max(0, sampling.dryPenaltyLastN),
                    "backend_sampling": sampling.backendSampling,
                    "seed": sampling.seed,
                    "max_tokens": maxTokens,
                    // Reuse the server-side KV cache for the unchanged history
                    // prefix so each turn only processes the new tokens.
                    "cache_prompt": true,
                    // Ask for a final usage chunk to drive the context meter.
                    "stream_options": ["include_usage": true],
                    // Stream prompt-processing progress (in `prompt_progress`).
                    "return_progress": true,
                    "timings_per_token": true,
                ]
                let samplerOrder = sampling.samplers.split(separator: ";")
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                if !samplerOrder.isEmpty { body["samplers"] = samplerOrder }
                if let activeModel { body["model"] = activeModel }
                if !availableTools.isEmpty {
                    body["tools"] = availableTools.compactMap(\.openAIDefinition)
                    body["tool_choice"] = "auto"
                }
                if reasoningOff {
                    body["chat_template_kwargs"] = ["enable_thinking": false]
                    // For templates that ignore enable_thinking (Qwen3.6 still
                    // prefills <think>): 0 forces the reasoning block to close now.
                    body["thinking_budget_tokens"] = 0
                } else {
                    var kwargs: [String: Any] = ["enable_thinking": true]
                    // "default" leaves the value out so the template picks its own.
                    if sampling.reasoningEffort != "default" {
                        kwargs["reasoning_effort"] = sampling.reasoningEffort
                    }
                    body["chat_template_kwargs"] = kwargs
                    if let budget = Self.reasoningBudget(for: sampling.reasoningEffort) {
                        body["thinking_budget_tokens"] = budget
                    }
                    body["reasoning_control"] = true
                }
                // Pin to slot 0 so the saved/restored KV always matches the chat.
                if self?.slotPersistEnabled == true { body["id_slot"] = 0 }
                if let data = sampling.customJSON.data(using: .utf8),
                   let custom = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    body.merge(custom) { _, customValue in customValue }
                }
                req.httpBody = try JSONSerialization.data(withJSONObject: body)

                let (bytes, response) = try await ChatStore.streamingSession.bytes(for: req)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                guard status == 200 else {
                    // The error body explains the cause (e.g. context overflow);
                    // surface it instead of a generic "bad server response".
                    var raw = ""
                    for try await line in bytes.lines {
                        raw += line
                        if raw.count > 4000 { break }
                    }
                    throw StreamError(message: Self.describeServerError(status: status, body: raw))
                }

                var completed = false
                var resumeError: Error?
                do {
                    completed = try await drain(bytes)
                } catch {
                    if error is CancellationError { throw error }
                    resumeError = error
                }

                var attempts = 0
                while !completed, attempts < 3, !Task.isCancelled {
                    attempts += 1
                    let before = bytesReceived
                    do {
                        guard let url = ChatStreamIdentity.resumeURL(
                            port: port, identity: streamIdentity, from: bytesReceived)
                        else { throw StreamError(message: "Invalid stream identity") }
                        var resumeRequest = URLRequest(url: url)
                        resumeRequest.timeoutInterval = 30
                        if let key = ServerSettings.activeAPIKey() {
                            resumeRequest.setValue("Bearer " + key,
                                                   forHTTPHeaderField: "Authorization")
                        }
                        let (resumeBytes, resumeResponse) = try await ChatStore.streamingSession.bytes(for: resumeRequest)
                        let resumeStatus = (resumeResponse as? HTTPURLResponse)?.statusCode ?? 0
                        guard resumeStatus == 200 else {
                            throw StreamError(message: "Stream resume failed (HTTP \(resumeStatus))")
                        }
                        completed = try await drain(resumeBytes)
                        if !completed, bytesReceived == before {
                            throw StreamError(message: "Stream resume returned no new data")
                        }
                    } catch {
                        if error is CancellationError { throw error }
                        resumeError = error
                    }
                }
                if !completed {
                    throw resumeError ?? StreamError(message: "The response stream ended before completion")
                }
            } catch {
                if error is CancellationError {
                    cancelled = true
                } else {
                    reportedError = true
                    AppLog.chat.error("stream failed: \(error.localizedDescription)")
                    let store = self
                    let raw = error.localizedDescription
                    // the engine kills the turn when the model writes the call in the wrong
                    // shape, and it does it on every tool, so the model stops getting them
                    let rejected = raw.contains("peg-native") && !availableTools.isEmpty
                    if rejected { ToolSupport.block(ToolSupport.currentModelIdentity) }
                    let message = rejected
                        ? "Este modelo escribe mal las llamadas a herramientas y el motor cortó la respuesta; se le han desactivado, vuelve a enviar / this model writes tool calls in the wrong shape and the engine stopped the answer; they are now off for it, send again"
                        : raw
                    await MainActor.run { store?.lastError = message }
                }
            }

            // Tell the pump to stop; the final transcript is published below.
            buffer.finish()

            let finalSpeed: Double? = tFirst.flatMap { start in
                let dt = Date().timeIntervalSince(start)
                return dt > 0.4 && nTokens > 1 ? Double(nTokens) / dt : nil
            }
            let ttft: Double? = tFirst.map { $0.timeIntervalSince(tSent) * 1000 }
            let hasVisibleAnswer = !accumulator.visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let streamedText = hasVisibleAnswer ? composed() : ""
            let finalUsage = accumulator.usage
            let finalAccept = accumulator.mtpAccept
            let finalTimings: ChatTimings? = {
                var t = accumulator.timings
                t?.timeToFirstTokenMilliseconds = ttft
                return t
            }()
            let finalFinishReason = accumulator.finishReason
            let wasCancelled = cancelled
            let didReportError = reportedError
            let hadReasoning = !accumulator.reasoning.isEmpty
            let finalToolCalls = accumulator.toolCalls.filter { !$0.name.isEmpty }
            let finalText = streamedText
            let nextAgentRun = AgentRunContext(
                port: port, temperature: temperature, maxTokens: maxTokens,
                system: system, thinking: thinking, sampling: sampling,
                modalities: modalities,
                remainingTurns: agentRun?.remainingTurns ?? agentTurnLimit,
                tools: availableTools, workingDirectory: toolCwd)
            let store = self
            let shouldDeliverQueued: Bool = await MainActor.run {
                if !wasCancelled && !didReportError && hadReasoning && !hasVisibleAnswer
                    && finalToolCalls.isEmpty {
                    store?.lastError = Self.emptyResponseMessage(finishReason: finalFinishReason)
                }
                if let finalUsage { store?.setContextUsed(finalUsage.prompt + finalUsage.completion, for: convID) }
                store?.finish(conversation: convID, text: finalText, speed: finalSpeed,
                              mtpAccept: finalAccept, timings: finalTimings,
                              toolCalls: finalToolCalls)
                let shouldDeliverQueued = store?.queuedMessage?.conversationID == convID
                if shouldDeliverQueued {
                    if !finalToolCalls.isEmpty { store?.interruptPendingToolCalls(conversation: convID) }
                } else if !wasCancelled && !didReportError && !finalToolCalls.isEmpty {
                    store?.beginToolPermissions(conversation: convID, context: nextAgentRun)
                } else {
                    store?.agentContext = nil
                    store?.compactIfNeeded(conversation: convID, port: port)
                }
                return shouldDeliverQueued
            }
            // Persist the conversation's KV after a real answer, so reopening it
            // (or restarting the engine) skips re-prefilling the history.
            if !wasCancelled && !didReportError && hasVisibleAnswer {
                await self?.saveSlot(convID: convID, port: port)
            }
            if shouldDeliverQueued {
                await MainActor.run {
                    store?.deliverQueuedMessage(conversation: convID, context: nextAgentRun)
                }
            }
            pump.cancel()
        }
    }

    private func finish(conversation id: UUID, text: String, speed: Double?,
                        mtpAccept: Double? = nil, timings: ChatTimings? = nil,
                        toolCalls: [ChatToolCall] = []) {
        if let i = conversations.firstIndex(where: { $0.id == id }) {
            if let j = conversations[i].messages.indices.last,
               conversations[i].messages[j].role == "assistant" {
                if text.isEmpty && toolCalls.isEmpty {
                    conversations[i].messages.removeLast()
                    Self.clampSummary(&conversations[i])
                } else {
                    conversations[i].messages[j].content = text
                    conversations[i].messages[j].genSpeed = speed
                    conversations[i].messages[j].mtpAccept = mtpAccept
                    conversations[i].messages[j].timings = timings
                    conversations[i].messages[j].toolCalls = toolCalls.isEmpty ? nil : toolCalls
                    conversations[i].messages[j].model = ServerSettings.activeRouterModel()
                }
            }
            conversations[i].updated = Date()
        }
        // Publish the completed transcript before replacing StreamingBubble.
        // Otherwise SwiftUI can briefly create and retain an empty bubble.
        generating = false
        generatingConvID = nil
        live.reset()
        task = nil
        watchdog?.cancel()
        watchdog = nil
        save()
    }

    private func beginToolPermissions(conversation id: UUID, context: AgentRunContext) {
        guard context.remainingTurns > 0 else {
            agentContext = context
            pendingAgentContinuation = PendingAgentContinuation(conversationID: id)
            return
        }
        agentContext = context
        advanceToolPermissions(conversation: id)
    }

    private func advanceToolPermissions(conversation id: UUID) {
        if queuedMessage?.conversationID == id, let context = agentContext {
            interruptPendingToolCalls(conversation: id)
            deliverQueuedMessage(conversation: id, context: context)
            return
        }
        guard let conversationIndex = conversations.firstIndex(where: { $0.id == id }),
              let messageIndex = conversations[conversationIndex].messages.lastIndex(where: {
                  $0.role == "assistant" && !($0.toolCalls ?? []).isEmpty
              }) else {
            agentContext = nil
            return
        }
        let message = conversations[conversationIndex].messages[messageIndex]
        if let call = message.toolCalls?.first(where: { $0.state == .pending }) {
            let info = agentContext?.tools.first(where: { $0.name == call.name })
            let request = PendingToolPermission(
                conversationID: id, messageID: message.id, callID: call.id,
                name: call.name, displayName: info?.displayName ?? call.name,
                arguments: call.arguments, writesData: info?.writesData ?? true,
                serverID: info?.mcpServerID,
                serverName: info?.mcpServerID.flatMap { serverID in
                    MCPServerStore.load().first(where: { $0.id == serverID })?.name
                })
            pendingToolPermission = request
            updateToolCall(request, state: .awaitingPermission)
            if ChatToolsService.isAlwaysAllowed(call.name) {
                respondToToolPermission(.once)
            }
            return
        }

        guard var context = agentContext else { return }
        context.remainingTurns -= 1
        agentContext = context
        guard context.remainingTurns > 0 else {
            pendingAgentContinuation = PendingAgentContinuation(conversationID: id)
            return
        }
        stream(into: conversationIndex, port: context.port, temperature: context.temperature,
               maxTokens: context.maxTokens, system: context.system, thinking: context.thinking,
               sampling: context.sampling, modalities: context.modalities, agentRun: context)
    }

    func queueMessage(text: String, attachments: [ChatAttachment], images: [String]) {
        guard let conversationID = currentID,
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !attachments.isEmpty || !images.isEmpty else { return }
        queuedMessage = QueuedChatMessage(conversationID: conversationID, text: text,
                                          attachments: attachments, imageURIs: images)
        if !generating, pendingToolPermission?.conversationID == conversationID,
           let context = agentContext {
            interruptPendingToolCalls(conversation: conversationID)
            deliverQueuedMessage(conversation: conversationID, context: context)
        }
    }

    func cancelQueuedMessage() {
        queuedMessage = nil
    }

    private func interruptPendingToolCalls(conversation id: UUID) {
        pendingToolPermission = nil
        guard let i = conversations.firstIndex(where: { $0.id == id }),
              let j = conversations[i].messages.lastIndex(where: {
                  $0.role == "assistant" && !($0.toolCalls ?? []).isEmpty
              }) else { return }
        let interruption = "Tool execution was interrupted by a new user message."
        var resultMessages: [ChatMessage] = []
        for index in conversations[i].messages[j].toolCalls?.indices ?? 0..<0 {
            guard conversations[i].messages[j].toolCalls?[index].state == .pending
                    || conversations[i].messages[j].toolCalls?[index].state == .awaitingPermission else { continue }
            conversations[i].messages[j].toolCalls?[index].state = .denied
            conversations[i].messages[j].toolCalls?[index].result = interruption
            conversations[i].messages[j].toolCalls?[index].finishedAt = Date()
            if let call = conversations[i].messages[j].toolCalls?[index] {
                resultMessages.append(ChatMessage(role: "tool", content: interruption,
                                                  toolCallID: call.serverID ?? call.id.uuidString))
            }
        }
        conversations[i].messages.append(contentsOf: resultMessages)
        save()
    }

    private func deliverQueuedMessage(conversation id: UUID, context: AgentRunContext) {
        guard let queued = queuedMessage, queued.conversationID == id,
              let i = conversations.firstIndex(where: { $0.id == id }) else { return }
        queuedMessage = nil
        pendingToolPermission = nil
        pendingAgentContinuation = nil
        agentContext = nil
        conversations[i].messages.append(ChatMessage(
            role: "user", content: queued.text,
            attachments: queued.attachments.isEmpty ? nil : queued.attachments,
            imageURIs: queued.imageURIs.isEmpty ? nil : queued.imageURIs))
        stream(into: i, port: context.port, temperature: context.temperature,
               maxTokens: context.maxTokens, system: context.system, thinking: context.thinking,
               sampling: context.sampling, modalities: context.modalities)
    }

    func respondToAgentContinuation(_ shouldContinue: Bool) {
        guard let pending = pendingAgentContinuation else { return }
        pendingAgentContinuation = nil
        lastError = nil
        guard shouldContinue, var context = agentContext,
              let conversationIndex = conversations.firstIndex(where: { $0.id == pending.conversationID }) else {
            agentContext = nil
            return
        }
        context.remainingTurns = Self.configuredAgentTurnLimit
        agentContext = context
        stream(into: conversationIndex, port: context.port, temperature: context.temperature,
               maxTokens: context.maxTokens, system: context.system, thinking: context.thinking,
               sampling: context.sampling, modalities: context.modalities, agentRun: context)
    }

    func respondToToolPermission(_ decision: ToolPermissionDecision) {
        guard let request = pendingToolPermission else { return }
        pendingToolPermission = nil
        if decision == .deny {
            let result = "Tool execution was denied by the user."
            completeToolCall(request, result: result, state: .denied)
            advanceToolPermissions(conversation: request.conversationID)
            return
        }
        if decision == .always {
            ChatToolsService.allowAlways(request.name)
        } else if decision == .alwaysServer, let serverID = request.serverID,
                  let context = agentContext {
            for tool in context.tools where tool.mcpServerID == serverID {
                ChatToolsService.allowAlways(tool.name)
            }
        }
        updateToolCall(request, state: .running, startedAt: Date())
        task = Task { [weak self] in
            do {
                let arguments = try ChatToolsService.parseArguments(request.arguments)
                guard let context = self?.agentContext else { return }
                let result: ToolExecutionResult
                if let tool = context.tools.first(where: { $0.name == request.name }),
                   let serverID = tool.mcpServerID, let remoteName = tool.remoteName {
                    result = try await ToshMCPService.shared.call(
                        serverID: serverID, name: remoteName, arguments: arguments)
                } else if request.name == JavaScriptSandboxService.toolName {
                    result = await JavaScriptSandboxService.execute(arguments: arguments)
                } else if ChatMemoryService.toolNames.contains(request.name) {
                    result = await MainActor.run {
                        self?.runMemoryTool(request.name, arguments: arguments)
                            ?? ToolExecutionResult(content: "No open conversation.", isError: true)
                    }
                } else if request.name == "exec_shell_command" {
                    result = try await ChatToolsService.executeStreaming(
                        name: request.name, arguments: arguments, port: context.port,
                        workingDirectory: context.workingDirectory
                    ) { [weak self] partial in
                        await MainActor.run { self?.updateToolCallResult(request, result: partial) }
                    }
                } else {
                    result = try await ChatToolsService.execute(
                        name: request.name, arguments: arguments, port: context.port,
                        workingDirectory: context.workingDirectory)
                }
                guard !Task.isCancelled else { return }
                self?.completeToolCall(request, result: result.content,
                                       state: result.isError ? .failed : .completed)
                self?.advanceToolPermissions(conversation: request.conversationID)
            } catch {
                guard !Task.isCancelled else { return }
                self?.completeToolCall(request, result: error.localizedDescription, state: .failed)
                self?.advanceToolPermissions(conversation: request.conversationID)
            }
        }
    }

    private func updateToolCall(_ request: PendingToolPermission, state: ChatToolCallState,
                                startedAt: Date? = nil) {
        guard let i = conversations.firstIndex(where: { $0.id == request.conversationID }),
              let j = conversations[i].messages.firstIndex(where: { $0.id == request.messageID }),
              let k = conversations[i].messages[j].toolCalls?.firstIndex(where: { $0.id == request.callID })
        else { return }
        conversations[i].messages[j].toolCalls?[k].state = state
        if let startedAt { conversations[i].messages[j].toolCalls?[k].startedAt = startedAt }
        save()
    }

    private func completeToolCall(_ request: PendingToolPermission, result: String,
                                  state: ChatToolCallState) {
        guard let i = conversations.firstIndex(where: { $0.id == request.conversationID }),
              let j = conversations[i].messages.firstIndex(where: { $0.id == request.messageID }),
              let k = conversations[i].messages[j].toolCalls?.firstIndex(where: { $0.id == request.callID })
        else { return }
        conversations[i].messages[j].toolCalls?[k].state = state
        conversations[i].messages[j].toolCalls?[k].result = result
        conversations[i].messages[j].toolCalls?[k].finishedAt = Date()
        let serverID = conversations[i].messages[j].toolCalls?[k].serverID ?? request.callID.uuidString
        conversations[i].messages.append(ChatMessage(role: "tool", content: result, toolCallID: serverID))
        conversations[i].updated = Date()
        save()
    }

    private func updateToolCallResult(_ request: PendingToolPermission, result: String) {
        guard let i = conversations.firstIndex(where: { $0.id == request.conversationID }),
              let j = conversations[i].messages.firstIndex(where: { $0.id == request.messageID }),
              let k = conversations[i].messages[j].toolCalls?.firstIndex(where: { $0.id == request.callID })
        else { return }
        conversations[i].messages[j].toolCalls?[k].result = result
    }

    // MARK: KV slot persistence

    /// Read live from defaults so toggling it in Settings takes effect next turn.
    nonisolated var slotPersistEnabled: Bool {
        let d = UserDefaults.standard
        guard d.bool(forKey: SettingsKeys.persistCache) else { return false }
        guard d.object(forKey: SettingsKeys.faAmd) as? Bool ?? ServerSettings.defaultFaAmd else { return false }
        // Only an actually loaded projector blocks slot save/restore; with the
        // vision eye off the model runs text-only and persistence works.
        guard d.object(forKey: SettingsKeys.loadVision) as? Bool ?? true else { return true }
        return ServerSettings.mmprojPath(forModel: d.string(forKey: SettingsKeys.modelPath) ?? "") == nil
    }

    nonisolated private static func slotFile(_ id: UUID) -> String { "\(id.uuidString).bin" }

    /// POST /slots/0?action=save|restore (best-effort; a missing file on restore
    /// just means a cold prefill, which is harmless).
    nonisolated private func slotAction(_ action: String, convID: UUID, port: Int) async {
        guard var comps = URLComponents(string: "http://127.0.0.1:\(port)/slots/0") else { return }
        comps.queryItems = [URLQueryItem(name: "action", value: action)]
        guard let url = comps.url else { return }
        let filename = Self.slotFile(convID)
        for attempt in 1...3 {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if let key = ServerSettings.activeAPIKey() {
                req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
            }
            guard let body = try? JSONSerialization.data(withJSONObject: ["filename": filename]) else {
                AppLog.chat.warning("slot \(action) attempt \(attempt): JSON encoding failed for \(convID.uuidString)")
                if attempt < 3 { try? await Task.sleep(for: .milliseconds(200)) }
                continue
            }
            req.httpBody = body
            do {
                _ = try await NetworkManager.session.data(for: req)
                if attempt > 1 {
                    AppLog.chat.info("slot \(action) succeeded on attempt \(attempt) for \(convID.uuidString)")
                }
                return
            } catch {
                AppLog.chat.warning("slot \(action) attempt \(attempt) failed for \(convID.uuidString): \(error.localizedDescription)")
                if attempt < 3 { try? await Task.sleep(for: .milliseconds(200)) }
            }
        }
        AppLog.chat.error("slot \(action) failed after 3 attempts for \(convID.uuidString)")
    }

    /// Before a turn: if slot 0 doesn't already hold this conversation, restore
    /// its persisted KV so only the new tokens get prefilled.
    func prepareSlot(convID: UUID, port: Int) async {
        guard slotPersistEnabled, slotConvID != convID else { return }
        // No file yet (new conversation or KV layout change): skip the failing restore.
        let file = ServerSettings.primarySlotCacheDir.appendingPathComponent(Self.slotFile(convID))
        guard FileManager.default.fileExists(atPath: file.path) else { slotConvID = convID; return }
        await slotAction("restore", convID: convID, port: port)
        slotConvID = convID
    }

    /// After a turn completes: persist the conversation's KV to disk.
    nonisolated func saveSlot(convID: UUID, port: Int) async {
        guard slotPersistEnabled else { return }
        await slotAction("save", convID: convID, port: port)
    }

    /// Drop slot files with no matching conversation (deleted while disabled, or
    /// left over). Bounds disk use to the conversations that still exist.
    private func pruneOrphanSlots() {
        let dir = ServerSettings.primarySlotCacheDir
        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { return }
        let ids = Set(conversations.map { $0.id.uuidString })
        for f in files where f.pathExtension == "bin" {
            let base = f.deletingPathExtension().lastPathComponent
            // Only prune per-conversation slot files (named by UUID); leave other
            // files like the external-client prefix (external.bin) untouched.
            guard UUID(uuidString: base) != nil else { continue }
            if !ids.contains(base) { try? FileManager.default.removeItem(at: f) }
        }
    }

    // MARK: stall watchdog

    /// Called from the stream whenever a token reaches the UI; resets the
    /// inactivity timer and marks that generation (not prefill) has begun.
    func noteStreamActivity() {
        lastStreamActivity = Date()
        sawFirstToken = true
    }

    private func startWatchdog(port: Int) {
        watchdog?.cancel()
        lastStreamActivity = Date()
        sawFirstToken = false
        watchdog = Task { [weak self] in
            while true {
                try? await Task.sleep(for: .seconds(5))
                guard let self, !Task.isCancelled, self.generating else { return }
                let idle = Date().timeIntervalSince(self.lastStreamActivity)
                let limit: TimeInterval = self.sawFirstToken ? 30 : 180
                if idle > limit {
                    self.handleStreamStall()
                    return
                }
            }
        }
    }

    /// The engine stopped producing tokens while alive: treat it as a driver
    /// deadlock, stop the engine to free its memory, and tell the user.
    private func handleStreamStall() {
        AppLog.chat.error("stream stalled; stopping engine")
        task?.cancel()
        task = nil
        watchdog = nil
        ServerManager.shared.active.stop()
        lastError = Self.stallMessage
        generating = false
        generatingConvID = nil
        live.reset()
        save()
    }

    nonisolated static var stallMessage: String {
        "El motor dejó de responder y se detuvo para liberar memoria. Suele pasar con modelos MoE grandes en GPU AMD: usa un modelo denso (8B) o sube 'Expertos MoE en CPU'. / The engine stopped responding and was stopped to free memory. This happens with large MoE models on AMD GPUs: use a dense model (8B) or raise 'MoE experts on CPU'."
    }

    // MARK: auto-compaction

    nonisolated static func requestHistory(system: String, summary: String?,
                                           messages: [ChatMessage], from start: Int,
                                           archived: [ArchivedBlock]? = nil,
                                           modalities: ModelModalities? = nil) -> [[String: Any]] {
        var history: [[String: Any]] = []
        var sys = system.trimmingCharacters(in: .whitespaces)
        if let summary, !summary.isEmpty {
            sys += (sys.isEmpty ? "" : "\n\n")
                + "Summary of the earlier part of this conversation:\n" + summary
        }
        if !sys.isEmpty { history.append(["role": "system", "content": sys]) }
        let safeStart = min(max(0, start), messages.count)
        let skipped = ChatMemoryService.archivedIndices(archived)
        history += messages[safeStart...].enumerated().compactMap { offset, m -> [String: Any]? in
            guard !skipped.contains(safeStart + offset) else { return nil }
            if m.role == "tool", let callID = m.toolCallID {
                return ["role": "tool", "tool_call_id": callID, "content": m.content]
            }
            let text = m.role == "assistant" ? m.parts.body : m.wireContent
            if m.role == "assistant", let calls = m.toolCalls, !calls.isEmpty {
                let payload: [[String: Any]] = calls.map { call in
                    ["id": call.serverID ?? call.id.uuidString,
                     "type": "function",
                     "function": ["name": call.name, "arguments": call.arguments]]
                }
                return ["role": "assistant", "content": text, "tool_calls": payload]
            }
            guard m.role != "assistant" || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            // A user turn with images uses the OpenAI multimodal content array
            // (text part + image_url parts); everything else stays a plain string.
            let media = m.attachments?.filter { $0.mediaKind != nil && $0.base64Payload != nil } ?? []
            if m.role == "user", !(m.imageURIs ?? []).isEmpty || !media.isEmpty {
                var parts: [[String: Any]] = []
                if !text.isEmpty { parts.append(["type": "text", "text": text]) }
                if modalities?.vision != false {
                    for uri in m.imageURIs ?? [] {
                        parts.append(["type": "image_url", "image_url": ["url": uri]])
                    }
                }
                for attachment in media {
                    guard let data = attachment.base64Payload else { continue }
                    if attachment.mediaKind == "audio" {
                        guard modalities?.audio != false else { continue }
                        parts.append(["type": "input_audio",
                                      "input_audio": ["data": data, "format": attachment.audioInputFormat]])
                    } else if modalities?.video != false {
                        parts.append(["type": "input_video",
                                      "input_video": ["data": data, "format": attachment.videoInputFormat]])
                    }
                }
                return ["role": m.role, "content": parts]
            }
            return ["role": m.role, "content": text]
        }
        return history
    }

    /// Plain text of a message's content, whether it's a string or the
    /// multimodal parts array. Used to detect a typed /no_think | /think switch.
    private static func messageText(_ content: Any?) -> String {
        if let s = content as? String { return s }
        if let parts = content as? [[String: Any]] {
            return parts.compactMap { $0["text"] as? String }.joined(separator: " ")
        }
        return ""
    }

    nonisolated static func compactionCutoff(messages: [ChatMessage], alreadyCompacted: Int,
                                             keepTokens: Int = 0) -> Int? {
        var cutoff = messages.count - 4
        if keepTokens > 0 {
            var kept = 0
            var i = messages.count - 1
            while i >= 0, kept < keepTokens {
                kept += messages[i].estimatedTokens
                i -= 1
            }
            cutoff = min(cutoff, i + 1)
        }
        while cutoff > 0 && messages[cutoff].role != "user" { cutoff -= 1 }
        guard cutoff >= alreadyCompacted + 2 else { return nil }
        return cutoff
    }


    func runMemoryTool(_ name: String, arguments: [String: Any]) -> ToolExecutionResult {
        guard ChatMemoryService.isEnabled else {
            return ToolExecutionResult(content: "Conversation memory tools are turned off.", isError: true)
        }
        guard let i = currentIndex else {
            return ToolExecutionResult(content: "No open conversation.", isError: true)
        }
        switch name {
        case ChatMemoryService.listName:   return memoryList(i)
        case ChatMemoryService.archiveName: return memoryArchive(i, arguments: arguments)
        case ChatMemoryService.recallName:  return memoryRecall(i, arguments: arguments)
        default:
            return ToolExecutionResult(content: "Unknown memory tool.", isError: true)
        }
    }

    private func memoryList(_ i: Int) -> ToolExecutionResult {
        let c = conversations[i]
        let skipped = ChatMemoryService.archivedIndices(c.archived)
        var lines: [String] = []
        if let covered = c.summarizedCount, covered > 0 {
            lines.append("0-\(covered - 1): already summarized, cannot be archived")
        }
        for (index, m) in c.messages.enumerated() where index >= (c.summarizedCount ?? 0) {
            let text = m.role == "assistant" ? m.parts.body : m.wireContent
            let state = skipped.contains(index) ? " [archived]" : ""
            lines.append("\(index) \(m.role)\(state): \(ChatMemoryService.preview(text))")
        }
        if let blocks = c.archived, !blocks.isEmpty {
            lines.append("")
            lines.append("Archived ranges:")
            for b in blocks { lines.append("  \(b.from)-\(b.to - 1): \(b.note)") }
        }
        return ToolExecutionResult(content: lines.joined(separator: "\n"), isError: false)
    }

    private func memoryArchive(_ i: Int, arguments: [String: Any]) -> ToolExecutionResult {
        guard let from = (arguments["from_index"] as? NSNumber)?.intValue,
              let to = (arguments["to_index"] as? NSNumber)?.intValue else {
            return ToolExecutionResult(content: "from_index and to_index are required.", isError: true)
        }
        let note = (arguments["note"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard let range = ChatMemoryService.validate(from: from, to: to,
                                                     messageCount: conversations[i].messages.count,
                                                     summarized: conversations[i].summarizedCount ?? 0) else {
            return ToolExecutionResult(
                content: "That range cannot be archived: it must be inside the conversation and leave the last exchange in place.",
                isError: true)
        }
        var blocks = conversations[i].archived ?? []
        blocks.append(ArchivedBlock(from: range.from, to: range.to,
                                    note: note.isEmpty ? "archived" : note))
        conversations[i].archived = blocks
        conversations[i].updated = Date()
        save()
        AppLog.chat.info("archived messages \(range.from)..<\(range.to)")
        // The turns leave the context here, so this is the one moment an external index can
        // still see them without reading the transcript file.
        MemoryArchiveHook.send(
            conversationID: conversations[i].id.uuidString,
            title: conversations[i].title,
            from: range.from, to: range.to,
            note: note.isEmpty ? "archived" : note,
            messages: conversations[i].messages[range.from..<range.to].enumerated().map {
                (index: range.from + $0.offset,
                 role: $0.element.role,
                 content: $0.element.role == "assistant" ? $0.element.parts.body : $0.element.wireContent)
            })
        let freed = conversations[i].messages[range.from..<range.to]
            .reduce(0) { $0 + $1.estimatedTokens }
        return ToolExecutionResult(
            content: "Archived turns \(range.from)-\(range.to - 1), about \(freed) tokens freed. Use memory_recall to bring them back.",
            isError: false)
    }

    private func memoryRecall(_ i: Int, arguments: [String: Any]) -> ToolExecutionResult {
        guard let query = (arguments["query"] as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines), !query.isEmpty else {
            return ToolExecutionResult(content: "query is required.", isError: true)
        }
        let limit = max(1, min(20, (arguments["max_results"] as? NSNumber)?.intValue ?? 4))
        let c = conversations[i]
        guard let blocks = c.archived, !blocks.isEmpty else {
            return ToolExecutionResult(content: "Nothing has been archived in this conversation.",
                                       isError: false)
        }
        var texts: [Int: String] = [:]
        for index in ChatMemoryService.archivedIndices(blocks) where index < c.messages.count {
            let m = c.messages[index]
            texts[index] = m.role == "assistant" ? m.parts.body : m.wireContent
        }
        let hits = ChatMemoryService.matches(query: query, in: blocks, texts: texts, limit: limit)
        guard !hits.isEmpty else {
            return ToolExecutionResult(content: "No archived turn matches \"\(query)\".", isError: false)
        }
        var out = ""
        for index in hits {
            let line = "[\(index)] \(c.messages[index].role): \(texts[index] ?? "")\n\n"
            if out.count + line.count > ChatMemoryService.maximumRecallCharacters { break }
            out += line
        }
        return ToolExecutionResult(content: out, isError: false)
    }

    private func compactIfNeeded(conversation id: UUID, port: Int) {
        let d = UserDefaults.standard
        let enabled = d.object(forKey: SettingsKeys.chatAutoCompact) == nil
            ? true : d.bool(forKey: SettingsKeys.chatAutoCompact)
        let limit = d.object(forKey: SettingsKeys.ctx) == nil
            ? 16384 : d.integer(forKey: SettingsKeys.ctx)
        guard enabled, !generating, !compacting, limit > 0,
              let i = conversations.firstIndex(where: { $0.id == id }),
              let used = conversations[i].contextUsed, Double(used) / Double(limit) > 0.7 else { return }
        let start = conversations[i].summarizedCount ?? 0
        guard let cutoff = Self.compactionCutoff(messages: conversations[i].messages,
                                                 alreadyCompacted: start,
                                                 keepTokens: limit / 4) else { return }
        compact(conversation: id, index: i, from: start, through: cutoff, port: port)
    }

    func compactCurrent(port: Int) {
        guard !generating, !compacting, let i = currentIndex,
              conversations[i].messages.last?.role == "assistant" else { return }
        let start = conversations[i].summarizedCount ?? 0
        let cutoff = conversations[i].messages.count
        guard cutoff > start else { return }
        compact(conversation: conversations[i].id, index: i,
                from: start, through: cutoff, port: port)
    }

    var canCompactCurrent: Bool {
        guard !generating, !compacting, let i = currentIndex,
              conversations[i].messages.last?.role == "assistant" else { return false }
        return conversations[i].messages.count > (conversations[i].summarizedCount ?? 0)
    }

    private func compact(conversation id: UUID, index i: Int,
                         from start: Int, through cutoff: Int, port: Int) {
        var prompt = ""
        if let prior = conversations[i].summary, !prior.isEmpty {
            prompt += "Previous summary:\n" + prior + "\n\n"
        }
        prompt += "Conversation to summarize:\n\n"
        for m in conversations[i].messages[start..<cutoff] {
            prompt += (m.role == "user" ? "User: " : "Assistant: ")
                + (m.role == "assistant" ? m.parts.body : m.content) + "\n\n"
        }

        compacting = true
        AppLog.chat.info("compacting conversation through message \(cutoff)")
        Task.detached(priority: .utility) { [weak self] in
            let summary = await Self.summarize(prompt: prompt, port: port)
            let store = self
            await MainActor.run {
                store?.applyCompaction(conversation: id, cutoff: cutoff, summary: summary)
            }
        }
    }

    private func applyCompaction(conversation id: UUID, cutoff: Int, summary: String?) {
        compacting = false
        guard let summary, let i = conversations.firstIndex(where: { $0.id == id }),
              cutoff <= conversations[i].messages.count else { return }
        conversations[i].summary = summary
        conversations[i].summarizedCount = cutoff
        save()
    }

    /// Non-streamed completion that condenses old turns. Returns nil on any
    /// failure; compaction is then retried after the next exchange.
    nonisolated private static func summarize(prompt: String, port: Int) async -> String? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions") else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = ServerSettings.activeAPIKey() {
            req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
        }
        let instructions = "You summarize conversations. Write a compact summary (at most ~250 words) of the conversation below, in the same language the conversation itself uses. Preserve key facts, decisions, names, numbers, code references and pending questions. Reply with the summary only."
        var body: [String: Any] = [
            "messages": [["role": "system", "content": instructions],
                         ["role": "user", "content": prompt]],
            "stream": false,
            "temperature": 0.3,
            "max_tokens": 512,
            "chat_template_kwargs": ["enable_thinking": false],
        ]
        if let model = ServerSettings.activeRouterModel() { body["model"] = model }
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        guard let (data, response) = try? await NetworkManager.session.data(for: req),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = obj["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            AppLog.chat.error("compaction summarize request failed")
            return nil
        }
        // Reasoning models may emit a think block anyway; keep only the body.
        let text = ChatMessage(role: "assistant",
                               content: content.trimmingCharacters(in: .whitespacesAndNewlines)).parts.body
        return text.isEmpty ? nil : text
    }

    /// Maps llama-server HTTP errors ({"error":{"message":…}}) to readable,
    /// actionable text, following the same bilingual style as Server.diagnose.
    nonisolated private static func describeServerError(status: Int, body: String) -> String {
        var message = body
        if let data = body.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let err = obj["error"] as? [String: Any],
           let m = err["message"] as? String {
            message = m
        }
        if message.lowercased().contains("reasoning effort")
            || message.lowercased().contains("reasoning_effort") {
            return "El modelo no acepta ese esfuerzo de razonamiento; elige otro o «Predeterminado del modelo» en los ajustes del chat. El motor respondió: \(message.prefix(200)) / the model does not accept that reasoning effort; pick another one or “Model default” in the chat settings. The engine answered: \(message.prefix(200))"
        }
        if message.lowercased().contains("context") {
            return "Contexto lleno: el mensaje, los archivos adjuntos y el historial juntos superan el contexto. Sube el contexto en Ajustes, adjunta menos o inicia un chat nuevo / context full: your message, attached files and history together exceed the context. Raise the context size in Settings, attach less, or start a new chat"
        }
        // std::bad_function_call comes from both a context overflow on a tool turn
        // and a missing ffmpeg/ffprobe on video; name both, not just video.
        if message.contains("bad_function_call") {
            return "La solicitud falló. Con herramientas activas suele ser que el archivo o la conversación superan el contexto: súbelo en Ajustes o usa un archivo más pequeño. Si adjuntaste un video, verifica que ffmpeg y ffprobe estén disponibles / the request failed. With tools enabled this is usually the file or conversation exceeding the context: raise it in Settings or use a smaller file. If you attached a video, make sure ffmpeg and ffprobe are available"
        }
        return "HTTP \(status): \(message.prefix(300))"
    }

    nonisolated static func streamedError(from object: [String: Any]) -> String? {
        guard let error = object["error"] else { return nil }
        if let details = error as? [String: Any],
           let message = details["message"] as? String, !message.isEmpty {
            return message
        }
        if let message = error as? String, !message.isEmpty { return message }
        return "El motor interrumpió la respuesta / the engine interrupted the response"
    }

    nonisolated static func emptyResponseMessage(finishReason: String?) -> String {
        if finishReason == "length" {
            return "El modelo agotó el máximo de tokens durante el razonamiento; aumenta Máx. o desactiva Razonamiento / the model used all max tokens while reasoning; raise Max or disable Reasoning"
        }
        return "El modelo terminó el razonamiento sin producir una respuesta visible; intenta regenerar o desactiva Razonamiento / the model finished reasoning without a visible answer; regenerate or disable Reasoning"
    }

    func stop() {
        watchdog?.cancel()
        watchdog = nil
        task?.cancel()
        if let stream = activeStream, let url = ChatStreamIdentity.stopURL(port: stream.port, identity: stream.identity) {
            activeStream = nil
            var req = URLRequest(url: url)
            req.httpMethod = "DELETE"
            req.timeoutInterval = 5
            if let key = ServerSettings.activeAPIKey() {
                req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
            }
            Task.detached { _ = try? await URLSession.shared.data(for: req) }
        }
    }

    /// Removes the last user message (and its response, if any) so it can be
    /// edited and resent. Returns the removed message, attachments included.
    func popLastExchange() -> ChatMessage? {
        guard !generating, let i = currentIndex else { return nil }
        if conversations[i].messages.last?.role == "assistant" {
            conversations[i].messages.removeLast()
        }
        guard conversations[i].messages.last?.role == "user" else { return nil }
        let message = conversations[i].messages.removeLast()
        Self.clampSummary(&conversations[i])
        save()
        return message
    }

    nonisolated static func clampSummary(_ c: inout Conversation) {
        c.archived = ChatMemoryService.clamped(c.archived, toCount: c.messages.count)
        guard let covered = c.summarizedCount else { return }
        if covered > c.messages.count { c.summarizedCount = c.messages.count }
        if c.summarizedCount == 0 { c.summary = nil; c.summarizedCount = nil }
    }

    func editMessage(_ messageID: UUID) -> ChatMessage? {
        guard !generating, let i = currentIndex,
              let j = conversations[i].messages.firstIndex(where: { $0.id == messageID }),
              conversations[i].messages[j].role == "user" else { return nil }
        let message = conversations[i].messages[j]
        beginAlternativeBranch(conversationIndex: i,
                               messages: Array(conversations[i].messages[..<j]))
        conversations[i].summary = nil
        conversations[i].summarizedCount = nil
        conversations[i].archived = nil
        conversations[i].updated = Date()
        conversations[i].contextUsed = nil
        save()
        return message
    }

    func deleteMessageAndFollowing(_ messageID: UUID) {
        guard !generating, let i = currentIndex,
              let j = conversations[i].messages.firstIndex(where: { $0.id == messageID }) else { return }
        conversations[i].messages.removeSubrange(j...)
        conversations[i].summary = nil
        conversations[i].summarizedCount = nil
        conversations[i].archived = nil
        conversations[i].updated = Date()
        conversations[i].contextUsed = nil
        save()
    }

    var currentBranchPosition: (index: Int, count: Int)? {
        guard let c = current, let branches = c.branches, branches.count > 1,
              let active = c.activeBranchID,
              let index = branches.firstIndex(where: { $0.id == active }) else { return nil }
        return (index + 1, branches.count)
    }

    func switchBranch(_ branchID: UUID) {
        guard !generating, let i = currentIndex,
              conversations[i].activateBranch(branchID) else { return }
        conversations[i].summary = nil
        conversations[i].summarizedCount = nil
        conversations[i].archived = nil
        conversations[i].updated = Date()
        conversations[i].contextUsed = nil
        save()
    }

    func toggleTool(_ name: String, allTools: [BuiltinToolInfo]) {
        guard let i = currentIndex else { return }
        var selected = Set(conversations[i].enabledToolNames ?? allTools.map(\.name))
        if selected.contains(name) { selected.remove(name) } else { selected.insert(name) }
        conversations[i].enabledToolNames = selected.sorted()
        save()
    }

    func enableAllTools() {
        guard let i = currentIndex else { return }
        conversations[i].enabledToolNames = nil
        save()
    }

    private func beginAlternativeBranch(conversationIndex i: Int, messages: [ChatMessage]) {
        conversations[i].beginAlternativeBranch(messages: messages)
    }

    @discardableResult
    func forkConversation(at messageID: UUID, title: String? = nil,
                          includeAttachments: Bool = true) -> Conversation? {
        guard let source = current,
              let j = source.messages.firstIndex(where: { $0.id == messageID }) else { return nil }
        var messages = Array(source.messages[...j])
        if !includeAttachments {
            for index in messages.indices {
                messages[index].attachments = nil
                messages[index].imageURIs = nil
            }
        }
        let proposed = title?.trimmingCharacters(in: .whitespacesAndNewlines)
        let name: String
        if let p = proposed, !p.isEmpty {
            name = p
        } else {
            name = "Fork of \(displayTitle(source))"
        }
        let fork = Conversation(title: name, messages: messages, created: Date(), updated: Date(),
                                projectID: source.projectID, systemPrompt: source.systemPrompt)
        conversations.insert(fork, at: 0)
        currentID = fork.id
        save()
        return fork
    }

    func updateDraft(conversationID: UUID, text: String,
                     attachments: [ChatAttachment], imageURIs: [String]) {
        guard let i = conversations.firstIndex(where: { $0.id == conversationID }) else { return }
        let draft = ChatDraft(text: text, attachments: attachments, imageURIs: imageURIs)
        conversations[i].draft = draft.isEmpty ? nil : draft
        save()
    }

    // MARK: persistence

    private func load() {
        if let data = try? Data(contentsOf: fileURL),
           let list = try? JSONDecoder().decode([Conversation].self, from: data) {
            conversations = list
        } else if let data = try? Data(contentsOf: backupURL),
                  let list = try? JSONDecoder().decode([Conversation].self, from: data) {
            // Fallback to backup if primary is corrupted
            conversations = list
            AppLog.chat.error("ChatStore: loaded from backup after primary read failure")
        } else {
            AppLog.chat.error("ChatStore: failed to load conversations (primary + backup)")
        }
        if let data = try? Data(contentsOf: projectsURL),
           let list = try? JSONDecoder().decode([ChatProject].self, from: data) {
            projects = list
        } else if let data = try? Data(contentsOf: backupProjectsURL),
                  let list = try? JSONDecoder().decode([ChatProject].self, from: data) {
            projects = list
            AppLog.chat.error("ChatStore: loaded projects from backup after primary read failure")
        } else {
            AppLog.chat.error("ChatStore: failed to load projects (primary + backup)")
        }
    }

    // Serial queue: keeps writes ordered while encoding off the main thread,
    // since the full history JSON grows with use and would cause hitches.
    private static let saveQueue = DispatchQueue(label: "dev.engel.toshllm.chat-save", qos: .utility)
    private static var saveWork: DispatchWorkItem?

    func save() {
        // Debounce: cancel previous pending save and schedule a new one after 150ms.
        // This collapses rapid successive calls (delete, rename, new conversation)
        // into a single disk write.
        Self.saveWork?.cancel()
        for i in conversations.indices {
            guard let active = conversations[i].activeBranchID,
                  let j = conversations[i].branches?.firstIndex(where: { $0.id == active }) else { continue }
            conversations[i].branches?[j].messages = conversations[i].messages
        }
        let snapshot = conversations
        let projectsSnapshot = projects
        let url = fileURL
        let pURL = projectsURL
        let bkURL = backupURL
        let bkPURL = backupProjectsURL
        let work = DispatchWorkItem { [oldWork = Self.saveWork] in
            // Snapshot previous version as backup before overwriting
            let fm = FileManager.default
            if fm.fileExists(atPath: url.path) {
                try? fm.removeItem(at: bkURL)
                do {
                    try fm.copyItem(at: url, to: bkURL)
                } catch {
                    AppLog.chat.error("ChatStore: failed to create conversations backup — data may be lost if write also fails: \(error.localizedDescription)")
                }
            }
            if fm.fileExists(atPath: pURL.path) {
                try? fm.removeItem(at: bkPURL)
                do {
                    try fm.copyItem(at: pURL, to: bkPURL)
                } catch {
                    AppLog.chat.error("ChatStore: failed to create projects backup — data may be lost if write also fails: \(error.localizedDescription)")
                }
            }
            if let data = try? JSONEncoder().encode(snapshot) {
                do {
                    try data.write(to: url, options: .atomic)
                } catch {
                    AppLog.chat.error("ChatStore: failed to write conversations file: \(error.localizedDescription)")
                }
            } else {
                AppLog.chat.error("ChatStore: failed to encode conversations")
            }
            if let data = try? JSONEncoder().encode(projectsSnapshot) {
                do {
                    try data.write(to: pURL, options: .atomic)
                } catch {
                    AppLog.chat.error("ChatStore: failed to write projects file: \(error.localizedDescription)")
                }
            } else {
                AppLog.chat.error("ChatStore: failed to encode projects")
            }
        }
        Self.saveWork = work
        Self.saveQueue.asyncAfter(deadline: .now() + 0.15, execute: work)
    }

    func exportText(_ c: Conversation, _ loc: Localizer) -> String {
        let user = loc.t("Tú", "You")
        let assistant = loc.t("Asistente", "Assistant")
        return c.messages.map { m in
            let who = m.role == "user" ? user : assistant
            return "## \(who)\n\n\(m.role == "assistant" ? m.parts.body : m.content)"
        }.joined(separator: "\n\n---\n\n")
    }

    func exportArchiveData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(ChatArchive(conversations: conversations, projects: projects))
    }

    func exportJSONLData() throws -> Data {
        try ChatJSONL.encode(conversations)
    }

    func importArchiveData(_ data: Data) throws -> Int {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let imported: ChatArchive
        if let archive = try? decoder.decode(ChatArchive.self, from: data) {
            imported = archive
        } else if let legacy = try? JSONDecoder().decode([Conversation].self, from: data) {
            imported = ChatArchive(conversations: legacy, projects: [])
        } else if let jsonl = try? ChatJSONL.decode(data) {
            imported = ChatArchive(conversations: jsonl, projects: [])
        } else {
            throw ChatArchiveError.unsupported
        }

        let existingConversationIDs = Set(conversations.map(\.id))
        let additions = imported.conversations.filter { !existingConversationIDs.contains($0.id) }
        let existingProjectIDs = Set(projects.map(\.id))
        projects.append(contentsOf: imported.projects.filter { !existingProjectIDs.contains($0.id) })
        conversations.insert(contentsOf: additions, at: 0)
        if currentID == nil { currentID = conversations.first?.id }
        save()
        return additions.count
    }
}
