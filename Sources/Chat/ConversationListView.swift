// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Conversation list (split-view sidebar)

enum ConversationSortOrder: String, CaseIterable {
    case lastUsed, created, title

    func label(_ loc: Localizer) -> String {
        switch self {
        case .lastUsed: return loc.t("Uso reciente", "Recently used")
        case .created: return loc.t("Fecha de creación", "Date created")
        case .title: return loc.t("Título (A-Z)", "Title (A-Z)")
        }
    }
}

struct ConversationListView: View {
    @EnvironmentObject var chat: ChatStore
    @EnvironmentObject var loc: Localizer
    @AppStorage(SettingsKeys.appAccent) private var accentRaw = AppTheme.defaultKey
    @State private var searchText = ""
    @State private var debouncedSearchText = ""
    @State private var renaming: Conversation?
    @State private var renameText = ""
    @State private var renamingProject: ChatProject?
    @State private var creatingProject = false
    @State private var newProjectName = ""
    @State private var promptProject: ChatProject?
    @State private var promptConversation: Conversation?
    @State private var archiveMessage: String?
    @State private var confirmDeleteAll = false
    @State private var hoveredConversationID: UUID?
    @State private var dropTargetProjectID: UUID?
    @State private var ungroupedDropIsTargeted = false
    @AppStorage(SettingsKeys.chatSortOrder) private var sortOrderRaw = ConversationSortOrder.lastUsed.rawValue

    private var sortOrder: ConversationSortOrder {
        ConversationSortOrder(rawValue: sortOrderRaw) ?? .lastUsed
    }

    private var searching: Bool { !debouncedSearchText.trimmingCharacters(in: .whitespaces).isEmpty }

    private func sorted(_ base: [Conversation]) -> [Conversation] {
        base.sorted { a, b in
            switch sortOrder {
            case .lastUsed: return a.updated > b.updated
            case .created: return a.created > b.created
            case .title: return chat.displayTitle(a).localizedCaseInsensitiveCompare(chat.displayTitle(b)) == .orderedAscending
            }
        }
    }

    private var searchResults: [Conversation] {
        let query = debouncedSearchText.trimmingCharacters(in: .whitespaces)
        return sorted(chat.conversations.filter { c in
            chat.displayTitle(c).localizedCaseInsensitiveContains(query) ||
            c.messages.contains { $0.content.localizedCaseInsensitiveContains(query) }
        })
    }

    /// Pinned chats surface in their own section; the rest stay in their group.
    private var pinnedChats: [Conversation] { sorted(chat.conversations.filter { $0.pinned ?? false }) }
    private var ungroupedChats: [Conversation] {
        sorted(chat.conversations.filter { !($0.pinned ?? false) && $0.projectID == nil })
    }
    private func projectChats(_ p: ChatProject) -> [Conversation] {
        sorted(chat.conversations.filter { !($0.pinned ?? false) && $0.projectID == p.id })
    }
    private var sortedProjects: [ChatProject] {
        chat.projects.sorted { a, b in
            if (a.pinned ?? false) != (b.pinned ?? false) { return a.pinned ?? false }
            return a.created > b.created
        }
    }

    /// Relative timestamp in the app's language, not the system locale — the
    /// default `.formatted(.relative…)` ignores the in-app language toggle.
    private func relativeDate(_ date: Date) -> String {
        date.formatted(Date.RelativeFormatStyle(
            presentation: .named,
            locale: Locale(identifier: loc.isSpanish ? "es" : "en")))
    }

    var body: some View {
        VStack(spacing: 0) {
            Button {
                chat.newConversation(in: chat.current?.projectID)
            } label: {
                HStack(spacing: 8) {
                    Label(loc.t("Nueva conversación", "New chat"), systemImage: "plus")
                    Spacer(minLength: 8)
                    Text("⌘ N")
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 9).padding(.vertical, 4)
                        .background(.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 6))
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(GlassPillButtonStyle(prominent: true))
            .keyboardShortcut("n", modifiers: .command)
            .padding(.horizontal, 12)
            .padding(.top, 12)
            .padding(.bottom, 10)
            .help(loc.t("Empieza una conversación nueva en el proyecto actual (⌘N).",
                        "Start a new conversation in the current project (⌘N)."))

            HStack(spacing: 8) {
                GlassSearchField(placeholder: loc.t("Buscar conversaciones…", "Search conversations…"), text: $searchText)

                Menu {
                    ForEach(ConversationSortOrder.allCases, id: \.self) { order in
                        Button {
                            sortOrderRaw = order.rawValue
                        } label: {
                            if order == sortOrder {
                                Label(order.label(loc), systemImage: "checkmark")
                            } else {
                                Text(order.label(loc))
                            }
                        }
                    }
                    Divider()
                    Button(loc.t("Importar archivo…", "Import archive…"),
                           systemImage: "square.and.arrow.down") { importArchive() }
                    Button(loc.t("Exportar todo…", "Export all…"),
                           systemImage: "square.and.arrow.up") { exportArchive() }
                    Button(loc.t("Exportar JSONL para llama.cpp…", "Export JSONL for llama.cpp…"),
                           systemImage: "doc.text") { exportJSONL() }
                    Divider()
                    Button(loc.t("Borrar todas las conversaciones…", "Delete all conversations…"),
                           systemImage: "trash", role: .destructive) { confirmDeleteAll = true }
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .menuStyle(.borderlessButton).menuIndicator(.hidden)
                .tint(.secondary)
                .frame(width: 28, height: 28)
                .glassSurface(in: Circle(), interactive: true)
                .overlay(Circle().strokeBorder(.primary.opacity(0.07)))
                .accessibilityLabel(loc.t("Acciones de conversaciones", "Conversation actions"))
                .help(loc.t("Ordenar, importar, exportar y borrar conversaciones",
                            "Sort, import, export and delete conversations"))
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 12)

            List {
                if searching {
                    ForEach(searchResults) { chatRow($0, showsProject: true) }
                } else {
                    if !pinnedChats.isEmpty {
                        Section {
                            ForEach(pinnedChats) { chatRow($0, showsProject: true) }
                        } header: {
                            sidebarHeader(loc.t("Fijados", "Pinned"))
                        }
                    }
                    if !chat.projects.isEmpty {
                        Section {
                            ForEach(sortedProjects) { p in
                                DisclosureGroup(isExpanded: expandBinding(p)) {
                                    let rows = projectChats(p)
                                    if rows.isEmpty {
                                        Text(loc.t("Sin conversaciones", "No conversations"))
                                            .font(.caption).foregroundStyle(.tertiary)
                                    } else {
                                        ForEach(rows) { chatRow($0, showsProject: false) }
                                    }
                                } label: {
                                    projectRow(p)
                                }
                            }
                        } header: {
                            sidebarHeader(loc.t("Proyectos", "Projects"), actionIcon: "plus") {
                                newProjectName = ""
                                creatingProject = true
                            }
                        }
                    } else {
                        Section {
                            EmptyView()
                        } header: {
                            sidebarHeader(loc.t("Proyectos", "Projects"), actionIcon: "plus") {
                                newProjectName = ""
                                creatingProject = true
                            }
                        }
                    }
                    Section {
                        ForEach(ungroupedChats) { chatRow($0, showsProject: false) }
                    } header: {
                        chatsHeader
                    }
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)

            Text("ToshLLM · macOS")
                .font(.system(size: 10, weight: .medium))
                .tracking(1.5)
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
        }
        .navigationSplitViewColumnWidth(min: 270, ideal: 330, max: 390)
        .task(id: searchText) {
            let value = searchText
            if value.trimmingCharacters(in: .whitespaces).isEmpty {
                debouncedSearchText = ""
                return
            }
            try? await Task.sleep(for: .milliseconds(180))
            guard !Task.isCancelled else { return }
            debouncedSearchText = value
        }
        .alert(loc.t("Renombrar conversación", "Rename conversation"),
               isPresented: Binding(get: { renaming != nil },
                                    set: { if !$0 { renaming = nil } })) {
            TextField(loc.t("Título", "Title"), text: $renameText)
            Button(loc.t("Guardar", "Save")) {
                if let c = renaming { chat.rename(c, to: renameText) }
                renaming = nil
            }
            Button(loc.t("Cancelar", "Cancel"), role: .cancel) { renaming = nil }
        }
        .alert(loc.t("Nuevo proyecto", "New project"), isPresented: $creatingProject) {
            TextField(loc.t("Nombre", "Name"), text: $newProjectName)
            Button(loc.t("Crear", "Create")) {
                let name = newProjectName.trimmingCharacters(in: .whitespaces)
                guard !name.isEmpty else { return }
                let p = chat.newProject(name: name)
                chat.newConversation(in: p.id)
            }
            Button(loc.t("Cancelar", "Cancel"), role: .cancel) {}
        } message: {
            Text(loc.t("Una carpeta para tus conversaciones. Si le defines un prompt de sistema al proyecto, todas las conversaciones dentro lo heredan.",
                       "A folder for your conversations. If you set a system prompt on the project, every conversation inside inherits it."))
        }
        .sheet(item: $renamingProject) { project in
            ProjectRenameSheet(initial: project.name) { name in
                chat.renameProject(project, to: name)
            }
            .environmentObject(loc)
        }
        .sheet(item: $promptProject) { p in
            PromptEditorSheet(
                title: loc.t("Prompt del proyecto \"%@\"", "Project prompt for \"%@\"", "\(p.name)"),
                hint: loc.t("Lo heredan todas las conversaciones del proyecto que no tengan prompt propio.",
                            "Inherited by every conversation in the project without its own prompt."),
                initial: p.systemPrompt
            ) { chat.setProjectPrompt(p, $0) }
        }
        .sheet(item: $promptConversation) { c in
            PromptEditorSheet(
                title: loc.t("Prompt de esta conversación", "This conversation's prompt"),
                hint: loc.t("Sustituye al prompt del proyecto y al global solo en esta conversación. Vacío = heredar.",
                            "Overrides the project and global prompts for this conversation only. Empty = inherit."),
                initial: c.systemPrompt ?? ""
            ) { chat.setConversationPrompt(c, $0) }
        }
        .alert(loc.t("Conversaciones", "Conversations"),
               isPresented: Binding(get: { archiveMessage != nil },
                                    set: { if !$0 { archiveMessage = nil } })) {
            Button(loc.t("Aceptar", "OK")) { archiveMessage = nil }
        } message: {
            Text(archiveMessage ?? "")
        }
        .confirmationDialog(
            loc.t("¿Borrar las %@ conversaciones?", "Delete all %@ conversations?", "\(chat.conversations.count)"),
            isPresented: $confirmDeleteAll, titleVisibility: .visible
        ) {
            Button(loc.t("Borrar todas", "Delete all"), role: .destructive) { chat.deleteAll() }
            Button(loc.t("Cancelar", "Cancel"), role: .cancel) {}
        } message: {
            Text(loc.t("No se puede deshacer. Tus proyectos y sus prompts se conservan; exporta antes si quieres una copia.",
                       "This cannot be undone. Your projects and their prompts are kept; export first if you want a copy."))
        }
    }

    private func expandBinding(_ p: ChatProject) -> Binding<Bool> {
        Binding(get: { !(p.collapsed ?? false) },
                set: { chat.setProjectCollapsed(p, !$0) })
    }

    @ViewBuilder
    private func sidebarHeader(_ title: String, actionIcon: String? = nil,
                               action: @escaping () -> Void = {}) -> some View {
        HStack {
            Text(title)
                .font(.callout.weight(.semibold))
                .foregroundStyle(.primary)
                .textCase(nil)
            Spacer()
            if let actionIcon {
                Button(action: action) {
                    Image(systemName: actionIcon)
                        .font(.system(size: 12, weight: .semibold))
                        .frame(width: 24, height: 24)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(loc.t("Nuevo proyecto", "New project"))
            }
        }
    }

    private var chatsHeader: some View {
        HStack {
            Text(loc.t("Conversaciones", "Chats"))
                .font(.callout.weight(.semibold))
                .foregroundStyle(.primary)
                .textCase(nil)
            Spacer()
            Text(ungroupedDropIsTargeted
                 ? loc.t("Soltar aquí", "Drop here")
                 : sortOrder.label(loc))
                .font(.caption)
                .foregroundStyle(ungroupedDropIsTargeted
                                 ? AnyShapeStyle(Color.appAccent)
                                 : AnyShapeStyle(.tertiary))
        }
        .padding(.vertical, 3)
        .contentShape(Rectangle())
        .dropDestination(for: String.self) { ids, _ in
            let moved = conversations(for: ids)
            guard !moved.isEmpty else { return false }
            moved.forEach { chat.move($0, toProject: nil) }
            return true
        } isTargeted: { ungroupedDropIsTargeted = $0 }
    }

    private func conversations(for ids: [String]) -> [Conversation] {
        ids.compactMap(UUID.init(uuidString:))
            .compactMap { id in chat.conversations.first { $0.id == id } }
    }

    private func conversationTitle(_ conversation: Conversation) -> String {
        if conversation.title.isEmpty && conversation.messages.isEmpty {
            return loc.t("Nueva conversación", "New chat")
        }
        return chat.displayTitle(conversation)
    }

    private func importArchive() {
        let panel = NSOpenPanel()
        var allowedTypes: [UTType] = [.json]
        if let jsonl = UTType(filenameExtension: "jsonl") { allowedTypes.append(jsonl) }
        panel.allowedContentTypes = allowedTypes
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let count = try chat.importArchiveData(Data(contentsOf: url))
            archiveMessage = loc.t("Se importaron %@ conversaciones nuevas.", "Imported %@ new conversations.", "\(count)")
        } catch {
            archiveMessage = error.localizedDescription
        }
    }

    private func exportArchive() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "ToshLLM-conversations.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try chat.exportArchiveData().write(to: url, options: .atomic)
            archiveMessage = loc.t("Historial exportado correctamente.",
                                   "Conversation history exported successfully.")
        } catch {
            archiveMessage = error.localizedDescription
        }
    }

    private func exportJSONL() {
        let panel = NSSavePanel()
        if let jsonl = UTType(filenameExtension: "jsonl") { panel.allowedContentTypes = [jsonl] }
        panel.nameFieldStringValue = "ToshLLM-conversations.jsonl"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try chat.exportJSONLData().write(to: url, options: .atomic)
            archiveMessage = loc.t("Historial JSONL compatible con llama.cpp exportado correctamente.",
                                   "llama.cpp-compatible JSONL history exported successfully.")
        } catch {
            archiveMessage = error.localizedDescription
        }
    }

    // MARK: rows

    @ViewBuilder private func projectRow(_ p: ChatProject) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "folder.fill")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(AppTheme.accent(accentRaw))
                .frame(width: 32, height: 32)
                .background(AppTheme.accent(accentRaw).opacity(0.10),
                            in: RoundedRectangle(cornerRadius: 9))
            Text(p.name)
                .font(.callout.weight(.medium)).lineLimit(1)
            if p.pinned ?? false {
                Image(systemName: "pin.fill")
                    .font(.system(size: 8)).foregroundStyle(.tertiary)
            }
            if !p.systemPrompt.trimmingCharacters(in: .whitespaces).isEmpty {
                Image(systemName: "text.bubble")
                    .font(.caption2).foregroundStyle(.tertiary)
                    .help(loc.t("Este proyecto tiene prompt de sistema propio.",
                                "This project has its own system prompt."))
            }
            Spacer(minLength: 4)
            Text("\(chat.conversations.filter { $0.projectID == p.id }.count)")
                .font(.caption2.weight(.medium)).foregroundStyle(.secondary)
                .padding(.horizontal, 6).padding(.vertical, 1)
                .background(.quaternary.opacity(0.6), in: Capsule())
        }
        .padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(dropTargetProjectID == p.id ? Color.appAccent.opacity(0.14) : .clear,
                    in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
            .strokeBorder(dropTargetProjectID == p.id ? Color.appAccent.opacity(0.7) : .clear))
        .contentShape(Rectangle())
        // The disclosure only toggles from its chevron; open from the whole row.
        .onTapGesture { withAnimation { expandBinding(p).wrappedValue.toggle() } }
        .dropDestination(for: String.self) { ids, _ in
            let moved = conversations(for: ids)
            guard !moved.isEmpty else { return false }
            moved.forEach { chat.move($0, toProject: p.id) }
            return true
        } isTargeted: { targeted in
            dropTargetProjectID = targeted ? p.id : (dropTargetProjectID == p.id ? nil : dropTargetProjectID)
        }
        .contextMenu {
            Button(loc.t("Nueva conversación aquí", "New chat here")) {
                chat.setProjectCollapsed(p, false)
                chat.newConversation(in: p.id)
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
            Button(loc.t("Renombrar…", "Rename…")) {
                renamingProject = p
            }
            Button((p.pinned ?? false) ? loc.t("Desfijar proyecto", "Unpin project")
                                       : loc.t("Fijar proyecto", "Pin project")) {
                chat.togglePinProject(p)
            }
            Divider()
            Button(loc.t("Eliminar proyecto", "Delete project"), role: .destructive) {
                chat.deleteProject(p)
            }
        }
    }

    @ViewBuilder private func chatRow(_ c: Conversation, showsProject: Bool) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "bubble.left")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 32, height: 32)
                .background(WorkspaceStyle.inset, in: Circle())
            if c.pinned ?? false {
                Image(systemName: "pin.fill")
                    .font(.system(size: 9)).foregroundStyle(AppTheme.accent(accentRaw).opacity(0.7))
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(conversationTitle(c))
                    .font(.callout).lineLimit(1)
                HStack(spacing: 5) {
                    if showsProject, let p = chat.project(id: c.projectID) {
                        Label(p.name, systemImage: "folder")
                            .font(.caption2).foregroundStyle(.tertiary)
                            .labelStyle(.titleAndIcon).lineLimit(1)
                    }
                    Text(relativeDate(c.updated))
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
            Spacer(minLength: 4)
            if !(c.systemPrompt ?? "").isEmpty {
                Image(systemName: "text.bubble")
                    .font(.caption2).foregroundStyle(.tertiary)
                    .help(loc.t("Esta conversación tiene prompt propio.",
                                "This conversation has its own prompt."))
            }
            Image(systemName: "line.3.horizontal")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(hoveredConversationID == c.id || chat.currentID == c.id
                                 ? AnyShapeStyle(Color.appAccent)
                                 : AnyShapeStyle(.quaternary))
                .frame(width: 18, height: 26)
                .help(loc.t("Puedes arrastrar toda la fila a un proyecto.",
                            "You can drag the entire row into a project."))
            Menu {
                Button((c.pinned ?? false) ? loc.t("Desfijar", "Unpin") : loc.t("Fijar", "Pin")) {
                    chat.togglePin(c)
                }
                Button(loc.t("Renombrar…", "Rename…")) {
                    renameText = conversationTitle(c)
                    renaming = c
                }
                Button(loc.t("Prompt de esta conversación…", "This conversation's prompt…")) {
                    promptConversation = c
                }
                Divider()
                Button(loc.t("Eliminar", "Delete"), role: .destructive) { chat.delete(c) }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 12, weight: .medium))
                    .frame(width: 26, height: 26)
                    .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .opacity(hoveredConversationID == c.id || chat.currentID == c.id ? 1 : 0)
            .accessibilityLabel(loc.t("Acciones de la conversación", "Conversation actions"))
        }
        .padding(.leading, 0)
        .padding(.trailing, 6)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(.interaction, RoundedRectangle(cornerRadius: 8, style: .continuous))
        .contentShape(.dragPreview, RoundedRectangle(cornerRadius: 8, style: .continuous))
        .onTapGesture { chat.currentID = c.id }
        .onHover { inside in
            if inside {
                hoveredConversationID = c.id
            } else if hoveredConversationID == c.id {
                hoveredConversationID = nil
            }
        }
        .onDrag {
            NSItemProvider(object: c.id.uuidString as NSString)
        } preview: {
            HStack(spacing: 8) {
                Image(systemName: "bubble.left.fill")
                Text(conversationTitle(c)).lineLimit(1)
            }
            .font(.callout.weight(.medium))
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(WorkspaceStyle.surface,
                        in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Color.appAccent.opacity(0.6)))
        }
        .listRowBackground(
            RoundedRectangle(cornerRadius: 8)
                .fill(chat.currentID == c.id
                      ? AppTheme.accent(accentRaw).opacity(0.26) : Color.clear)
                .padding(.horizontal, 13))
        .contextMenu {
            Button((c.pinned ?? false) ? loc.t("Desfijar", "Unpin") : loc.t("Fijar", "Pin")) {
                chat.togglePin(c)
            }
            Button(loc.t("Renombrar…", "Rename…")) {
                renameText = conversationTitle(c)
                renaming = c
            }
            Button(loc.t("Prompt de esta conversación…", "This conversation's prompt…")) {
                promptConversation = c
            }
            Menu(loc.t("Mover a proyecto", "Move to project")) {
                Button(loc.t("Ninguno", "None")) { chat.move(c, toProject: nil) }
                ForEach(sortedProjects) { p in
                    Button(p.name + (c.projectID == p.id ? " ✓" : "")) {
                        chat.move(c, toProject: p.id)
                    }
                }
            }
            Divider()
            Button(loc.t("Copiar conversación", "Copy conversation")) {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(chat.exportText(c, loc), forType: .string)
            }
            Button(loc.t("Exportar a Markdown…", "Export to Markdown…")) {
                let panel = NSSavePanel()
                panel.nameFieldStringValue = chat.displayTitle(c)
                    .replacingOccurrences(of: "/", with: "-") + ".md"
                if panel.runModal() == .OK, let url = panel.url {
                    try? chat.exportText(c, loc).write(to: url, atomically: true, encoding: .utf8)
                }
            }
            Button(loc.t("Eliminar", "Delete"), role: .destructive) { chat.delete(c) }
        }
    }
}

/// Shared editor for project and per-conversation system prompts.
struct PromptEditorSheet: View {
    @EnvironmentObject var loc: Localizer
    @Environment(\.dismiss) private var dismiss
    let title: String
    let hint: String
    let initial: String
    let onSave: (String) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 13) {
                SectionGlyph(systemName: "text.bubble")
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.title3.weight(.semibold))
                        .lineLimit(2)
                    Text(loc.t("Instrucciones para esta parte del chat.",
                               "Instructions for this part of Chat."))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 16)
            .background(WorkspaceStyle.surface)

            Divider()

            VStack(alignment: .leading, spacing: 12) {
                TextEditor(text: $text)
                    .font(.system(size: 13))
                    .focused($focused)
                    .frame(height: 190)
                    .workspaceFieldSurface(cornerRadius: 10)
                Label(hint, systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(11)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(WorkspaceStyle.inset,
                                in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .strokeBorder(WorkspaceStyle.border))
            }
            .padding(18)

            Divider()

            HStack(spacing: 10) {
                Spacer()
                Button(loc.t("Cancelar", "Cancel")) { dismiss() }
                    .glassButton()
                    .keyboardShortcut(.cancelAction)
                Button(loc.t("Guardar", "Save"), systemImage: "checkmark") {
                    onSave(text)
                    dismiss()
                }
                .glassButton(prominent: true)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20).padding(.vertical, 14)
            .background(WorkspaceStyle.surface)
        }
        .frame(width: 540)
        .background(WorkspaceStyle.canvas)
        .onAppear {
            text = initial
            Task { await Task.yield(); focused = true }
        }
    }
}

/// Compact project-name editor using the same surfaces as the rest of Chat.
private struct ProjectRenameSheet: View {
    @EnvironmentObject private var loc: Localizer
    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @FocusState private var focused: Bool
    let onSave: (String) -> Void

    init(initial: String, onSave: @escaping (String) -> Void) {
        _name = State(initialValue: initial)
        self.onSave = onSave
    }

    private var trimmedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 13) {
                SectionGlyph(systemName: "folder")
                VStack(alignment: .leading, spacing: 3) {
                    Text(loc.t("Renombrar proyecto", "Rename project"))
                        .font(.title3.weight(.semibold))
                    Text(loc.t("El nuevo nombre aparecerá en la barra lateral.",
                               "The new name will appear in the sidebar."))
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 20).padding(.vertical, 16)
            .background(WorkspaceStyle.surface)

            Divider()

            SettingsRowGroup {
                SettingsRow(icon: "tag", title: loc.t("Nombre", "Name")) {
                    TextField(loc.t("Nombre del proyecto", "Project name"), text: $name)
                        .focused($focused)
                        .workspaceTextField(width: 280)
                }
            }
            .padding(18)

            Divider()

            HStack(spacing: 10) {
                Spacer()
                Button(loc.t("Cancelar", "Cancel")) { dismiss() }
                    .glassButton()
                    .keyboardShortcut(.cancelAction)
                Button(loc.t("Guardar", "Save"), systemImage: "checkmark", action: saveName)
                    .glassButton(prominent: true)
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmedName.isEmpty)
            }
            .padding(.horizontal, 20).padding(.vertical, 14)
            .background(WorkspaceStyle.surface)
        }
        .frame(width: 520)
        .background(WorkspaceStyle.canvas)
        .onAppear {
            Task { await Task.yield(); focused = true }
        }
    }

    private func saveName() {
        guard !trimmedName.isEmpty else { return }
        onSave(trimmedName)
        dismiss()
    }
}
