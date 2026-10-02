// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AppKit
import UniformTypeIdentifiers

/// Save-panel plumbing shared by every "export…" action.
///
/// The panel setup and the write-then-report step were repeated across the chat
/// archive, the settings backup, the transcripts and the log reports, and each
/// copy handled the failure case slightly differently. One helper keeps the
/// panel behaviour identical and reports every write failure the same way.
enum FileExport {
    /// Runs a save panel and writes `data` to the chosen location.
    static func write(_ data: Data,
                      suggestedName: String,
                      contentTypes: [UTType] = [.json],
                      directory: URL? = nil) -> FileExportResult {
        guard let url = chooseDestination(suggestedName: suggestedName,
                                          contentTypes: contentTypes,
                                          directory: directory) else { return .cancelled }
        do {
            try data.write(to: url, options: .atomic)
            return .written
        } catch {
            return report(error, at: url)
        }
    }

    /// As `write(_:suggestedName:contentTypes:directory:)`, for UTF-8 text.
    static func write(_ text: String,
                      suggestedName: String,
                      contentTypes: [UTType] = [.plainText],
                      directory: URL? = nil) -> FileExportResult {
        guard let url = chooseDestination(suggestedName: suggestedName,
                                          contentTypes: contentTypes,
                                          directory: directory) else { return .cancelled }
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
            return .written
        } catch {
            return report(error, at: url)
        }
    }

    /// Copies an already-written file (a render, a video) to the chosen location.
    static func copy(_ source: URL,
                     suggestedName: String,
                     contentTypes: [UTType] = [],
                     directory: URL? = nil) -> FileExportResult {
        guard let dest = chooseDestination(suggestedName: suggestedName,
                                           contentTypes: contentTypes,
                                           directory: directory) else { return .cancelled }
        do {
            // The panel can hand back an existing name, so clear it first.
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: source, to: dest)
            return .written
        } catch {
            return report(error, at: dest)
        }
    }

    private static func report(_ error: any Error, at url: URL) -> FileExportResult {
        AppLog.app.error("export to \(url.lastPathComponent, privacy: .public) failed: \(error.localizedDescription, privacy: .public)")
        return .failed(error.localizedDescription)
    }

    private static func chooseDestination(suggestedName: String,
                                          contentTypes: [UTType],
                                          directory: URL?) -> URL? {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedName
        if !contentTypes.isEmpty { panel.allowedContentTypes = contentTypes }
        panel.directoryURL = directory
        return panel.runModal() == .OK ? panel.url : nil
    }
}

/// Open-panel plumbing, the counterpart to `FileExport`.
enum FileImport {
    /// Runs an open panel for a single file.
    static func choose(contentTypes: [UTType],
                       directory: URL? = nil,
                       allowsMultipleSelection: Bool = false) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = contentTypes
        panel.allowsMultipleSelection = allowsMultipleSelection
        panel.canChooseDirectories = false
        panel.directoryURL = directory
        guard panel.runModal() == .OK else { return nil }
        return allowsMultipleSelection ? panel.urls.first : panel.url
    }

    /// Audio or video, as offered by the transcription entry points.
    static func chooseAudioOrVideo() -> URL? {
        choose(contentTypes: [.audio, .movie, .audiovisualContent])
    }
}

/// Outcome of a save panel run. `cancelled` and `written` stay distinct because
/// some call sites show a "saved successfully" confirmation and must not claim
/// one after the user backed out.
enum FileExportResult: Equatable {
    case cancelled
    case written
    case failed(String)
}