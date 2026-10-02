// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// A memoised `contentsOfDirectory` for the span of one models scan.
///
/// Pairing a projector and looking for an MTP head both start by listing the
/// model's own directory, and a warm pass asks about every model in turn, so a
/// flat folder of N models used to run roughly 3N full directory listings — the
/// single largest syscall multiplier in a scan.
///
/// The cache exists **only inside `withCache`**. The same lookups are also made
/// at launch and from view bodies, where a stale listing would be a wrong answer
/// rather than a slow one: pairing a projector reads the directory, decides, and
/// is then asked again after a download changed what is there. So outside a pass
/// every read goes to the file system, exactly as before.
///
/// Values are value types and no caller mutates them in place, so handing the
/// same array to several readers is safe.
enum ModelDirectoryListing {
    private static let lock = NSLock()
    /// nil outside a pass, which is what makes an uncached read the default.
    nonisolated(unsafe) private static var cache: [String: [URL]]?

    /// Directory entries, or an empty array when the directory cannot be read —
    /// the same result the call sites already handled by falling back to `[]`.
    static func entries(in dir: URL) -> [URL] {
        let key = dir.standardizedFileURL.path
        lock.lock()
        if let cache, let cached = cache[key] {
            lock.unlock()
            return cached
        }
        let caching = cache != nil
        lock.unlock()

        let listed = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []

        if caching {
            lock.lock()
            cache?[key] = listed
            lock.unlock()
        }
        return listed
    }

    /// True when `dir` holds an entry with this name, so a caller can skip the
    /// probe for a subdirectory it already listed.
    static func contains(_ name: String, in dir: URL) -> Bool {
        entries(in: dir).contains { $0.lastPathComponent == name }
    }

    /// Runs `body` with listings reused, then drops them.
    static func withCache<T>(_ body: () -> T) -> T {
        lock.lock()
        let outer = cache
        cache = [:]
        lock.unlock()
        defer {
            lock.lock()
            cache = outer
            lock.unlock()
        }
        return body()
    }
}

/// What a downloaded GGUF brings with it: experts, vision projector, MTP head, DFlash draft.
struct ModelTraits {
    let isMoE: Bool
    let hasVision: Bool
    let hasMTP: Bool
    let hasDflash: Bool

    /// Shown until the background warm has read the header.
    static let unknown = ModelTraits(isMoE: false, hasVision: false, hasMTP: false, hasDflash: false)

    static func of(path: String) -> ModelTraits {
        ModelTraits(isMoE: ServerSettings.modelIsMoE(at: path),
                    hasVision: ServerSettings.mmprojPath(forModel: path) != nil,
                    hasMTP: ServerSettings.modelUsesMTP(at: path),
                    hasDflash: ServerSettings.dflashDraftPath(forModel: path) != nil)
    }

    /// Compact markers for menu rows, where a badge can't be drawn.
    func pickerSuffix(spanish: Bool) -> String {
        var marks: [String] = []
        if isMoE { marks.append("MoE") }
        if hasVision { marks.append(spanish ? "Visión" : "Vision") }
        if hasMTP { marks.append("MTP") }
        if hasDflash { marks.append("DFlash") }
        return marks.isEmpty ? "" : "  ·  " + marks.joined(separator: " · ")
    }
}

/// Each read hits the file system and the GGUF header, and every model card and
/// model menu asks on redraw, so results are held until the folder is rescanned.
enum ModelTraitsCache {
    private static let lock = NSLock()
    private static var entries: [String: ModelTraits] = [:]

    static func traits(for path: String) -> ModelTraits {
        lock.lock()
        if let cached = entries[path] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let traits = ModelTraits.of(path: path)
        lock.lock()
        entries[path] = traits
        lock.unlock()
        return traits
    }

    /// For view bodies: never reads the header, so drawing cannot block on it.
    static func cached(for path: String) -> ModelTraits? {
        lock.lock()
        defer { lock.unlock() }
        return entries[path]
    }

    static func warm(paths: [String], then done: @escaping () -> Void) {
        Task.detached(priority: .utility) {
            // One listing per directory for the whole pass, not one per model.
            ModelDirectoryListing.withCache {
                for path in paths {
                    autoreleasepool {
                        _ = traits(for: path)
                    }
                }
            }
            await MainActor.run { done() }
        }
    }

    static func invalidate() {
        lock.lock()
        entries.removeAll()
        lock.unlock()
    }
}

struct ModelTraitBadges: View {
    let traits: ModelTraits
    @EnvironmentObject var loc: Localizer

    var body: some View {
        if traits.isMoE {
            TagBadge(text: "MoE", icon: "square.stack.3d.up", color: .appAccent)
                .help(loc.t("Mezcla de expertos: puedes descargar parte de los expertos a la RAM.",
                            "Mixture of experts: some experts can be offloaded to RAM."))
        }
        if traits.hasVision {
            TagBadge(text: loc.t("Visión", "Vision"), icon: "eye", color: .purple)
                .help(loc.t("Lee imágenes: tiene su proyector (mmproj) emparejado.",
                            "Reads images: its projector (mmproj) is paired."))
        }
        if traits.hasMTP {
            TagBadge(text: "MTP", icon: "hare", color: .green)
                .help(loc.t("El GGUF trae el cabezal de predicción multi-token: la decodificación especulativa se activa sola cuando conviene.",
                            "The GGUF ships the multi-token prediction head: speculative decoding turns on by itself when it pays off."))
        }
        if traits.hasDflash {
            TagBadge(text: "DFlash", icon: "bolt", color: .orange)
                .help(loc.t("Tiene descargado su modelo borrador DFlash para decodificación especulativa.",
                            "Its DFlash draft model for speculative decoding is downloaded."))
        }
    }
}
