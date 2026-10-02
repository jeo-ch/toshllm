import Foundation

/// Posts the turns an archive set aside to a URL the user configures, so an external index can
/// keep them. Deliveries survive a quit: each one is written to disk before it is attempted and
/// only removed once the receiver accepts it, so nothing is lost if the app closes mid-flight.
actor MemoryArchiveHook {
    static let shared = MemoryArchiveHook()

    /// The payload is ours and stays stable: one JSON object per archive.
    ///
    ///     {
    ///       "event": "memory_archive",
    ///       "delivery_id": "UUID",
    ///       "sent_at": "2026-09-06T21:14:03Z",
    ///       "conversation": { "id": "...", "title": "..." },
    ///       "range": { "from": 12, "to": 30, "note": "..." },
    ///       "messages": [ { "index": 12, "role": "user", "content": "..." } ]
    ///     }
    ///
    /// `delivery_id` is stable across retries, so a receiver can drop a duplicate it already
    /// stored rather than index the same turns twice.
    struct Payload: Codable {
        struct Conversation: Codable { let id: String; let title: String }
        struct Range: Codable { let from: Int; let to: Int; let note: String }
        struct Message: Codable { let index: Int; let role: String; let content: String }

        var event = "memory_archive"
        var delivery_id: String
        var sent_at: String
        var conversation: Conversation
        var range: Range
        var messages: [Message]
    }

    /// Beyond this the oldest are dropped: a receiver that has been down for days must not fill
    /// the disk, and the newest turns are the ones worth keeping.
    private static let queueLimit = 500
    private static let attemptTimeout: TimeInterval = 15

    private var draining = false

    private var queueDirectory: URL {
        let dir = AppSupport.directory.appendingPathComponent("memory-archive-hook", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    nonisolated static var url: URL? {
        guard let raw = UserDefaults.standard.string(forKey: SettingsKeys.memoryArchiveHookURL)?
                .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty,
              let url = URL(string: raw), url.host != nil,
              url.scheme == "http" || url.scheme == "https" else { return nil }
        return url
    }

    /// Sent as `Authorization: Bearer …`, so it is a credential rather than a
    /// preference. It lives in the Keychain: a bearer token in UserDefaults is a
    /// plaintext plist any process running as this user can read with `defaults
    /// read`, which is the same exposure the API key did not have. A value written
    /// before this existed is still read once from UserDefaults so nobody has to
    /// retype it, and then removed from there.
    private nonisolated static var secret: String? {
        if let fromKeychain = Keychain.get(Self.secretAccount)?
            .trimmingCharacters(in: .whitespacesAndNewlines), !fromKeychain.isEmpty {
            // Also drop a plaintext copy left behind by an earlier build, or by a
            // settings import: the Keychain is the source of truth, so anything
            // still in the plist is a stale secret nothing should keep serving.
            UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
            return fromKeychain
        }
        // One-time migration off the plist, so the token is not left lying there.
        let legacy = UserDefaults.standard.string(forKey: SettingsKeys.memoryArchiveHookSecret)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let legacy, !legacy.isEmpty else { return nil }
        Keychain.setThisDeviceOnly(legacy, account: Self.secretAccount)
        UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
        return legacy
    }

    /// Mirrors a value typed in settings into the Keychain, or removes the
    /// Keychain copy when it is cleared.
    @discardableResult
    nonisolated static func storeSecret(_ raw: String) -> String {
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            Keychain.delete(Self.secretAccount)
            UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
            return ""
        }
        Keychain.setThisDeviceOnly(value, account: Self.secretAccount)
        UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
        return value
    }

    /// The current secret, for display in the settings field.
    nonisolated static func currentSecret() -> String {
        secret ?? ""
    }

    private nonisolated static let secretAccount = "memoryArchiveHookSecret"

    /// Queues one archive. Returns immediately: the chat must not wait on a receiver.
    nonisolated static func send(conversationID: String, title: String,
                                 from: Int, to: Int, note: String,
                                 messages: [(index: Int, role: String, content: String)]) {
        guard url != nil else { return }
        let payload = Payload(
            delivery_id: UUID().uuidString,
            sent_at: ISO8601DateFormatter().string(from: Date()),
            conversation: .init(id: conversationID, title: title),
            range: .init(from: from, to: to, note: note),
            messages: messages.map { .init(index: $0.index, role: $0.role, content: $0.content) })
        Task { await shared.enqueue(payload) }
    }

    /// Picks up anything left from a previous run.
    nonisolated static func resume() {
        Task { await shared.drain() }
    }

    private func enqueue(_ payload: Payload) async {
        guard let data = try? JSONEncoder().encode(payload) else { return }
        // The name carries the timestamp so the directory listing sorts into delivery order.
        let name = String(format: "%013.0f-%@.json", Date().timeIntervalSince1970 * 1000, payload.delivery_id)
        try? data.write(to: queueDirectory.appendingPathComponent(name), options: .atomic)
        trim()
        await drain()
    }

    private func pending() -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: queueDirectory, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func trim() {
        let files = pending()
        guard files.count > Self.queueLimit else { return }
        for f in files.prefix(files.count - Self.queueLimit) {
            try? FileManager.default.removeItem(at: f)
        }
        AppLog.chat.warning("memory archive hook queue over \(Self.queueLimit), dropped the oldest")
    }

    private func drain() async {
        guard !draining else { return }
        draining = true
        defer { draining = false }

        var backoff: UInt64 = 1
        var consecutiveFailures = 0
        while let file = pending().first {
            guard let url = Self.url else { return }   // hook turned off: leave the queue for later
            guard let data = try? Data(contentsOf: file) else {
                try? FileManager.default.removeItem(at: file); continue
            }
            switch await post(data, to: url) {
            case .delivered:
                try? FileManager.default.removeItem(at: file)
                backoff = 1
                consecutiveFailures = 0
            case .rejected:
                // The receiver understood and refused it; another attempt would be refused too.
                try? FileManager.default.removeItem(at: file)
                backoff = 1
                consecutiveFailures = 0
            case .retry:
                // A receiver that never comes back used to keep this loop alive
                // for the rest of the session, growing the backoff to its ceiling
                // and holding `draining` true, so every later archive returned
                // immediately and nothing was ever delivered. Give up for now and
                // leave the queue intact: resume() picks it up again next launch,
                // and the oldest entries are still there to deliver.
                consecutiveFailures += 1
                guard consecutiveFailures <= Self.maxConsecutiveFailures else {
                    AppLog.chat.error("memory archive hook unreachable after \(Self.maxConsecutiveFailures) attempts; \(self.pending().count) file(s) kept for the next run")
                    return
                }
                try? await Task.sleep(for: .seconds(min(backoff, 300)))
                backoff *= 2
            }
        }
    }

    /// Attempts on one archive before the drain gives up until the next launch.
    /// The backoff totals roughly half an hour at this ceiling.
    private static let maxConsecutiveFailures = 8

    private enum Outcome { case delivered, rejected, retry }

    private func post(_ body: Data, to url: URL) async -> Outcome {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("memory_archive", forHTTPHeaderField: "X-ToshLLM-Event")
        if let id = (try? JSONDecoder().decode(Payload.self, from: body))?.delivery_id {
            request.setValue(id, forHTTPHeaderField: "X-ToshLLM-Delivery")
        }
        if let secret = Self.secret {
            request.setValue("Bearer \(secret)", forHTTPHeaderField: "Authorization")
        }
        request.timeoutInterval = Self.attemptTimeout
        request.httpBody = body

        do {
            let (_, response) = try await NetworkManager.session.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            switch code {
            case 200..<300: return .delivered
            // 408 and 429 are the receiver asking for another try; the rest of 4xx is a refusal.
            case 408, 429:  return .retry
            case 400..<500:
                AppLog.chat.warning("memory archive hook refused a delivery with \(code), dropping it")
                return .rejected
            default:        return .retry
            }
        } catch {
            AppLog.chat.warning("memory archive hook could not deliver: \(error.localizedDescription)")
            return .retry
        }
    }
}
