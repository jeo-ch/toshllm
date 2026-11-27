// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

/// Turns names that arrive from URLs or remote listings into a single path
/// component under a folder the app owns.
///
/// `URL.lastPathComponent` decodes percent-escapes, so `%2e%2e` reaches the
/// downloader as `..`, and a caller-supplied preferred name can carry a
/// separator outright. Either one used to let the destination, and the staging
/// file beside it, land outside the models folder.
enum DownloadName {
    /// APFS allows 255 UTF-8 bytes per component.
    static let maximumBytes = 255

    /// A name safe to append to a directory, or nil when it is not exactly one
    /// visible component.
    static func safe(_ raw: String) -> String? {
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name != ".", name != "..",
              !name.hasPrefix("."),
              !name.contains("/"), !name.contains("\\"),
              !name.contains(":"), !name.contains("\0"),
              name.rangeOfCharacter(from: .controlCharacters) == nil,
              name.utf8.count <= maximumBytes else { return nil }
        return name
    }

    /// The file `name` would occupy directly inside `directory`, or nil when the
    /// resolved path would leave it.
    static func destination(_ name: String, in directory: URL) -> URL? {
        guard let name = safe(name) else { return nil }
        let root = directory.standardizedFileURL
        let url = root.appendingPathComponent(name).standardizedFileURL
        guard url.deletingLastPathComponent().path == root.path else { return nil }
        return url
    }

    /// A file path taken from a repository listing: relative, and every segment
    /// a real name.
    ///
    /// Rejects the traversal and URL syntax that would change which host or path
    /// the interpolated download URL names.
    static func safeRelativePath(_ raw: String) -> String? {
        let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty, !path.hasPrefix("/"),
              !path.contains("\\"), !path.contains("?"), !path.contains("#"),
              !path.contains("%"), !path.contains("\0"),
              path.rangeOfCharacter(from: .controlCharacters) == nil else { return nil }
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        guard segments.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { return nil }
        return path
    }
}
