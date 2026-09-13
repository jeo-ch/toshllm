// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation

enum WhisperVADModel {
    static let name = "Silero VAD 6.2"
    static let fileName = "ggml-silero-v6.2.0.bin"
    static let sizeKB = 864
    static var downloadURL: String {
        DownloadSource.current.downloadURL(repo: "ggml-org/whisper-vad", file: fileName)
    }

    static func url(in directory: URL) -> URL {
        directory.appendingPathComponent(fileName)
    }
}
