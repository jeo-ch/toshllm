// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import AVFoundation
import Combine
import QuartzCore

@MainActor
final class SubtitleVideoExporter: ObservableObject {
    @Published private(set) var isExporting = false
    @Published private(set) var progress = 0.0
    @Published private(set) var error: String?
    @Published private(set) var completedURL: URL?

    private var session: AVAssetExportSession?
    private var progressTask: Task<Void, Never>?
    private var exportTask: Task<Void, Never>?
    /// Identifies the export that owns `session`, `progress` and `isExporting`.
    /// `cancel()` clears it immediately, so a late completion from the old
    /// export cannot overwrite the state of an export started afterwards.
    private var runID: UUID?

    func export(sourceURL: URL, cues: [SubtitleCue], outputURL: URL,
                style: SubtitleStyle = .load()) {
        guard !isExporting, !cues.isEmpty else { return }
        let id = UUID()
        runID = id
        isExporting = true
        progress = 0
        error = nil
        completedURL = nil
        exportTask = Task { [weak self] in
            guard let self else { return }
            do {
                try await self.performExport(sourceURL: sourceURL, cues: cues,
                                             outputURL: outputURL, style: style, runID: id)
                guard self.runID == id else { return }
                self.progress = 1
                self.completedURL = outputURL
            } catch {
                guard self.runID == id else { return }
                self.error = error.localizedDescription
            }
            guard self.runID == id else { return }
            self.progressTask?.cancel()
            self.progressTask = nil
            self.session = nil
            self.exportTask = nil
            self.runID = nil
            self.isExporting = false
        }
    }

    func cancel() {
        // Invalidate first: the old task may already be past its last await and
        // about to publish progress or a completed URL.
        runID = nil
        exportTask?.cancel()
        exportTask = nil
        session?.cancelExport()
        session = nil
        progressTask?.cancel()
        progressTask = nil
        isExporting = false
    }

    private func ensureCurrent(_ id: UUID) throws {
        try Task.checkCancellation()
        guard runID == id else { throw CancellationError() }
    }

    private func performExport(sourceURL: URL, cues: [SubtitleCue], outputURL: URL,
                               style: SubtitleStyle, runID id: UUID) async throws {
        try ensureCurrent(id)
        let asset = AVURLAsset(url: sourceURL)
        let composition = AVMutableComposition()
        let duration = try await asset.load(.duration)
        try ensureCurrent(id)
        let range = CMTimeRange(start: .zero, duration: duration)
        guard let sourceVideo = try await asset.loadTracks(withMediaType: .video).first,
              let videoTrack = composition.addMutableTrack(withMediaType: .video,
                                                           preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw CocoaError(.fileReadUnsupportedScheme,
                             userInfo: [NSLocalizedDescriptionKey: "El archivo no contiene vídeo / the file contains no video."])
        }
        try ensureCurrent(id)
        try videoTrack.insertTimeRange(range, of: sourceVideo, at: .zero)
        videoTrack.preferredTransform = try await sourceVideo.load(.preferredTransform)
        try ensureCurrent(id)
        if let sourceAudio = try await asset.loadTracks(withMediaType: .audio).first,
           let audioTrack = composition.addMutableTrack(withMediaType: .audio,
                                                        preferredTrackID: kCMPersistentTrackID_Invalid) {
            try audioTrack.insertTimeRange(range, of: sourceAudio, at: .zero)
        }
        try ensureCurrent(id)

        let videoComposition = AVMutableVideoComposition(propertiesOf: composition)
        let size = videoComposition.renderSize
        let parentLayer = CALayer()
        let videoLayer = CALayer()
        parentLayer.frame = CGRect(origin: .zero, size: size)
        videoLayer.frame = parentLayer.frame
        parentLayer.addSublayer(videoLayer)
        cues.forEach { parentLayer.addSublayer(Self.subtitleLayer(for: $0, size: size, style: style)) }
        videoComposition.animationTool = AVVideoCompositionCoreAnimationTool(
            postProcessingAsVideoLayer: videoLayer, in: parentLayer
        )

        try await Self.preserveColour(from: sourceVideo, on: videoComposition)
        try ensureCurrent(id)

        let preset = await Self.preset(for: composition, source: sourceVideo)
        try ensureCurrent(id)
        guard let export = AVAssetExportSession(asset: composition, presetName: preset) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try? FileManager.default.removeItem(at: outputURL)
        export.outputURL = outputURL
        export.outputFileType = .mov
        export.shouldOptimizeForNetworkUse = true
        export.videoComposition = videoComposition
        // Cancellation can only happen on this actor, and there is no await
        // between the check and publishing the session, so the export cannot
        // be installed after `cancel()` has already cleared it.
        try ensureCurrent(id)
        session = export
        progressTask = Task { [weak self, weak export] in
            while !Task.isCancelled, let export {
                guard let self, self.runID == id else { return }
                self.progress = Double(export.progress)
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        await withCheckedContinuation { continuation in
            export.exportAsynchronously { continuation.resume() }
        }
        try ensureCurrent(id)
        if export.status != .completed {
            throw export.error ?? CocoaError(.fileWriteUnknown)
        }
    }

    /// Burning text in means re-encoding, but not necessarily down a generation:
    /// an HEVC source re-encoded as H.264 pays for nothing.
    nonisolated private static func preset(for composition: AVAsset,
                                           source: AVAssetTrack) async -> String {
        let hevc = AVAssetExportPresetHEVCHighestQuality
        let h264 = AVAssetExportPresetHighestQuality
        // Asked per preset against the file type actually written below, rather
        // than listing what the composition supports: exportPresets(compatibleWith:)
        // was deprecated in macOS 13 for exactly this replacement.
        let supports = await AVAssetExportSession.compatibility(
            ofExportPreset: hevc, with: composition, outputFileType: .mov)
        guard supports,
              let formats = try? await source.load(.formatDescriptions),
              formats.contains(where: { CMFormatDescriptionGetMediaSubType($0) == kCMVideoCodecType_HEVC }) else {
            return h264
        }
        return hevc
    }

    /// Without carrying the source's colour tags across, a wide-gamut or HDR clip
    /// comes back flattened.
    nonisolated private static func preserveColour(from source: AVAssetTrack,
                                                   on composition: AVMutableVideoComposition) async throws {
        guard let format = try await source.load(.formatDescriptions).first else { return }
        let ext = CMFormatDescriptionGetExtensions(format) as? [CFString: Any]
        guard let primaries = ext?[kCMFormatDescriptionExtension_ColorPrimaries] as? String,
              let transfer = ext?[kCMFormatDescriptionExtension_TransferFunction] as? String,
              let matrix = ext?[kCMFormatDescriptionExtension_YCbCrMatrix] as? String else { return }
        composition.colorPrimaries = primaries
        composition.colorTransferFunction = transfer
        composition.colorYCbCrMatrix = matrix
    }

    nonisolated private static func subtitleLayer(for cue: SubtitleCue, size: CGSize,
                                                  style: SubtitleStyle) -> CALayer {
        let l = SubtitleLayout.layout(text: cue.text, style: style, frame: size)
        let text = CATextLayer()
        text.string = l.attributed
        text.alignmentMode = .center
        text.isWrapped = true
        text.contentsScale = 2

        guard style.background == .box else {
            text.frame = l.box.insetBy(dx: l.textInset, dy: l.textInset)
            animate(text, cue: cue)
            return text
        }
        let plate = CALayer()
        plate.frame = l.box
        plate.backgroundColor = CGColor(gray: 0, alpha: style.backgroundOpacity)
        plate.cornerRadius = 8
        text.frame = CGRect(x: l.textInset, y: l.textInset,
                            width: l.box.width - l.textInset * 2,
                            height: l.box.height - l.textInset * 2)
        plate.addSublayer(text)
        animate(plate, cue: cue)
        return plate
    }

    nonisolated private static func animate(_ layer: CALayer, cue: SubtitleCue) {
        layer.opacity = 0
        let animation = CAKeyframeAnimation(keyPath: "opacity")
        animation.values = [0, 1, 1, 0]
        animation.keyTimes = [0, 0.02, 0.98, 1]
        animation.beginTime = AVCoreAnimationBeginTimeAtZero + cue.start
        animation.duration = max(0.1, cue.end - cue.start)
        animation.isRemovedOnCompletion = false
        animation.fillMode = .both
        layer.add(animation, forKey: "visibility")
    }
}
