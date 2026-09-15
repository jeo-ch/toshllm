// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// Unified speculative decoding control for any decoder type.
/// Replaces separate MTPControl and DflashControl with a single component.
struct SpeculativeControl: View {
    enum Layout { case inline, settings, detail }
    
    let modelPath: String
    var layout: Layout = .inline
    @EnvironmentObject var loc: Localizer
    @EnvironmentObject var server: ServerController
    
    private var availableDecoders: [SpeculativeDecoder] {
        SpeculativeDecoderManager.availableDecoders(forModel: modelPath)
    }
    
    private var preferredDecoder: SpeculativeDecoder? {
        SpeculativeDecoderManager.preferredDecoder(forModel: modelPath)
    }
    
    private var isActive: Bool {
        SpeculativeDecoderManager.anyEnabled(forModel: modelPath)
    }
    
    @ViewBuilder
    var body: some View {
        if availableDecoders.isEmpty {
            // No speculative decoding available for this model
            EmptyView()
        } else if let decoder = preferredDecoder {
            switch layout {
            case .settings:
                settingsSection(decoder: decoder)
            case .detail:
                detailRow(decoder: decoder)
            case .inline:
                inlineRow(decoder: decoder)
            }
        }
    }
    
    // MARK: - Settings Layout
    
    private func settingsSection(decoder: SpeculativeDecoder) -> some View {
        Group {
            if let dflashDecoder = decoder as? DFlashDecoder {
                dflashSettingsRows(decoder: dflashDecoder)
            } else {
                mtpSettingsRows(decoder: decoder)
            }
        }
    }
    
    private func mtpSettingsRows(decoder: SpeculativeDecoder) -> some View {
        Group {
            LabeledContent(loc.t("Decodificación especulativa", "Speculative Decoding")) {
                Picker(loc.t("MTP", "MTP"), selection: Binding(
                    get: { decoder.isEnabled(forModel: modelPath) },
                    set: { decoder.setEnabled($0, forModel: modelPath) }
                )) {
                    Text(loc.t("Activado", "Enabled")).tag(true)
                    Text(loc.t("Desactivado", "Disabled")).tag(false)
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
            }
            if decoder.isEnabled(forModel: modelPath) {
                LabeledContent(loc.t("Tipo", "Type")) {
                    Text(decoder.displayName).monospacedDigit()
                }
            }
        }
        .help(loc.t("MTP usa el cabezal MTP integrado o un borrador externo para decodificación más rápida.",
                    "MTP uses the embedded MTP head or external draft for faster decoding."))
    }
    
    private func dflashSettingsRows(decoder: DFlashDecoder) -> some View {
        Group {
            LabeledContent(loc.t("Decodificación especulativa", "Speculative Decoding")) {
                Picker(loc.t("DFlash", "DFlash"), selection: Binding(
                    get: { decoder.mode(forModel: modelPath) },
                    set: { decoder.setMode($0, forModel: modelPath) }
                )) {
                    Text(loc.t("Apagado", "Off")).tag(DflashMode.off)
                    Text("Auto").tag(DflashMode.auto)
                    Text(loc.t("Forzado", "Forced")).tag(DflashMode.forced)
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
            }
            if decoder.isEnabled(forModel: modelPath), let acc = server.dflashAcceptance {
                LabeledContent(loc.t("Aceptación", "Acceptance")) {
                    Text(acc.formatted(.percent.precision(.fractionLength(0)))).monospacedDigit()
                }
            }
        }
        .help(loc.t("Auto usa DFlash cuando hay un draft compatible y el planificador deja memoria suficiente. Forzado ignora la reserva de seguridad y avisa si la VRAM supera 95 %.",
                    "Auto uses DFlash when a compatible draft is installed and the memory planner leaves enough headroom. Forced ignores the safety reserve and warns if VRAM exceeds 95%."))
    }
    
    // MARK: - Detail Layout
    
    private func detailRow(decoder: SpeculativeDecoder) -> some View {
        Group {
            if let dflashDecoder = decoder as? DFlashDecoder {
                DflashControl(modelPath: modelPath, layout: .detail)
            } else {
                MTPControl(modelPath: modelPath)
            }
        }
    }
    
    // MARK: - Inline Layout
    
    private func inlineRow(decoder: SpeculativeDecoder) -> some View {
        HStack(spacing: 8) {
            if let dflashDecoder = decoder as? DFlashDecoder {
                DflashControl(modelPath: modelPath, layout: .inline)
            } else {
                mtpInlineRow(decoder: decoder)
            }
        }
    }
    
    private func mtpInlineRow(decoder: SpeculativeDecoder) -> some View {
        let enabled = decoder.isEnabled(forModel: modelPath)
        return HStack(spacing: 8) {
            Image(systemName: decoder.systemImage)
                .foregroundStyle(enabled ? Color.accentColor : Color.secondary)
            Text(decoder.displayName)
                .foregroundStyle(enabled ? .primary : .secondary)
                .fixedSize()
            Picker(decoder.displayName, selection: Binding(
                get: { enabled },
                set: { decoder.setEnabled($0, forModel: modelPath) }
            )) {
                Text(loc.t("Off", "Off")).tag(false)
                Text(loc.t("On", "On")).tag(true)
            }
            .labelsHidden()
            .fixedSize()
        }
        .font(.caption)
        .help(loc.t("MTP usa el cabezal MTP integrado o un borrador externo para decodificación más rápida.",
                    "MTP uses the embedded MTP head or external draft for faster decoding."))
    }
}

// MARK: - Preview

#Preview {
    SpeculativeControl(modelPath: "/path/to/model.gguf", layout: .inline)
        .environmentObject(Localizer())
        .environmentObject(ServerController())
}
