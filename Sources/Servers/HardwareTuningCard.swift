// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import SwiftUI

/// A starting point for the manual offload knobs, derived from the detected
/// GPU/VRAM/model geometry. It sits beside the real controls so the measured
/// values can be applied without hunting through Settings.
struct HardwareTuningCard: View {
    let advice: HardwareTuningAdvice
    let isApplied: Bool
    let onApply: () -> Void

    @EnvironmentObject private var loc: Localizer

    private var hardwareLine: String {
        var parts: [String] = []
        if !advice.gpuName.isEmpty {
            parts.append("\(advice.gpuName) · \(String(format: "%.0f GB VRAM", Double(advice.vramMB) / 1024))")
        }
        if let architecture = advice.architecture {
            parts.append(advice.isWave64 ? "\(architecture) (wave64)" : architecture)
        }
        if advice.gpuCount > 1 {
            parts.append(loc.t("%@ GPUs", "%@ GPUs", "\(advice.gpuCount)"))
        }
        if advice.physicalCores > 0 {
            parts.append(loc.t("%@ núcleos · %@ GB RAM",
                               "%@ cores · %@ GB RAM",
                               "\(advice.physicalCores)",
                               "\(Int(advice.ramGB.rounded()))"))
        }
        return parts.joined(separator: " · ")
    }

    private var estimatedLine: String {
        let reserve = max(0, advice.vramMB - Int((advice.usableVRAMGB * 1024).rounded()))
        return loc.t("Uso estimado: ~%@ de %@ GB. Reserva del motor: %@ MB.",
                     "Estimated use: ~%@ of %@ GB. Engine reserve: %@ MB.",
                     String(format: "%.1f", advice.estimatedVRAMGB),
                     String(format: "%.1f", advice.usableVRAMGB),
                     "\(reserve)")
    }

    private var reasonsHelp: String {
        advice.reasons.map(reasonText).joined(separator: "\n")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Label(loc.t("Ajuste recomendado para este Mac", "Recommended for this Mac"),
                      systemImage: "wand.and.stars")
                    .font(.system(size: 13, weight: .semibold))
                if advice.measuredProfile {
                    TagBadge(text: loc.t("Perfil medido", "Measured profile"),
                             icon: "checkmark.seal", color: .green)
                }
                Spacer(minLength: 8)
                Button(action: onApply) {
                    Label(isApplied ? loc.t("En uso", "In use") : loc.t("Aplicar", "Apply"),
                          systemImage: isApplied ? "checkmark" : "arrow.down.to.line")
                }
                .glassButton(prominent: !isApplied)
                .controlSize(.small)
                .disabled(isApplied)
            }

            Text(hardwareLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 112), spacing: 8)],
                      alignment: .leading, spacing: 8) {
                metric(loc.t("Capas GPU", "GPU layers"),
                       advice.ngl >= advice.modelLayers ? loc.t("Todas", "All") : "\(advice.ngl)")
                metric(loc.t("Expertos en CPU", "Experts on CPU"), "\(advice.ncmoe)")
                metric(loc.t("Contexto", "Context"), ServerSettings.contextLabel(advice.ctx))
                metric(loc.t("Micro-lote", "Micro-batch"),
                       advice.ubatch == 0 ? loc.t("Auto", "Auto") : "\(advice.ubatch)")
                metric(loc.t("Caché KV", "KV cache"), "\(advice.cacheTypeK)/\(advice.cacheTypeV)")
                metric(loc.t("Hilos", "Threads"), "\(advice.threads)")
            }

            ForEach(advice.warnings.indices, id: \.self) { index in
                Label(warningText(advice.warnings[index]), systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text(estimatedLine)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(WorkspaceStyle.inset, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous)
            .strokeBorder(WorkspaceStyle.border.opacity(0.7)))
        .help(reasonsHelp)
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func warningText(_ warning: HardwareTuningWarning) -> String {
        switch warning {
        case let .ubatchAboveSafeLimit(current, safe):
            return loc.t("El micro-lote actual (%@) supera el límite seguro medido (%@) para esta VRAM; el arranque puede fallar al reservar el buffer de prompt.",
                         "The current micro-batch (%@) is above the measured safe limit (%@) for this VRAM; startup can fail while allocating the prompt buffer.",
                         "\(current)", "\(safe)")
        case let .quantizedKVOnWave64(k, v):
            return loc.t("KV cuantizado (%@/%@) en una tarjeta GCN wave64: la atención cae a la CPU y el prompt se lee ~42% más lento. La recomendación usa f16/f16.",
                         "Quantized KV (%@/%@) on a wave64 GCN card: attention falls to the CPU and prompt reading is ~42% slower. The recommendation uses f16/f16.",
                         k, v)
        case let .expertsAboveGPULayers(ncmoe, ngl):
            return loc.t("Expertos en CPU (%@) por encima de las capas GPU (%@): esa combinación midió 1.6 t/s. Aplica una distribución coherente.",
                         "Experts on CPU (%@) above GPU layers (%@): that combination measured 1.6 t/s. Apply a consistent split.",
                         "\(ncmoe)", "\(ngl)")
        case let .contextAboveEstimatedFit(requested, safe):
            return loc.t("El contexto actual (%@) no cabe con estas capas; el cálculo estima %@ como máximo. La recomendación lo baja para que el modelo cargue.",
                         "The current context (%@) does not fit with these layers; the estimate tops out at %@. The recommendation lowers it so the model loads.",
                         ServerSettings.contextLabel(requested), ServerSettings.contextLabel(safe))
        case .cpuOnly:
            return loc.t("El modelo no cabe en la GPU; la recomendación lo ejecuta en CPU.",
                         "The model does not fit on the GPU; the recommendation runs it on the CPU.")
        case .dynamicMoeActive:
            return loc.t("Dynamic MoE está activo y decide sus propios valores. Aplicar lo desactiva en este servidor para que manden estos parámetros.",
                         "Dynamic MoE is active and picks its own values. Apply turns it off for this server so these parameters take effect.")
        case .fullGPUForced:
            return loc.t("El modo GPU completa está forzado. Aplicar vuelve al plan manual para que manden estos parámetros.",
                         "Full GPU mode is forced. Apply returns to the manual plan so these parameters take effect.")
        case .routerMayShrinkContext:
            return loc.t("El router puede reducir el contexto de los modelos que no caben. Esta recomendación es para un modelo cargado a la vez.",
                         "Router mode may shrink the context of models that do not fit. This recommendation is for one model loaded at a time.")
        case .multiGPUSplit:
            return loc.t("El presupuesto suma las GPUs seleccionadas; revisa el modo de reparto antes de aplicar.",
                         "The budget adds the selected GPUs together; check the split mode before applying.")
        }
    }

    private func reasonText(_ reason: HardwareTuningReason) -> String {
        switch reason {
        case .measuredWave64Profile:
            return loc.t("Perfil medido en una iMac 4 GB gfx803: 10 capas, 6 expertos en CPU, 16k, f16 y 6 hilos.",
                         "Measured on a 4 GB gfx803 iMac: 10 layers, 6 experts on CPU, 16k, f16 and 6 threads.")
        case .fullGPU:
            return loc.t("El modelo cabe entero en la VRAM.",
                         "The whole model fits in VRAM.")
        case .partialMoE:
            return loc.t("Las capas de atención se quedan en la GPU y los expertos se reparten entre VRAM y RAM.",
                         "Attention layers stay on the GPU while experts are split between VRAM and RAM.")
        case .partialDense:
            return loc.t("Solo las capas que caben se quedan en la GPU.",
                         "Only the layers that fit stay on the GPU.")
        case .wave64KVF16:
            return loc.t("En GCN/Vega (wave64), f16 midió 47.9 t/s de prompt frente a 27.7 con KV cuantizado.",
                         "On GCN/Vega (wave64), f16 measured 47.9 t/s prompt vs 27.7 with quantized KV.")
        case let .ubatchClipped(from, to):
            return loc.t("Micro-lote reducido de %@ a %@: el buffer de prompt crece ~2.26 MiB por token.",
                         "Micro-batch lowered from %@ to %@: the prompt buffer grows ~2.26 MiB per token.",
                         "\(from)", "\(to)")
        case .ubatchForExpertUploads:
            return loc.t("Un micro-lote mayor amortiza la subida de expertos durante el prompt.",
                         "A larger micro-batch amortizes expert uploads during prompt processing.")
        case let .trainedContextCap(trained):
            return loc.t("Contexto limitado al máximo entrenado del modelo (%@).",
                         "Context capped to the model's trained maximum (%@).",
                         ServerSettings.contextLabel(trained))
        case .physicalCores:
            return loc.t("Hilos fijados a los núcleos físicos; el SMT no acelera la decodificación.",
                         "Threads set to physical cores; SMT does not speed up decode.")
        case let .reserve(reserve):
            return loc.t("Reserva de VRAM del motor: %@ MB.",
                         "Engine VRAM reserve: %@ MB.", "\(reserve)")
        }
    }
}
