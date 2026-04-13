// ProfilingSession.swift - Central profiling session coordinator
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import os

/// Collects profiling events during an inference run.
///
/// Tracks phase timings, memory snapshots, GPU/CPU utilization, and exports
/// Chrome Trace JSON for visualization in [Perfetto UI](https://ui.perfetto.dev/).
public final class ProfilingSession: @unchecked Sendable {
    public let sessionId: String
    public let startTime: Date
    public let config: ProfilingConfig
    public let deviceArchitecture: String
    public let systemRAMGB: Int

    /// Free-form metadata (model name, quantization, resolution, etc.)
    public var metadata: [String: String] = [:]

    /// Report title (e.g., "LTX-2.3 PROFILING REPORT")
    public var title: String = "MLX PROFILING REPORT"

    private var events: [ProfilingEvent] = []
    private var memoryTimeline: [MemoryTimelineEntry] = []
    private let lock = NSLock()
    private let sessionStartTime: CFAbsoluteTime

    // os_signpost for Instruments integration
    private let signposter: OSSignposter
    private var activeSignpostIDs: [String: (id: OSSignpostID, state: OSSignpostIntervalState)] = [:]

    public init(config: ProfilingConfig = .singleRun, subsystem: String = "com.mlxprofiler") {
        self.sessionId = UUID().uuidString
        self.startTime = Date()
        self.config = config
        self.sessionStartTime = CFAbsoluteTimeGetCurrent()
        self.deviceArchitecture = GPU.deviceInfo().architecture
        self.systemRAMGB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
        self.signposter = OSSignposter(subsystem: subsystem, category: .pointsOfInterest)
    }

    private func currentTimestampUs() -> UInt64 {
        UInt64((CFAbsoluteTimeGetCurrent() - sessionStartTime) * 1_000_000)
    }

    // MARK: - Event Recording

    public func beginPhase(_ name: String, category: ProfilingCategory) {
        let ts = currentTimestampUs()
        let snapshot = config.trackMemory ? takeSnapshot() : nil
        let spID = signposter.makeSignpostID()
        let state = signposter.beginInterval("Phase", id: spID, "\(name)")

        lock.lock()
        activeSignpostIDs[name] = (id: spID, state: state)
        events.append(ProfilingEvent(
            name: name, category: category, phase: .begin, timestampUs: ts,
            mlxActiveBytes: snapshot?.mlx.activeBytes, mlxCacheBytes: snapshot?.mlx.cacheBytes,
            mlxPeakBytes: snapshot?.mlx.peakBytes, processFootprintBytes: snapshot?.processFootprint
        ))
        if config.trackMemory, let snap = snapshot {
            appendTimeline(ts: ts, context: "begin:\(name)", snapshot: snap)
        }
        lock.unlock()
    }

    public func endPhase(_ name: String, category: ProfilingCategory) {
        let ts = currentTimestampUs()
        let snapshot = config.trackMemory ? takeSnapshot() : nil

        lock.lock()
        if let entry = activeSignpostIDs.removeValue(forKey: name) {
            signposter.endInterval("Phase", entry.state, "\(name) done")
        }
        events.append(ProfilingEvent(
            name: name, category: category, phase: .end, timestampUs: ts,
            mlxActiveBytes: snapshot?.mlx.activeBytes, mlxCacheBytes: snapshot?.mlx.cacheBytes,
            mlxPeakBytes: snapshot?.mlx.peakBytes, processFootprintBytes: snapshot?.processFootprint
        ))
        if config.trackMemory, let snap = snapshot {
            appendTimeline(ts: ts, context: "end:\(name)", snapshot: snap)
        }
        lock.unlock()
    }

    public func recordComplete(_ name: String, category: ProfilingCategory, durationUs: UInt64) {
        let ts = currentTimestampUs()
        let startTs = ts >= durationUs ? ts - durationUs : 0
        lock.lock()
        events.append(ProfilingEvent(name: name, category: category, phase: .complete, timestampUs: startTs, durationUs: durationUs))
        lock.unlock()
    }

    /// Record a step (denoising step, generation token, TTS frame, etc.)
    public func recordStep(index: Int, total: Int, durationUs: UInt64, category: ProfilingCategory = .denoisingStep) {
        signposter.emitEvent("Step", id: signposter.makeSignpostID(), "Step \(index)/\(total) \(durationUs / 1000)ms")

        let ts = currentTimestampUs()
        let startTs = ts >= durationUs ? ts - durationUs : 0
        let gpuUtil = SystemMetrics.gpuUtilization()
        let cpuTime = SystemMetrics.processCPUTime()
        let mlx = SystemMetrics.mlxMemory()
        let footprint = SystemMetrics.processFootprint()

        lock.lock()
        events.append(ProfilingEvent(
            name: "Step \(index)/\(total)", category: category, phase: .complete,
            timestampUs: startTs, durationUs: durationUs,
            mlxActiveBytes: mlx.activeBytes, mlxCacheBytes: mlx.cacheBytes,
            mlxPeakBytes: mlx.peakBytes, processFootprintBytes: footprint,
            stepIndex: index, totalSteps: total
        ))
        appendTimeline(ts: ts, context: "step:\(index)/\(total)",
                        snapshot: RawSnapshot(mlx: mlx, processFootprint: footprint, cpuTime: cpuTime, gpuUtil: gpuUtil))
        lock.unlock()
    }

    // MARK: - Counter Events (pour courbes dans Chrome Trace)

    /// Ajoute un counter event pour visualiser des metriques en courbe dans Perfetto UI
    public func addCounterEvent(name: String, timestampUs: UInt64, values: [String: Double]) {
        lock.lock()
        // Les counter events sont representes comme des events "C" avec des args
        events.append(ProfilingEvent(
            name: name, category: .custom, phase: .counter,
            timestampUs: timestampUs
        ))
        // Stocker les valeurs dans le metadata pour l'export Chrome Trace
        _counterValues.append(CounterValue(name: name, timestampUs: timestampUs, values: values))
        lock.unlock()
    }

    /// Expose le timestamp courant pour les extensions
    public func currentTimestampUsPublic() -> UInt64 {
        currentTimestampUs()
    }

    /// Retourne les counter values pour l'export
    public func getCounterValues() -> [CounterValue] {
        lock.lock(); defer { lock.unlock() }
        return _counterValues
    }

    private var _counterValues: [CounterValue] = []

    // MARK: - Data Access

    public func getEvents() -> [ProfilingEvent] {
        lock.lock(); defer { lock.unlock() }; return events
    }

    public func getMemoryTimeline() -> [MemoryTimelineEntry] {
        lock.lock(); defer { lock.unlock() }; return memoryTimeline
    }

    public var elapsedSeconds: TimeInterval {
        CFAbsoluteTimeGetCurrent() - sessionStartTime
    }

    // MARK: - Snapshot Internals

    private struct RawSnapshot {
        let mlx: SystemMetrics.MLXMemorySnapshot
        let processFootprint: Int64
        let cpuTime: Double
        let gpuUtil: Int
    }

    private func takeSnapshot() -> RawSnapshot {
        RawSnapshot(
            mlx: SystemMetrics.mlxMemory(),
            processFootprint: SystemMetrics.processFootprint(),
            cpuTime: SystemMetrics.processCPUTime(),
            gpuUtil: SystemMetrics.gpuUtilization()
        )
    }

    private func appendTimeline(ts: UInt64, context: String, snapshot snap: RawSnapshot) {
        memoryTimeline.append(MemoryTimelineEntry(
            timestampUs: ts, context: context,
            mlxActiveMB: snap.mlx.activeMB, mlxCacheMB: snap.mlx.cacheMB,
            mlxPeakMB: snap.mlx.peakMB,
            processFootprintMB: Double(snap.processFootprint) / 1_048_576,
            cpuTimeSeconds: snap.cpuTime, gpuUtilization: snap.gpuUtil
        ))
    }

    // MARK: - Report Generation

    public func generateReport() -> String {
        let events = getEvents()
        let timeline = getMemoryTimeline()

        var phases: [(name: String, category: ProfilingCategory, durationMs: Double, cpuPct: Double, gpuPct: Int)] = []
        var stepDurations: [Double] = []

        var cpuTimeAtPoint: [String: Double] = [:]
        var gpuSamplesPerPhase: [String: [Int]] = [:]
        var currentPhase: String? = nil
        for entry in timeline {
            cpuTimeAtPoint[entry.context] = entry.cpuTimeSeconds
            if entry.context.hasPrefix("begin:") {
                currentPhase = String(entry.context.dropFirst(6))
                gpuSamplesPerPhase[currentPhase!, default: []].append(entry.gpuUtilization)
            } else if entry.context.hasPrefix("end:") {
                let phase = String(entry.context.dropFirst(4))
                gpuSamplesPerPhase[phase, default: []].append(entry.gpuUtilization)
                if currentPhase == phase { currentPhase = nil }
            } else if entry.context.hasPrefix("step:"), let phase = currentPhase {
                gpuSamplesPerPhase[phase, default: []].append(entry.gpuUtilization)
            }
        }

        var beginTimestamps: [String: (ts: UInt64, cat: ProfilingCategory)] = [:]
        for event in events {
            switch event.phase {
            case .begin:
                beginTimestamps[event.name] = (event.timestampUs, event.category)
            case .end:
                if let begin = beginTimestamps[event.name] {
                    let wallMs = Double(event.timestampUs - begin.ts) / 1000.0
                    let wallSec = wallMs / 1000.0
                    let cpuBegin = cpuTimeAtPoint["begin:\(event.name)"] ?? 0
                    let cpuEnd = cpuTimeAtPoint["end:\(event.name)"] ?? 0
                    let cpuPct = wallSec > 0 ? ((cpuEnd - cpuBegin) / wallSec) * 100 : 0
                    let gpuSamples = gpuSamplesPerPhase[event.name] ?? []
                    let gpuAvg = gpuSamples.isEmpty ? 0 : gpuSamples.reduce(0, +) / gpuSamples.count
                    phases.append((name: event.name, category: begin.cat, durationMs: wallMs, cpuPct: cpuPct, gpuPct: gpuAvg))
                    beginTimestamps.removeValue(forKey: event.name)
                }
            case .complete:
                if [.denoisingStep, .generationStep, .semanticCodeGen, .flowMatching, .codecDecode].contains(event.category),
                   let dur = event.durationUs {
                    stepDurations.append(Double(dur) / 1000.0)
                }
            default: break
            }
        }

        phases.sort { $0.category.sortOrder < $1.category.sortOrder }
        let totalMs = phases.reduce(0.0) { $0 + $1.durationMs }

        // Build metadata line
        let metaStr = metadata.sorted(by: { $0.key < $1.key }).map { "\($0.key): \($0.value)" }.joined(separator: "  ")

        var report = """

        \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256E}
        \u{2502}  \(title.padding(toLength: 64, withPad: " ", startingAt: 0))\u{2502}
        \u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}

        """
        if !metaStr.isEmpty { report += "  \(metaStr)\n" }
        report += "  Device: \(deviceArchitecture)  RAM: \(systemRAMGB)GB\n\n"

        report += "  PHASE TIMINGS                                    GPU%   CPU%\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        for phase in phases {
            let pct = totalMs > 0 ? (phase.durationMs / totalMs) * 100 : 0
            let bar = String(repeating: "\u{2588}", count: min(10, Int(pct / 10)))
            let name = phase.name.padding(toLength: 22, withPad: " ", startingAt: 0)
            let gpuStr = String(format: "%3d%%", phase.gpuPct)
            let cpuStr = phase.cpuPct > 0 ? String(format: "%5.1f%%", phase.cpuPct) : "   -  "
            report += "  \(name) \(formatDuration(phase.durationMs / 1000))  \(String(format: "%5.1f", pct))% \(bar)  \(gpuStr)  \(cpuStr)\n"
        }
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        report += "  \("TOTAL".padding(toLength: 22, withPad: " ", startingAt: 0)) \(formatDuration(totalMs / 1000))  100.0%\n"

        if !stepDurations.isEmpty {
            let avgMs = stepDurations.reduce(0, +) / Double(stepDurations.count)
            let minMs = stepDurations.min() ?? 0
            let maxMs = stepDurations.max() ?? 0
            let variance = stepDurations.map { ($0 - avgMs) * ($0 - avgMs) }.reduce(0, +) / max(1, Double(stepDurations.count - 1))

            report += "\n  STEP STATISTICS\n"
            report += "  \(String(repeating: "\u{2500}", count: 66))\n"
            report += "  Steps: \(stepDurations.count)\n"
            report += "  Average: \(formatDuration(avgMs / 1000))  Std: \(formatDuration(sqrt(variance) / 1000))\n"
            report += "  Min: \(formatDuration(minMs / 1000))  Max: \(formatDuration(maxMs / 1000))\n"
        }

        if !timeline.isEmpty {
            let peakActive = timeline.map(\.mlxActiveMB).max() ?? 0
            let peakProcess = timeline.map(\.processFootprintMB).max() ?? 0
            report += "\n  MEMORY\n"
            report += "  \(String(repeating: "\u{2500}", count: 66))\n"
            report += "  Peak MLX Active: \(String(format: "%.1f", peakActive)) MB\n"
            report += "  Peak Process: \(String(format: "%.1f", peakProcess)) MB\n"

            let keyPoints = timeline.filter { $0.context.hasPrefix("begin:") || $0.context.hasPrefix("end:") }
            if !keyPoints.isEmpty {
                report += "\n  Memory Timeline:\n"
                for entry in keyPoints {
                    report += "    \(entry.context.padding(toLength: 35, withPad: " ", startingAt: 0)) MLX: \(String(format: "%7.1f", entry.mlxActiveMB)) MB\n"
                }
            }
        }

        report += "\n\u{2570}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256F}\n"
        return report
    }
}

/// Category inference from phase names
extension ProfilingSession {
    public static func inferCategory(_ phaseName: String) -> ProfilingCategory {
        let name = phaseName.lowercased()
        // Check "unload" before "load" — "unload text" contains "load text"
        if name.contains("unload text") || name.contains("unload gemma") { return .textEncoderUnload }
        if name.contains("unload transformer") { return .transformerUnload }
        if name.contains("load text") || name.contains("load gemma") { return .textEncoderLoad }
        if name.contains("vlm") || name.contains("prompt enhancement") { return .vlmInterpretation }
        if name.contains("tokeniz") { return .tokenization }
        if name.contains("text encod") { return .textEncoding }
        if name.contains("load transformer") { return .transformerLoad }
        if name.contains("load vae") { return .vaeLoad }
        if name.contains("load audio") || name.contains("audio model") { return .audioLoad }
        if name.contains("audio denois") { return .audioDenoise }
        // Speech / TTS categories
        if name.contains("mel") || name.contains("spectrogram") { return .melSpectrogram }
        if name.contains("audio feature") || name.contains("feature extract") { return .audioFeatureExtract }
        if name.contains("semantic code") || name.contains("semantic gen") { return .semanticCodeGen }
        if name.contains("flow match") { return .flowMatching }
        if name.contains("codec") || name.contains("waveform decode") { return .codecDecode }
        if name.contains("voice embed") { return .voiceEmbedding }
        if name.contains("audio write") || name.contains("wav write") { return .audioWrite }
        if name.contains("upscal") { return .upscaler }
        if name.contains("denois") { return .denoisingLoop }
        if name.contains("vae decode") || name.contains("vae forward") { return .vaeDecode }
        if name.contains("frame conver") { return .frameConversion }
        if name.contains("video write") { return .videoWrite }
        if name.contains("prefill") { return .prefill }
        if name.contains("decod") { return .decoding }
        if name.contains("generat") { return .generation }
        if name.contains("vision") { return .visionEncode }
        if name.contains("audio encod") { return .audioEncode }
        if name.contains("post") || name.contains("export") { return .postProcess }
        return .custom
    }
}

/// Counter value for Chrome Trace counter events (loss curves, memory, etc.)
public struct CounterValue: Sendable {
    public let name: String
    public let timestampUs: UInt64
    public let values: [String: Double]
}

/// Shared duration formatter
public func formatDuration(_ duration: TimeInterval) -> String {
    if duration < 1 {
        return String(format: "%7.1fms", duration * 1000)
    } else if duration < 60 {
        return String(format: "%7.2fs ", duration)
    } else {
        let minutes = Int(duration / 60)
        let seconds = duration.truncatingRemainder(dividingBy: 60)
        return String(format: "%dm %04.1fs", minutes, seconds)
    }
}
