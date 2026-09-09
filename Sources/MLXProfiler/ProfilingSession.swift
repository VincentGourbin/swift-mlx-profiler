// ProfilingSession.swift - Central profiling session coordinator
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import os

/// Collects profiling events during an inference run.
///
/// Tracks phase timings, memory snapshots, GPU/CPU utilization, and exports
/// Chrome Trace JSON for visualization in [Perfetto UI](https://ui.perfetto.dev/).
///
/// ## Cost of a phase boundary
///
/// Since 1.5, `beginPhase`/`endPhase` do nothing but read a clock and emit an
/// `os_signpost`; GPU, CPU and memory come from a background sampler running at a
/// fixed rate. Before that, each boundary took a full snapshot — about 4.7 ms per
/// begin/end pair, which is invisible on a phase that lasts a second and ruinous
/// on one that lasts five milliseconds. Set
/// ``ProfilingConfig/snapshotAtPhaseBoundaries`` to get the old behaviour back
/// where you need memory pinned to an exact boundary.
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

    /// Subsystem the signposts are emitted under. Metal System Trace merging uses
    /// it to find this session's phases in an Instruments trace.
    public let signpostSubsystem: String

    // Sampling and run hygiene
    private let sampler: MetricsSampler?
    private var sleepAssertion: IdleSleepAssertion?
    private var finished = false

    /// Power source at session start (`AC Power`, `Battery Power`, …), if known.
    public let powerSource: String?
    /// Sleep-related `pmset -g` settings recorded at session start.
    public let powerManagementSettings: [String: String]

    /// GPU kernel intervals merged in from a Metal System Trace, if any.
    internal var gpuKernelIntervals: [GPUKernelInterval] = []
    /// Metal captures written during this session, as (phase, file).
    internal var gpuCaptures: [(phase: String, url: URL)] = []
    /// The capture currently in flight, if any.
    internal var activeCapture: (phase: String, url: URL, startUs: UInt64)?
    /// Summary of the last merged Metal System Trace, if any.
    internal var gpuKernelSummary: GPUKernelSummary?

    public init(config: ProfilingConfig = .singleRun, subsystem: String = "com.mlxprofiler") {
        let sessionId = UUID().uuidString
        let started = CFAbsoluteTimeGetCurrent()

        self.sessionId = sessionId
        self.startTime = Date()
        self.config = config
        self.sessionStartTime = started
        self.deviceArchitecture = GPU.deviceInfo().architecture
        self.systemRAMGB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
        self.signpostSubsystem = subsystem
        self.signposter = OSSignposter(subsystem: subsystem, category: .pointsOfInterest)

        self.powerSource = config.recordPowerManagement ? RunEnvironment.powerSource() : nil
        self.powerManagementSettings = config.recordPowerManagement
            ? RunEnvironment.powerManagementSettings() : [:]

        if config.enableSampling && config.trackMemory {
            // Capture the start instant rather than `self`, so the sampler's clock
            // closure does not retain the session.
            let sampler = MetricsSampler(
                intervalMs: config.samplingIntervalMs,
                systemMemoryIntervalMs: config.systemMemorySamplingIntervalMs,
                trackSystemMemory: config.trackSystemMemory,
                gpuBackend: config.gpuBackend,
                clock: { UInt64(max(0, (CFAbsoluteTimeGetCurrent() - started) * 1_000_000)) }
            )
            self.sampler = sampler
            sampler.start()
        } else {
            self.sampler = nil
        }

        if config.preventIdleSleep {
            self.sleepAssertion = IdleSleepAssertion(reason: "MLXProfiler session \(sessionId)")
        }
    }

    deinit { finish() }

    /// Stops sampling and releases the idle-sleep assertion.
    ///
    /// Idempotent, and called automatically on deinit. Call it explicitly before
    /// generating a report if you want the last sample to land inside the run
    /// rather than after it.
    public func finish() {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        lock.unlock()

        sampler?.stop()
        sleepAssertion?.release()
        sleepAssertion = nil
    }

    private func currentTimestampUs() -> UInt64 {
        UInt64(max(0, (CFAbsoluteTimeGetCurrent() - sessionStartTime) * 1_000_000))
    }

    // MARK: - Event Recording

    public func beginPhase(_ name: String, category: ProfilingCategory) {
        // Read the clock first and do everything else after, so whatever the
        // bookkeeping costs it lands outside the measured interval.
        let ts = currentTimestampUs()
        let snapshot = config.snapshotAtPhaseBoundaries ? takeSnapshot() : nil
        let spID = signposter.makeSignpostID()
        let state = signposter.beginInterval("Phase", id: spID, "\(name)")

        lock.lock()
        activeSignpostIDs[name] = (id: spID, state: state)
        events.append(ProfilingEvent(
            name: name, category: category, phase: .begin, timestampUs: ts,
            mlxActiveBytes: snapshot?.mlx.activeBytes, mlxCacheBytes: snapshot?.mlx.cacheBytes,
            mlxPeakBytes: snapshot?.mlx.peakBytes, processFootprintBytes: snapshot?.processFootprint
        ))
        if let snap = snapshot { appendTimeline(ts: ts, context: "begin:\(name)", snapshot: snap) }
        lock.unlock()
    }

    public func endPhase(_ name: String, category: ProfilingCategory) {
        let ts = currentTimestampUs()
        let snapshot = config.snapshotAtPhaseBoundaries ? takeSnapshot() : nil

        lock.lock()
        if let entry = activeSignpostIDs.removeValue(forKey: name) {
            signposter.endInterval("Phase", entry.state, "\(name) done")
        }
        events.append(ProfilingEvent(
            name: name, category: category, phase: .end, timestampUs: ts,
            mlxActiveBytes: snapshot?.mlx.activeBytes, mlxCacheBytes: snapshot?.mlx.cacheBytes,
            mlxPeakBytes: snapshot?.mlx.peakBytes, processFootprintBytes: snapshot?.processFootprint
        ))
        if let snap = snapshot { appendTimeline(ts: ts, context: "end:\(name)", snapshot: snap) }
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
    ///
    /// Memory here is read from cheap Mach calls only; GPU utilization comes from
    /// the sampler, which is why this no longer touches the IO registry.
    public func recordStep(index: Int, total: Int, durationUs: UInt64, category: ProfilingCategory = .denoisingStep) {
        let ts = currentTimestampUs()
        signposter.emitEvent("Step", id: signposter.makeSignpostID(), "Step \(index)/\(total) \(durationUs / 1000)ms")

        let startTs = ts >= durationUs ? ts - durationUs : 0
        let mlx = SystemMetrics.mlxMemory()
        let footprint = SystemMetrics.processFootprint()
        let cpuTime = SystemMetrics.processCPUTime()

        lock.lock()
        events.append(ProfilingEvent(
            name: "Step \(index)/\(total)", category: category, phase: .complete,
            timestampUs: startTs, durationUs: durationUs,
            mlxActiveBytes: mlx.activeBytes, mlxCacheBytes: mlx.cacheBytes,
            mlxPeakBytes: mlx.peakBytes, processFootprintBytes: footprint,
            stepIndex: index, totalSteps: total
        ))
        if config.trackPerStepMemory {
            appendTimeline(ts: ts, context: "step:\(index)/\(total)",
                           snapshot: RawSnapshot(mlx: mlx, processFootprint: footprint,
                                                 cpuTime: cpuTime, gpuUtil: 0, system: nil))
        }
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

    /// Memory and utilization timeline, merging boundary/step snapshots with the
    /// background sampler's readings (context `sample`), ordered by timestamp.
    public func getMemoryTimeline() -> [MemoryTimelineEntry] {
        lock.lock()
        var merged = memoryTimeline
        lock.unlock()

        if let sampler {
            merged.append(contentsOf: sampler.samples().map { sample in
                MemoryTimelineEntry(
                    timestampUs: sample.timestampUs, context: "sample",
                    mlxActiveMB: sample.mlxActiveMB, mlxCacheMB: sample.mlxCacheMB,
                    mlxPeakMB: sample.mlxPeakMB, processFootprintMB: sample.processFootprintMB,
                    cpuTimeSeconds: sample.cpuTimeSeconds, gpuUtilization: sample.gpuUtilization,
                    systemMemory: sample.systemMemory
                )
            })
        }
        merged.sort { $0.timestampUs < $1.timestampUs }
        return merged
    }

    /// Raw sampler readings, or an empty array if sampling is off.
    public func getSamples() -> [MetricsSample] { sampler?.samples() ?? [] }

    /// Stretches with no sample at all — a system sleep, or a stopped process.
    public func getStalls() -> [SamplingGap] {
        sampler?.gaps(thresholdSeconds: config.stallThresholdSeconds) ?? []
    }

    /// The GPU backend in use, after any fallback.
    public var activeGPUBackend: GPUUtilizationBackend? { sampler?.gpuBackend }

    public var elapsedSeconds: TimeInterval {
        CFAbsoluteTimeGetCurrent() - sessionStartTime
    }

    // MARK: - Snapshot Internals

    internal struct RawSnapshot {
        let mlx: SystemMetrics.MLXMemorySnapshot
        let processFootprint: Int64
        let cpuTime: Double
        let gpuUtil: Int
        let system: SystemMemorySnapshot?
    }

    private func takeSnapshot() -> RawSnapshot {
        RawSnapshot(
            mlx: SystemMetrics.mlxMemory(),
            processFootprint: SystemMetrics.processFootprint(),
            cpuTime: SystemMetrics.processCPUTime(),
            gpuUtil: SystemMetrics.gpuUtilization(),
            system: config.trackSystemMemory ? SystemMetrics.systemMemory() : nil
        )
    }

    /// Caller must hold `lock`.
    private func appendTimeline(ts: UInt64, context: String, snapshot snap: RawSnapshot) {
        memoryTimeline.append(MemoryTimelineEntry(
            timestampUs: ts, context: context,
            mlxActiveMB: snap.mlx.activeMB, mlxCacheMB: snap.mlx.cacheMB,
            mlxPeakMB: snap.mlx.peakMB,
            processFootprintMB: Double(snap.processFootprint) / 1_048_576,
            cpuTimeSeconds: snap.cpuTime, gpuUtilization: snap.gpuUtil,
            systemMemory: snap.system
        ))
    }

    /// Records an instant event on the trace — used for stalls and captures.
    internal func recordInstant(_ name: String, category: ProfilingCategory, timestampUs: UInt64) {
        lock.lock()
        events.append(ProfilingEvent(name: name, category: category, phase: .instant, timestampUs: timestampUs))
        lock.unlock()
    }
}

/// A phase resolved to an interval, with its sampled statistics.
public struct PhaseSummary: Sendable {
    public let name: String
    public let category: ProfilingCategory
    public let startUs: UInt64
    public let endUs: UInt64
    public let durationMs: Double
    public let cpuPercent: Double?
    public let gpu: SampleAggregate?
    /// Growth of compressor-occupied memory across the phase, in MB.
    public let compressorGrowthMB: Double?
    public let peakMLXActiveMB: Double?
    public let peakProcessMB: Double?
}

// MARK: - Analysis

extension ProfilingSession {

    /// Pairs begin/end events into intervals and attaches the sampled statistics
    /// that fall inside each one.
    public func phaseSummaries() -> [PhaseSummary] {
        let events = getEvents()
        let samples = getSamples()
        let gpuCoverage = sampler?.gpuCoverage ?? .midpoint

        let gpuPoints = samples.map { (timestampUs: $0.timestampUs, value: Double($0.gpuUtilization)) }
        let cpuSeries = samples.map { (timestampUs: $0.timestampUs, value: $0.cpuTimeSeconds) }
        let compressorSeries = samples.compactMap { sample -> (timestampUs: UInt64, value: Double)? in
            guard let system = sample.systemMemory else { return nil }
            return (sample.timestampUs, system.compressorOccupiedMB)
        }

        // Phases can repeat and can nest, so match each end to the most recent
        // unmatched begin of the same name.
        var openPhases: [String: [(ts: UInt64, cat: ProfilingCategory)]] = [:]
        var summaries: [PhaseSummary] = []

        for event in events {
            switch event.phase {
            case .begin:
                openPhases[event.name, default: []].append((event.timestampUs, event.category))
            case .end:
                guard var stack = openPhases[event.name], let begin = stack.popLast() else { continue }
                openPhases[event.name] = stack
                let startUs = begin.ts
                let endUs = max(event.timestampUs, begin.ts)
                let wallMs = Double(endUs - startUs) / 1000.0

                let inWindow = samples.filter { $0.timestampUs >= startUs && $0.timestampUs <= endUs }
                summaries.append(PhaseSummary(
                    name: event.name,
                    category: begin.cat,
                    startUs: startUs,
                    endUs: endUs,
                    durationMs: wallMs,
                    cpuPercent: cpuPercent(series: cpuSeries, startUs: startUs, endUs: endUs),
                    gpu: TimeWeightedStatistics.aggregate(
                        points: gpuPoints, startUs: startUs, endUs: endUs, coverage: gpuCoverage),
                    compressorGrowthMB: growth(series: compressorSeries, startUs: startUs, endUs: endUs),
                    peakMLXActiveMB: inWindow.map(\.mlxActiveMB).max(),
                    peakProcessMB: inWindow.map(\.processFootprintMB).max()
                ))
            default:
                break
            }
        }
        return summaries
    }

    /// CPU% across a window, from the cumulative CPU-time series.
    ///
    /// Interpolating the cumulative counter at the two ends is exact regardless of
    /// where samples happened to fall, which matters for phases shorter than the
    /// sampling period.
    private func cpuPercent(
        series: [(timestampUs: UInt64, value: Double)], startUs: UInt64, endUs: UInt64
    ) -> Double? {
        guard endUs > startUs,
              let begin = interpolate(series: series, at: startUs),
              let end = interpolate(series: series, at: endUs)
        else { return nil }
        let wallSeconds = Double(endUs - startUs) / 1_000_000
        guard wallSeconds > 0 else { return nil }
        return ((end - begin) / wallSeconds) * 100
    }

    private func growth(
        series: [(timestampUs: UInt64, value: Double)], startUs: UInt64, endUs: UInt64
    ) -> Double? {
        guard let begin = interpolate(series: series, at: startUs),
              let end = interpolate(series: series, at: endUs)
        else { return nil }
        return end - begin
    }

    /// Linear interpolation of a monotonically sampled series, clamped at the ends.
    private func interpolate(
        series: [(timestampUs: UInt64, value: Double)], at timestamp: UInt64
    ) -> Double? {
        guard !series.isEmpty else { return nil }
        if timestamp <= series[0].timestampUs { return series[0].value }
        if let last = series.last, timestamp >= last.timestampUs { return last.value }

        var low = 0, high = series.count - 1
        while high - low > 1 {
            let mid = (low + high) / 2
            if series[mid].timestampUs <= timestamp { low = mid } else { high = mid }
        }
        let a = series[low], b = series[high]
        let span = Double(b.timestampUs - a.timestampUs)
        guard span > 0 else { return a.value }
        let ratio = Double(timestamp - a.timestampUs) / span
        return a.value + (b.value - a.value) * ratio
    }
}

// MARK: - Report Generation

extension ProfilingSession {

    public func generateReport() -> String {
        let events = getEvents()
        let timeline = getMemoryTimeline()
        let samples = getSamples()

        var phases = phaseSummaries()
        var stepDurations: [Double] = []
        for event in events where event.phase == .complete {
            if [.denoisingStep, .generationStep, .semanticCodeGen, .flowMatching, .codecDecode]
                .contains(event.category), let dur = event.durationUs {
                stepDurations.append(Double(dur) / 1000.0)
            }
        }

        phases.sort { $0.category.sortOrder < $1.category.sortOrder }
        let totalMs = phases.reduce(0.0) { $0 + $1.durationMs }

        let metaStr = metadata.sorted(by: { $0.key < $1.key }).map { "\($0.key): \($0.value)" }.joined(separator: "  ")

        var report = """

        \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256E}
        \u{2502}  \(title.padding(toLength: 64, withPad: " ", startingAt: 0))\u{2502}
        \u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}

        """
        report += buildWarningSection()
        if !metaStr.isEmpty { report += "  \(metaStr)\n" }
        report += environmentLines(sampleCount: samples.count)
        report += "\n"

        report += "  PHASE TIMINGS                                GPU%   p50   CPU%\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        for phase in phases {
            let pct = totalMs > 0 ? (phase.durationMs / totalMs) * 100 : 0
            let bar = String(repeating: "\u{2588}", count: min(6, Int(pct / 16.7)))
            let name = phase.name.padding(toLength: 22, withPad: " ", startingAt: 0)
            let gpuStr = phase.gpu.map { String(format: "%4.0f%%", $0.mean) } ?? "   -  "
            let p50Str = phase.gpu.map { String(format: "%4.0f%%", $0.median) } ?? "   -  "
            let cpuStr = phase.cpuPercent.map { String(format: "%5.1f%%", $0) } ?? "   -  "
            report += "  \(name) \(formatDuration(phase.durationMs / 1000))  \(String(format: "%5.1f", pct))% \(bar.padding(toLength: 6, withPad: " ", startingAt: 0)) \(gpuStr) \(p50Str) \(cpuStr)\n"
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

        report += buildMemorySection(timeline: timeline, samples: samples, phases: phases)
        report += buildGPUKernelSection()
        report += buildStallSection()

        report += "\n\u{2570}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256F}\n"
        return report
    }

    /// Anything that makes the numbers below untrustworthy, said before them.
    private func buildWarningSection() -> String {
        guard RunEnvironment.isDebugBuild else { return "" }
        let red = "\u{1B}[31m", reset = "\u{1B}[0m"
        return """
          \(red)\u{26A0}  DEBUG BUILD - THESE TIMINGS ARE NOT A BENCHMARK\(reset)
             MLX's C++ is compiled at -O0 here; host cost runs 1.2x to 1.8x
             higher than Release, and not by a constant factor across phases.
             Rebuild with `swift build -c release` before drawing conclusions.


        """
    }

    private func environmentLines(sampleCount: Int) -> String {
        var line = "  Device: \(deviceArchitecture)  RAM: \(systemRAMGB)GB"
        line += "  Build: \(RunEnvironment.buildConfiguration)"
        if let powerSource { line += "  Power: \(powerSource)" }
        line += "\n"

        if let backend = activeGPUBackend {
            line += "  Sampler: \(config.samplingIntervalMs) ms, \(backend.rawValue), \(sampleCount) samples\n"
        } else {
            line += "  Sampler: off (phase GPU/CPU statistics unavailable)\n"
        }

        if !powerManagementSettings.isEmpty {
            let settings = powerManagementSettings.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
            line += "  pmset: \(settings)"
            line += config.preventIdleSleep ? "  (idle sleep held off)\n" : "\n"
        }
        return line
    }

    private func buildMemorySection(
        timeline: [MemoryTimelineEntry], samples: [MetricsSample], phases: [PhaseSummary]
    ) -> String {
        guard !timeline.isEmpty else { return "" }
        var report = "\n  MEMORY\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"

        let peakActive = timeline.map(\.mlxActiveMB).max() ?? 0
        let peakProcess = timeline.map(\.processFootprintMB).max() ?? 0
        report += "  Peak MLX Active: \(String(format: "%.1f", peakActive)) MB"
        if let at = timeline.first(where: { $0.mlxActiveMB == peakActive })?.timestampUs {
            report += "  at \(formatTimestamp(at))"
        }
        report += "\n"
        report += "  Peak Process: \(String(format: "%.1f", peakProcess)) MB"
        if let at = timeline.first(where: { $0.processFootprintMB == peakProcess })?.timestampUs {
            report += "  at \(formatTimestamp(at))"
        }
        report += "\n"

        // System-wide memory: what "Process (MB)" alone could never explain.
        let systemSamples = samples.compactMap(\.systemMemory)
        if !systemSamples.isEmpty {
            report += "\n  System-wide (peaks):\n"
            func peak(_ keyPath: KeyPath<SystemMemorySnapshot, Double>) -> Double {
                systemSamples.map { $0[keyPath: keyPath] }.max() ?? 0
            }
            report += "    Anonymous:   \(String(format: "%8.1f", peak(\.anonymousMB))) MB"
            report += "    Wired: \(String(format: "%8.1f", peak(\.wiredMB))) MB\n"
            report += "    File-backed: \(String(format: "%8.1f", peak(\.fileBackedMB))) MB"
            report += "    Speculative: \(String(format: "%6.1f", peak(\.speculativeMB))) MB\n"

            let peakCompressor = peak(\.compressorOccupiedMB)
            report += "    Compressor:  \(String(format: "%8.1f", peakCompressor)) MB occupied"
            let peakStored = peak(\.compressorStoredMB)
            if peakCompressor > 0 {
                report += ", holding \(String(format: "%.1f", peakStored)) MB"
                report += " (\(String(format: "%.2f", peakStored / peakCompressor))x)"
            }
            report += "\n"

            let swapPeaks = systemSamples.compactMap(\.swapUsedMB)
            if let peakSwap = swapPeaks.max() {
                report += "    Swap used:   \(String(format: "%8.1f", peakSwap)) MB\n"
            }
            if let lowestLevel = systemSamples.compactMap(\.memoryStatusLevel).min() {
                report += "    Lowest kern.memorystatus_level: \(lowestLevel)"
                report += lowestLevel < 20 ? "  \u{26A0} jetsam territory\n" : "\n"
            }
        }

        // A phase during which the compressor grew by a gigabyte was not measuring
        // the model; it was measuring the machine running out of room.
        let thrashing = phases.filter { ($0.compressorGrowthMB ?? 0) > 1024 }
        if !thrashing.isEmpty {
            report += "\n"
            for phase in thrashing {
                let growthGB = (phase.compressorGrowthMB ?? 0) / 1024
                report += "  \u{26A0} Compressor grew \(String(format: "%.1f", growthGB)) GB during "
                report += "\"\(phase.name)\" - that phase includes compression stalls,\n"
                report += "    not just model work.\n"
            }
        }

        let keyPoints = timeline.filter { $0.context.hasPrefix("begin:") || $0.context.hasPrefix("end:") }
        if !keyPoints.isEmpty {
            report += "\n  Memory Timeline:\n"
            for entry in keyPoints {
                report += "    \(entry.context.padding(toLength: 35, withPad: " ", startingAt: 0)) MLX: \(String(format: "%7.1f", entry.mlxActiveMB)) MB\n"
            }
        }
        return report
    }

    private func buildStallSection() -> String {
        let stalls = getStalls()
        guard !stalls.isEmpty else { return "" }
        var report = "\n  STALLS\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        report += "  \u{26A0} \(stalls.count) gap(s) of \(String(format: "%.0f", config.stallThresholdSeconds))s or more with no sample.\n"
        report += "    The sampler runs at \(config.samplingIntervalMs) ms and does not simply stop:\n"
        report += "    this is a system sleep, a suspended process, or severe starvation.\n"
        for stall in stalls.prefix(10) {
            report += "    at \(formatTimestamp(stall.startUs)) for \(String(format: "%.1f", stall.durationSeconds))s\n"
        }
        if stalls.count > 10 { report += "    ... and \(stalls.count - 10) more\n" }
        return report
    }
}

/// Formats a session-relative timestamp as mm:ss.mmm
public func formatTimestamp(_ timestampUs: UInt64) -> String {
    let totalSeconds = Double(timestampUs) / 1_000_000
    let minutes = Int(totalSeconds / 60)
    let seconds = totalSeconds.truncatingRemainder(dividingBy: 60)
    return String(format: "%02d:%06.3f", minutes, seconds)
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
