// ProfilingConfig.swift - Configuration for profiling sessions
// Copyright 2026 Vincent Gourbin

import Foundation

/// Configuration for what to profile and how
public struct ProfilingConfig: Sendable {
    /// Collect memory and utilization data at all.
    public var trackMemory: Bool

    /// Record per-step memory during inference
    public var trackPerStepMemory: Bool

    /// Output directory for trace files
    public var outputDirectory: URL?

    /// Whether to export Chrome Trace JSON
    public var exportChromeTrace: Bool

    /// Whether to print summary to console
    public var printSummary: Bool

    // MARK: - Sampling

    /// Collect GPU/CPU/memory on a background thread at a fixed rate.
    ///
    /// This is what keeps `beginPhase`/`endPhase` cheap. Turning it off leaves
    /// phases as bare timestamps with no utilization data attached.
    public var enableSampling: Bool

    /// Sampling period in milliseconds. 16 ms matches the cadence the Chrome
    /// Trace counters were already drawn at.
    public var samplingIntervalMs: Int

    /// How often to refresh system-wide memory, which is heavier than the rest.
    public var systemMemorySamplingIntervalMs: Int

    /// Collect system-wide memory (compressor, swap, wired, pressure level) in
    /// addition to this process's footprint.
    public var trackSystemMemory: Bool

    /// Also take a full snapshot on every phase boundary — the pre-1.5 behaviour.
    ///
    /// Costs roughly 4.7 ms per begin/end pair, so a phase per layer on a
    /// 48-layer model adds ~225 ms per token. Only worth it when you need memory
    /// pinned to an exact boundary and the phases are coarse.
    public var snapshotAtPhaseBoundaries: Bool

    /// How GPU busy-ness is measured. See ``GPUUtilizationBackend``.
    public var gpuBackend: GPUUtilizationBackend

    // MARK: - Run hygiene

    /// Hold an idle-sleep assertion for the lifetime of the session.
    public var preventIdleSleep: Bool

    /// Sampling gaps at least this long are flagged as a stall — a system sleep,
    /// a stopped process, or severe starvation.
    public var stallThresholdSeconds: Double

    /// Record power source and `pmset` settings in the report and trace metadata.
    public var recordPowerManagement: Bool

    public init(
        trackMemory: Bool = true,
        trackPerStepMemory: Bool = false,
        outputDirectory: URL? = nil,
        exportChromeTrace: Bool = true,
        printSummary: Bool = true,
        enableSampling: Bool = true,
        samplingIntervalMs: Int = 16,
        systemMemorySamplingIntervalMs: Int = 500,
        trackSystemMemory: Bool = true,
        snapshotAtPhaseBoundaries: Bool = false,
        gpuBackend: GPUUtilizationBackend = .deviceUtilization,
        preventIdleSleep: Bool = true,
        stallThresholdSeconds: Double = 5.0,
        recordPowerManagement: Bool = true
    ) {
        self.trackMemory = trackMemory
        self.trackPerStepMemory = trackPerStepMemory
        self.outputDirectory = outputDirectory
        self.exportChromeTrace = exportChromeTrace
        self.printSummary = printSummary
        self.enableSampling = enableSampling
        self.samplingIntervalMs = samplingIntervalMs
        self.systemMemorySamplingIntervalMs = systemMemorySamplingIntervalMs
        self.trackSystemMemory = trackSystemMemory
        self.snapshotAtPhaseBoundaries = snapshotAtPhaseBoundaries
        self.gpuBackend = gpuBackend
        self.preventIdleSleep = preventIdleSleep
        self.stallThresholdSeconds = stallThresholdSeconds
        self.recordPowerManagement = recordPowerManagement
    }

    /// Default config for a single profiled run
    public static let singleRun = ProfilingConfig()

    /// Config for benchmarking (no trace export)
    public static func benchmark(runs: Int = 3, warmup: Int = 1) -> ProfilingConfig {
        ProfilingConfig(trackMemory: true, trackPerStepMemory: false, exportChromeTrace: false, printSummary: true)
    }

    /// Config for detailed profiling with per-step memory
    public static let detailed = ProfilingConfig(
        trackMemory: true, trackPerStepMemory: true, exportChromeTrace: true, printSummary: true
    )

    /// Config for timing very fine phases — one per layer, say.
    ///
    /// Phase boundaries stay at a bare timestamp and the sampler carries the
    /// utilization data, so the profiler's own cost does not scale with the
    /// number of phases. This is the config to use when the phase duration is
    /// itself the measurement.
    public static let fineGrained = ProfilingConfig(
        trackMemory: true,
        trackPerStepMemory: false,
        exportChromeTrace: true,
        printSummary: true,
        enableSampling: true,
        samplingIntervalMs: 16,
        snapshotAtPhaseBoundaries: false,
        gpuBackend: .ioReportResidency
    )
}
