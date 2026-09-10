// MetricsSampler.swift - Fixed-interval background sampling of GPU/CPU/memory
// Copyright 2026 Vincent Gourbin
//
// Why this exists: reading GPU utilization, CPU time and memory *on a phase
// boundary* cost ~4.7 ms per begin/end pair. With one phase per transformer
// layer that is ~225 ms of pure profiler per token on a 48-layer model — enough
// to inflate a 5.5 ms layer to 10.2 ms and make the resulting trace a
// measurement of the profiler rather than of the model. Boundaries are now just
// timestamps; everything expensive happens here, off to the side, at a fixed
// rate that does not scale with how finely the caller carves up its phases.

import Foundation

/// One periodic reading of the machine's state.
public struct MetricsSample: Sendable {
    public let timestampUs: UInt64
    public let gpuUtilization: Int
    public let cpuTimeSeconds: Double
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int
    public let processFootprintBytes: Int64
    /// Present only on ticks where system memory was refreshed (it runs at a
    /// slower cadence than the main sample rate).
    public let systemMemory: SystemMemorySnapshot?

    public var mlxActiveMB: Double { Double(mlxActiveBytes) / 1_048_576 }
    public var mlxCacheMB: Double { Double(mlxCacheBytes) / 1_048_576 }
    public var mlxPeakMB: Double { Double(mlxPeakBytes) / 1_048_576 }
    public var processFootprintMB: Double { Double(processFootprintBytes) / 1_048_576 }
}

/// Summary statistics over a set of samples, weighted by how much time each one
/// stands for rather than by how many of them happened to land in the window.
public struct SampleAggregate: Sendable {
    public let count: Int
    public let mean: Double
    public let median: Double
    public let p10: Double
    public let p90: Double
    public let minimum: Double
    public let maximum: Double
}

/// How much of the timeline a single sample speaks for.
public enum SampleCoverage: Sendable {
    /// The value describes the interval that *ended* at the sample's timestamp.
    /// Correct for counters read as deltas, such as IOReport residency.
    case trailing
    /// The value is an instant, so it stands for the time nearest to it —
    /// halfway to the previous sample and halfway to the next.
    case midpoint
}

/// A stretch of wall-clock during which no sample was taken.
///
/// Long gaps are almost never the sampler's fault: they mean the machine slept,
/// or the process was stopped or badly starved. Three days of the Flash-Next
/// campaign were lost to the Mac going to sleep mid-run with nothing in the
/// trace to show for it.
public struct SamplingGap: Sendable {
    public let startUs: UInt64
    public let endUs: UInt64
    public var durationSeconds: Double { Double(endUs - startUs) / 1_000_000 }
}

/// Samples the machine on a fixed interval on a dedicated thread.
public final class MetricsSampler: @unchecked Sendable {

    private let intervalUs: UInt64
    private let systemMemoryIntervalUs: UInt64
    private let trackSystemMemory: Bool
    private let clock: @Sendable () -> UInt64
    private let gpuReader: GPUUtilizationReader

    private let lock = NSLock()
    private var _samples: [MetricsSample] = []
    private var thread: Thread?
    private var stopping = false
    private var running = false

    /// The GPU backend actually in use, after any fallback.
    public var gpuBackend: GPUUtilizationBackend { gpuReader.backend }
    /// Why the requested GPU backend was not used, if it wasn't.
    public var gpuBackendFallbackReason: String? { gpuReader.fallbackReason }
    /// Coverage semantics implied by the active GPU backend.
    public var gpuCoverage: SampleCoverage {
        gpuReader.backend == .ioReportResidency ? .trailing : .midpoint
    }

    public init(
        intervalMs: Int = 16,
        systemMemoryIntervalMs: Int = 500,
        trackSystemMemory: Bool = true,
        gpuBackend: GPUUtilizationBackend = .deviceUtilization,
        clock: @escaping @Sendable () -> UInt64
    ) {
        self.intervalUs = UInt64(max(1, intervalMs)) * 1000
        self.systemMemoryIntervalUs = UInt64(max(1, systemMemoryIntervalMs)) * 1000
        self.trackSystemMemory = trackSystemMemory
        self.clock = clock
        self.gpuReader = GPUUtilizationReader(preferred: gpuBackend)
    }

    // MARK: - Lifecycle

    public func start() {
        lock.lock()
        guard thread == nil else { lock.unlock(); return }
        stopping = false
        let thread = Thread { [weak self] in self?.loop() }
        thread.name = "com.mlxprofiler.sampler"
        // Above default so a saturated app does not starve the sampler and
        // manufacture gaps that look like a system sleep.
        thread.qualityOfService = .userInitiated
        self.thread = thread
        running = true
        lock.unlock()
        thread.start()
    }

    public func stop() {
        lock.lock()
        guard thread != nil else { lock.unlock(); return }
        stopping = true
        lock.unlock()

        // Wait for the loop to actually leave, so a stop/start pair cannot end up
        // with two sampling threads running at once.
        let deadline = Date().addingTimeInterval(1.0)
        while Date() < deadline {
            lock.lock(); let stillRunning = running; lock.unlock()
            if !stillRunning { break }
            usleep(1000)
        }

        // One last reading so the final phase is covered all the way to its end.
        appendSample(includeSystemMemory: trackSystemMemory)
        lock.lock(); thread = nil; lock.unlock()
    }

    private func loop() {
        defer { lock.lock(); running = false; lock.unlock() }
        var nextTick = DispatchTime.now().uptimeNanoseconds
        var nextSystemMemory: UInt64 = 0

        while true {
            lock.lock(); let shouldStop = stopping; lock.unlock()
            if shouldStop { return }

            let now = clock()
            let wantsSystemMemory = trackSystemMemory && now >= nextSystemMemory
            if wantsSystemMemory { nextSystemMemory = now + systemMemoryIntervalUs }
            appendSample(includeSystemMemory: wantsSystemMemory, at: now)

            // Absolute deadlines, so a slow tick does not push every later one out.
            nextTick &+= intervalUs &* 1000
            if !sleepUntil(nextTick) { return }
            let current = DispatchTime.now().uptimeNanoseconds
            if nextTick < current { nextTick = current }  // fell behind; resynchronize
        }
    }

    /// Sleeps until `deadline` (uptime nanoseconds), in slices, so that a stop
    /// request is noticed promptly whatever the sampling interval is.
    ///
    /// Sleeping the whole interval in one go would make `stop()` wait a full
    /// period; with an interval above a second it would outlast `stop()`'s own
    /// timeout and leave the thread running after `stop()` returned.
    ///
    /// Returns false if the sampler was asked to stop.
    private func sleepUntil(_ deadline: UInt64) -> Bool {
        let maximumSliceNs: UInt64 = 50_000_000
        while true {
            lock.lock(); let shouldStop = stopping; lock.unlock()
            if shouldStop { return false }

            let now = DispatchTime.now().uptimeNanoseconds
            guard deadline > now else { return true }
            let slice = min(deadline - now, maximumSliceNs)
            Thread.sleep(forTimeInterval: Double(slice) / 1_000_000_000)
        }
    }

    private func appendSample(includeSystemMemory: Bool, at timestamp: UInt64? = nil) {
        let ts = timestamp ?? clock()
        let mlx = SystemMetrics.mlxMemory()
        let sample = MetricsSample(
            timestampUs: ts,
            gpuUtilization: gpuReader.read(),
            cpuTimeSeconds: SystemMetrics.processCPUTime(),
            mlxActiveBytes: mlx.activeBytes,
            mlxCacheBytes: mlx.cacheBytes,
            mlxPeakBytes: mlx.peakBytes,
            processFootprintBytes: SystemMetrics.processFootprint(),
            systemMemory: includeSystemMemory ? SystemMetrics.systemMemory() : nil
        )
        lock.lock(); _samples.append(sample); lock.unlock()
    }

    // MARK: - Access

    public func samples() -> [MetricsSample] {
        lock.lock(); defer { lock.unlock() }; return _samples
    }

    /// Intervals longer than `threshold` with no sample in them.
    public func gaps(thresholdSeconds: Double) -> [SamplingGap] {
        let samples = self.samples()
        guard samples.count > 1 else { return [] }
        let thresholdUs = UInt64(thresholdSeconds * 1_000_000)
        var gaps: [SamplingGap] = []
        for i in 1..<samples.count {
            let start = samples[i - 1].timestampUs
            let end = samples[i].timestampUs
            if end > start, end - start >= thresholdUs {
                gaps.append(SamplingGap(startUs: start, endUs: end))
            }
        }
        return gaps
    }
}

// MARK: - Time-weighted statistics

public enum TimeWeightedStatistics {

    /// Aggregate `points` over `[startUs, endUs)`, weighting each value by the
    /// span of the window it actually accounts for.
    ///
    /// A plain mean over "samples whose timestamp falls inside the window" is
    /// wrong in two ways at once: it drops the sample that covers the start of
    /// the window, and it over-weights whichever region happened to be sampled
    /// more densely. Both matter for short phases, which is where the old
    /// boundary readings went astray.
    public static func aggregate(
        points: [(timestampUs: UInt64, value: Double)],
        startUs: UInt64,
        endUs: UInt64,
        coverage: SampleCoverage
    ) -> SampleAggregate? {
        guard endUs > startUs, !points.isEmpty else { return nil }
        let sorted = points.sorted { $0.timestampUs < $1.timestampUs }

        var weighted: [(value: Double, weight: Double)] = []
        for (index, point) in sorted.enumerated() {
            let previous = index > 0 ? sorted[index - 1].timestampUs : nil
            let next = index + 1 < sorted.count ? sorted[index + 1].timestampUs : nil
            let span = self.span(at: point.timestampUs, previous: previous, next: next, coverage: coverage)

            let lower = max(span.lowerBound, Double(startUs))
            let upper = min(span.upperBound, Double(endUs))
            let weight = upper - lower
            if weight > 0 { weighted.append((point.value, weight)) }
        }
        guard !weighted.isEmpty else { return nil }

        let totalWeight = weighted.reduce(0.0) { $0 + $1.weight }
        let mean = weighted.reduce(0.0) { $0 + $1.value * $1.weight } / totalWeight
        let values = weighted.map(\.value)

        return SampleAggregate(
            count: weighted.count,
            mean: mean,
            median: weightedPercentile(weighted, 0.5),
            p10: weightedPercentile(weighted, 0.1),
            p90: weightedPercentile(weighted, 0.9),
            minimum: values.min() ?? 0,
            maximum: values.max() ?? 0
        )
    }

    /// The slice of timeline a sample stands for, in microseconds.
    private static func span(
        at timestamp: UInt64, previous: UInt64?, next: UInt64?, coverage: SampleCoverage
    ) -> ClosedRange<Double> {
        let t = Double(timestamp)
        switch coverage {
        case .trailing:
            // Covers what came before it. With no predecessor, fall back to the
            // following interval so the first sample still carries some weight.
            let lower: Double
            if let previous { lower = Double(previous) }
            else if let next { lower = t - (Double(next) - t) }
            else { lower = t }
            return lower...t
        case .midpoint:
            let lower = previous.map { (Double($0) + t) / 2 } ?? t
            let upper = next.map { (Double($0) + t) / 2 } ?? t
            // A lone sample has no width; give it a nominal one so it is not
            // silently discarded.
            return lower == upper ? (t - 0.5)...(t + 0.5) : lower...upper
        }
    }

    private static func weightedPercentile(
        _ weighted: [(value: Double, weight: Double)], _ percentile: Double
    ) -> Double {
        let sorted = weighted.sorted { $0.value < $1.value }
        let total = sorted.reduce(0.0) { $0 + $1.weight }
        guard total > 0 else { return sorted.first?.value ?? 0 }
        let target = total * percentile
        var cumulative = 0.0
        for entry in sorted {
            cumulative += entry.weight
            if cumulative >= target { return entry.value }
        }
        return sorted.last?.value ?? 0
    }
}
