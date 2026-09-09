// SamplingTests.swift - Phase boundary cost, sampling, and time-weighted statistics
// Copyright 2026 Vincent Gourbin

import Testing
import Foundation
@testable import MLXProfiler

@Suite("Phase boundary cost")
struct PhaseBoundaryCostTests {

    private func microsecondsPerPair(_ config: ProfilingConfig, pairs: Int) -> Double {
        let session = ProfilingSession(config: config)
        for _ in 0..<500 {
            session.beginPhase("warmup", category: .custom)
            session.endPhase("warmup", category: .custom)
        }
        let start = Date()
        for _ in 0..<pairs {
            session.beginPhase("Layer", category: .custom)
            session.endPhase("Layer", category: .custom)
        }
        let elapsed = Date().timeIntervalSince(start)
        session.finish()
        return elapsed / Double(pairs) * 1_000_000
    }

    /// The regression this release exists to prevent.
    ///
    /// A begin/end pair used to take ~4.7 ms because each side took a full
    /// snapshot. At one phase per layer over 48 layers that is ~225 ms per token
    /// of pure profiler, which is what made a 5.5 ms layer measure 10.2 ms. The
    /// threshold is deliberately loose — this needs to catch a return to
    /// milliseconds, not police microseconds on a shared CI machine.
    @Test func testBoundaryIsCheap() {
        let cost = microsecondsPerPair(ProfilingConfig(), pairs: 20_000)
        #expect(cost < 100, "begin/end cost \(cost) us per pair; it should be tens of us at most")
    }

    /// And that the expensive path is still available, and still expensive —
    /// if this stops being true the comparison above has stopped meaning anything.
    @Test func testBoundarySnapshotsCostMore() {
        let light = microsecondsPerPair(ProfilingConfig(), pairs: 20_000)
        let heavy = microsecondsPerPair(
            ProfilingConfig(snapshotAtPhaseBoundaries: true), pairs: 2_000)
        #expect(heavy > light * 2)
    }
}

@Suite("GPU utilization reader")
struct GPUUtilizationReaderTests {

    /// Guards the 90x regression waiting to be reintroduced.
    ///
    /// Reading `Device Utilization %` via `IORegistryEntryCreateCFProperties`
    /// materializes the driver's whole property dictionary and costs ~1.5 ms;
    /// asking for the single `PerformanceStatistics` property costs ~17 us. At a
    /// 16 ms sampling rate that is the difference between 0.1 % of a core and
    /// 9.5 % of one.
    @Test func testReadIsCheap() {
        let reader = GPUUtilizationReader(preferred: .deviceUtilization)
        _ = reader.read()
        let start = Date()
        let reads = 2_000
        for _ in 0..<reads { _ = reader.read() }
        let microseconds = Date().timeIntervalSince(start) / Double(reads) * 1_000_000
        #expect(microseconds < 300, "GPU read cost \(microseconds) us; the whole-dictionary fetch is back")
    }

    @Test func testReadIsInRange() {
        let reader = GPUUtilizationReader(preferred: .deviceUtilization)
        for _ in 0..<20 {
            let value = reader.read()
            #expect(value >= 0 && value <= 100)
        }
    }

    /// IOReport is private API and may simply not be there; when it is not, the
    /// reader has to fall back rather than fail.
    @Test func testIOReportFallsBackCleanly() {
        let reader = GPUUtilizationReader(preferred: .ioReportResidency)
        if reader.backend == .deviceUtilization {
            #expect(reader.fallbackReason != nil)
        } else {
            #expect(reader.fallbackReason == nil)
        }
        #expect(reader.read() >= 0)
    }
}

@Suite("MetricsSampler")
struct MetricsSamplerTests {

    @Test func testSamplerCollectsOnItsOwnThread() {
        let origin = Date()
        let sampler = MetricsSampler(intervalMs: 5, systemMemoryIntervalMs: 20) {
            UInt64(Date().timeIntervalSince(origin) * 1_000_000)
        }
        sampler.start()
        Thread.sleep(forTimeInterval: 0.2)
        sampler.stop()

        let samples = sampler.samples()
        #expect(samples.count > 5)
        #expect(samples.map(\.timestampUs) == samples.map(\.timestampUs).sorted())
        #expect(samples.contains { $0.systemMemory != nil })
    }

    @Test func testStopIsIdempotentAndJoins() {
        let sampler = MetricsSampler(intervalMs: 5) { 0 }
        sampler.start()
        Thread.sleep(forTimeInterval: 0.05)
        sampler.stop()
        let count = sampler.samples().count
        sampler.stop()
        // A second stop must not start collecting again.
        #expect(sampler.samples().count == count)
    }

    @Test func testGapDetection() {
        let sampler = MetricsSampler(intervalMs: 5) { 0 }
        sampler.start()
        Thread.sleep(forTimeInterval: 0.05)
        sampler.stop()
        // Every sample shares timestamp 0, so nothing looks like a stall.
        #expect(sampler.gaps(thresholdSeconds: 1).isEmpty)
    }
}

@Suite("Time-weighted statistics")
struct TimeWeightedStatisticsTests {

    /// The bug this replaces: a plain mean over the samples inside the window
    /// weights each *sample* equally, so a densely sampled stretch dominates a
    /// sparsely sampled one no matter how little time it occupied.
    ///
    /// Here ten readings of 100 % are packed into the first 90 ms of a one-second
    /// window and five readings of 0 % cover the remaining 900 ms. Counting
    /// samples says the GPU was busy two thirds of the time; the clock says it
    /// was busy about a seventh of it.
    @Test func testDenseSamplingDoesNotOutweighTime() {
        var points: [(timestampUs: UInt64, value: Double)] = []
        for i in 0..<10 { points.append((UInt64(i * 10_000), 100)) }
        for i in 1...5 { points.append((UInt64(i * 200_000), 0)) }

        let plainMean = points.map(\.value).reduce(0, +) / Double(points.count)
        let aggregate = TimeWeightedStatistics.aggregate(
            points: points, startUs: 0, endUs: 1_000_000, coverage: .midpoint)!

        #expect(plainMean > 60)          // what counting samples would say
        #expect(aggregate.mean < 20)     // what the timeline actually shows
    }

    /// Pins the convention for point samples: a reading stands for the time
    /// closest to it, reaching halfway back to the previous reading and halfway
    /// on to the next. With readings far apart that assigns half of the gap to
    /// each side — which is the honest answer, since nothing was observed in
    /// between.
    @Test func testMidpointCoverageSplitsGapsEvenly() {
        let points: [(timestampUs: UInt64, value: Double)] = [(0, 0), (1_000_000, 100)]
        let aggregate = TimeWeightedStatistics.aggregate(
            points: points, startUs: 0, endUs: 1_000_000, coverage: .midpoint)!
        #expect(abs(aggregate.mean - 50) < 1)
    }

    @Test func testTrailingCoverageAttributesToPrecedingInterval() {
        let points: [(timestampUs: UInt64, value: Double)] = [
            (0, 0), (500_000, 100), (1_000_000, 0),
        ]
        // With trailing semantics the reading at 500 ms describes 0-500 ms, so
        // the first half of the window is the busy half.
        let firstHalf = TimeWeightedStatistics.aggregate(
            points: points, startUs: 0, endUs: 500_000, coverage: .trailing)
        let secondHalf = TimeWeightedStatistics.aggregate(
            points: points, startUs: 500_000, endUs: 1_000_000, coverage: .trailing)
        #expect(firstHalf!.mean > secondHalf!.mean)
    }

    @Test func testPercentilesAreWeighted() {
        var points: [(timestampUs: UInt64, value: Double)] = []
        for i in 0..<100 { points.append((UInt64(i * 10_000), Double(i))) }
        let aggregate = TimeWeightedStatistics.aggregate(
            points: points, startUs: 0, endUs: 1_000_000, coverage: .midpoint)!
        #expect(aggregate.minimum == 0)
        #expect(aggregate.maximum == 99)
        #expect(abs(aggregate.median - 50) < 5)
        #expect(aggregate.p10 < aggregate.median)
        #expect(aggregate.p90 > aggregate.median)
    }

    @Test func testEmptyAndDegenerateWindows() {
        #expect(TimeWeightedStatistics.aggregate(
            points: [], startUs: 0, endUs: 100, coverage: .midpoint) == nil)
        #expect(TimeWeightedStatistics.aggregate(
            points: [(0, 1)], startUs: 100, endUs: 100, coverage: .midpoint) == nil)
        // A window with no samples anywhere near it yields nothing rather than 0.
        #expect(TimeWeightedStatistics.aggregate(
            points: [(0, 1)], startUs: 500_000, endUs: 600_000, coverage: .midpoint) == nil)
    }
}

@Suite("Phase summaries")
struct PhaseSummaryTests {

    @Test func testRepeatedPhasesArePairedIndividually() {
        let session = ProfilingSession(config: ProfilingConfig(enableSampling: false))
        for _ in 0..<3 {
            session.beginPhase("Layer", category: .custom)
            Thread.sleep(forTimeInterval: 0.01)
            session.endPhase("Layer", category: .custom)
        }
        session.finish()

        let summaries = session.phaseSummaries()
        #expect(summaries.count == 3)
        #expect(summaries.allSatisfy { $0.durationMs > 5 })
    }

    @Test func testNestedPhasesPairInnermostFirst() {
        let session = ProfilingSession(config: ProfilingConfig(enableSampling: false))
        session.beginPhase("Outer", category: .custom)
        session.beginPhase("Inner", category: .custom)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Inner", category: .custom)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Outer", category: .custom)
        session.finish()

        let summaries = session.phaseSummaries()
        #expect(summaries.count == 2)
        let outer = summaries.first { $0.name == "Outer" }!
        let inner = summaries.first { $0.name == "Inner" }!
        #expect(outer.durationMs > inner.durationMs)
        #expect(outer.startUs <= inner.startUs)
        #expect(outer.endUs >= inner.endUs)
    }

    @Test func testCPUPercentComesFromTheSampler() {
        let session = ProfilingSession(config: ProfilingConfig(samplingIntervalMs: 5))
        session.beginPhase("Busy", category: .custom)
        // Keep one core busy so CPU% is unambiguously above zero.
        let deadline = Date().addingTimeInterval(0.2)
        var counter = 0.0
        while Date() < deadline { counter += 1 }
        session.endPhase("Busy", category: .custom)
        session.finish()
        #expect(counter > 0)

        let summary = session.phaseSummaries().first { $0.name == "Busy" }
        #expect(summary?.cpuPercent != nil)
        #expect((summary?.cpuPercent ?? 0) > 50)
    }
}

@Suite("Run environment")
struct RunEnvironmentTests {

    @Test func testBuildConfigurationIsReported() {
        let session = ProfilingSession(config: ProfilingConfig(enableSampling: false))
        session.finish()
        let report = session.generateReport()
        #expect(report.contains("Build: \(RunEnvironment.buildConfiguration)"))
        // A Debug run has to say so before it shows any timing.
        if RunEnvironment.isDebugBuild {
            #expect(report.contains("DEBUG BUILD"))
        }
    }

    @Test func testSystemMemorySnapshotIsPlausible() throws {
        let snapshot = try #require(SystemMetrics.systemMemory())
        #expect(snapshot.anonymousBytes > 0)
        #expect(snapshot.wiredBytes > 0)
        // Anonymous memory alone cannot exceed physical RAM.
        #expect(snapshot.anonymousBytes < Int64(ProcessInfo.processInfo.physicalMemory))
        if let level = snapshot.memoryStatusLevel { #expect(level >= 0 && level <= 100) }
    }

    @Test func testIdleSleepAssertionReleaseIsIdempotent() {
        let assertion = IdleSleepAssertion(reason: "test")
        assertion.release()
        assertion.release()
    }
}
