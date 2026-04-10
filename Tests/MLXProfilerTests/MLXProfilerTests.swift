// MLXProfilerTests.swift
// Copyright 2026 Vincent Gourbin

import Testing
import Foundation
@testable import MLXProfiler

// MARK: - SystemMetrics

@Suite("SystemMetrics")
struct SystemMetricsTests {
    @Test func testGPUUtilization() {
        let gpu = SystemMetrics.gpuUtilization()
        #expect(gpu >= 0 && gpu <= 100)
    }

    @Test func testCPUTime() {
        let cpu = SystemMetrics.processCPUTime()
        #expect(cpu > 0)
    }

    @Test func testProcessFootprint() {
        let footprint = SystemMetrics.processFootprint()
        #expect(footprint > 0)
    }
}

// MARK: - ProfilingCategory

@Suite("ProfilingCategory")
struct ProfilingCategoryTests {
    @Test func testThreadIds() {
        let categories: [ProfilingCategory] = [
            .textEncoderLoad, .textEncoding, .denoisingLoop, .vaeDecode,
            .prefill, .generation, .custom
        ]
        for cat in categories {
            #expect(cat.threadId > 0)
            #expect(!cat.threadName.isEmpty)
        }
    }

    @Test func testSortOrder() {
        #expect(ProfilingCategory.textEncoding.sortOrder < ProfilingCategory.denoisingLoop.sortOrder)
        #expect(ProfilingCategory.denoisingLoop.sortOrder < ProfilingCategory.vaeDecode.sortOrder)
    }

    @Test func testPhaseRawValues() {
        #expect(ProfilingPhase.begin.rawValue == "B")
        #expect(ProfilingPhase.end.rawValue == "E")
        #expect(ProfilingPhase.complete.rawValue == "X")
    }
}

// MARK: - ProfilingConfig

@Suite("ProfilingConfig")
struct ProfilingConfigTests {
    @Test func testDefaults() {
        let config = ProfilingConfig()
        #expect(config.trackMemory == true)
        #expect(config.trackPerStepMemory == false)
        #expect(config.exportChromeTrace == true)
    }

    @Test func testPresets() {
        #expect(ProfilingConfig.singleRun.trackMemory == true)
        #expect(ProfilingConfig.detailed.trackPerStepMemory == true)
    }
}

// MARK: - ProfilingSession

@Suite("ProfilingSession")
struct ProfilingSessionTests {
    @Test func testInit() {
        let session = ProfilingSession()
        #expect(!session.sessionId.isEmpty)
        #expect(session.systemRAMGB > 0)
    }

    @Test func testBeginEndPhase() {
        let session = ProfilingSession()
        session.beginPhase("Test", category: .textEncoding)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Test", category: .textEncoding)

        let events = session.getEvents()
        #expect(events.count == 2)
        #expect(events[0].phase == .begin)
        #expect(events[1].phase == .end)
    }

    @Test func testRecordStep() {
        let session = ProfilingSession()
        session.recordStep(index: 1, total: 8, durationUs: 100_000)
        let events = session.getEvents()
        #expect(events.count == 1)
        #expect(events[0].stepIndex == 1)
    }

    @Test func testMetadata() {
        let session = ProfilingSession()
        session.metadata = ["model": "LTX-2.3", "quant": "qint8"]
        session.title = "LTX-2.3 PROFILING REPORT"
        #expect(session.metadata["model"] == "LTX-2.3")
        #expect(session.title.contains("LTX"))
    }

    @Test func testGenerateReport() {
        let session = ProfilingSession()
        session.title = "TEST REPORT"
        session.metadata = ["model": "test"]
        session.beginPhase("Denoising", category: .denoisingLoop)
        session.recordStep(index: 1, total: 2, durationUs: 50_000)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Denoising", category: .denoisingLoop)

        let report = session.generateReport()
        #expect(report.contains("TEST REPORT"))
        #expect(report.contains("Denoising"))
        #expect(report.contains("GPU%"))
    }

    @Test func testMemoryTimeline() {
        let session = ProfilingSession()
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)
        let timeline = session.getMemoryTimeline()
        #expect(timeline.count == 2)
        #expect(timeline[0].gpuUtilization >= 0)
        #expect(timeline[0].cpuTimeSeconds > 0)
    }
}

// MARK: - Category Inference

@Suite("Category Inference")
struct CategoryInferenceTests {
    @Test func testInference() {
        #expect(ProfilingSession.inferCategory("Load Text Encoder") == .textEncoderLoad)
        #expect(ProfilingSession.inferCategory("Denoising loop") == .denoisingLoop)
        #expect(ProfilingSession.inferCategory("VAE Decode") == .vaeDecode)
        #expect(ProfilingSession.inferCategory("Tokenization") == .tokenization)
        #expect(ProfilingSession.inferCategory("Prefill") == .prefill)
        #expect(ProfilingSession.inferCategory("Frame conversion") == .frameConversion)
        #expect(ProfilingSession.inferCategory("unknown") == .custom)
    }
}

// MARK: - ChromeTraceExporter

@Suite("ChromeTraceExporter")
struct ChromeTraceExporterTests {
    @Test func testExport() {
        let session = ProfilingSession()
        session.metadata = ["model": "test"]
        session.beginPhase("Test", category: .textEncoding)
        session.endPhase("Test", category: .textEncoding)

        let data = ChromeTraceExporter.export(session: session)
        #expect(data.count > 0)

        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json != nil)
        let events = json?["traceEvents"] as? [[String: Any]]
        #expect((events?.count ?? 0) > 0)
    }

    @Test func testComparison() {
        let s1 = ProfilingSession()
        s1.beginPhase("A", category: .textEncoding)
        s1.endPhase("A", category: .textEncoding)
        let s2 = ProfilingSession()
        s2.beginPhase("A", category: .textEncoding)
        s2.endPhase("A", category: .textEncoding)

        let data = ChromeTraceExporter.exportComparison(sessions: [
            (label: "Run1", session: s1), (label: "Run2", session: s2)
        ])
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let pids = Set((json?["traceEvents"] as? [[String: Any]])?.compactMap { $0["pid"] as? Int } ?? [])
        #expect(pids.contains(1) && pids.contains(2))
    }
}

// MARK: - BenchmarkAggregator

@Suite("BenchmarkAggregator")
struct BenchmarkAggregatorTests {
    @Test func testEmpty() {
        let result = BenchmarkAggregator.aggregate(sessions: [], warmupCount: 0)
        #expect(result.measuredRuns == 0)
    }

    @Test func testSingleSession() {
        let session = ProfilingSession()
        session.metadata = ["model": "test"]
        session.beginPhase("Phase", category: .denoisingLoop)
        Thread.sleep(forTimeInterval: 0.01)
        session.endPhase("Phase", category: .denoisingLoop)

        let result = BenchmarkAggregator.aggregate(sessions: [session], warmupCount: 0)
        #expect(result.measuredRuns == 1)
        #expect(!result.phaseStats.isEmpty)
    }
}

// MARK: - MLXProfiler Singleton

@Suite("MLXProfiler", .serialized)
struct MLXProfilerSingletonTests {
    @Test func testEnableDisable() {
        let p = MLXProfiler.shared
        p.disable(); p.reset()
        #expect(p.isEnabled == false)
        p.enable()
        #expect(p.isEnabled == true)
        p.disable()
    }

    @Test func testStartEnd() {
        let p = MLXProfiler.shared
        p.enable()
        p.start("Test")
        Thread.sleep(forTimeInterval: 0.01)
        p.end("Test")
        let timings = p.getTimings()
        #expect(timings.count >= 1)
        #expect(timings.last?.name == "Test")
        p.disable(); p.reset()
    }

    @Test func testSessionBridge() {
        let p = MLXProfiler.shared
        let session = ProfilingSession()
        p.enable()
        p.activeSession = session
        p.start("Bridged")
        Thread.sleep(forTimeInterval: 0.01)
        p.end("Bridged")
        let events = session.getEvents()
        #expect(events.count == 2)
        p.activeSession = nil; p.disable(); p.reset()
    }
}

// MARK: - TimingEntry

@Suite("TimingEntry")
struct TimingEntryTests {
    @Test func testFormat() {
        let entry = TimingEntry(name: "fast", duration: 0.05, startTime: Date(), endTime: Date())
        #expect(entry.durationMs == 50.0)
        #expect(entry.durationFormatted.contains("ms"))
    }
}
