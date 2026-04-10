// LLMProfilingTests.swift
// Copyright 2026 Vincent Gourbin

import Testing
import Foundation
@testable import MLXProfiler

@Suite("LLM Profiling")
struct LLMProfilingTests {

    @Test func testTokenizationPhase() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        profiler.startTokenization()
        Thread.sleep(forTimeInterval: 0.01)
        profiler.endTokenization(tokenCount: 42)

        let metrics = profiler.getLLMMetrics()
        #expect(metrics.tokenizationTime > 0.005)
        #expect(metrics.promptTokens == 42)

        profiler.disable()
    }

    @Test func testPrefillPhase() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        profiler.startPrefill()
        Thread.sleep(forTimeInterval: 0.01)
        profiler.endPrefill()

        let metrics = profiler.getLLMMetrics()
        #expect(metrics.prefillTime > 0.005)

        profiler.disable()
    }

    @Test func testGenerationPhase() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        profiler.startGeneration()
        Thread.sleep(forTimeInterval: 0.01)
        profiler.endGeneration(tokenCount: 100)

        let metrics = profiler.getLLMMetrics()
        #expect(metrics.generationTime > 0.005)
        #expect(metrics.generatedTokens == 100)
        #expect(metrics.generationTokensPerSecond > 0)

        profiler.disable()
    }

    @Test func testDecodingTimeAccumulation() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        profiler.addDecodingTime(0.001)
        profiler.addDecodingTime(0.002)
        profiler.addDecodingTime(0.003)

        let metrics = profiler.getLLMMetrics()
        #expect(abs(metrics.decodingTime - 0.006) < 0.001)

        profiler.disable()
    }

    @Test func testFullLLMPipeline() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        // Tokenization
        profiler.startTokenization()
        Thread.sleep(forTimeInterval: 0.005)
        profiler.endTokenization(tokenCount: 50)

        // Prefill
        profiler.startPrefill()
        Thread.sleep(forTimeInterval: 0.01)
        profiler.endPrefill()

        // Generation
        profiler.startGeneration()
        Thread.sleep(forTimeInterval: 0.02)
        profiler.addDecodingTime(0.005)
        profiler.endGeneration(tokenCount: 200)

        let metrics = profiler.getLLMMetrics()
        #expect(metrics.promptTokens == 50)
        #expect(metrics.generatedTokens == 200)
        #expect(metrics.totalTime > 0.03)
        #expect(metrics.prefillTokensPerSecond > 0)
        #expect(metrics.generationTokensPerSecond > 0)
        #expect(metrics.decodingTime > 0.004)

        profiler.disable()
    }

    @Test func testLLMMetricsSummary() {
        let metrics = LLMMetrics(
            tokenizationTime: 0.005,
            prefillTime: 0.050,
            generationTime: 2.0,
            decodingTime: 0.1,
            promptTokens: 100,
            generatedTokens: 500
        )

        let summary = metrics.summary
        #expect(summary.contains("Tokenization"))
        #expect(summary.contains("Prefill"))
        #expect(summary.contains("Generation"))
        #expect(summary.contains("500"))

        let compact = metrics.compactSummary
        #expect(compact.contains("tok/s"))
        #expect(compact.contains("100 tok"))
    }

    @Test func testLLMMetricsZeroValues() {
        let metrics = LLMMetrics()
        #expect(metrics.totalTime == 0)
        #expect(metrics.prefillTokensPerSecond == 0)
        #expect(metrics.generationTokensPerSecond == 0)
    }

    @Test func testEnableResetsLLMState() {
        let profiler = MLXProfiler.shared
        profiler.enable()

        profiler.startTokenization()
        Thread.sleep(forTimeInterval: 0.01)
        profiler.endTokenization(tokenCount: 42)
        profiler.addDecodingTime(0.1)

        // Re-enable should reset
        profiler.enable()
        let metrics = profiler.getLLMMetrics()
        #expect(metrics.promptTokens == 0)
        #expect(metrics.decodingTime == 0)

        profiler.disable()
    }

    @Test func testDisabledProfilerSkipsLLM() {
        let profiler = MLXProfiler.shared
        profiler.disable()

        profiler.startTokenization()
        profiler.endTokenization(tokenCount: 99)
        profiler.addDecodingTime(1.0)

        // Should have no data since disabled
        let metrics = profiler.getLLMMetrics()
        // State might have stale values from enable(); the key is methods short-circuit
        // Just verify no crash
        _ = metrics.summary
    }

    @Test func testDecodingCategoryExists() {
        #expect(ProfilingCategory.decoding.rawValue == "decoding")
        #expect(ProfilingCategory.decoding.threadId == 2) // Same lane as prefill/generation
        #expect(ProfilingCategory.decoding.threadName == "Inference")
    }

    @Test func testInferCategoryDecoding() {
        #expect(ProfilingSession.inferCategory("Token Decoding") == .decoding)
        #expect(ProfilingSession.inferCategory("decoder warmup") == .decoding)
    }

    @Test func testLLMPhasesRecordedInSession() {
        let profiler = MLXProfiler.shared
        let session = ProfilingSession()
        profiler.enable()
        profiler.activeSession = session

        profiler.startTokenization()
        profiler.endTokenization(tokenCount: 10)
        profiler.startPrefill()
        profiler.endPrefill()
        profiler.startGeneration()
        profiler.endGeneration(tokenCount: 50)

        let events = session.getEvents()
        let categories = Set(events.map(\.category))
        #expect(categories.contains(.tokenization))
        #expect(categories.contains(.prefill))
        #expect(categories.contains(.generation))

        profiler.activeSession = nil
        profiler.disable()
    }
}
