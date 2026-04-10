// LLMProfiling.swift - LLM convenience methods for MLXProfiler
// Copyright 2026 Vincent Gourbin

import Foundation

// MARK: - LLM Metrics

/// Performance metrics for a single LLM generation (tokenization + prefill + generation).
///
/// Captures timing, throughput, and token counts for the three standard LLM phases.
/// Memory snapshots are captured separately via `ProfilingSession`.
public struct LLMMetrics: Sendable {
    public let tokenizationTime: TimeInterval
    public let prefillTime: TimeInterval
    public let generationTime: TimeInterval
    public let decodingTime: TimeInterval

    public let promptTokens: Int
    public let generatedTokens: Int

    public var totalTime: TimeInterval {
        tokenizationTime + prefillTime + generationTime
    }

    public var prefillTokensPerSecond: Double {
        prefillTime > 0 ? Double(promptTokens) / prefillTime : 0
    }

    public var generationTokensPerSecond: Double {
        generationTime > 0 ? Double(generatedTokens) / generationTime : 0
    }

    public init(
        tokenizationTime: TimeInterval = 0,
        prefillTime: TimeInterval = 0,
        generationTime: TimeInterval = 0,
        decodingTime: TimeInterval = 0,
        promptTokens: Int = 0,
        generatedTokens: Int = 0
    ) {
        self.tokenizationTime = tokenizationTime
        self.prefillTime = prefillTime
        self.generationTime = generationTime
        self.decodingTime = decodingTime
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
    }

    public var summary: String {
        """
        Tokenization: \(String(format: "%.1f", tokenizationTime * 1000)) ms
        Prefill: \(String(format: "%.1f", prefillTime * 1000)) ms (\(promptTokens) tokens, \(String(format: "%.0f", prefillTokensPerSecond)) tok/s)
        Generation: \(String(format: "%.1f", generationTime * 1000)) ms (\(generatedTokens) tokens, \(String(format: "%.1f", generationTokensPerSecond)) tok/s)
        Decoding: \(String(format: "%.1f", decodingTime * 1000)) ms
        Total: \(String(format: "%.1f", totalTime * 1000)) ms
        """
    }

    public var compactSummary: String {
        "Prefill: \(String(format: "%.0f", prefillTokensPerSecond)) tok/s (\(promptTokens) tok) | " +
        "Gen: \(String(format: "%.1f", generationTokensPerSecond)) tok/s (\(generatedTokens) tok)"
    }
}

// MARK: - LLM Profiling Extension

/// LLM convenience methods for profiling tokenization, prefill, and generation phases.
///
/// Usage:
/// ```swift
/// let profiler = MLXProfiler.shared
/// profiler.enable()
///
/// profiler.startTokenization()
/// let tokens = tokenizer.encode(prompt)
/// profiler.endTokenization(tokenCount: tokens.count)
///
/// profiler.startPrefill()
/// let logits = model.forward(tokens)
/// profiler.endPrefill()
///
/// profiler.startGeneration()
/// // ... generation loop ...
/// profiler.endGeneration(tokenCount: generatedTokens.count)
///
/// let metrics = profiler.getLLMMetrics()
/// print(metrics.summary)
/// ```
extension MLXProfiler {

    // MARK: - Internal LLM State

    /// Thread-safe storage for LLM profiling state.
    /// Uses a dedicated lock separate from the main profiler lock.
    private nonisolated(unsafe) static let llmLock = NSLock()
    private nonisolated(unsafe) static var _llmState = LLMProfilingState()

    private struct LLMProfilingState {
        var tokenizationStart: CFAbsoluteTime = 0
        var tokenizationEnd: CFAbsoluteTime = 0
        var prefillStart: CFAbsoluteTime = 0
        var prefillEnd: CFAbsoluteTime = 0
        var generationStart: CFAbsoluteTime = 0
        var generationEnd: CFAbsoluteTime = 0
        var promptTokenCount: Int = 0
        var generatedTokenCount: Int = 0
        var decodingTime: TimeInterval = 0

        mutating func reset() {
            tokenizationStart = 0; tokenizationEnd = 0
            prefillStart = 0; prefillEnd = 0
            generationStart = 0; generationEnd = 0
            promptTokenCount = 0; generatedTokenCount = 0
            decodingTime = 0
        }
    }

    // MARK: - Tokenization

    public func startTokenization() {
        guard isEnabled else { return }
        Self.llmLock.lock()
        Self._llmState.tokenizationStart = CFAbsoluteTimeGetCurrent()
        Self.llmLock.unlock()
        start("Tokenization")
    }

    public func endTokenization(tokenCount: Int) {
        guard isEnabled else { return }
        end("Tokenization")
        Self.llmLock.lock()
        Self._llmState.tokenizationEnd = CFAbsoluteTimeGetCurrent()
        Self._llmState.promptTokenCount = tokenCount
        Self.llmLock.unlock()
    }

    // MARK: - Prefill

    public func startPrefill() {
        guard isEnabled else { return }
        Self.llmLock.lock()
        Self._llmState.prefillStart = CFAbsoluteTimeGetCurrent()
        Self.llmLock.unlock()
        start("Prefill")
    }

    public func endPrefill() {
        guard isEnabled else { return }
        end("Prefill")
        Self.llmLock.lock()
        Self._llmState.prefillEnd = CFAbsoluteTimeGetCurrent()
        Self.llmLock.unlock()
    }

    // MARK: - Generation

    public func startGeneration() {
        guard isEnabled else { return }
        Self.llmLock.lock()
        Self._llmState.generationStart = CFAbsoluteTimeGetCurrent()
        Self.llmLock.unlock()
        start("Generation")
    }

    public func endGeneration(tokenCount: Int) {
        guard isEnabled else { return }
        end("Generation")
        Self.llmLock.lock()
        Self._llmState.generationEnd = CFAbsoluteTimeGetCurrent()
        Self._llmState.generatedTokenCount = tokenCount
        Self.llmLock.unlock()
    }

    // MARK: - Decoding Time

    /// Accumulate tokenizer decoding time (called incrementally during generation).
    public func addDecodingTime(_ time: TimeInterval) {
        guard isEnabled else { return }
        Self.llmLock.lock()
        Self._llmState.decodingTime += time
        Self.llmLock.unlock()
    }

    // MARK: - Metrics

    /// Get LLM generation metrics from the current profiling state.
    public func getLLMMetrics() -> LLMMetrics {
        Self.llmLock.lock()
        let state = Self._llmState
        Self.llmLock.unlock()

        return LLMMetrics(
            tokenizationTime: state.tokenizationEnd - state.tokenizationStart,
            prefillTime: state.prefillEnd - state.prefillStart,
            generationTime: state.generationEnd - state.generationStart,
            decodingTime: state.decodingTime,
            promptTokens: state.promptTokenCount,
            generatedTokens: state.generatedTokenCount
        )
    }

    /// Reset LLM profiling state (called automatically by `enable()`).
    public func resetLLMState() {
        Self.llmLock.lock()
        Self._llmState.reset()
        Self.llmLock.unlock()
    }
}
