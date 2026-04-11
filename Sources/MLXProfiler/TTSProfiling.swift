// TTSProfiling.swift - TTS convenience methods for MLXProfiler
// Copyright 2026 Vincent Gourbin

import Foundation

// MARK: - TTS Metrics

/// Performance metrics for a TTS synthesis run (semantic gen + flow matching + codec decode).
///
/// Captures timing, throughput, and audio output metrics for the standard TTS phases.
/// Memory snapshots are captured separately via `ProfilingSession`.
public struct TTSMetrics: Sendable {
    public let tokenizationTime: TimeInterval
    public let semanticGenTime: TimeInterval
    public let flowMatchingTime: TimeInterval
    public let codecDecodeTime: TimeInterval
    public let timeToFirstToken: TimeInterval

    public let numFrames: Int
    public let audioDurationSeconds: Double

    public var totalTime: TimeInterval {
        tokenizationTime + semanticGenTime + flowMatchingTime + codecDecodeTime
    }

    public var realtimeFactor: Double {
        totalTime > 0 ? audioDurationSeconds / totalTime : 0
    }

    public var framesPerSecond: Double {
        semanticGenTime > 0 ? Double(numFrames) / semanticGenTime : 0
    }

    public init(
        tokenizationTime: TimeInterval = 0,
        semanticGenTime: TimeInterval = 0,
        flowMatchingTime: TimeInterval = 0,
        codecDecodeTime: TimeInterval = 0,
        timeToFirstToken: TimeInterval = 0,
        numFrames: Int = 0,
        audioDurationSeconds: Double = 0
    ) {
        self.tokenizationTime = tokenizationTime
        self.semanticGenTime = semanticGenTime
        self.flowMatchingTime = flowMatchingTime
        self.codecDecodeTime = codecDecodeTime
        self.timeToFirstToken = timeToFirstToken
        self.numFrames = numFrames
        self.audioDurationSeconds = audioDurationSeconds
    }

    public var summary: String {
        """
        Tokenization: \(String(format: "%.1f", tokenizationTime * 1000)) ms
        Semantic Gen: \(String(format: "%.1f", semanticGenTime * 1000)) ms (\(numFrames) frames, \(String(format: "%.1f", framesPerSecond)) fr/s)
        Flow Matching: \(String(format: "%.1f", flowMatchingTime * 1000)) ms
        Codec Decode: \(String(format: "%.1f", codecDecodeTime * 1000)) ms
        TTFT: \(String(format: "%.1f", timeToFirstToken * 1000)) ms
        Total: \(String(format: "%.1f", totalTime * 1000)) ms
        Audio: \(String(format: "%.2f", audioDurationSeconds))s | RT factor: \(String(format: "%.2f", realtimeFactor))x
        """
    }

    public var compactSummary: String {
        "Semantic: \(String(format: "%.1f", framesPerSecond)) fr/s (\(numFrames) fr) | " +
        "RT: \(String(format: "%.2f", realtimeFactor))x | " +
        "TTFT: \(String(format: "%.0f", timeToFirstToken * 1000))ms"
    }
}

// MARK: - TTS Profiling Extension

/// TTS convenience methods for profiling semantic generation, flow matching, and codec decode phases.
///
/// Usage:
/// ```swift
/// let profiler = MLXProfiler.shared
/// profiler.enable()
///
/// profiler.startTokenization()
/// let tokens = tokenizer.encode(text)
/// profiler.endTokenization(tokenCount: tokens.count)
///
/// profiler.startSemanticGen()
/// let codes = model.generateSemanticCodes(tokens)
/// profiler.endSemanticGen(frameCount: codes.count)
///
/// profiler.startFlowMatching()
/// let acoustic = model.flowMatchingDecode(codes)
/// profiler.endFlowMatching()
///
/// profiler.startCodecDecode()
/// let waveform = codec.decode(acoustic)
/// profiler.endCodecDecode()
///
/// profiler.setAudioDuration(waveform.count / sampleRate)
/// let metrics = profiler.getTTSMetrics()
/// print(metrics.summary)
/// ```
extension MLXProfiler {

    // MARK: - Internal TTS State

    private nonisolated(unsafe) static let ttsLock = NSLock()
    private nonisolated(unsafe) static var _ttsState = TTSProfilingState()

    private struct TTSProfilingState {
        var tokenizationStart: CFAbsoluteTime = 0
        var tokenizationEnd: CFAbsoluteTime = 0
        var semanticGenStart: CFAbsoluteTime = 0
        var semanticGenEnd: CFAbsoluteTime = 0
        var flowMatchingStart: CFAbsoluteTime = 0
        var flowMatchingEnd: CFAbsoluteTime = 0
        var codecDecodeStart: CFAbsoluteTime = 0
        var codecDecodeEnd: CFAbsoluteTime = 0
        var timeToFirstToken: TimeInterval = 0
        var numFrames: Int = 0
        var audioDurationSeconds: Double = 0

        mutating func reset() {
            tokenizationStart = 0; tokenizationEnd = 0
            semanticGenStart = 0; semanticGenEnd = 0
            flowMatchingStart = 0; flowMatchingEnd = 0
            codecDecodeStart = 0; codecDecodeEnd = 0
            timeToFirstToken = 0; numFrames = 0; audioDurationSeconds = 0
        }
    }

    // MARK: - Semantic Code Generation

    public func startSemanticGen() {
        guard isEnabled else { return }
        Self.ttsLock.lock()
        Self._ttsState.semanticGenStart = CFAbsoluteTimeGetCurrent()
        Self.ttsLock.unlock()
        start("Semantic Code Generation")
    }

    public func endSemanticGen(frameCount: Int) {
        guard isEnabled else { return }
        end("Semantic Code Generation")
        Self.ttsLock.lock()
        Self._ttsState.semanticGenEnd = CFAbsoluteTimeGetCurrent()
        Self._ttsState.numFrames = frameCount
        Self.ttsLock.unlock()
    }

    // MARK: - Flow Matching

    public func startFlowMatching() {
        guard isEnabled else { return }
        Self.ttsLock.lock()
        Self._ttsState.flowMatchingStart = CFAbsoluteTimeGetCurrent()
        Self.ttsLock.unlock()
        start("Flow Matching")
    }

    public func endFlowMatching() {
        guard isEnabled else { return }
        end("Flow Matching")
        Self.ttsLock.lock()
        Self._ttsState.flowMatchingEnd = CFAbsoluteTimeGetCurrent()
        Self.ttsLock.unlock()
    }

    // MARK: - Codec Decode

    public func startCodecDecode() {
        guard isEnabled else { return }
        Self.ttsLock.lock()
        Self._ttsState.codecDecodeStart = CFAbsoluteTimeGetCurrent()
        Self.ttsLock.unlock()
        start("Codec Decode")
    }

    public func endCodecDecode() {
        guard isEnabled else { return }
        end("Codec Decode")
        Self.ttsLock.lock()
        Self._ttsState.codecDecodeEnd = CFAbsoluteTimeGetCurrent()
        Self.ttsLock.unlock()
    }

    // MARK: - Metadata

    /// Set the time-to-first-token (TTFT) measured during generation.
    public func setTTFT(_ ttft: TimeInterval) {
        guard isEnabled else { return }
        Self.ttsLock.lock()
        Self._ttsState.timeToFirstToken = ttft
        Self.ttsLock.unlock()
    }

    /// Set the audio output duration in seconds.
    public func setAudioDuration(_ duration: Double) {
        guard isEnabled else { return }
        Self.ttsLock.lock()
        Self._ttsState.audioDurationSeconds = duration
        Self.ttsLock.unlock()
    }

    // MARK: - Metrics

    /// Get TTS synthesis metrics from the current profiling state.
    public func getTTSMetrics() -> TTSMetrics {
        Self.ttsLock.lock()
        let state = Self._ttsState
        Self.ttsLock.unlock()

        // Reuse tokenization time from LLM state if TTS tokenization wasn't tracked separately
        let tokTime: TimeInterval
        if state.tokenizationEnd > state.tokenizationStart {
            tokTime = state.tokenizationEnd - state.tokenizationStart
        } else {
            let llmMetrics = getLLMMetrics()
            tokTime = llmMetrics.tokenizationTime
        }

        return TTSMetrics(
            tokenizationTime: tokTime,
            semanticGenTime: state.semanticGenEnd - state.semanticGenStart,
            flowMatchingTime: state.flowMatchingEnd - state.flowMatchingStart,
            codecDecodeTime: state.codecDecodeEnd - state.codecDecodeStart,
            timeToFirstToken: state.timeToFirstToken,
            numFrames: state.numFrames,
            audioDurationSeconds: state.audioDurationSeconds
        )
    }

    /// Reset TTS profiling state.
    public func resetTTSState() {
        Self.ttsLock.lock()
        Self._ttsState.reset()
        Self.ttsLock.unlock()
    }
}
