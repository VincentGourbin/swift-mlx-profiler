// ProfilingEvent.swift - Event types for MLX pipeline profiling
// Copyright 2026 Vincent Gourbin

import Foundation

/// Category of a profiling event (maps to visual lanes in Chrome Trace).
///
/// Covers LLM, image diffusion, and video diffusion pipelines.
/// Use `.custom` for project-specific phases not covered by the built-in categories.
public enum ProfilingCategory: String, Codable, Sendable {
    // MARK: - Shared
    case modelLoad = "model_load"
    case modelUnload = "model_unload"

    // MARK: - Text / Vision encoding
    case textEncoderLoad = "text_encoder_load"
    case textEncoding = "text_encoding"
    case textEncoderUnload = "text_encoder_unload"
    case vlmInterpretation = "vlm_interpretation"
    case tokenization = "tokenization"

    // MARK: - Diffusion
    case transformerLoad = "transformer_load"
    case denoisingLoop = "denoising_loop"
    case denoisingStep = "denoising_step"
    case transformerUnload = "transformer_unload"
    case upscaler = "upscaler"

    // MARK: - VAE
    case vaeLoad = "vae_load"
    case vaeDecode = "vae_decode"

    // MARK: - Video / Audio
    case audioLoad = "audio_load"
    case audioDenoise = "audio_denoise"
    case frameConversion = "frame_conversion"
    case videoWrite = "video_write"

    // MARK: - Speech / TTS
    case melSpectrogram = "mel_spectrogram"
    case audioFeatureExtract = "audio_feature_extract"
    case semanticCodeGen = "semantic_code_gen"
    case flowMatching = "flow_matching"
    case codecDecode = "codec_decode"
    case voiceEmbedding = "voice_embedding"
    case audioWrite = "audio_write"

    // MARK: - LLM
    case prefill = "prefill"
    case generation = "generation"
    case generationStep = "generation_step"
    case decoding = "decoding"
    case kvCache = "kv_cache"
    case visionEncode = "vision_encode"
    case audioEncode = "audio_encode"

    // MARK: - Infrastructure
    case postProcess = "post_process"
    case evalSync = "eval_sync"
    case memoryOp = "memory_op"
    case custom = "custom"

    /// Thread ID for Chrome Trace lane grouping
    public var threadId: Int {
        switch self {
        case .textEncoderLoad, .textEncoding, .textEncoderUnload, .vlmInterpretation, .tokenization: return 1
        case .transformerLoad, .denoisingLoop, .denoisingStep, .transformerUnload: return 2
        case .prefill, .generation, .generationStep, .decoding, .kvCache: return 2
        case .upscaler: return 3
        case .vaeLoad, .vaeDecode: return 4
        case .audioLoad, .audioDenoise, .audioEncode: return 5
        case .melSpectrogram, .audioFeatureExtract, .codecDecode, .voiceEmbedding: return 5
        case .semanticCodeGen, .flowMatching: return 2
        case .frameConversion, .videoWrite, .audioWrite, .postProcess: return 6
        case .visionEncode: return 5
        case .memoryOp: return 7
        case .evalSync: return 8
        case .modelLoad, .modelUnload: return 1
        case .custom: return 9
        }
    }

    /// Human-readable thread name for Chrome Trace
    public var threadName: String {
        switch self {
        case .textEncoderLoad, .textEncoding, .textEncoderUnload, .vlmInterpretation, .tokenization: return "Text Encoding"
        case .transformerLoad, .denoisingLoop, .denoisingStep, .transformerUnload: return "Transformer"
        case .prefill, .generation, .generationStep, .decoding, .kvCache: return "Inference"
        case .upscaler: return "Upscaler"
        case .vaeLoad, .vaeDecode: return "VAE"
        case .audioLoad, .audioDenoise, .audioEncode: return "Audio"
        case .melSpectrogram, .audioFeatureExtract, .codecDecode, .voiceEmbedding: return "Audio"
        case .semanticCodeGen, .flowMatching: return "Inference"
        case .frameConversion, .videoWrite, .audioWrite, .postProcess: return "Post-processing"
        case .visionEncode: return "Vision"
        case .memoryOp: return "Memory"
        case .evalSync: return "eval() Syncs"
        case .modelLoad, .modelUnload: return "Model"
        case .custom: return "Other"
        }
    }

    /// Sort order for deterministic report output
    public var sortOrder: Int {
        switch self {
        case .modelLoad: return 0
        case .textEncoderLoad: return 1
        case .tokenization: return 2
        case .textEncoding: return 3
        case .textEncoderUnload: return 4
        case .vlmInterpretation: return 5
        case .transformerLoad: return 6
        case .prefill: return 7
        case .denoisingLoop: return 8
        case .denoisingStep: return 9
        case .generation: return 8
        case .generationStep: return 9
        case .transformerUnload: return 10
        case .upscaler: return 11
        case .vaeLoad: return 12
        case .vaeDecode: return 13
        case .audioLoad: return 14
        case .audioDenoise: return 15
        case .audioEncode: return 14
        case .melSpectrogram: return 14
        case .audioFeatureExtract: return 14
        case .voiceEmbedding: return 15
        case .semanticCodeGen: return 8
        case .flowMatching: return 9
        case .codecDecode: return 16
        case .visionEncode: return 5
        case .decoding: return 10
        case .kvCache: return 10
        case .frameConversion: return 17
        case .videoWrite: return 18
        case .audioWrite: return 18
        case .postProcess: return 19
        case .evalSync: return 20
        case .memoryOp: return 21
        case .modelUnload: return 22
        case .custom: return 23
        }
    }
}

/// Phase type matching Chrome Trace Event Format
public enum ProfilingPhase: String, Codable, Sendable {
    case begin = "B"
    case end = "E"
    case complete = "X"
    case instant = "i"
    case counter = "C"
    case metadata = "M"
}

/// A single profiling event with timing and optional memory snapshot
public struct ProfilingEvent: Sendable, Codable {
    public let name: String
    public let category: ProfilingCategory
    public let phase: ProfilingPhase
    public let timestampUs: UInt64
    public let durationUs: UInt64?
    public let threadId: Int
    public let mlxActiveBytes: Int?
    public let mlxCacheBytes: Int?
    public let mlxPeakBytes: Int?
    public let processFootprintBytes: Int64?
    public let stepIndex: Int?
    public let totalSteps: Int?

    public init(
        name: String, category: ProfilingCategory, phase: ProfilingPhase,
        timestampUs: UInt64, durationUs: UInt64? = nil, threadId: Int? = nil,
        mlxActiveBytes: Int? = nil, mlxCacheBytes: Int? = nil,
        mlxPeakBytes: Int? = nil, processFootprintBytes: Int64? = nil,
        stepIndex: Int? = nil, totalSteps: Int? = nil
    ) {
        self.name = name
        self.category = category
        self.phase = phase
        self.timestampUs = timestampUs
        self.durationUs = durationUs
        self.threadId = threadId ?? category.threadId
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
        self.processFootprintBytes = processFootprintBytes
        self.stepIndex = stepIndex
        self.totalSteps = totalSteps
    }
}

/// Memory and utilization timeline entry for counter events
public struct MemoryTimelineEntry: Sendable, Codable {
    public let timestampUs: UInt64
    public let context: String
    public let mlxActiveMB: Double
    public let mlxCacheMB: Double
    public let mlxPeakMB: Double
    public let processFootprintMB: Double
    public let cpuTimeSeconds: Double
    public let gpuUtilization: Int
}
