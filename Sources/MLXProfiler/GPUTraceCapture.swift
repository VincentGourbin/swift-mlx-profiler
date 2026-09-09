// GPUTraceCapture.swift - Bounded .gputrace captures around a named phase
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Why a Metal capture could not be taken.
public enum GPUCaptureError: Error, CustomStringConvertible {
    case captureNotEnabled
    case destinationExists(URL)
    case notProduced(URL)
    case alreadyCapturing(phase: String)
    case notCapturing

    public var description: String {
        switch self {
        case .captureNotEnabled:
            return """
            Metal capture is not enabled. Set MTL_CAPTURE_ENABLED=1 in the \
            environment of the process being profiled.
            """
        case .destinationExists(let url):
            return "\(url.path) already exists; Metal refuses to overwrite a capture."
        case .notProduced(let url):
            return """
            No capture appeared at \(url.path). MLX has to be built with \
            MLX_METAL_DEBUG for GPU.startCapture to do anything: add \
            .define("MLX_METAL_DEBUG") to the Cmlx target's cxxSettings in the \
            mlx-swift Package.swift and rebuild.
            """
        case .alreadyCapturing(let phase):
            return "A capture is already running for phase \"\(phase)\"."
        case .notCapturing:
            return "No capture is running."
        }
    }
}

/// Capture and Metal System Trace merging are orchestration, not hot paths, and
/// are **not** thread-safe against each other: drive them from a single thread.
/// Phase and step recording remain safe to call from anywhere.
extension ProfilingSession {

    /// Captures the Metal work done by `body` into a `.gputrace` file.
    ///
    /// The file opens in Xcode, where the individual kernels *are* named — which
    /// is the one thing neither the sampler nor a command-line Metal System Trace
    /// can tell you. Scope it tightly: a capture of a whole run is unusable, a
    /// capture of one layer is exactly the question you wanted answered.
    ///
    /// Two build-time preconditions apply, both checked here rather than left to
    /// fail silently: `MTL_CAPTURE_ENABLED=1` must be in the environment, and MLX
    /// must have been compiled with `MLX_METAL_DEBUG`.
    @discardableResult
    public func captureGPUTrace<T>(
        phase: String, to url: URL? = nil, _ body: () throws -> T
    ) throws -> T {
        try beginGPUCapture(phase: phase, to: url)
        var stopped = false
        // If `body` throws, the capture still has to be closed or every later one
        // fails with "already capturing".
        defer { if !stopped { try? endGPUCapture() } }
        let result = try body()
        stopped = true
        _ = try endGPUCapture()
        return result
    }

    /// Starts a capture bounded to `phase`. Pair with ``endGPUCapture()``.
    @discardableResult
    public func beginGPUCapture(phase: String, to url: URL? = nil) throws -> URL {
        guard RunEnvironment.isMetalCaptureEnabled else { throw GPUCaptureError.captureNotEnabled }
        if let active = activeCapture { throw GPUCaptureError.alreadyCapturing(phase: active.phase) }

        let destination = url ?? defaultCaptureURL(for: phase)
        if FileManager.default.fileExists(atPath: destination.path) {
            throw GPUCaptureError.destinationExists(destination)
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        let timestamp = currentTimestampUsPublic()
        GPU.startCapture(url: destination)
        activeCapture = (phase: phase, url: destination, startUs: timestamp)
        recordInstant("GPU capture start: \(phase)", category: .custom, timestampUs: timestamp)
        return destination
    }

    /// Stops the running capture and returns the file it produced.
    @discardableResult
    public func endGPUCapture() throws -> URL {
        guard let active = activeCapture else { throw GPUCaptureError.notCapturing }
        activeCapture = nil

        GPU.stopCapture(url: active.url)
        recordInstant("GPU capture end: \(active.phase)", category: .custom,
                      timestampUs: currentTimestampUsPublic())

        guard FileManager.default.fileExists(atPath: active.url.path) else {
            throw GPUCaptureError.notProduced(active.url)
        }
        gpuCaptures.append((phase: active.phase, url: active.url))
        return active.url
    }

    /// Captures written during this session.
    public var gpuTraceCaptures: [(phase: String, url: URL)] { gpuCaptures }

    private func defaultCaptureURL(for phase: String) -> URL {
        let directory = config.outputDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let safePhase = phase.unicodeScalars
            .map { allowed.contains($0) ? Character($0) : "-" }
            .reduce(into: "") { $0.append($1) }
        let shortId = sessionId.prefix(8)
        return directory.appendingPathComponent("\(shortId)-\(safePhase).gputrace")
    }
}
