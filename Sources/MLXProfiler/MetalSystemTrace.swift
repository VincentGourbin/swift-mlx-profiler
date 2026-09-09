// MetalSystemTrace.swift - Drive xctrace and fold its GPU track into the session
// Copyright 2026 Vincent Gourbin
//
// Nothing the profiler measures says how many Metal kernels a layer dispatches,
// or where the GPU is idle inside a phase that looks busy. That view only comes
// from Instruments. This file automates what was previously done by hand:
// `xctrace record --attach` on the running process, `xctrace export` of the GPU
// tables, and a merge back into the session's Chrome Trace so the GPU track lines
// up with the phases instead of living in a separate window.
//
// Known limitation, and it is the tool's, not this code's: `xctrace record
// --template` cannot enable Shader Timeline from the command line, and without it
// `metal-gpu-intervals` names encoders ("Command Buffer 12:Compute Command 0"),
// not MLX kernels. You get the GPU's busy/idle structure and command-buffer
// counts, which is what the "is the GPU actually working" question needs; you do
// not get a per-kernel-family breakdown. For that, take a `.gputrace` with
// ``ProfilingSession/captureGPUTrace(phase:to:_:)`` and open it in Xcode, or
// sample the process and aggregate by primitive.

import Foundation

/// One GPU interval lifted from a Metal System Trace, in session time.
public struct GPUKernelInterval: Sendable {
    /// Session-relative start, after alignment.
    public let startUs: UInt64
    public let durationUs: UInt64
    /// The encoder label, e.g. `Command Buffer 12:Compute Command 0`.
    public let label: String
    /// `Compute`, `Vertex`, `Fragment`, `Blit`, …
    public let channel: String
    public let commandBufferId: String?

    public var endUs: UInt64 { startUs &+ durationUs }

    /// The label with counters stripped, so repeated encoders group together.
    public var family: String { MetalSystemTrace.family(of: label) }
}

/// What a merge produced.
public struct GPUKernelSummary: Sendable {
    public let intervalCount: Int
    public let commandBufferCount: Int
    /// Window the trace covers, in session time.
    public let windowStartUs: UInt64
    public let windowEndUs: UInt64
    /// Union of all GPU intervals — overlapping encoders counted once.
    public let busyUs: UInt64
    /// Plain sum of every interval's duration.
    ///
    /// Larger than ``busyUs`` whenever encoders overlap or nest, which they
    /// routinely do — Instruments reports a command buffer and the encoders
    /// inside it as separate rows. Summing is the easy mistake and it inflates
    /// GPU busy-ness: on the Flash-Next bench it read 85.4 % against a true
    /// 75.7 %. Reported here only so the gap between the two is visible.
    public let sumOfDurationsUs: UInt64
    /// How the session clock was matched to the trace clock.
    public let alignment: MetalSystemTrace.Alignment
    public let byChannel: [(channel: String, count: Int, busyUs: UInt64)]
    public let byFamily: [(family: String, count: Int, totalUs: UInt64)]

    public var windowUs: UInt64 { windowEndUs > windowStartUs ? windowEndUs - windowStartUs : 0 }
    public var busyPercent: Double {
        windowUs > 0 ? Double(busyUs) / Double(windowUs) * 100 : 0
    }
    /// What summing durations instead of unioning them would have claimed.
    public var summedPercent: Double {
        windowUs > 0 ? Double(sumOfDurationsUs) / Double(windowUs) * 100 : 0
    }
}

public enum MetalSystemTraceError: Error, CustomStringConvertible {
    case xctraceUnavailable
    case recordingFailed(status: Int32, output: String)
    case exportFailed(schema: String, status: Int32, output: String)
    case destinationExists(URL)
    case noGPUIntervals
    case cannotAlign

    public var description: String {
        switch self {
        case .xctraceUnavailable:
            return "xcrun xctrace not found — Xcode (not just the Command Line Tools) is required."
        case .recordingFailed(let status, let output):
            return "xctrace record failed (exit \(status)): \(output)"
        case .exportFailed(let schema, let status, let output):
            return "xctrace export of \(schema) failed (exit \(status)): \(output)"
        case .destinationExists(let url):
            return "\(url.path) already exists; xctrace will not overwrite it."
        case .noGPUIntervals:
            return """
            The trace has no metal-gpu-intervals rows for this process. Either no \
            Metal work ran inside the recording window, or the recording did not \
            attach in time — call waitUntilRecording() before starting the work.
            """
        case .cannotAlign:
            return """
            Could not align the trace clock to the session clock: no matching \
            profiler signposts in the trace, and no usable start date in its \
            table of contents.
            """
        }
    }
}

public enum MetalSystemTrace {

    /// How the trace clock was tied to the session clock.
    public enum Alignment: Sendable {
        /// Matched on the profiler's own phase signposts — exact.
        case signposts(matched: Int)
        /// Derived from the trace's start date. Good to a few milliseconds.
        case traceStartDate
    }

    static let gpuIntervalsSchema = "metal-gpu-intervals"
    static let commandBufferSchema = "metal-application-command-buffer-submissions"
    static let signpostSchema = "os-signpost"

    // MARK: - Recording

    /// A running `xctrace record`, attached to a live process.
    ///
    /// Recording has to happen *while* the work runs, so this starts xctrace and
    /// returns; the caller runs its pipeline and then calls ``stop()``.
    public final class Recorder {
        public let output: URL
        private let process: Process
        private let logURL: URL
        private var stopped = false
        private let lock = NSLock()

        init(output: URL, process: Process, logURL: URL) {
            self.output = output
            self.process = process
            self.logURL = logURL
        }

        /// Whatever xctrace has printed so far.
        public func log() -> String {
            (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
        }

        /// Blocks until xctrace says it is recording.
        ///
        /// Attaching takes a second or two. Work started before that is simply
        /// missing from the trace, which then looks like an idle GPU — so this is
        /// worth waiting for rather than guessing at with a sleep.
        @discardableResult
        public func waitUntilRecording(timeout: TimeInterval = 30) -> Bool {
            let deadline = Date().addingTimeInterval(timeout)
            while Date() < deadline {
                let text = log().lowercased()
                if text.contains("starting recording") || text.contains("ctrl-c to stop") {
                    // The message lands slightly before the first samples do.
                    Thread.sleep(forTimeInterval: 0.5)
                    return true
                }
                if !process.isRunning { return false }
                usleep(100_000)
            }
            return false
        }

        /// Stops the recording and returns the finished `.trace` bundle.
        @discardableResult
        public func stop(timeout: TimeInterval = 120) throws -> URL {
            lock.lock()
            defer { lock.unlock() }
            guard !stopped else { return output }
            stopped = true

            // SIGINT is how xctrace is meant to be stopped; it finalizes the
            // bundle on the way out. terminate() would leave it unreadable.
            if process.isRunning { kill(process.processIdentifier, SIGINT) }

            let deadline = Date().addingTimeInterval(timeout)
            while process.isRunning, Date() < deadline { usleep(50_000) }
            if process.isRunning {
                process.terminate()
                throw MetalSystemTraceError.recordingFailed(status: -1, output: log())
            }
            guard FileManager.default.fileExists(atPath: output.path) else {
                throw MetalSystemTraceError.recordingFailed(
                    status: process.terminationStatus, output: log())
            }
            return output
        }
    }

    /// Starts recording `pid` into `output`.
    ///
    /// `timeLimit` is a backstop so a crashed caller cannot leave xctrace running
    /// forever; normal termination is ``Recorder/stop()``.
    public static func startRecording(
        attachingTo pid: Int32 = ProcessInfo.processInfo.processIdentifier,
        output: URL,
        template: String = "Metal System Trace",
        timeLimit: TimeInterval = 600
    ) throws -> Recorder {
        guard FileManager.default.fileExists(atPath: "/usr/bin/xcrun") else {
            throw MetalSystemTraceError.xctraceUnavailable
        }
        if FileManager.default.fileExists(atPath: output.path) {
            throw MetalSystemTraceError.destinationExists(output)
        }
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(), withIntermediateDirectories: true)

        let logURL = output.deletingPathExtension().appendingPathExtension("xctrace.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        guard let logHandle = try? FileHandle(forWritingTo: logURL) else {
            throw MetalSystemTraceError.xctraceUnavailable
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = [
            "xctrace", "record",
            "--template", template,
            "--attach", String(pid),
            "--output", output.path,
            "--time-limit", "\(Int(timeLimit))s",
        ]
        // Straight to a file: an export can run to hundreds of megabytes and a
        // Pipe nobody is draining would deadlock the child.
        process.standardOutput = logHandle
        process.standardError = logHandle

        do { try process.run() } catch {
            throw MetalSystemTraceError.recordingFailed(status: -1, output: "\(error)")
        }
        return Recorder(output: output, process: process, logURL: logURL)
    }

    // MARK: - Export

    /// Runs `xctrace export` for one schema and parses the result.
    public static func exportTables(from trace: URL, schema: String) throws -> [XCTraceTable] {
        let xpath = "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"\(schema)\"]"
        let data = try runExport(trace: trace, arguments: ["--xpath", xpath], schema: schema)
        return try XCTraceExport.parse(data: data)
    }

    /// GPU intervals for `pid`, in trace time (nanoseconds from trace start).
    static func rawGPUIntervals(from trace: URL, pid: Int32) throws -> [(startNs: Int64, interval: GPUKernelInterval)] {
        let tables = try exportTables(from: trace, schema: gpuIntervalsSchema)
        var results: [(startNs: Int64, interval: GPUKernelInterval)] = []

        for table in tables where table.schema == gpuIntervalsSchema {
            for row in table.rows {
                guard let rowPid = table.value("process", in: row)?
                        .firstDescendant(named: "pid")?.integerValue,
                      Int32(rowPid) == pid,
                      let startNs = table.value("start", in: row)?.integerValue,
                      let durationNs = table.value("duration", in: row)?.integerValue
                else { continue }

                let label = table.value("event-label", in: row)?.fmt ?? ""
                let channel = table.value("channel-name", in: row)?.fmt
                    ?? table.value("channel-name", in: row)?.text ?? "unknown"
                let commandBuffer = table.value("cmdbuffer-id", in: row)?.fmt

                results.append((startNs, GPUKernelInterval(
                    startUs: 0,  // filled in once the clocks are aligned
                    durationUs: UInt64(max(0, durationNs)) / 1000,
                    label: label,
                    channel: channel,
                    commandBufferId: commandBuffer
                )))
            }
        }
        return results
    }

    /// Count of command-buffer submissions by `pid`.
    static func commandBufferCount(from trace: URL, pid: Int32) -> Int {
        guard let tables = try? exportTables(from: trace, schema: commandBufferSchema) else { return 0 }
        var count = 0
        for table in tables where table.schema == commandBufferSchema {
            for row in table.rows {
                guard let rowPid = table.value("process", in: row)?
                        .firstDescendant(named: "pid")?.integerValue else { continue }
                if Int32(rowPid) == pid { count += 1 }
            }
        }
        return count
    }

    /// Absolute start date of the recording, from the table of contents.
    static func traceStartDate(from trace: URL) -> Date? {
        guard let data = try? runExport(trace: trace, arguments: ["--toc"], schema: "toc"),
              let xml = String(data: data, encoding: .utf8),
              let open = xml.range(of: "<start-date>"),
              let close = xml.range(of: "</start-date>", range: open.upperBound..<xml.endIndex)
        else { return nil }

        let text = String(xml[open.upperBound..<close.lowerBound])
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: text) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: text)
    }

    private static func runExport(trace: URL, arguments: [String], schema: String) throws -> Data {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("xctrace-export-\(UUID().uuidString).xml")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: temporary) }

        guard let handle = try? FileHandle(forWritingTo: temporary) else {
            throw MetalSystemTraceError.exportFailed(schema: schema, status: -1, output: "no temp file")
        }
        let errorPipe = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["xctrace", "export", "--input", trace.path] + arguments
        process.standardOutput = handle
        process.standardError = errorPipe

        do { try process.run() } catch {
            throw MetalSystemTraceError.exportFailed(schema: schema, status: -1, output: "\(error)")
        }
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        try? handle.close()

        guard process.terminationStatus == 0 else {
            throw MetalSystemTraceError.exportFailed(
                schema: schema, status: process.terminationStatus,
                output: String(data: errorData, encoding: .utf8) ?? "")
        }
        return (try? Data(contentsOf: temporary)) ?? Data()
    }

    // MARK: - Helpers

    /// Strips counters and pointers from an encoder label so repeats group.
    ///
    /// `Command Buffer 12:Compute Command 0` and `Command Buffer 13:Compute
    /// Command 1` are the same kind of work and belong on the same line.
    static func family(of label: String) -> String {
        let tail = label.split(separator: ":").last.map(String.init) ?? label
        let withoutParenthetical = tail.split(separator: "(").first.map(String.init) ?? tail
        let tokens = withoutParenthetical
            .split(separator: " ")
            .filter { token in
                !token.allSatisfy(\.isNumber) && !token.hasPrefix("0x")
            }
        let cleaned = tokens.joined(separator: " ").trimmingCharacters(in: .whitespaces)
        return cleaned.isEmpty ? "unnamed" : cleaned
    }

    /// Total time covered by the union of `intervals` — overlaps counted once.
    ///
    /// Summing durations would double-count concurrent encoders and can exceed
    /// the window, which is how a GPU ends up "137 % busy".
    static func unionDuration(of intervals: [GPUKernelInterval]) -> UInt64 {
        guard !intervals.isEmpty else { return 0 }
        let sorted = intervals.sorted { $0.startUs < $1.startUs }
        var total: UInt64 = 0
        var currentStart = sorted[0].startUs
        var currentEnd = sorted[0].endUs

        for interval in sorted.dropFirst() {
            if interval.startUs > currentEnd {
                total &+= currentEnd - currentStart
                currentStart = interval.startUs
                currentEnd = interval.endUs
            } else {
                currentEnd = max(currentEnd, interval.endUs)
            }
        }
        return total &+ (currentEnd - currentStart)
    }
}
