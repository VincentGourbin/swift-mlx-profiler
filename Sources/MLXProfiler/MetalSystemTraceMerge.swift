// MetalSystemTraceMerge.swift - Align an Instruments trace to the session clock
// Copyright 2026 Vincent Gourbin

import Foundation

extension ProfilingSession {

    /// Starts a Metal System Trace attached to this process.
    ///
    /// Recording runs alongside the work rather than instead of it:
    ///
    /// ```swift
    /// let recorder = try session.startMetalSystemTrace()
    /// recorder.waitUntilRecording()
    /// // ... run the pipeline ...
    /// let trace = try recorder.stop()
    /// let summary = try session.mergeMetalSystemTrace(trace)
    /// ```
    ///
    /// After the merge the GPU track appears in the exported Chrome Trace on its
    /// own lane, on the same timeline as the phases.
    public func startMetalSystemTrace(
        output: URL? = nil,
        template: String = "Metal System Trace",
        timeLimit: TimeInterval = 600
    ) throws -> MetalSystemTrace.Recorder {
        let directory = config.outputDirectory
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let destination = output
            ?? directory.appendingPathComponent("\(sessionId.prefix(8))-metal-system.trace")
        return try MetalSystemTrace.startRecording(
            output: destination, template: template, timeLimit: timeLimit)
    }

    /// Folds a recorded trace's GPU intervals into this session.
    @discardableResult
    public func mergeMetalSystemTrace(
        _ trace: URL,
        pid: Int32 = ProcessInfo.processInfo.processIdentifier
    ) throws -> GPUKernelSummary {
        let raw = try MetalSystemTrace.rawGPUIntervals(from: trace, pid: pid)
        guard !raw.isEmpty else { throw MetalSystemTraceError.noGPUIntervals }

        let (offsetUs, alignment) = try resolveAlignment(trace: trace, pid: pid)

        let intervals: [GPUKernelInterval] = raw.map { entry in
            let shifted = Int64(entry.startNs / 1000) + offsetUs
            return GPUKernelInterval(
                startUs: UInt64(max(0, shifted)),
                durationUs: entry.interval.durationUs,
                label: entry.interval.label,
                channel: entry.interval.channel,
                commandBufferId: entry.interval.commandBufferId
            )
        }.sorted { $0.startUs < $1.startUs }

        let commandBuffers = MetalSystemTrace.commandBufferCount(from: trace, pid: pid)
        gpuKernelIntervals = intervals

        var byChannel: [String: (count: Int, intervals: [GPUKernelInterval])] = [:]
        var byFamily: [String: (count: Int, totalUs: UInt64)] = [:]
        for interval in intervals {
            byChannel[interval.channel, default: (0, [])].count += 1
            byChannel[interval.channel]!.intervals.append(interval)
            byFamily[interval.family, default: (0, 0)].count += 1
            byFamily[interval.family]!.totalUs &+= interval.durationUs
        }

        let summary = GPUKernelSummary(
            intervalCount: intervals.count,
            commandBufferCount: commandBuffers,
            windowStartUs: intervals.first?.startUs ?? 0,
            windowEndUs: intervals.map(\.endUs).max() ?? 0,
            busyUs: MetalSystemTrace.unionDuration(of: intervals),
            sumOfDurationsUs: intervals.reduce(UInt64(0)) { $0 &+ $1.durationUs },
            alignment: alignment,
            byChannel: byChannel
                .map { ($0.key, $0.value.count, MetalSystemTrace.unionDuration(of: $0.value.intervals)) }
                .sorted { $0.2 > $1.2 },
            byFamily: byFamily
                .map { ($0.key, $0.value.count, $0.value.totalUs) }
                .sorted { $0.2 > $1.2 }
        )
        gpuKernelSummary = summary
        return summary
    }

    /// GPU intervals merged into this session, in session time.
    public var mergedGPUKernelIntervals: [GPUKernelInterval] { gpuKernelIntervals }

    // MARK: - Clock alignment

    /// Session-relative microsecond offset to add to a trace timestamp.
    private func resolveAlignment(
        trace: URL, pid: Int32
    ) throws -> (offsetUs: Int64, alignment: MetalSystemTrace.Alignment) {
        // Preferred: the profiler's own phase signposts appear in the trace, so
        // the same instant is timestamped on both clocks and the offset is exact.
        if let matched = signpostOffset(trace: trace, pid: pid) {
            return (matched.offsetUs, .signposts(matched: matched.matchCount))
        }
        // Otherwise fall back to the recording's wall-clock start date. Good to a
        // few milliseconds, which is fine for phases but not for single kernels.
        if let traceStart = MetalSystemTrace.traceStartDate(from: trace) {
            let offset = traceStart.timeIntervalSince(startTime) * 1_000_000
            return (Int64(offset), .traceStartDate)
        }
        throw MetalSystemTraceError.cannotAlign
    }

    /// Matches phase signposts in the trace against this session's own phase
    /// events and returns the median offset between the two clocks.
    private func signpostOffset(trace: URL, pid: Int32) -> (offsetUs: Int64, matchCount: Int)? {
        guard let tables = try? MetalSystemTrace.exportTables(
            from: trace, schema: MetalSystemTrace.signpostSchema) else { return nil }

        // Our own begin events, grouped by phase name and in order.
        var sessionBegins: [String: [UInt64]] = [:]
        for event in getEvents() where event.phase == .begin {
            sessionBegins[event.name, default: []].append(event.timestampUs)
        }
        guard !sessionBegins.isEmpty else { return nil }
        let knownNames = sessionBegins.keys.sorted { $0.count > $1.count }

        var traceBegins: [String: [UInt64]] = [:]
        for table in tables where table.schema == MetalSystemTrace.signpostSchema {
            for row in table.rows {
                guard let rowPid = table.value("process", in: row)?
                        .firstDescendant(named: "pid")?.integerValue,
                      Int32(rowPid) == pid,
                      table.value("subsystem", in: row)?.text == signpostSubsystem,
                      table.value("name", in: row)?.text == "Phase",
                      (table.value("event-type", in: row)?.fmt ?? "") == "Begin",
                      let timeNs = table.value("time", in: row)?.integerValue
                else { continue }

                let message = table.value("message", in: row)?.fmt
                    ?? table.value("message", in: row)?.text ?? ""
                // Longest name first, so "Layer 1" cannot swallow "Layer 12".
                guard let name = knownNames.first(where: { message.contains($0) }) else { continue }
                traceBegins[name, default: []].append(UInt64(max(0, timeNs)) / 1000)
            }
        }
        guard !traceBegins.isEmpty else { return nil }

        var offsets: [Int64] = []
        for (name, var traceTimes) in traceBegins {
            guard var sessionTimes = sessionBegins[name] else { continue }
            traceTimes.sort()
            sessionTimes.sort()
            for (traceTime, sessionTime) in zip(traceTimes, sessionTimes) {
                offsets.append(Int64(sessionTime) - Int64(traceTime))
            }
        }
        guard !offsets.isEmpty else { return nil }
        offsets.sort()
        return (offsets[offsets.count / 2], offsets.count)
    }
}

// MARK: - Report section

extension ProfilingSession {

    func buildGPUKernelSection() -> String {
        var report = ""

        if !gpuCaptures.isEmpty {
            report += "\n  METAL CAPTURES\n"
            report += "  \(String(repeating: "\u{2500}", count: 66))\n"
            for capture in gpuCaptures {
                report += "  \(capture.phase): \(capture.url.lastPathComponent)\n"
            }
            report += "  Open in Xcode for per-kernel detail.\n"
        }

        guard let summary = gpuKernelSummary, summary.intervalCount > 0 else { return report }

        report += "\n  GPU KERNELS (Metal System Trace)\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"

        switch summary.alignment {
        case .signposts(let matched):
            report += "  Aligned on \(matched) phase signpost(s) - exact.\n"
        case .traceStartDate:
            report += "  Aligned on the trace start date - accurate to a few ms;\n"
            report += "  reliable for phases, not for individual kernels.\n"
        }

        let windowSeconds = Double(summary.windowUs) / 1_000_000
        report += "  Intervals: \(summary.intervalCount)"
        report += "   Command buffers: \(summary.commandBufferCount)\n"
        report += "  Window: \(String(format: "%.3f", windowSeconds))s"
        report += "   GPU busy: \(String(format: "%.1f", summary.busyPercent))%"
        report += " (union of intervals, not a sample average)\n"
        if summary.sumOfDurationsUs > summary.busyUs {
            report += "  Durations sum to \(String(format: "%.1f", summary.summedPercent))% of the window:"
            report += " encoders overlap or nest,\n"
            report += "  so adding them up would overstate GPU busy-ness.\n"
        }

        if !summary.byChannel.isEmpty {
            report += "\n  By channel:\n"
            for entry in summary.byChannel.prefix(8) {
                let percent = summary.windowUs > 0
                    ? Double(entry.busyUs) / Double(summary.windowUs) * 100 : 0
                report += "    \(entry.channel.padding(toLength: 24, withPad: " ", startingAt: 0))"
                report += " \(String(format: "%6d", entry.count)) intervals"
                report += "  \(String(format: "%5.1f", percent))% of window\n"
            }
        }

        if !summary.byFamily.isEmpty {
            report += "\n  By encoder label:\n"
            for entry in summary.byFamily.prefix(10) {
                report += "    \(entry.family.padding(toLength: 24, withPad: " ", startingAt: 0))"
                report += " \(String(format: "%6d", entry.count))"
                report += "  \(formatDuration(Double(entry.totalUs) / 1_000_000)) total\n"
            }
            report += "\n  These are Metal encoders, not MLX kernels: `xctrace record\n"
            report += "  --template` cannot turn on Shader Timeline, so individual\n"
            report += "  kernels are unnamed. Use captureGPUTrace(phase:) for those.\n"
        }
        return report
    }
}
