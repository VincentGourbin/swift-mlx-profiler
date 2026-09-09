// ChromeTraceExporter.swift - Export profiling data to Chrome Trace JSON
// Copyright 2026 Vincent Gourbin
//
// View in Perfetto UI (https://ui.perfetto.dev/) or chrome://tracing

import Foundation

/// Exports ProfilingSession data to Chrome Trace JSON format
public struct ChromeTraceExporter {

    public static func export(session: ProfilingSession) -> Data {
        var traceEvents: [[String: Any]] = []
        let pid = 1

        let processName = session.metadata["model"] ?? "MLX Pipeline"
        traceEvents.append(metadataEvent(name: "process_name", pid: pid, tid: 0, args: ["name": processName]))

        for (tid, name) in [(1, "Text Encoding"), (2, "Transformer"), (3, "Upscaler"),
                            (4, "VAE"), (5, "Audio"), (6, "Post-processing"), (7, "Memory"),
                            (8, "eval() Syncs"), (10, "System Memory")] {
            traceEvents.append(metadataEvent(name: "thread_name", pid: pid, tid: tid, args: ["name": name]))
        }

        for event in session.getEvents() {
            var traceEvent: [String: Any] = [
                "name": event.name, "cat": event.category.rawValue,
                "ph": event.phase.rawValue, "ts": Int(event.timestampUs),
                "pid": pid, "tid": event.threadId,
            ]
            if let dur = event.durationUs, event.phase == .complete { traceEvent["dur"] = Int(dur) }

            var args: [String: Any] = [:]
            if let v = event.mlxActiveBytes { args["mlx_active_mb"] = String(format: "%.1f", Double(v) / 1_048_576) }
            if let v = event.mlxCacheBytes { args["mlx_cache_mb"] = String(format: "%.1f", Double(v) / 1_048_576) }
            if let v = event.mlxPeakBytes { args["mlx_peak_mb"] = String(format: "%.1f", Double(v) / 1_048_576) }
            if let v = event.processFootprintBytes { args["process_mb"] = String(format: "%.1f", Double(v) / 1_048_576) }
            if let v = event.stepIndex { args["step"] = v }
            if let v = event.totalSteps { args["total_steps"] = v }
            if !args.isEmpty { traceEvent["args"] = args }
            if event.phase == .instant { traceEvent["s"] = "g" }
            traceEvents.append(traceEvent)
        }

        let timeline = session.getMemoryTimeline()
        for entry in timeline {
            traceEvents.append([
                "name": "Memory" as Any, "cat": "memory" as Any, "ph": "C" as Any,
                "ts": Int(entry.timestampUs) as Any, "pid": pid as Any, "tid": 7 as Any,
                "args": [
                    "MLX Active (MB)": round(entry.mlxActiveMB * 10) / 10,
                    "MLX Cache (MB)": round(entry.mlxCacheMB * 10) / 10,
                    "Process (MB)": round(entry.processFootprintMB * 10) / 10,
                ] as [String: Any],
            ])

            // System-wide memory on its own lane. `Process (MB)` alone explained
            // neither the low-memory kills nor the CPU-bound decode; these are the
            // counters that did.
            guard let anonymous = entry.systemAnonymousMB else { continue }
            var systemArgs: [String: Any] = ["Anonymous (MB)": round(anonymous * 10) / 10]
            if let v = entry.systemCompressorOccupiedMB { systemArgs["Compressor occupied (MB)"] = round(v * 10) / 10 }
            if let v = entry.systemCompressorStoredMB { systemArgs["Compressor stored (MB)"] = round(v * 10) / 10 }
            if let v = entry.systemWiredMB { systemArgs["Wired (MB)"] = round(v * 10) / 10 }
            if let v = entry.systemFileBackedMB { systemArgs["File-backed (MB)"] = round(v * 10) / 10 }
            if let v = entry.systemSpeculativeMB { systemArgs["Speculative (MB)"] = round(v * 10) / 10 }
            if let v = entry.systemSwapUsedMB { systemArgs["Swap used (MB)"] = round(v * 10) / 10 }
            if let v = entry.systemMemoryStatusLevel { systemArgs["memorystatus_level"] = v }
            traceEvents.append([
                "name": "System Memory" as Any, "cat": "memory" as Any, "ph": "C" as Any,
                "ts": Int(entry.timestampUs) as Any, "pid": pid as Any, "tid": 10 as Any,
                "args": systemArgs as [String: Any],
            ])
        }

        // CPU% and GPU% counters.
        //
        // CPU% is a difference of a cumulative counter, so at the sampler's 16 ms
        // cadence a per-sample delta is mostly quantization noise. Differencing
        // over at least 100 ms gives a curve that means something without losing
        // the shape.
        let cpuWindowUs: UInt64 = 100_000
        var base = 0
        // `1..<count` traps on an empty timeline, which is reachable now that
        // phase boundaries no longer force a snapshot.
        for i in stride(from: 1, to: timeline.count, by: 1) {
            let curr = timeline[i]
            // The furthest-forward entry that is still a full window behind `i`.
            // Monotone in `i`, so the whole pass stays linear.
            while base + 1 < i, curr.timestampUs - timeline[base + 1].timestampUs >= cpuWindowUs {
                base += 1
            }
            let prev = timeline[base]
            let wallDelta = Double(curr.timestampUs - prev.timestampUs) / 1_000_000
            let cpuDelta = curr.cpuTimeSeconds - prev.cpuTimeSeconds
            let cpuPct = wallDelta > 0 ? min((cpuDelta / wallDelta) * 100, 800) : 0
            traceEvents.append([
                "name": "Utilization" as Any, "cat": "utilization" as Any, "ph": "C" as Any,
                "ts": Int(curr.timestampUs) as Any, "pid": pid as Any, "tid": 8 as Any,
                "args": [
                    "GPU (%)": curr.gpuUtilization,
                    "CPU (%, all threads)": round(cpuPct * 10) / 10,
                ] as [String: Any],
            ])
        }

        traceEvents.append(contentsOf: gpuKernelEvents(session: session, pid: pid))
        traceEvents.append(contentsOf: stallEvents(session: session, pid: pid))

        // Counter values (training loss curves, custom metrics)
        for counter in session.getCounterValues() {
            var args: [String: Any] = [:]
            for (key, value) in counter.values {
                args[key] = round(value * 10000) / 10000
            }
            traceEvents.append([
                "name": counter.name as Any, "cat": "training" as Any, "ph": "C" as Any,
                "ts": Int(counter.timestampUs) as Any, "pid": pid as Any, "tid": 9 as Any,
                "args": args as [String: Any],
            ])
        }
        if !session.getCounterValues().isEmpty {
            traceEvents.append(metadataEvent(name: "thread_name", pid: pid, tid: 9, args: ["name": "Training Metrics"]))
        }

        // Session metadata
        traceEvents.append([
            "name": "Session Info" as Any, "cat": "metadata" as Any, "ph": "i" as Any,
            "ts": 0 as Any, "pid": pid as Any, "tid": 0 as Any, "s": "g" as Any,
            "args": session.metadata.merging(sessionInfo(session)) { $1 } as [String: Any],
        ])

        let trace: [String: Any] = ["traceEvents": traceEvents]
        return (try? JSONSerialization.data(withJSONObject: trace, options: [.prettyPrinted, .sortedKeys]))
            ?? "{ \"traceEvents\": [] }".data(using: .utf8)!
    }

    public static func exportComparison(sessions: [(label: String, session: ProfilingSession)]) -> Data {
        var traceEvents: [[String: Any]] = []
        for (index, entry) in sessions.enumerated() {
            let pid = index + 1
            traceEvents.append(metadataEvent(name: "process_name", pid: pid, tid: 0, args: ["name": entry.label]))
            for (tid, name) in [(1, "Text Encoding"), (2, "Transformer"), (3, "Upscaler"), (4, "VAE"), (5, "Audio"), (7, "Memory")] {
                traceEvents.append(metadataEvent(name: "thread_name", pid: pid, tid: tid, args: ["name": name]))
            }
            for event in entry.session.getEvents() {
                var ev: [String: Any] = ["name": event.name, "cat": event.category.rawValue, "ph": event.phase.rawValue,
                                         "ts": Int(event.timestampUs), "pid": pid, "tid": event.threadId]
                if let dur = event.durationUs, event.phase == .complete { ev["dur"] = Int(dur) }
                if event.phase == .instant { ev["s"] = "g" }
                traceEvents.append(ev)
            }
            for m in entry.session.getMemoryTimeline() {
                traceEvents.append(["name": "Memory" as Any, "cat": "memory" as Any, "ph": "C" as Any,
                    "ts": Int(m.timestampUs) as Any, "pid": pid as Any, "tid": 7 as Any,
                    "args": ["MLX Active (MB)": round(m.mlxActiveMB * 10) / 10, "Process (MB)": round(m.processFootprintMB * 10) / 10] as [String: Any]])
            }
        }
        let trace: [String: Any] = ["traceEvents": traceEvents]
        return (try? JSONSerialization.data(withJSONObject: trace, options: [.prettyPrinted, .sortedKeys]))
            ?? "{ \"traceEvents\": [] }".data(using: .utf8)!
    }

    private static func metadataEvent(name: String, pid: Int, tid: Int, args: [String: Any]) -> [String: Any] {
        ["name": name, "ph": "M", "pid": pid, "tid": tid, "args": args]
    }

    /// Everything needed to judge whether these numbers are worth trusting.
    private static func sessionInfo(_ session: ProfilingSession) -> [String: String] {
        var info: [String: String] = [
            "device": session.deviceArchitecture,
            "ram_gb": String(session.systemRAMGB),
            "session_id": session.sessionId,
            "build_configuration": RunEnvironment.buildConfiguration,
        ]
        if RunEnvironment.isDebugBuild {
            info["WARNING"] = "Debug build - MLX C++ at -O0, timings are not a benchmark"
        }
        if let powerSource = session.powerSource { info["power_source"] = powerSource }
        for (key, value) in session.powerManagementSettings { info["pmset_\(key)"] = value }
        if let backend = session.activeGPUBackend {
            info["gpu_backend"] = backend.rawValue
            info["sampling_interval_ms"] = String(session.config.samplingIntervalMs)
        }
        info["idle_sleep_prevented"] = String(session.config.preventIdleSleep)
        return info
    }

    /// GPU intervals merged in from a Metal System Trace, on their own lane.
    private static func gpuKernelEvents(session: ProfilingSession, pid: Int) -> [[String: Any]] {
        let intervals = session.mergedGPUKernelIntervals
        guard !intervals.isEmpty else { return [] }

        var events: [[String: Any]] = [
            metadataEvent(name: "thread_name", pid: pid, tid: 11, args: ["name": "GPU Kernels"])
        ]
        for interval in intervals {
            var args: [String: Any] = ["label": interval.label, "channel": interval.channel]
            if let commandBuffer = interval.commandBufferId { args["command_buffer"] = commandBuffer }
            events.append([
                "name": interval.family as Any, "cat": "gpu_kernel" as Any, "ph": "X" as Any,
                "ts": Int(interval.startUs) as Any, "dur": Int(interval.durationUs) as Any,
                "pid": pid as Any, "tid": 11 as Any, "args": args as [String: Any],
            ])
        }
        return events
    }

    /// Stretches with no sample at all, marked so a frozen run is visible in the
    /// trace instead of looking like a very slow phase.
    private static func stallEvents(session: ProfilingSession, pid: Int) -> [[String: Any]] {
        session.getStalls().map { stall in
            [
                "name": "STALL \(String(format: "%.1f", stall.durationSeconds))s (no samples)" as Any,
                "cat": "stall" as Any, "ph": "X" as Any,
                "ts": Int(stall.startUs) as Any,
                "dur": Int(stall.endUs - stall.startUs) as Any,
                "pid": pid as Any, "tid": 12 as Any,
                "args": ["hint": "system sleep, suspended process, or severe starvation"] as [String: Any],
            ]
        }
    }
}
