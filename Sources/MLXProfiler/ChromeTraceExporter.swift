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
                            (4, "VAE"), (5, "Audio"), (6, "Post-processing"), (7, "Memory"), (8, "eval() Syncs")] {
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
        }

        // CPU% and GPU% counters
        for i in 1..<timeline.count {
            let prev = timeline[i - 1]
            let curr = timeline[i]
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
            "args": session.metadata.merging([
                "device": session.deviceArchitecture,
                "ram_gb": String(session.systemRAMGB),
                "session_id": session.sessionId,
            ]) { $1 } as [String: Any],
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
}
