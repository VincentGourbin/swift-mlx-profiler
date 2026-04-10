// BenchmarkRunner.swift - Statistical benchmarking
// Copyright 2026 Vincent Gourbin

import Foundation

/// Result of a benchmark run with statistical analysis
public struct BenchmarkResult: Sendable {
    public struct PhaseStats: Sendable {
        public let name: String
        public let category: ProfilingCategory
        public let meanMs: Double
        public let stdMs: Double
        public let minMs: Double
        public let maxMs: Double
        public let count: Int
    }

    public let phaseStats: [PhaseStats]
    public let stepStats: PhaseStats?
    public let totalStats: PhaseStats
    public let peakMLXActiveMB: Double
    public let peakProcessMB: Double
    public let warmupRuns: Int
    public let measuredRuns: Int
    public let metadata: [String: String]

    public func generateReport() -> String {
        let metaStr = metadata.sorted(by: { $0.key < $1.key }).map { "\($0.key): \($0.value)" }.joined(separator: "  ")

        var report = """

        \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256E}
        \u{2502}  MLX BENCHMARK REPORT                                                  \u{2502}
        \u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}

        """
        if !metaStr.isEmpty { report += "  \(metaStr)\n" }
        report += "  Warm-up: \(warmupRuns)  Measured runs: \(measuredRuns)\n\n"

        report += "  PHASE TIMINGS (mean \u{00B1} std)\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        for phase in phaseStats {
            let name = phase.name.padding(toLength: 25, withPad: " ", startingAt: 0)
            report += "  \(name) \(formatDuration(phase.meanMs / 1000)) \u{00B1} \(formatDuration(phase.stdMs / 1000))"
            report += "  [\(formatDuration(phase.minMs / 1000)) - \(formatDuration(phase.maxMs / 1000))]\n"
        }
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        report += "  \("TOTAL".padding(toLength: 25, withPad: " ", startingAt: 0)) \(formatDuration(totalStats.meanMs / 1000)) \u{00B1} \(formatDuration(totalStats.stdMs / 1000))\n"

        if let step = stepStats {
            report += "\n  STEP\n"
            report += "  \(String(repeating: "\u{2500}", count: 66))\n"
            report += "  Average per step: \(formatDuration(step.meanMs / 1000)) \u{00B1} \(formatDuration(step.stdMs / 1000))\n"
        }

        report += "\n  MEMORY\n"
        report += "  \(String(repeating: "\u{2500}", count: 66))\n"
        report += "  Peak MLX Active: \(String(format: "%.1f", peakMLXActiveMB)) MB\n"
        report += "  Peak Process: \(String(format: "%.1f", peakProcessMB)) MB\n"

        report += "\n\u{2570}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{256F}\n"
        return report
    }
}

/// Aggregates multiple ProfilingSessions into a BenchmarkResult
public struct BenchmarkAggregator {

    public static func aggregate(sessions: [ProfilingSession], warmupCount: Int) -> BenchmarkResult {
        guard let first = sessions.first else {
            return BenchmarkResult(
                phaseStats: [], stepStats: nil,
                totalStats: .init(name: "TOTAL", category: .custom, meanMs: 0, stdMs: 0, minMs: 0, maxMs: 0, count: 0),
                peakMLXActiveMB: 0, peakProcessMB: 0, warmupRuns: warmupCount, measuredRuns: 0, metadata: [:])
        }

        var phaseDurations: [String: (cat: ProfilingCategory, values: [Double])] = [:]
        var allStepDurations: [Double] = []
        var totalDurations: [Double] = []
        var peakMLXActive: Double = 0
        var peakProcess: Double = 0

        for session in sessions {
            var beginTimestamps: [String: (ts: UInt64, cat: ProfilingCategory)] = [:]
            var sessionTotal: Double = 0
            for event in session.getEvents() {
                switch event.phase {
                case .begin:
                    beginTimestamps[event.name] = (event.timestampUs, event.category)
                case .end:
                    if let begin = beginTimestamps[event.name] {
                        let ms = Double(event.timestampUs - begin.ts) / 1000.0
                        phaseDurations[event.name, default: (cat: begin.cat, values: [])].values.append(ms)
                        sessionTotal += ms
                        beginTimestamps.removeValue(forKey: event.name)
                    }
                case .complete:
                    if event.category == .denoisingStep || event.category == .generationStep,
                       let dur = event.durationUs {
                        allStepDurations.append(Double(dur) / 1000.0)
                    }
                default: break
                }
            }
            totalDurations.append(sessionTotal)
            for entry in session.getMemoryTimeline() {
                peakMLXActive = max(peakMLXActive, entry.mlxActiveMB)
                peakProcess = max(peakProcess, entry.processFootprintMB)
            }
        }

        let phaseStats = phaseDurations.map { (name, data) in
            computeStats(name: name, category: data.cat, values: data.values)
        }.sorted { $0.category.sortOrder < $1.category.sortOrder }

        let stepStats = allStepDurations.isEmpty ? nil :
            computeStats(name: "Step", category: .denoisingStep, values: allStepDurations)

        return BenchmarkResult(
            phaseStats: phaseStats, stepStats: stepStats,
            totalStats: computeStats(name: "TOTAL", category: .custom, values: totalDurations),
            peakMLXActiveMB: peakMLXActive, peakProcessMB: peakProcess,
            warmupRuns: warmupCount, measuredRuns: sessions.count, metadata: first.metadata)
    }

    private static func computeStats(name: String, category: ProfilingCategory, values: [Double]) -> BenchmarkResult.PhaseStats {
        guard !values.isEmpty else { return .init(name: name, category: category, meanMs: 0, stdMs: 0, minMs: 0, maxMs: 0, count: 0) }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / max(1, Double(values.count - 1))
        return .init(name: name, category: category, meanMs: mean, stdMs: sqrt(variance),
                     minMs: values.min() ?? 0, maxMs: values.max() ?? 0, count: values.count)
    }
}
