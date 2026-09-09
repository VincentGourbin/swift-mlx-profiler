# swift-mlx-profiler

Performance profiling framework for [MLX](https://github.com/ml-explore/mlx-swift) models on Apple Silicon.

[![](https://img.shields.io/badge/Platform-macOS_15+-blue)](https://developer.apple.com/macos/) [![](https://img.shields.io/badge/Swift-6.2-orange)](https://swift.org) [![Website](https://img.shields.io/badge/Website-www.vinceforge.com-blue)](https://www.vinceforge.com)

## Features

- **Cheap phase boundaries** — `beginPhase`/`endPhase` cost ~2 µs, so you can put a phase around every layer without the profiler becoming the thing you measure
- **Background sampling** of GPU/CPU/memory at a fixed rate, with time-weighted per-phase statistics (mean, median, p10/p90)
- **Memory tracking**: MLX active/cache/peak, process footprint, and system-wide compressor/swap/wired/pressure
- **Chrome Trace export** for [Perfetto UI](https://ui.perfetto.dev/) visualization
- **Metal System Trace merging** — fold Instruments' GPU track into the same timeline as your phases
- **`.gputrace` capture** bounded to a named phase
- **Benchmark runner**: multi-run statistics (mean/std/min/max)
- **os_signpost**: Instruments integration
- **Debug-build and stall warnings**, so an invalid measurement says so
- **Thread-safe** singleton profiler (NSLock)
- Supports **LLM**, **image diffusion**, and **video diffusion** pipelines

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.0.0"),
]

targets: [
    .target(dependencies: [
        .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
    ]),
]
```

## Usage

### Basic profiling

```swift
import MLXProfiler

let profiler = MLXProfiler.shared
let session = ProfilingSession()
session.title = "My Model Profiling"
session.metadata = ["model": "LTX-2.3", "quant": "qint8"]

profiler.enable()
profiler.activeSession = session

profiler.start("Text Encoding")
// ... encode text ...
profiler.end("Text Encoding")

profiler.start("Denoising")
profiler.setTotalSteps(8)
for step in 0..<8 {
    let t = Date()
    // ... denoising step ...
    profiler.recordStep(duration: Date().timeIntervalSince(t))
}
profiler.end("Denoising")

// Print report with GPU%, CPU%, memory timeline
print(session.generateReport())

// Export Chrome Trace for Perfetto UI
let trace = ChromeTraceExporter.export(session: session)
try trace.write(to: URL(fileURLWithPath: "trace.json"))
```

### Example output

```
╭──────────────────────────────────────────────────────────────────╮
│  My Model Profiling                                              │
├──────────────────────────────────────────────────────────────────┤
  model: LTX-2.3  quant: qint8
  Device: applegpu_g15s  RAM: 96GB

  PHASE TIMINGS                                    GPU%   CPU%
  ──────────────────────────────────────────────────────────────────
  Text Encoding             5.8s     1.4%    94%   49.2%
  Denoising              5m 03.2s   87.1% ████████   79%    3.4%
  VAE Decode               39.0s    11.2% █   41%   88.5%
  ──────────────────────────────────────────────────────────────────
  TOTAL                     5m 48.0s  100.0%
╰──────────────────────────────────────────────────────────────────╯
```

### System metrics

```swift
import MLXProfiler

// GPU utilization (0-100%, from IOKit, no root required)
let gpu = SystemMetrics.gpuUtilization()

// CPU time (user + system, all threads)
let cpu = SystemMetrics.processCPUTime()

// MLX GPU memory
let mem = SystemMetrics.mlxMemory()
print("Active: \(mem.activeMB) MB, Peak: \(mem.peakMB) MB")

// System-wide memory: what `Process (MB)` alone cannot explain — a
// `low on memory` kill, or a decode that turns CPU-bound on the compressor.
if let system = SystemMetrics.systemMemory() {
    print("Compressor: \(system.compressorOccupiedMB) MB holding \(system.compressorStoredMB) MB")
    print("Swap: \(system.swapUsedMB ?? 0) MB, headroom level: \(system.memoryStatusLevel ?? -1)")
}
```

### Measuring fine-grained phases

Phase boundaries are a timestamp and an `os_signpost` — about 2 µs. Everything
expensive is read by a background sampler at a fixed rate, so the profiler's cost
does not grow with how finely you carve the work up.

```swift
let session = ProfilingSession(config: .fineGrained)   // 16 ms sampler, IOReport GPU backend

for (index, layer) in layers.enumerated() {
    session.beginPhase("Layer \(index)", category: .custom)
    hidden = layer(hidden)
    session.endPhase("Layer \(index)", category: .custom)
}

session.finish()                       // stops the sampler, releases the sleep assertion
for phase in session.phaseSummaries() {
    // GPU is a time-weighted aggregate over the phase, not a boundary sample
    print(phase.name, phase.durationMs, phase.gpu?.mean ?? 0, phase.gpu?.median ?? 0)
}
```

Before 1.5 each boundary took a full snapshot — a GPU read, a CPU read and a
memory snapshot on both sides, about 4.7 ms per pair. One phase per layer over 48
layers added ~225 ms per token, which is how a 5.5 ms layer came to measure
10.2 ms. Set `snapshotAtPhaseBoundaries: true` to get that behaviour back where
you genuinely need memory pinned to an exact boundary.

### GPU utilization backends

| Backend | Cost per read | Semantics |
|---|---|---|
| `.deviceUtilization` (default) | ~17 µs | Instantaneous sample of the AGX driver's `Device Utilization %` |
| `.ioReportResidency` | ~112 µs | GPU performance-state residency since the previous read — a true interval average |

A point sample read at a phase boundary is close to meaningless on a short phase:
on the same benchmark it reported 14–38 % where the interval truth was ~0 %, and
41–49 % where it was ~82 %. The sampler fixes this by averaging over time;
`.ioReportResidency` fixes it at the source, at the cost of reaching into a
private framework (loaded with `dlopen`, with automatic fallback).

### Metal kernels and GPU dead time

Neither the sampler nor the phase timings can say how many Metal kernels a layer
dispatches or where the GPU sits idle inside a phase. Two routes, cheapest first.

A `.gputrace` bounded to one phase, opened in Xcode, where kernels *are* named:

```swift
// Requires MTL_CAPTURE_ENABLED=1 and an MLX built with MLX_METAL_DEBUG.
// Both preconditions are checked and reported rather than failing silently.
try session.captureGPUTrace(phase: "Layer 0") {
    hidden = layers[0](hidden)
}
```

Or record a full Metal System Trace alongside the run and merge its GPU track
into the session, so it lands on the same timeline as the phases:

```swift
let recorder = try session.startMetalSystemTrace()
recorder.waitUntilRecording()          // attaching takes a second or two
runPipeline()
let trace = try recorder.stop()

let summary = try session.mergeMetalSystemTrace(trace)
print(summary.busyPercent)             // union of GPU intervals over the window
```

The session's own phase signposts appear in the Instruments trace, so the two
clocks are matched exactly; failing that it falls back to the trace start date
and says so.

Two things worth knowing:

- **GPU-busy is a union, not a sum.** Instruments reports a command buffer and
  the encoders inside it as separate, overlapping rows. Adding their durations
  double-counts: on the Flash-Next benchmark that reads 85.4 % against a true
  75.7 %. `busyPercent` unions; `summedPercent` is reported alongside it only so
  the gap is visible.
- **Kernels are not named this way.** `xctrace record --template` cannot enable
  Shader Timeline from the command line, so `metal-gpu-intervals` gives you
  encoders (`Command Buffer 12:Compute Command 0`), not MLX kernels. Use
  `captureGPUTrace(phase:)` for those.

### Runs that invalidate themselves

Some measurements are worthless and the report should say so before showing any
numbers:

- **Debug builds.** SwiftPM builds MLX's C++ at `-O0` alongside your code; host
  cost runs 1.2–1.8× higher, and not by a constant factor. The report leads with
  a warning and the trace carries it in `Session Info`.
- **The machine going to sleep.** A session holds an idle-sleep assertion by
  default (`preventIdleSleep`), records the power source and `pmset` settings,
  and marks any stretch longer than `stallThresholdSeconds` with no sample as a
  stall — in the report and as a block in the trace.
- **The compressor.** A phase during which compressor-occupied memory grew by
  more than a gigabyte was measuring the machine running out of room, not the
  model; the report calls that phase out by name.

### Benchmarking

```swift
let sessions: [ProfilingSession] = // ... multiple runs ...
let result = BenchmarkAggregator.aggregate(sessions: sessions, warmupCount: 1)
print(result.generateReport())
```

## Used by

- [ltx-video-swift-mlx](https://github.com/VincentGourbin/ltx-video-swift-mlx) — LTX-2.3 video generation
- [flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) — Flux.2 image generation

## Requirements

- macOS 15+ (Sequoia) or iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- mlx-swift 0.31.3+
- Xcode (not just the Command Line Tools) for Metal System Trace recording

On iOS, GPU utilization is unavailable (IOKit is not) and returns 0; phase
timings, memory and the Chrome Trace export work as they do on macOS.

## License

MIT
