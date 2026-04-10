# swift-mlx-profiler

Performance profiling framework for [MLX](https://github.com/ml-explore/mlx-swift) models on Apple Silicon.

[![](https://img.shields.io/badge/Platform-macOS_15+-blue)](https://developer.apple.com/macos/) [![](https://img.shields.io/badge/Swift-6.0-orange)](https://swift.org) [![Website](https://img.shields.io/badge/Website-www.vinceforge.com-blue)](https://www.vinceforge.com)

## Features

- **Phase timing** with GPU% and CPU% per phase (IOKit + rusage)
- **Memory tracking**: MLX active/cache/peak + process footprint
- **Chrome Trace export** for [Perfetto UI](https://ui.perfetto.dev/) visualization
- **Benchmark runner**: multi-run statistics (mean/std/min/max)
- **os_signpost**: Instruments Metal System Trace integration
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
```

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

- macOS 15+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- mlx-swift 0.31.3+

## License

MIT
