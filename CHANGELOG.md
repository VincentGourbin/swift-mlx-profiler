# Changelog

## 1.5.0

Per-phase Metal profiling, and a set of fixes to measurements that were quietly
wrong. Motivated by the Flash-Next performance campaign on qwen38-mlx-swift,
where three limitations of the 1.4 profiler cost several days of diagnosis.

### Phase boundaries no longer cost anything

`beginPhase`/`endPhase` used to take a full snapshot on each side — a GPU read, a
CPU read and a memory snapshot — about **4.7 ms per begin/end pair**. With one
phase per layer over 48 layers that is ~225 ms per token of pure profiler: a
5.5 ms layer measured 10.2 ms, and the conclusion drawn from it ("25 ms/layer")
was 20 % wrong.

A boundary is now a timestamp and an `os_signpost`, **~2 µs per pair**. GPU, CPU
and memory come from a background sampler running at a fixed rate, so the
profiler's cost no longer scales with how finely the work is carved up.

- New `ProfilingConfig.enableSampling`, `samplingIntervalMs` (default 16 ms),
  `systemMemorySamplingIntervalMs`, `trackSystemMemory`.
- `snapshotAtPhaseBoundaries` (default `false`) restores the old behaviour.
- New `ProfilingConfig.fineGrained` preset for one-phase-per-layer work.
- New `ProfilingSession.finish()` — stops the sampler and releases the idle-sleep
  assertion. Called automatically on `deinit`.

### GPU utilization is measured over time, not sampled at a boundary

`Device Utilization %` is an instantaneous reading. Sampled at the edge of a 5 ms
phase it is a coin flip: it reported 14–38 % where the interval truth was ~0 %,
and 41–49 % where it was ~82 %.

- Per-phase GPU is now a **time-weighted aggregate** over the sampler's readings,
  with mean, median and p10/p90 (`ProfilingSession.phaseSummaries()`).
- Reading `Device Utilization %` no longer fetches the driver's whole property
  dictionary: **17 µs instead of 1511 µs**, a 90x reduction, which is what makes
  a 16 ms sampler affordable (~0.1 % of a core).
- New `.ioReportResidency` GPU backend: performance-state residency read through
  IOReport, a true interval average. Opt-in, `dlopen`-loaded, falls back
  automatically.

### GPU kernels and dead time

- `ProfilingSession.captureGPUTrace(phase:to:_:)` wraps `MLX.GPU.startCapture`
  into a capture bounded to a named phase. Both build-time preconditions
  (`MTL_CAPTURE_ENABLED=1`, MLX built with `MLX_METAL_DEBUG`) are checked and
  reported instead of failing silently.
- `startMetalSystemTrace()` / `mergeMetalSystemTrace(_:)` drive
  `xctrace record --attach` alongside a run and fold the GPU track into the
  session's Chrome Trace, on the same timeline as the phases. Clocks are matched
  exactly via the session's own phase signposts, falling back to the trace start
  date.
- GPU-busy is reported as the **union** of GPU intervals. Summing durations
  double-counts nested and concurrent encoders: on the Flash-Next bench that
  reads 85.4 % against a true 75.7 %. Both are reported so the gap is visible.
- Known tooling limit: `xctrace record --template` cannot enable Shader Timeline,
  so `metal-gpu-intervals` names encoders, not MLX kernels. Use
  `captureGPUTrace(phase:)` for per-kernel detail.

### Measurements that invalidate themselves now say so

- **Debug builds** are called out at the top of the report and in the trace's
  `Session Info`. SwiftPM compiles MLX's C++ at `-O0` alongside your code; host
  cost runs 1.2–1.8x higher, and not by a constant factor.
- **System-wide memory** joins process footprint: anonymous, compressor
  (stored and occupied), swap, wired, file-backed, speculative, and
  `kern.memorystatus_level` — in the report, and on their own Chrome Trace lane.
  A phase during which the compressor grew by more than 1 GB is named explicitly.
- **Sleep and stalls**: a session holds an idle-sleep assertion by default,
  records power source and `pmset` settings, and marks any stretch longer than
  `stallThresholdSeconds` with no sample as a stall — in the report and as a
  block in the trace.

### Behaviour changes

- `getMemoryTimeline()` now returns sampler readings with `context == "sample"`.
  Boundary entries (`begin:`/`end:`) appear only when
  `snapshotAtPhaseBoundaries` is enabled. The type and method are unchanged.
- `recordStep` no longer reads GPU utilization; that comes from the sampler.
  Per-step timeline entries are written only when `trackPerStepMemory` is set.
- Constructing a `ProfilingSession` starts a sampling thread and takes an
  idle-sleep assertion. Both stop at `finish()` or `deinit`.
- The report gained columns (GPU p50) and sections; its layout is not stable API.

`LLMMetrics`, `MLXProfiler.start`/`end`, `beginPhase`/`endPhase` and
`ProfilingConfig.init` keep their existing signatures.

## 1.4.x and earlier

See the git history.
