// SystemMetrics.swift - GPU, CPU, and memory monitoring for Apple Silicon
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Low-level system metrics for Apple Silicon.
///
/// Provides GPU utilization (via IOKit on macOS), CPU time (via rusage), and memory
/// footprint (via task_info) without requiring root access.
public enum SystemMetrics {

    /// Shared reader, so the IO registry is walked once for the whole process
    /// rather than on every call.
    private static let sharedGPUReader = GPUUtilizationReader(preferred: .deviceUtilization)

    /// Instantaneous GPU utilization % (0-100).
    ///
    /// On macOS: reads `Device Utilization %` from the AGX accelerator driver via IOKit.
    /// On iOS: returns 0 (IOKit is not available; use Instruments for GPU profiling).
    ///
    /// This is a *point* sample and a poor description of a short interval — see
    /// ``GPUUtilizationBackend/deviceUtilization``. Prefer the session's sampler,
    /// which averages over time, for anything you intend to draw a conclusion from.
    public static func gpuUtilization() -> Int {
        sharedGPUReader.read()
    }

    /// Cumulative CPU time (user + system) for this process in seconds.
    ///
    /// Use two measurements with elapsed wall time to compute CPU%:
    /// `CPU% = (cpuTime2 - cpuTime1) / wallTimeDelta * 100`
    ///
    /// Values >100% indicate multi-threaded CPU usage.
    public static func processCPUTime() -> Double {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let userSec = Double(usage.ru_utime.tv_sec) + Double(usage.ru_utime.tv_usec) / 1_000_000
        let sysSec = Double(usage.ru_stime.tv_sec) + Double(usage.ru_stime.tv_usec) / 1_000_000
        return userSec + sysSec
    }

    /// Physical memory footprint of the current process in bytes.
    public static func processFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        return result == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
    }

    /// MLX GPU memory snapshot.
    public struct MLXMemorySnapshot: Sendable {
        public let activeBytes: Int
        public let cacheBytes: Int
        public let peakBytes: Int

        public var activeMB: Double { Double(activeBytes) / 1_048_576 }
        public var cacheMB: Double { Double(cacheBytes) / 1_048_576 }
        public var peakMB: Double { Double(peakBytes) / 1_048_576 }
    }

    /// Current MLX GPU memory usage.
    public static func mlxMemory() -> MLXMemorySnapshot {
        MLXMemorySnapshot(
            activeBytes: Memory.activeMemory,
            cacheBytes: Memory.cacheMemory,
            peakBytes: Memory.peakMemory
        )
    }
}
