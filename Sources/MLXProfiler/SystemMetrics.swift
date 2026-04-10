// SystemMetrics.swift - GPU, CPU, and memory monitoring for Apple Silicon
// Copyright 2026 Vincent Gourbin

import Foundation
import IOKit
import MLX

/// Low-level system metrics for Apple Silicon.
///
/// Provides GPU utilization (via IOKit), CPU time (via rusage), and memory
/// footprint (via task_info) without requiring root access.
public enum SystemMetrics {

    /// Instantaneous GPU utilization % (0-100) from IOKit.
    ///
    /// Reads `Device Utilization %` from the AGX accelerator driver.
    /// Works on Apple Silicon without root. Returns 0 if unavailable.
    public static func gpuUtilization() -> Int {
        var iterator: io_iterator_t = 0
        let matching = IOServiceMatching("IOAccelerator")
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else { return 0 }
        defer { IOObjectRelease(iterator) }

        var service = IOIteratorNext(iterator)
        while service != 0 {
            var properties: Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(service, &properties, kCFAllocatorDefault, 0) == KERN_SUCCESS,
               let dict = properties?.takeRetainedValue() as? [String: Any],
               let perfStats = dict["PerformanceStatistics"] as? [String: Any],
               let utilization = perfStats["Device Utilization %"] as? Int {
                IOObjectRelease(service)
                return utilization
            }
            IOObjectRelease(service)
            service = IOIteratorNext(iterator)
        }
        return 0
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
