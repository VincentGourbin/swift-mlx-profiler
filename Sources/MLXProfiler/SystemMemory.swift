// SystemMemory.swift - System-wide memory pressure metrics
// Copyright 2026 Vincent Gourbin

import Foundation
import Darwin

/// System-wide memory state, as opposed to this process's footprint.
///
/// `SystemMetrics.processFootprint()` answers "how much does *this* process hold".
/// It does not explain a `low on memory` kill, nor a decode that turns CPU-bound
/// because the compressor is thrashing. These counters do: anonymous pages, the
/// compressor (stored vs occupied), swap, wired, file-backed and speculative
/// pages, plus the kernel's own pressure level.
///
/// Equivalent to `vm_stat` + `sysctl vm.swapusage kern.memorystatus_level`.
public struct SystemMemorySnapshot: Sendable, Codable {
    /// Anonymous (process-private, not file-backed) pages — `internal_page_count`.
    public let anonymousBytes: Int64
    /// Physical memory the compressor currently occupies — `compressor_page_count`.
    public let compressorOccupiedBytes: Int64
    /// Uncompressed size of what the compressor holds — `total_uncompressed_pages_in_compressor`.
    public let compressorStoredBytes: Int64
    /// Kernel-wired, never pageable — `wire_count`.
    public let wiredBytes: Int64
    /// Clean file-backed pages, cheap to evict — `external_page_count`.
    public let fileBackedBytes: Int64
    /// Speculatively read-ahead pages — `speculative_count`.
    public let speculativeBytes: Int64
    /// Free pages — `free_count`.
    public let freeBytes: Int64
    /// Bytes currently swapped out (`vm.swapusage`), nil where unavailable.
    public let swapUsedBytes: Int64?
    /// Total swap file size (`vm.swapusage`), nil where unavailable.
    public let swapTotalBytes: Int64?
    /// `kern.memorystatus_level` — the headroom the kernel still sees, roughly a
    /// percentage. It falls toward 0 under pressure and jetsam fires well before
    /// it gets there. nil where the sysctl is unavailable.
    public let memoryStatusLevel: Int?

    public var anonymousMB: Double { Double(anonymousBytes) / 1_048_576 }
    public var compressorOccupiedMB: Double { Double(compressorOccupiedBytes) / 1_048_576 }
    public var compressorStoredMB: Double { Double(compressorStoredBytes) / 1_048_576 }
    public var wiredMB: Double { Double(wiredBytes) / 1_048_576 }
    public var fileBackedMB: Double { Double(fileBackedBytes) / 1_048_576 }
    public var speculativeMB: Double { Double(speculativeBytes) / 1_048_576 }
    public var freeMB: Double { Double(freeBytes) / 1_048_576 }
    public var swapUsedMB: Double? { swapUsedBytes.map { Double($0) / 1_048_576 } }

    /// How well the compressor is doing: stored / occupied. ~1.0 means it is
    /// holding incompressible data and buying almost nothing.
    public var compressionRatio: Double? {
        guard compressorOccupiedBytes > 0 else { return nil }
        return Double(compressorStoredBytes) / Double(compressorOccupiedBytes)
    }
}

extension SystemMetrics {

    /// System-wide memory snapshot, or nil if the Mach call fails.
    ///
    /// Costs one `host_statistics64` trap plus two sysctls — cheap enough for a
    /// 1-2 Hz sampler, too heavy to sit on a phase boundary.
    public static func systemMemory() -> SystemMemorySnapshot? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)

        let result = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }

        let pageSize = hostPageSize()
        func bytes(_ pages: UInt32) -> Int64 { Int64(pages) * pageSize }
        func bytes(_ pages: UInt64) -> Int64 { Int64(clamping: pages) * pageSize }

        let swap = swapUsage()
        return SystemMemorySnapshot(
            anonymousBytes: bytes(stats.internal_page_count),
            compressorOccupiedBytes: bytes(stats.compressor_page_count),
            compressorStoredBytes: bytes(stats.total_uncompressed_pages_in_compressor),
            wiredBytes: bytes(stats.wire_count),
            fileBackedBytes: bytes(stats.external_page_count),
            speculativeBytes: bytes(stats.speculative_count),
            freeBytes: bytes(stats.free_count),
            swapUsedBytes: swap?.used,
            swapTotalBytes: swap?.total,
            memoryStatusLevel: memoryStatusLevel()
        )
    }

    /// `vm_kernel_page_size` is a mutable global and so off-limits under strict
    /// concurrency; ask the kernel instead.
    private static func hostPageSize() -> Int64 {
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS, pageSize > 0 else {
            return 16384
        }
        return Int64(pageSize)
    }

    private static func swapUsage() -> (used: Int64, total: Int64)? {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return nil }
        return (Int64(clamping: usage.xsu_used), Int64(clamping: usage.xsu_total))
    }

    private static func memoryStatusLevel() -> Int? {
        var level: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("kern.memorystatus_level", &level, &size, nil, 0) == 0 else { return nil }
        return Int(level)
    }
}
