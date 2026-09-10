// GPUUtilization.swift - GPU busy-ness backends for Apple Silicon
// Copyright 2026 Vincent Gourbin

import Foundation
#if canImport(IOKit)
import IOKit
#endif

/// How GPU busy-ness is measured.
public enum GPUUtilizationBackend: String, Sendable, Codable {
    /// `Device Utilization %` from the AGX accelerator's `PerformanceStatistics`.
    ///
    /// An *instantaneous* reading. Sampled at a phase boundary it is a coin flip:
    /// on a 5 ms phase it reported 14-38 % where the interval truth was ~0 %, and
    /// 41-49 % where it was ~82 %. Only meaningful averaged over many samples.
    case deviceUtilization

    /// GPU performance-state residency, read from IOReport.
    ///
    /// Each read returns the fraction of time the GPU spent out of its idle states
    /// *since the previous read* — an interval average rather than a point sample,
    /// which is what a phase actually wants to know. Uses a private framework
    /// (`libIOReport.dylib`), loaded with `dlopen`, so nothing links against it
    /// unless you ask for this backend; falls back to ``deviceUtilization`` if
    /// unavailable.
    case ioReportResidency
}

/// Reads GPU busy-ness, keeping whatever handles the backend needs open.
///
/// What made the old reader expensive was not finding the accelerator — that
/// costs almost nothing — but `IORegistryEntryCreateCFProperties`, which
/// materializes the driver's *entire* property dictionary just to read one
/// integer out of it: 1511 us per call, measured. Asking for the single
/// `PerformanceStatistics` property instead costs 16.7 us, a 90x difference, and
/// that is what this reader does. The cached service handle is a smaller, real
/// saving on top: no iterator or matching dictionary per tick.
public final class GPUUtilizationReader: @unchecked Sendable {

    /// The backend actually in use — may differ from the one requested if it was
    /// unavailable on this machine.
    public let backend: GPUUtilizationBackend

    /// Set when the requested backend could not be used, explaining why.
    public let fallbackReason: String?

    private let lock = NSLock()
    #if canImport(IOKit)
    private var service: io_service_t = 0
    #endif
    private var ioReport: IOReportGPUResidency?

    public init(preferred: GPUUtilizationBackend = .deviceUtilization) {
        var chosen = preferred
        var reason: String? = nil

        if preferred == .ioReportResidency {
            if let reader = IOReportGPUResidency() {
                self.ioReport = reader
            } else {
                chosen = .deviceUtilization
                reason = "libIOReport unavailable or no GPU performance-state channel; "
                    + "fell back to Device Utilization %"
            }
        }

        self.backend = chosen
        self.fallbackReason = reason
        #if canImport(IOKit)
        if chosen == .deviceUtilization { self.service = Self.matchAccelerator() }
        #endif
    }

    deinit {
        #if canImport(IOKit)
        if service != 0 { IOObjectRelease(service) }
        #endif
    }

    /// GPU busy-ness in percent (0-100).
    ///
    /// For ``GPUUtilizationBackend/ioReportResidency`` this is the average since
    /// the previous call; for ``GPUUtilizationBackend/deviceUtilization`` it is an
    /// instantaneous sample.
    public func read() -> Int {
        lock.lock(); defer { lock.unlock() }

        if let ioReport { return ioReport.readActiveResidencyPercent() }

        #if canImport(IOKit)
        if service == 0 { service = Self.matchAccelerator() }
        guard service != 0 else { return 0 }
        if let value = Self.deviceUtilization(of: service) { return value }
        // The accelerator can go away (eGPU unplug, driver restart) — re-match once.
        IOObjectRelease(service)
        service = Self.matchAccelerator()
        guard service != 0 else { return 0 }
        return Self.deviceUtilization(of: service) ?? 0
        #else
        return 0
        #endif
    }

    #if canImport(IOKit)
    /// Retained accelerator service, or 0. Caller owns the reference.
    private static func matchAccelerator() -> io_service_t {
        var iterator: io_iterator_t = 0
        let matching = IOServiceMatching("IOAccelerator")
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
            return 0
        }
        defer { IOObjectRelease(iterator) }

        var candidate = IOIteratorNext(iterator)
        while candidate != 0 {
            if deviceUtilization(of: candidate) != nil { return candidate }  // keep the reference
            IOObjectRelease(candidate)
            candidate = IOIteratorNext(iterator)
        }
        return 0
    }

    /// Reads just the one property. Never widen this to
    /// `IORegistryEntryCreateCFProperties`: fetching the whole dictionary is
    /// ~90x more expensive and every other field in it is unused.
    private static func deviceUtilization(of service: io_service_t) -> Int? {
        guard let property = IORegistryEntryCreateCFProperty(
                service, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0),
              let perfStats = property.takeRetainedValue() as? [String: Any],
              let utilization = perfStats["Device Utilization %"] as? Int
        else { return nil }
        return utilization
    }
    #endif
}
