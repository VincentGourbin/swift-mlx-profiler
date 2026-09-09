// IOReportGPUResidency.swift - GPU active residency via the private IOReport framework
// Copyright 2026 Vincent Gourbin
//
// `Device Utilization %` is a point sample. What a phase wants is "what fraction
// of this interval was the GPU out of its idle power states", which is exactly
// what the GPU performance-state residency counters report. Those live behind
// IOReport, a private framework — so it is loaded with `dlopen` and every symbol
// is optional. Nothing links against it; if anything is missing the reader simply
// fails to construct and the caller falls back to `Device Utilization %`.

import Foundation

#if os(macOS)

/// Reads GPU active residency from IOReport's `GPU Performance States` channels.
///
/// Each ``readActiveResidencyPercent()`` returns the share of time spent in a
/// non-idle power state *since the previous call*, so consecutive reads tile the
/// timeline without gaps or double counting.
final class IOReportGPUResidency {

    // MARK: - Dynamically bound symbols

    private typealias CopyChannelsInGroup = @convention(c) (
        CFString?, CFString?, UInt64, UInt64, UInt64) -> Unmanaged<CFMutableDictionary>?
    private typealias CreateSubscription = @convention(c) (
        UnsafeMutableRawPointer?, CFMutableDictionary,
        UnsafeMutablePointer<Unmanaged<CFMutableDictionary>?>, UInt64, CFTypeRef?
    ) -> UnsafeMutableRawPointer?
    private typealias CreateSamples = @convention(c) (
        UnsafeMutableRawPointer, CFMutableDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias CreateSamplesDelta = @convention(c) (
        CFDictionary, CFDictionary, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias Iterate = @convention(c) (
        CFDictionary, @convention(block) (CFDictionary) -> Int32) -> Void
    private typealias StateGetCount = @convention(c) (CFDictionary) -> Int32
    private typealias StateGetNameForIndex = @convention(c) (CFDictionary, Int32) -> Unmanaged<CFString>?
    private typealias StateGetResidency = @convention(c) (CFDictionary, Int32) -> Int64

    private let createSamples: CreateSamples
    private let createSamplesDelta: CreateSamplesDelta
    private let iterate: Iterate
    private let stateGetCount: StateGetCount
    private let stateGetNameForIndex: StateGetNameForIndex
    private let stateGetResidency: StateGetResidency

    private let subscription: UnsafeMutableRawPointer
    private let subscribedChannels: CFMutableDictionary
    private var previousSample: CFDictionary

    /// Power-state names that mean "not doing work". Everything else is a P-state.
    private static let idleStateNames: Set<String> = ["IDLE", "OFF", "DOWN"]

    init?() {
        guard let handle = dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY) else { return nil }

        func symbol<T>(_ name: String, as type: T.Type) -> T? {
            guard let pointer = dlsym(handle, name) else { return nil }
            return unsafeBitCast(pointer, to: type)
        }

        guard let copyChannels = symbol("IOReportCopyChannelsInGroup", as: CopyChannelsInGroup.self),
              let createSubscription = symbol("IOReportCreateSubscription", as: CreateSubscription.self),
              let createSamples = symbol("IOReportCreateSamples", as: CreateSamples.self),
              let createSamplesDelta = symbol("IOReportCreateSamplesDelta", as: CreateSamplesDelta.self),
              let iterate = symbol("IOReportIterate", as: Iterate.self),
              let stateGetCount = symbol("IOReportStateGetCount", as: StateGetCount.self),
              let stateGetNameForIndex = symbol("IOReportStateGetNameForIndex", as: StateGetNameForIndex.self),
              let stateGetResidency = symbol("IOReportStateGetResidency", as: StateGetResidency.self)
        else { return nil }

        guard let channels = copyChannels(
            "GPU Stats" as CFString, "GPU Performance States" as CFString, 0, 0, 0
        )?.takeRetainedValue() else { return nil }

        var subscribed: Unmanaged<CFMutableDictionary>?
        guard let subscription = createSubscription(nil, channels, &subscribed, 0, nil),
              let subscribedChannels = subscribed?.takeRetainedValue()
        else { return nil }

        // A baseline, so the first real read already covers a known interval.
        guard let baseline = createSamples(subscription, subscribedChannels, nil)?.takeRetainedValue()
        else { return nil }

        self.createSamples = createSamples
        self.createSamplesDelta = createSamplesDelta
        self.iterate = iterate
        self.stateGetCount = stateGetCount
        self.stateGetNameForIndex = stateGetNameForIndex
        self.stateGetResidency = stateGetResidency
        self.subscription = subscription
        self.subscribedChannels = subscribedChannels
        self.previousSample = baseline

        // A channel set that reports no states at all is useless to us; say so now
        // rather than returning 0 forever.
        guard residencySplit(of: baseline) != nil else { return nil }
    }

    /// Percentage of the interval since the previous call that the GPU spent in a
    /// non-idle power state (0-100).
    func readActiveResidencyPercent() -> Int {
        guard let current = createSamples(subscription, subscribedChannels, nil)?.takeRetainedValue(),
              let delta = createSamplesDelta(previousSample, current, nil)?.takeRetainedValue()
        else { return 0 }
        previousSample = current

        guard let split = residencySplit(of: delta), split.total > 0 else { return 0 }
        let percent = Double(split.active) / Double(split.total) * 100
        return min(100, max(0, Int(percent.rounded())))
    }

    /// Sums residency ticks across every performance-state channel in `sample`,
    /// split into active and idle. nil when the sample exposes no states.
    private func residencySplit(of sample: CFDictionary) -> (active: Int64, total: Int64)? {
        var active: Int64 = 0
        var total: Int64 = 0
        var sawState = false

        iterate(sample) { channel in
            let stateCount = self.stateGetCount(channel)
            guard stateCount > 0 else { return 0 }
            sawState = true
            for index in 0..<stateCount {
                let residency = self.stateGetResidency(channel, index)
                guard residency > 0 else { continue }
                total &+= residency
                let name = self.stateGetNameForIndex(channel, index)?
                    .takeUnretainedValue() as String? ?? ""
                if !Self.idleStateNames.contains(name.uppercased()) { active &+= residency }
            }
            return 0
        }

        return sawState ? (active, total) : nil
    }
}

#else

/// IOReport is macOS-only; on other platforms construction always fails and the
/// caller falls back to ``GPUUtilizationBackend/deviceUtilization``.
final class IOReportGPUResidency {
    init?() { return nil }
    func readActiveResidencyPercent() -> Int { 0 }
}

#endif
