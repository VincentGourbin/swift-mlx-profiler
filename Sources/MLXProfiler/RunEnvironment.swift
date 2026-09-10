// RunEnvironment.swift - Build configuration, power state and idle-sleep control
// Copyright 2026 Vincent Gourbin
//
// Two things silently invalidated weeks of measurement during the Flash-Next
// campaign: every run between 2026-08-29 and 2026-09-07 was a Debug build (MLX's
// C++ compiled at -O0, host cost x1.2 to x1.8) with nothing in the report to say
// so, and three days of runs froze because the Mac went to sleep, again with
// nothing in the trace to show for it. Both are recorded here, up front.

import Foundation
#if os(macOS)
import IOKit.ps
#endif

/// What the profiler can tell about how this binary was built and what machine
/// it is running on.
public struct RunEnvironment: Sendable {

    /// True when the profiler was compiled without optimization.
    ///
    /// SwiftPM builds dependencies in the same configuration as the root package,
    /// so this also tells you MLX's C++ was compiled at `-O0`. A Debug timing is
    /// not a benchmark; it is off by a factor, and not a constant one.
    public static let isDebugBuild: Bool = {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }()

    public static var buildConfiguration: String { isDebugBuild ? "Debug" : "Release" }

    /// Whether the process is running under a Metal capture-enabled environment,
    /// a precondition for ``ProfilingSession/captureGPUTrace(phase:to:)``.
    public static var isMetalCaptureEnabled: Bool {
        ProcessInfo.processInfo.environment["MTL_CAPTURE_ENABLED"] == "1"
    }

    /// `AC Power`, `Battery Power`, `UPS Power`, or nil where unknown.
    ///
    /// Apple Silicon throttles hard on battery; a run that changed power source
    /// halfway through is not one run.
    public static func powerSource() -> String? {
        #if os(macOS)
        guard let type = IOPSGetProvidingPowerSourceType(nil)?.takeUnretainedValue() else { return nil }
        return type as String
        #else
        return nil
        #endif
    }

    /// The sleep-related settings from `pmset -g`, as `key: value`.
    ///
    /// Best effort: sandboxed processes cannot spawn it, and that is fine — the
    /// field is simply absent from the report.
    ///
    /// Read once per process. These settings barely ever change mid-run, and a
    /// session must not pay for a subprocess every time one is constructed.
    public static func powerManagementSettings() -> [String: String] {
        cachedPowerManagementSettings
    }

    private static let cachedPowerManagementSettings: [String: String] = readPowerManagementSettings()

    private static func readPowerManagementSettings() -> [String: String] {
        #if os(macOS)
        let keys = ["sleep", "displaysleep", "disksleep", "powernap", "standby", "lowpowermode"]
        guard let output = runCommand("/usr/bin/pmset", ["-g"], timeout: 2.0) else { return [:] }

        var settings: [String: String] = [:]
        for line in output.split(separator: "\n") {
            let fields = line.split(separator: " ", omittingEmptySubsequences: true)
            guard fields.count >= 2 else { continue }
            let key = String(fields[0]).lowercased()
            if keys.contains(key) { settings[key] = String(fields[1]) }
        }
        return settings
        #else
        return [:]
        #endif
    }

    #if os(macOS)
    /// Runs a command and returns stdout, or nil on failure or timeout.
    private static func runCommand(_ path: String, _ arguments: [String], timeout: TimeInterval) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do { try process.run() } catch { return nil }

        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning, Date() < deadline { usleep(20_000) }
        if process.isRunning { process.terminate(); return nil }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }
    #endif
}

/// Keeps the system awake for as long as it is held.
///
/// Wraps `ProcessInfo.beginActivity(options:reason:)` with
/// `.idleSystemSleepDisabled`, which stops the *system* sleeping without keeping
/// the display on. It does not survive the lid closing, and it does not override
/// a forced sleep — but it covers the case that actually bit: a long run on an
/// idle machine with the default energy settings.
public final class IdleSleepAssertion: @unchecked Sendable {
    private var token: NSObjectProtocol?
    private let lock = NSLock()

    /// Takes the assertion immediately.
    public init(reason: String) {
        self.token = ProcessInfo.processInfo.beginActivity(
            options: [.idleSystemSleepDisabled, .suddenTerminationDisabled],
            reason: reason
        )
    }

    /// Releases it. Idempotent; also runs on deinit.
    public func release() {
        lock.lock(); defer { lock.unlock() }
        guard let token else { return }
        ProcessInfo.processInfo.endActivity(token)
        self.token = nil
    }

    deinit { release() }
}
