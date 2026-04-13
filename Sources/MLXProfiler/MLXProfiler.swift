// MLXProfiler.swift - Thread-safe profiler singleton
// Copyright 2026 Vincent Gourbin

import Foundation

/// Thread-safe profiler singleton for MLX pipelines.
///
/// Bridges with `ProfilingSession` for rich Chrome Trace export,
/// GPU/CPU utilization tracking, and memory timeline.
///
/// ```swift
/// let profiler = MLXProfiler.shared
/// profiler.enable()
/// profiler.activeSession = ProfilingSession()
///
/// profiler.start("Denoising")
/// for step in 0..<8 {
///     let t = Date()
///     // ... step work ...
///     profiler.recordStep(duration: Date().timeIntervalSince(t))
/// }
/// profiler.end("Denoising")
///
/// print(profiler.activeSession!.generateReport())
/// ```
public final class MLXProfiler: @unchecked Sendable {
    public static let shared = MLXProfiler()

    // Note: lock is internal so TrainingProfiling extension can use it
    internal let lock = NSLock()
    private var _isEnabled = false
    private var _activeSession: ProfilingSession? = nil
    private var timings: [TimingEntry] = []
    private var stepTimes: [TimeInterval] = []
    private var stepCount: Int = 0
    private var totalStepsCount: Int = 0
    private var activeTimers: [String: Date] = [:]

    // Training metrics storage (accessed under lock)
    internal var trainingSteps: [TrainingStepMetrics] = []
    internal var validationEntries: [ValidationMetrics] = []
    internal var trainingStartTime: Date? = nil

    private init() {}

    public var isEnabled: Bool {
        lock.lock(); defer { lock.unlock() }; return _isEnabled
    }

    public var activeSession: ProfilingSession? {
        get { lock.lock(); defer { lock.unlock() }; return _activeSession }
        set { lock.lock(); _activeSession = newValue; lock.unlock() }
    }

    public func enable() {
        lock.lock()
        _isEnabled = true
        timings.removeAll()
        activeTimers.removeAll()
        stepTimes.removeAll()
        stepCount = 0
        totalStepsCount = 0
        lock.unlock()
        resetLLMState()
        resetTTSState()
    }

    public func disable() {
        lock.lock(); _isEnabled = false; lock.unlock()
    }

    public func setTotalSteps(_ total: Int) {
        lock.lock(); totalStepsCount = total; stepCount = 0; lock.unlock()
    }

    public func start(_ name: String) {
        lock.lock()
        guard _isEnabled else { lock.unlock(); return }
        let session = _activeSession
        activeTimers[name] = Date()
        lock.unlock()

        let category = ProfilingSession.inferCategory(name)
        session?.beginPhase(name, category: category)
    }

    public func end(_ name: String) {
        let endTime = Date()
        lock.lock()
        guard _isEnabled else { lock.unlock(); return }
        let session = _activeSession
        guard let startTime = activeTimers[name] else { lock.unlock(); return }
        activeTimers.removeValue(forKey: name)
        timings.append(TimingEntry(name: name, duration: endTime.timeIntervalSince(startTime), startTime: startTime, endTime: endTime))
        lock.unlock()

        let category = ProfilingSession.inferCategory(name)
        session?.endPhase(name, category: category)
    }

    public func record(_ name: String, duration: TimeInterval) {
        let now = Date()
        lock.lock()
        guard _isEnabled else { lock.unlock(); return }
        timings.append(TimingEntry(name: name, duration: duration, startTime: now.addingTimeInterval(-duration), endTime: now))
        lock.unlock()
    }

    public func recordStep(duration: TimeInterval) {
        lock.lock()
        guard _isEnabled else { lock.unlock(); return }
        stepTimes.append(duration)
        stepCount += 1
        let currentStep = stepCount
        let total = totalStepsCount
        let session = _activeSession
        lock.unlock()

        session?.recordStep(index: currentStep, total: total, durationUs: UInt64(duration * 1_000_000))
    }

    @discardableResult
    public func measure<T>(_ name: String, _ operation: () throws -> T) rethrows -> T {
        guard isEnabled else { return try operation() }
        let startTime = Date()
        let result = try operation()
        let endTime = Date()
        lock.lock()
        timings.append(TimingEntry(name: name, duration: endTime.timeIntervalSince(startTime), startTime: startTime, endTime: endTime))
        lock.unlock()
        return result
    }

    public func reset() {
        lock.lock()
        timings.removeAll(); activeTimers.removeAll(); stepTimes.removeAll()
        stepCount = 0; totalStepsCount = 0
        lock.unlock()
    }

    public func getTimings() -> [TimingEntry] {
        lock.lock(); defer { lock.unlock() }; return timings
    }

    public func getStepTimes() -> [TimeInterval] {
        lock.lock(); defer { lock.unlock() }; return stepTimes
    }
}

/// Timing entry for a profiled operation
public struct TimingEntry: Sendable {
    public let name: String
    public let duration: TimeInterval
    public let startTime: Date
    public let endTime: Date

    public var durationMs: Double { duration * 1000 }

    public var durationFormatted: String { formatDuration(duration) }
}
