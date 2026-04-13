// TrainingProfiling.swift - Training metrics tracking for LoRA/QLoRA fine-tuning
// Copyright 2026 Vincent Gourbin

import Foundation

/// Metriques capturees a chaque step d'entrainement
public struct TrainingStepMetrics: Sendable, Codable {
    public let iteration: Int
    public let loss: Float
    public let tokensPerSecond: Double
    public let learningRate: Float
    public let mlxActiveBytes: Int
    public let mlxPeakBytes: Int
    public let gpuUtilization: Int
    public let durationUs: UInt64

    public init(
        iteration: Int,
        loss: Float,
        tokensPerSecond: Double,
        learningRate: Float,
        mlxActiveBytes: Int,
        mlxPeakBytes: Int,
        gpuUtilization: Int,
        durationUs: UInt64
    ) {
        self.iteration = iteration
        self.loss = loss
        self.tokensPerSecond = tokensPerSecond
        self.learningRate = learningRate
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxPeakBytes = mlxPeakBytes
        self.gpuUtilization = gpuUtilization
        self.durationUs = durationUs
    }
}

/// Metriques de validation
public struct ValidationMetrics: Sendable, Codable {
    public let iteration: Int
    public let loss: Float
    public let duration: TimeInterval
}

/// Resume complet d'une session de training
public struct TrainingSummary: Sendable {
    public let totalIterations: Int
    public let finalLoss: Float
    public let bestLoss: Float
    public let bestIteration: Int
    public let avgTokensPerSecond: Double
    public let peakMemoryMB: Double
    public let totalTrainingTime: TimeInterval
}

// MARK: - Extension MLXProfiler pour le training

extension MLXProfiler {

    /// Demarre une session de profiling training
    public func startTrainingSession(config: [String: String]) {
        lock.lock()
        trainingSteps.removeAll()
        validationEntries.removeAll()
        trainingStartTime = Date()
        lock.unlock()

        if let session = activeSession {
            session.metadata.merge(config) { _, new in new }
            session.title = "TRAINING PROFILING REPORT"
            session.beginPhase("Training", category: .custom)
        }
    }

    /// Enregistre les metriques d'un step d'entrainement
    public func recordTrainingStep(_ metrics: TrainingStepMetrics) {
        lock.lock()
        trainingSteps.append(metrics)
        lock.unlock()

        activeSession?.recordComplete(
            "Train Step \(metrics.iteration)",
            category: .custom,
            durationUs: metrics.durationUs
        )

        if let session = activeSession {
            let ts = session.currentTimestampUsPublic()
            session.addCounterEvent(
                name: "Training Loss",
                timestampUs: ts,
                values: [
                    "loss": Double(metrics.loss),
                    "tokens_per_sec": metrics.tokensPerSecond,
                ]
            )
            session.addCounterEvent(
                name: "GPU Memory (MB)",
                timestampUs: ts,
                values: [
                    "active": Double(metrics.mlxActiveBytes) / 1_048_576,
                    "peak": Double(metrics.mlxPeakBytes) / 1_048_576,
                ]
            )
        }
    }

    /// Enregistre les metriques d'une evaluation de validation
    public func recordValidation(iteration: Int, loss: Float, duration: TimeInterval) {
        let metrics = ValidationMetrics(iteration: iteration, loss: loss, duration: duration)
        lock.lock()
        validationEntries.append(metrics)
        lock.unlock()

        activeSession?.recordComplete(
            "Validation (iter \(iteration))",
            category: .custom,
            durationUs: UInt64(duration * 1_000_000)
        )

        if let session = activeSession {
            let ts = session.currentTimestampUsPublic()
            session.addCounterEvent(
                name: "Validation Loss",
                timestampUs: ts,
                values: ["val_loss": Double(loss)]
            )
        }
    }

    /// Retourne toutes les metriques de training enregistrees
    public func getTrainingMetrics() -> [TrainingStepMetrics] {
        lock.lock(); defer { lock.unlock() }
        return trainingSteps
    }

    /// Genere un resume de la session de training
    public func getTrainingSummary() -> TrainingSummary {
        lock.lock()
        let steps = trainingSteps
        let startTime = trainingStartTime ?? Date()
        lock.unlock()

        let totalTime = Date().timeIntervalSince(startTime)

        guard !steps.isEmpty else {
            return TrainingSummary(
                totalIterations: 0, finalLoss: 0, bestLoss: 0,
                bestIteration: 0, avgTokensPerSecond: 0,
                peakMemoryMB: 0, totalTrainingTime: totalTime
            )
        }

        let bestStep = steps.min(by: { $0.loss < $1.loss })!
        let peakMem = steps.max(by: { $0.mlxPeakBytes < $1.mlxPeakBytes })!
        let avgTokPerSec = steps.reduce(0.0) { $0 + $1.tokensPerSecond } / Double(steps.count)

        return TrainingSummary(
            totalIterations: steps.last!.iteration + 1,
            finalLoss: steps.last!.loss,
            bestLoss: bestStep.loss,
            bestIteration: bestStep.iteration + 1,
            avgTokensPerSecond: avgTokPerSec,
            peakMemoryMB: Double(peakMem.mlxPeakBytes) / 1_048_576,
            totalTrainingTime: totalTime
        )
    }
}
