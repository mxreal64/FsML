namespace FsML.Optim

open System

module Schedulers =

    /// Step learning rate scheduler: decays the learning rate of optimizer by gamma every stepSize epochs
    type StepLR(optimizer: IOptimizer, stepSize: int, ?gamma: float32) =
        let decay = defaultArg gamma 0.1f
        let baseLr = optimizer.LearningRate
        let mutable epoch = 0

        member _.Step() =
            epoch <- epoch + 1
            let factor = MathF.Pow(decay, float32 (epoch / stepSize))
            optimizer.LearningRate <- baseLr * factor

    /// Cosine Annealing learning rate scheduler
    type CosineAnnealingLR(optimizer: IOptimizer, tMax: int, ?etaMin: float32) =
        let minLr = defaultArg etaMin 0.0f
        let baseLr = optimizer.LearningRate
        let mutable currentT = 0

        member _.Step() =
            currentT <- currentT + 1
            let fraction = float32 currentT / float32 tMax
            let cosTerm = 1.0f + MathF.Cos(MathF.PI * fraction)
            optimizer.LearningRate <- minLr + 0.5f * (baseLr - minLr) * cosTerm
