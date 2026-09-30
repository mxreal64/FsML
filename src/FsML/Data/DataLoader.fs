namespace FsML.Data

open System
open FsML

/// A pair of input features and target labels
type Sample<'TInput, 'TTarget> = {
    Features: 'TInput
    Target: 'TTarget
}

/// Abstract in-memory dataset
type Dataset<'TInput, 'TTarget>(samples: Sample<'TInput, 'TTarget>[]) =
    member _.Length : int = samples.Length
    member _.Item(i: int) : Sample<'TInput, 'TTarget> = samples.[i]
    member _.Samples : Sample<'TInput, 'TTarget>[] = samples

/// High-throughput streaming DataLoader for mini-batching and shuffling
type DataLoader(features: Tensor, targets: int[], batchSize: int, ?shuffle: bool, ?seed: int) =
    let numSamples = features.Shape.[0]
    let featureDim = features.Shape.[1]
    let doShuffle = defaultArg shuffle true
    let rng = match seed with Some s -> Random(s) | None -> Random()

    member _.NumBatches : int =
        int (MathF.Ceiling(float32 numSamples / float32 batchSize))

    member _.GetBatches() : seq<Tensor * int[]> =
        seq {
            let indices = Array.init numSamples id
            if doShuffle then
                for i in numSamples - 1 .. -1 .. 1 do
                    let j = rng.Next(i + 1)
                    let tmp = indices.[i]
                    indices.[i] <- indices.[j]
                    indices.[j] <- tmp

            let mutable startIdx = 0
            while startIdx < numSamples do
                let curBatchSize = min batchSize (numSamples - startIdx)
                let batchFeatures = Array.zeroCreate<float32> (curBatchSize * featureDim)
                let batchTargets = Array.zeroCreate<int> curBatchSize

                for b in 0 .. curBatchSize - 1 do
                    let sampleIdx = indices.[startIdx + b]
                    let srcOffset = sampleIdx * featureDim
                    let dstOffset = b * featureDim
                    Array.Copy(features.Data, srcOffset, batchFeatures, dstOffset, featureDim)
                    batchTargets.[b] <- targets.[sampleIdx]

                startIdx <- startIdx + curBatchSize
                yield (Tensor([| curBatchSize; featureDim |], batchFeatures), batchTargets)
        }

module Datasets =

    /// Generates non-linear 2D Spiral Dataset (classic ML benchmark)
    let makeSpiral (numSamplesPerClass: int) (numClasses: int) (seed: int option) : Tensor * int[] =
        let rng = match seed with Some s -> Random(s) | None -> Random()
        let totalSamples = numSamplesPerClass * numClasses
        let features = Array.zeroCreate<float32> (totalSamples * 2)
        let targets = Array.zeroCreate<int> totalSamples

        for c in 0 .. numClasses - 1 do
            for i in 0 .. numSamplesPerClass - 1 do
                let idx = c * numSamplesPerClass + i
                let r = float32 i / float32 numSamplesPerClass
                let theta = (float32 c * 4.0f) + (r * 4.0f) + (float32 (rng.NextDouble()) * 0.2f)

                features.[idx * 2] <- r * MathF.Sin(theta)
                features.[idx * 2 + 1] <- r * MathF.Cos(theta)
                targets.[idx] <- c

        (Tensor([| totalSamples; 2 |], features), targets)

    /// Generates XOR dataset
    let makeXor (numSamples: int) (noise: float32 option) (seed: int option) : Tensor * int[] =
        let n = defaultArg noise 0.05f
        let rng = match seed with Some s -> Random(s) | None -> Random()
        let features = Array.zeroCreate<float32> (numSamples * 2)
        let targets = Array.zeroCreate<int> numSamples

        for i in 0 .. numSamples - 1 do
            let x1 = if rng.Next(2) = 1 then 1.0f else -1.0f
            let x2 = if rng.Next(2) = 1 then 1.0f else -1.0f
            let n1 = (float32 (rng.NextDouble()) * 2.0f - 1.0f) * n
            let n2 = (float32 (rng.NextDouble()) * 2.0f - 1.0f) * n

            features.[i * 2] <- x1 + n1
            features.[i * 2 + 1] <- x2 + n2
            targets.[i] <- if (x1 > 0.0f) <> (x2 > 0.0f) then 1 else 0

        (Tensor([| numSamples; 2 |], features), targets)
