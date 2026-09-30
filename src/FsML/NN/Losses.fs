namespace FsML.NN

open System
open FsML
open FsML.Autograd

module Losses =

    /// Mean Squared Error (MSE) loss: (1/N) * sum((pred - target)^2)
    let mseLoss (pred: Value) (target: Value) : Value =
        let diff = pred - target
        let sq = diff * diff
        Ops.mean sq

    /// Cross Entropy Loss with LogSoftmax for multi-class classification
    /// pred: [Batch, Classes] logits
    /// targets: 1D array of integer class labels [Batch]
    let crossEntropyLoss (pred: Value) (targetClasses: int[]) : Value =
        if pred.Rank <> 2 then failwith "crossEntropyLoss expects 2D logits [Batch, NumClasses]"
        let b, c = pred.Shape.[0], pred.Shape.[1]
        if targetClasses.Length <> b then
            failwith (sprintf "Target count (%d) must match batch size (%d)" targetClasses.Length b)

        let logProbs = Array.zeroCreate<float32> (b * c)
        let probs = Array.zeroCreate<float32> (b * c)
        let mutable totalLoss = 0.0f
        let predFlat = pred.Data.Data

        for i in 0 .. b - 1 do
            let offset = i * c
            let mutable maxV = predFlat.[offset]
            for j in 1 .. c - 1 do
                if predFlat.[offset + j] > maxV then maxV <- predFlat.[offset + j]

            let mutable sumExp = 0.0f
            for j in 0 .. c - 1 do
                let ev = MathF.Exp(predFlat.[offset + j] - maxV)
                probs.[offset + j] <- ev
                sumExp <- sumExp + ev

            let logSumExp = MathF.Log(sumExp) + maxV
            for j in 0 .. c - 1 do
                let lp = predFlat.[offset + j] - logSumExp
                logProbs.[offset + j] <- lp
                probs.[offset + j] <- probs.[offset + j] / sumExp

            let targetClass = targetClasses.[i]
            totalLoss <- totalLoss - logProbs.[offset + targetClass]

        let meanLoss = totalLoss / float32 b
        let lossVal = Value(Tensor.scalar meanLoss, [pred], "crossEntropy", pred.RequiresGrad)

        lossVal.Backward <- fun () ->
            match lossVal.Grad with
            | Some g ->
                if pred.RequiresGrad then
                    let factor = g.Data.[0] / float32 b
                    let gradPred = Array.zeroCreate<float32> (b * c)
                    for i in 0 .. b - 1 do
                        let offset = i * c
                        let targetClass = targetClasses.[i]
                        for j in 0 .. c - 1 do
                            let p = probs.[offset + j]
                            let indicator = if j = targetClass then 1.0f else 0.0f
                            gradPred.[offset + j] <- (p - indicator) * factor

                    let gradTensor = Tensor([| b; c |], gradPred)
                    match pred.Grad with
                    | Some curr -> pred.Grad <- Some (curr + gradTensor)
                    | None -> pred.Grad <- Some gradTensor
            | None -> ()

        lossVal

    /// Binary Cross Entropy loss with logits: max(x, 0) - x * z + log(1 + exp(-|x|))
    let binaryCrossEntropyWithLogits (pred: Value) (target: Value) : Value =
        let maxVal = Tensor.create pred.Shape (Array.map (fun x -> MathF.Max(x, 0.0f)) pred.Data.Data)
        let absVal = Tensor.create pred.Shape (Array.map MathF.Abs pred.Data.Data)
        let logTerm = Tensor.create pred.Shape (Array.map (fun x -> MathF.Log(1.0f + MathF.Exp(-x))) absVal.Data)

        let lossData = maxVal - (pred.Data * target.Data) + logTerm
        let meanLoss = Tensor.mean lossData

        let out = Value(Tensor.scalar meanLoss, [pred; target], "bceWithLogits", pred.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                if pred.RequiresGrad then
                    let count = float32 pred.Length
                    let factor = g.Data.[0] / count
                    let sigm = Tensor.sigmoid pred.Data
                    let gradPred = (sigm - target.Data) * factor
                    match pred.Grad with
                    | Some curr -> pred.Grad <- Some (curr + gradPred)
                    | None -> pred.Grad <- Some gradPred
            | None -> ()
        out
