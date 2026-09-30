namespace FsML.Optim

open System
open FsML
open FsML.NN

/// Base interface for optimizers
type IOptimizer =
    abstract member Step : unit -> unit
    abstract member ZeroGrad : unit -> unit
    abstract member LearningRate : float32 with get, set

/// Stochastic Gradient Descent with optional Momentum and L2 Weight Decay
type SGD(parameters: Parameter list, lr: float32, ?momentum: float32, ?weightDecay: float32) =
    let mutable currentLr = lr
    let mom = defaultArg momentum 0.0f
    let decay = defaultArg weightDecay 0.0f
    let velocities = parameters |> List.map (fun p -> Tensor.zeros p.Shape)

    interface IOptimizer with
        member _.LearningRate
            with get() = currentLr
            and set(v) = currentLr <- v

        member _.ZeroGrad() =
            for p in parameters do p.ZeroGrad()

        member _.Step() =
            for i in 0 .. parameters.Length - 1 do
                let p = parameters.[i]
                let v = velocities.[i]
                match p.Grad with
                | Some g ->
                    let gradWithDecay =
                        if decay > 0.0f then g + (p.Data * decay)
                        else g

                    for j in 0 .. p.Data.Length - 1 do
                        let newVel = mom * v.Data.[j] + gradWithDecay.Data.[j]
                        v.Data.[j] <- newVel
                        p.Data.Data.[j] <- p.Data.Data.[j] - currentLr * newVel
                | None -> ()

/// AdamW Optimizer with decoupled weight decay
type AdamW(parameters: Parameter list, lr: float32, ?beta1: float32, ?beta2: float32, ?eps: float32, ?weightDecay: float32) =
    let mutable currentLr = lr
    let b1 = defaultArg beta1 0.9f
    let b2 = defaultArg beta2 0.999f
    let epsilon = defaultArg eps 1e-8f
    let decay = defaultArg weightDecay 0.01f

    let mList = parameters |> List.map (fun p -> Tensor.zeros p.Shape)
    let vList = parameters |> List.map (fun p -> Tensor.zeros p.Shape)
    let mutable stepCount = 0

    interface IOptimizer with
        member _.LearningRate
            with get() = currentLr
            and set(v) = currentLr <- v

        member _.ZeroGrad() =
            for p in parameters do p.ZeroGrad()

        member _.Step() =
            stepCount <- stepCount + 1
            let biasCorrection1 = 1.0f - MathF.Pow(b1, float32 stepCount)
            let biasCorrection2 = 1.0f - MathF.Pow(b2, float32 stepCount)

            for i in 0 .. parameters.Length - 1 do
                let p = parameters.[i]
                let m = mList.[i]
                let v = vList.[i]

                match p.Grad with
                | Some g ->
                    for j in 0 .. p.Data.Length - 1 do
                        let gj = g.Data.[j]

                        // First moment: m = b1 * m + (1 - b1) * g
                        let mj = b1 * m.Data.[j] + (1.0f - b1) * gj
                        m.Data.[j] <- mj

                        // Second moment: v = b2 * v + (1 - b2) * g^2
                        let vj = b2 * v.Data.[j] + (1.0f - b2) * (gj * gj)
                        v.Data.[j] <- vj

                        // Bias corrected moments
                        let mHat = mj / biasCorrection1
                        let vHat = vj / biasCorrection2

                        // Decoupled weight decay update
                        let decayTerm = p.Data.Data.[j] * currentLr * decay

                        // Parameter update
                        let stepTerm = (currentLr * mHat) / (MathF.Sqrt(vHat) + epsilon)
                        p.Data.Data.[j] <- p.Data.Data.[j] - decayTerm - stepTerm
                | None -> ()

/// Standard Adam Optimizer
type Adam(parameters: Parameter list, lr: float32, ?beta1: float32, ?beta2: float32, ?eps: float32) =
    inherit AdamW(parameters, lr, ?beta1=beta1, ?beta2=beta2, ?eps=eps, weightDecay=0.0f)
