namespace FsML.Autograd

open System
open FsML
open FsML.Core

module Ops =

    let inline private accumulateGrad (target: Value) (delta: Tensor) =
        if target.RequiresGrad then
            let reducedDelta =
                if target.Shape = delta.Shape then
                    delta
                elif target.Rank = 1 && delta.Rank = 2 && target.Shape.[0] = delta.Shape.[1] then
                    Tensor.sumDim2D 0 delta |> Tensor.reshape target.Shape
                elif target.Length = 1 && delta.Length > 1 then
                    Tensor.sumTensor delta
                else
                    failwith (sprintf "Cannot accumulate grad of shape %A into target of shape %A" delta.Shape target.Shape)

            match target.Grad with
            | Some current -> target.Grad <- Some (current + reducedDelta)
            | None -> target.Grad <- Some (reducedDelta.Clone())

    // --- Basic Arithmetic Functions ---

    let add (a: Value) (b: Value) : Value = a + b
    let addScalar (s: float32) (a: Value) : Value = a + s
    let sub (a: Value) (b: Value) : Value = a - b
    let mul (a: Value) (b: Value) : Value = a * b
    let scale (s: float32) (a: Value) : Value = a * s
    let div (a: Value) (b: Value) : Value = a / b
    let neg (a: Value) : Value = -a

    // --- Matrix Operations ---

    let matmul (a: Value) (b: Value) : Value =
        let outData = Tensor.matmul a.Data b.Data
        let out = Value(outData, [a; b], "matmul", a.RequiresGrad || b.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                if a.RequiresGrad then
                    let bT = Tensor.transpose b.Data
                    let gradA = Tensor.matmul g bT
                    accumulateGrad a gradA

                if b.RequiresGrad then
                    let aT = Tensor.transpose a.Data
                    let gradB = Tensor.matmul aT g
                    accumulateGrad b gradB
            | None -> ()
        out

    let transpose (a: Value) : Value =
        let outData = Tensor.transpose a.Data
        let out = Value(outData, [a], "transpose", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Tensor.transpose g
                accumulateGrad a gradA
            | None -> ()
        out

    let reshape (newShape: int[]) (a: Value) : Value =
        let outData = Tensor.reshape newShape a.Data
        let out = Value(outData, [a], "reshape", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Tensor.reshape a.Shape g
                accumulateGrad a gradA
            | None -> ()
        out

    // --- Reductions ---

    let sum (a: Value) : Value =
        let out = Value(Tensor.sumTensor a.Data, [a], "sum", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Tensor.ones a.Shape * g.Data.[0]
                accumulateGrad a gradA
            | None -> ()
        out

    let mean (a: Value) : Value =
        let out = Value(Tensor.meanTensor a.Data, [a], "mean", a.RequiresGrad)
        let numEl = float32 a.Length
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let factor = g.Data.[0] / numEl
                let gradA = Tensor.ones a.Shape * factor
                accumulateGrad a gradA
            | None -> ()
        out

    // --- Non-Linear Activations ---

    let relu (a: Value) : Value =
        let outData = Tensor.relu a.Data
        let out = Value(outData, [a], "relu", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Array.zeroCreate<float32> a.Length
                Kernels.reluBackward g.Data a.Data.Data gradA
                accumulateGrad a (Tensor(a.Shape, gradA))
            | None -> ()
        out

    let leakyRelu (alpha: float32) (a: Value) : Value =
        let outData = Tensor.leakyRelu alpha a.Data
        let out = Value(outData, [a], sprintf "leakyRelu(%f)" alpha, a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Array.zeroCreate<float32> a.Length
                Kernels.leakyReluBackward alpha g.Data a.Data.Data gradA
                accumulateGrad a (Tensor(a.Shape, gradA))
            | None -> ()
        out

    let sigmoid (a: Value) : Value =
        let outData = Tensor.sigmoid a.Data
        let out = Value(outData, [a], "sigmoid", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Array.zeroCreate<float32> a.Length
                Kernels.sigmoidBackward g.Data outData.Data gradA
                accumulateGrad a (Tensor(a.Shape, gradA))
            | None -> ()
        out

    let tanh (a: Value) : Value =
        let outData = Tensor.tanh a.Data
        let out = Value(outData, [a], "tanh", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Array.zeroCreate<float32> a.Length
                Kernels.tanhBackward g.Data outData.Data gradA
                accumulateGrad a (Tensor(a.Shape, gradA))
            | None -> ()
        out

    let gelu (a: Value) : Value =
        let outData = Tensor.gelu a.Data
        let out = Value(outData, [a], "gelu", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let sqrt2OverPi = MathF.Sqrt(2.0f / MathF.PI)
                let gradA = Array.zeroCreate<float32> a.Length
                for i in 0 .. a.Length - 1 do
                    let x = a.Data.Data.[i]
                    let x3 = x * x * x
                    let inner = sqrt2OverPi * (x + 0.044715f * x3)
                    let t = MathF.Tanh(inner)
                    let sech2 = 1.0f - t * t
                    let dInner = sqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x)
                    let dGelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * dInner
                    gradA.[i] <- g.Data.[i] * dGelu
                accumulateGrad a (Tensor(a.Shape, gradA))
            | None -> ()
        out

    let exp (a: Value) : Value =
        let outData = Tensor.exp a.Data
        let out = Value(outData, [a], "exp", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g -> accumulateGrad a (g * outData)
            | None -> ()
        out

    let log (a: Value) : Value =
        let outData = Tensor.log a.Data
        let out = Value(outData, [a], "log", a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g -> accumulateGrad a (g / a.Data)
            | None -> ()
        out

    let pow (p: float32) (a: Value) : Value =
        let outData = Tensor.create a.Shape (Array.map (fun x -> MathF.Pow(x, p)) a.Data.Data)
        let out = Value(outData, [a], sprintf "pow(%f)" p, a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                let gradA = Array.zeroCreate<float32> a.Length
                for i in 0 .. a.Length - 1 do
                    let x = a.Data.Data.[i]
                    gradA.[i] <- g.Data.[i] * p * MathF.Pow(x, p - 1.0f)
                accumulateGrad a (Tensor(a.Shape, gradA))
            | None -> ()
        out
