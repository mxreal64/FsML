namespace FsML.NN

open System
open FsML
open FsML.Autograd

/// Base interface for all neural network modules
type ILayer =
    abstract member Forward : Value -> Value
    abstract member Parameters : Parameter list
    abstract member SetTraining : bool -> unit

/// Dense / Fully Connected affine layer: y = x * W + b
type Linear(inFeatures: int, outFeatures: int, ?useBias: bool, ?seed: int) =
    let hasBias = defaultArg useBias true
    let weight = Init.kaimingNormal inFeatures outFeatures seed
    let biasOpt =
        if hasBias then
            let bBound = 1.0f / MathF.Sqrt(float32 inFeatures)
            Some (Init.uniform [| 1; outFeatures |] bBound seed)
        else None

    member _.Weight : Parameter = weight
    member _.Bias : Parameter option = biasOpt

    member _.Parameters : Parameter list =
        match biasOpt with
        | Some b -> [ weight; b ]
        | None -> [ weight ]

    member _.Forward(x: Value) : Value =
        let affine = Ops.matmul x weight.Value
        match biasOpt with
        | Some b -> affine + b.Value
        | None -> affine

    interface ILayer with
        member this.Forward(x) = this.Forward(x)
        member this.Parameters = this.Parameters
        member _.SetTraining(_) = ()

/// Dropout layer for regularization during training
type Dropout(p: float32, ?seed: int) =
    let mutable isTraining = true
    let rng = match seed with Some s -> Random(s) | None -> Random()

    member _.IsTraining
        with get() = isTraining
        and set(v) = isTraining <- v

    member _.Parameters : Parameter list = []

    member _.Forward(x: Value) : Value =
        if not isTraining || p <= 0.0f then x
        else
            let mask = Array.init x.Length (fun _ -> if rng.NextDouble() >= float p then 1.0f / (1.0f - p) else 0.0f)
            let maskVal = Value.tensor (Tensor(x.Shape, mask))
            x * maskVal

    interface ILayer with
        member this.Forward(x) = this.Forward(x)
        member this.Parameters = this.Parameters
        member _.SetTraining(t) = isTraining <- t

/// Layer Normalization across the feature dimension
type LayerNorm(features: int, ?eps: float32) =
    let epsilon = defaultArg eps 1e-5f
    let gamma = Parameter(Value.param (Tensor.ones [| 1; features |]))
    let beta = Parameter(Value.param (Tensor.zeros [| 1; features |]))

    member _.Parameters : Parameter list = [ gamma; beta ]

    member _.Forward(x: Value) : Value =
        if x.Rank <> 2 then failwith "LayerNorm currently expects 2D input [Batch, Features]"
        let b, f = x.Shape.[0], x.Shape.[1]
        let mean = Tensor.meanDim2D 1 x.Data
        let diff = x.Data - mean
        let var = Tensor.meanDim2D 1 (diff * diff)
        let stdInv = Tensor.create [| b; 1 |] (Array.map (fun v -> 1.0f / MathF.Sqrt(v + epsilon)) var.Data)

        let normData = diff * stdInv
        let normVal = Value(normData, [x], "layerNorm", x.RequiresGrad)
        normVal.Backward <- fun () ->
            match normVal.Grad with
            | Some g ->
                if x.RequiresGrad then
                    let gradX = g * stdInv
                    match x.Grad with
                    | Some curr -> x.Grad <- Some (curr + gradX)
                    | None -> x.Grad <- Some gradX
            | None -> ()

        (normVal * gamma.Value) + beta.Value

    interface ILayer with
        member this.Forward(x) = this.Forward(x)
        member this.Parameters = this.Parameters
        member _.SetTraining(_) = ()

/// Sequential container for chaining layers
type Sequential(layers: ILayer list) =
    member _.Layers : ILayer list = layers

    member _.Forward(input: Value) : Value =
        layers |> List.fold (fun acc layer -> layer.Forward(acc)) input

    member _.Parameters : Parameter list =
        layers |> List.collect (fun l -> l.Parameters)

    member _.SetTraining(isTraining: bool) : unit =
        for l in layers do l.SetTraining(isTraining)

    interface ILayer with
        member this.Forward(x) = this.Forward(x)
        member this.Parameters = this.Parameters
        member this.SetTraining(t) = this.SetTraining(t)

/// Embedding lookup layer: [Batch, SeqLen] -> [Batch, SeqLen, EmbeddingDim]
type Embedding(numEmbeddings: int, embeddingDim: int, ?seed: int) =
    let weight = Init.uniform [| numEmbeddings; embeddingDim |] 1.0f seed

    member _.Weight : Parameter = weight
    member _.Parameters : Parameter list = [ weight ]

    member _.Forward(tokenIds: int[]) : Value =
        let count = tokenIds.Length
        let outData = Array.zeroCreate<float32> (count * embeddingDim)
        for i in 0 .. count - 1 do
            let tokenId = tokenIds.[i]
            let srcOffset = tokenId * embeddingDim
            let dstOffset = i * embeddingDim
            Array.Copy(weight.Data.Data, srcOffset, outData, dstOffset, embeddingDim)

        let out = Value(Tensor([| count; embeddingDim |], outData), [weight.Value], "embedding", weight.Value.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                if weight.Value.RequiresGrad then
                    let gradW = match weight.Grad with Some existing -> existing | None -> Tensor.zeros weight.Shape
                    for i in 0 .. count - 1 do
                        let tokenId = tokenIds.[i]
                        let srcOffset = i * embeddingDim
                        let dstOffset = tokenId * embeddingDim
                        for j in 0 .. embeddingDim - 1 do
                            gradW.Data.[dstOffset + j] <- gradW.Data.[dstOffset + j] + g.Data.[srcOffset + j]
                    weight.Value.Grad <- Some gradW
            | None -> ()
        out

    interface ILayer with
        member this.Forward(_) = failwith "Embedding forward requires integer token array"
        member this.Parameters = this.Parameters
        member _.SetTraining(_) = ()
