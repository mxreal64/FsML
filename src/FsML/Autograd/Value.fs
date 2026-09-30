namespace FsML.Autograd

open System
open System.Threading
open FsML
open FsML.Core

[<AllowNullLiteral>]
type Value(data: Tensor, parents: Value list, opName: string, requiresGrad: bool) =
    static let mutable nextId = 0L
    let id = Interlocked.Increment(&nextId)

    let mutable grad : Tensor option = None
    let mutable backwardFn : unit -> unit = fun () -> ()

    new(data: Tensor) = Value(data, [], "leaf", false)
    new(data: Tensor, requiresGrad: bool) = Value(data, [], "leaf", requiresGrad)
    new(scalar: float32, requiresGrad: bool) = Value(Tensor.scalar scalar, [], "leaf", requiresGrad)

    member _.Id : int64 = id
    member _.Data : Tensor = data
    member _.Shape : int[] = data.Shape
    member _.Rank : int = data.Rank
    member _.Length : int = data.Length

    member _.Parents : Value list = parents
    member _.OpName : string = opName
    member _.RequiresGrad : bool = requiresGrad

    member _.Grad
        with get() : Tensor option = grad
        and set(g: Tensor option) = grad <- g

    member _.Backward
        with get() : unit -> unit = backwardFn
        and set(fn: unit -> unit) = backwardFn <- fn

    member this.ZeroGrad() =
        grad <- None

    member this.Item
        with get(i: int) = data.[i]
        and set(i: int) v = data.[i] <- v

    member this.Item
        with get(r: int, c: int) = data.[r, c]
        and set(r: int, c: int) v = data.[r, c] <- v

    override this.ToString() =
        let gStr = match grad with Some g -> g.ToString() | None -> "None"
        sprintf "Value(data=%s, grad=%s, op='%s')" (data.ToString()) gStr opName

    // --- Static Constructors ---

    static member scalar(v: float32, ?requiresGrad: bool) : Value =
        let rg = defaultArg requiresGrad false
        Value(Tensor.scalar v, [], "leaf", rg)

    static member tensor(t: Tensor, ?requiresGrad: bool) : Value =
        let rg = defaultArg requiresGrad false
        Value(t, [], "leaf", rg)

    static member param(t: Tensor) : Value =
        Value(t, [], "param", true)

    static member zeros(shape: int[], ?requiresGrad: bool) : Value =
        let rg = defaultArg requiresGrad false
        Value(Tensor.zeros shape, [], "zeros", rg)

    static member ones(shape: int[], ?requiresGrad: bool) : Value =
        let rg = defaultArg requiresGrad false
        Value(Tensor.ones shape, [], "ones", rg)

    static member randn(shape: int[], ?requiresGrad: bool, ?seed: int) : Value =
        let rg = defaultArg requiresGrad false
        Value(Tensor.randn shape seed, [], "randn", rg)

    // --- Core Gradient Accumulator ---

    static member AccumulateGrad(target: Value, delta: Tensor) =
        if target.RequiresGrad then
            let reducedDelta =
                if target.Shape = delta.Shape then
                    delta
                elif delta.Rank = 2 && target.Rank = 2 && target.Shape.[0] = 1 && target.Shape.[1] = delta.Shape.[1] then
                    // Broadcasted row vector [1, N] across batch [B, N] -> sum across batch
                    Tensor.sumDim2D 0 delta
                elif delta.Rank = 2 && target.Rank = 1 && target.Shape.[0] = delta.Shape.[1] then
                    // Broadcasted 1D vector [N] across batch [B, N]
                    Tensor.sumDim2D 0 delta |> Tensor.reshape target.Shape
                elif delta.Rank = 2 && target.Rank = 2 && target.Shape.[1] = 1 && target.Shape.[0] = delta.Shape.[0] then
                    // Broadcasted col vector [M, 1] across [M, N]
                    Tensor.sumDim2D 1 delta
                elif target.Length = 1 && delta.Length > 1 then
                    Tensor.sumTensor delta
                else
                    failwith (sprintf "Cannot accumulate grad of shape %A into target of shape %A" delta.Shape target.Shape)

            match target.Grad with
            | Some current -> target.Grad <- Some (current + reducedDelta)
            | None -> target.Grad <- Some (reducedDelta.Clone())

    // --- Operator Overloads ---

    static member (+) (a: Value, b: Value) : Value =
        let out = Value(a.Data + b.Data, [a; b], "+", a.RequiresGrad || b.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                Value.AccumulateGrad(a, g)
                Value.AccumulateGrad(b, g)
            | None -> ()
        out

    static member (+) (a: Value, s: float32) : Value =
        let out = Value(a.Data + s, [a], sprintf "+%f" s, a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g -> Value.AccumulateGrad(a, g)
            | None -> ()
        out

    static member (+) (s: float32, a: Value) : Value =
        a + s

    static member (-) (a: Value, b: Value) : Value =
        let out = Value(a.Data - b.Data, [a; b], "-", a.RequiresGrad || b.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                Value.AccumulateGrad(a, g)
                Value.AccumulateGrad(b, -g)
            | None -> ()
        out

    static member (-) (a: Value, s: float32) : Value =
        a + (-s)

    static member (-) (s: float32, a: Value) : Value =
        let negA = a * -1.0f
        negA + s

    static member (*) (a: Value, b: Value) : Value =
        let out = Value(a.Data * b.Data, [a; b], "*", a.RequiresGrad || b.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                Value.AccumulateGrad(a, g * b.Data)
                Value.AccumulateGrad(b, g * a.Data)
            | None -> ()
        out

    static member (*) (a: Value, s: float32) : Value =
        let out = Value(a.Data * s, [a], sprintf "*%f" s, a.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g -> Value.AccumulateGrad(a, g * s)
            | None -> ()
        out

    static member (*) (s: float32, a: Value) : Value =
        a * s

    static member (/) (a: Value, b: Value) : Value =
        let out = Value(a.Data / b.Data, [a; b], "/", a.RequiresGrad || b.RequiresGrad)
        out.Backward <- fun () ->
            match out.Grad with
            | Some g ->
                Value.AccumulateGrad(a, g / b.Data)
                let gradB = -g * a.Data / (b.Data * b.Data)
                Value.AccumulateGrad(b, gradB)
            | None -> ()
        out

    static member (/) (a: Value, s: float32) : Value =
        a * (1.0f / s)

    static member (~-) (a: Value) : Value =
        a * -1.0f
