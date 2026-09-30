namespace FsML

open System
open System.Text
open FsML.Core

type Tensor(shape: int[], data: float32[]) =
    let numElements =
        if shape.Length = 0 then 1
        else shape |> Array.fold (*) 1

    do
        if data.Length <> numElements then
            let shapeText = String.Join(", ", shape)
            invalidArg "data" (sprintf "Data length (%d) does not match shape [%s] with %d elements." data.Length shapeText numElements)

    // Compute contiguous row-major strides
    let strides =
        let s = Array.zeroCreate<int> shape.Length
        if shape.Length > 0 then
            s.[shape.Length - 1] <- 1
            for i in shape.Length - 2 .. -1 .. 0 do
                s.[i] <- s.[i + 1] * shape.[i + 1]
        s

    member _.Shape : int[] = shape
    member _.Dimensions : int[] = shape
    member _.Strides : int[] = strides
    member _.Data : float32[] = data
    member _.Length : int = numElements
    member _.Rank : int = shape.Length

    // --- Indexing ---

    /// Flat 1D element access
    member _.Item
        with get (i: int) : float32 = data.[i]
        and set (i: int) (value: float32) = data.[i] <- value

    /// 2D Matrix element access [row, col]
    member _.Item
        with get (r: int, c: int) : float32 =
            let offset = r * strides.[0] + c * strides.[1]
            data.[offset]
        and set (r: int, c: int) (value: float32) =
            let offset = r * strides.[0] + c * strides.[1]
            data.[offset] <- value

    /// 3D Tensor element access [d0, d1, d2]
    member _.Item
        with get (d0: int, d1: int, d2: int) : float32 =
            let offset = d0 * strides.[0] + d1 * strides.[1] + d2 * strides.[2]
            data.[offset]
        and set (d0: int, d1: int, d2: int) (value: float32) =
            let offset = d0 * strides.[0] + d1 * strides.[1] + d2 * strides.[2]
            data.[offset] <- value

    /// 4D Vision Tensor element access [batch, channel, height, width]
    member _.Item
        with get (d0: int, d1: int, d2: int, d3: int) : float32 =
            let offset = d0 * strides.[0] + d1 * strides.[1] + d2 * strides.[2] + d3 * strides.[3]
            data.[offset]
        and set (d0: int, d1: int, d2: int, d3: int) (value: float32) =
            let offset = d0 * strides.[0] + d1 * strides.[1] + d2 * strides.[2] + d3 * strides.[3]
            data.[offset] <- value

    /// Generic N-dimensional element access
    member this.Item
        with get (indices: int[]) : float32 =
            let mutable offset = 0
            for i in 0 .. indices.Length - 1 do
                offset <- offset + indices.[i] * strides.[i]
            data.[offset]
        and set (indices: int[]) (value: float32) =
            let mutable offset = 0
            for i in 0 .. indices.Length - 1 do
                offset <- offset + indices.[i] * strides.[i]
            data.[offset] <- value

    // --- Clone & Memory ---

    member _.Clone() : Tensor =
        let copy = Array.copy data
        Tensor(Array.copy shape, copy)

    member _.ToArray() : float32[] =
        Array.copy data

    override this.ToString() : string =
        let sb = StringBuilder()
        let shapeText = String.Join(", ", shape)
        sb.Append(sprintf "Tensor(shape=[%s], data=[" shapeText) |> ignore
        let maxDisplay = min 10 numElements
        for i in 0 .. maxDisplay - 1 do
            sb.Append(data.[i].ToString("G4")) |> ignore
            if i < maxDisplay - 1 then sb.Append(", ") |> ignore
        if numElements > maxDisplay then
            sb.Append(sprintf ", ... (%d more)" (numElements - maxDisplay)) |> ignore
        sb.Append("])") |> ignore
        sb.ToString()

    // --- Operator Overloads ---

    static member (+) (a: Tensor, b: Tensor) : Tensor =
        if a.Shape = b.Shape then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.add a.Data b.Data res
            Tensor(Array.copy a.Shape, res)
        elif a.Length = 1 then
            let res = Array.zeroCreate<float32> b.Length
            Kernels.addScalar a.Data.[0] b.Data res
            Tensor(Array.copy b.Shape, res)
        elif b.Length = 1 then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.addScalar b.Data.[0] a.Data res
            Tensor(Array.copy a.Shape, res)
        else
            // Matrix + Vector row-wise broadcast (e.g. [M, N] + [N] or [M, N] + [1, N])
            if a.Rank = 2 && (b.Rank = 1 && b.Shape.[0] = a.Shape.[1] || (b.Rank = 2 && b.Shape.[0] = 1 && b.Shape.[1] = a.Shape.[1])) then
                let m = a.Shape.[0]
                let n = a.Shape.[1]
                let res = Array.zeroCreate<float32> a.Length
                for i in 0 .. m - 1 do
                    let offset = i * n
                    for j in 0 .. n - 1 do
                        res.[offset + j] <- a.Data.[offset + j] + b.Data.[j]
                Tensor(Array.copy a.Shape, res)
            else
                let sA = String.Join(", ", a.Shape)
                let sB = String.Join(", ", b.Shape)
                failwith (sprintf "Shape mismatch in addition: [%s] + [%s]" sA sB)

    static member (+) (a: Tensor, scalar: float32) : Tensor =
        let res = Array.zeroCreate<float32> a.Length
        Kernels.addScalar scalar a.Data res
        Tensor(Array.copy a.Shape, res)

    static member (+) (scalar: float32, a: Tensor) : Tensor =
        a + scalar

    static member (-) (a: Tensor, b: Tensor) : Tensor =
        if a.Shape = b.Shape then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.sub a.Data b.Data res
            Tensor(Array.copy a.Shape, res)
        elif b.Length = 1 then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.addScalar (-b.Data.[0]) a.Data res
            Tensor(Array.copy a.Shape, res)
        elif a.Length = 1 then
            let res = Array.zeroCreate<float32> b.Length
            for i in 0 .. b.Length - 1 do
                res.[i] <- a.Data.[0] - b.Data.[i]
            Tensor(Array.copy b.Shape, res)
        else
            let sA = String.Join(", ", a.Shape)
            let sB = String.Join(", ", b.Shape)
            failwith (sprintf "Shape mismatch in subtraction: [%s] - [%s]" sA sB)

    static member (-) (a: Tensor, scalar: float32) : Tensor =
        a + (-scalar)

    static member (-) (scalar: float32, a: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> a.Length
        for i in 0 .. a.Length - 1 do
            res.[i] <- scalar - a.Data.[i]
        Tensor(Array.copy a.Shape, res)

    static member (*) (a: Tensor, b: Tensor) : Tensor =
        if a.Shape = b.Shape then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.mul a.Data b.Data res
            Tensor(Array.copy a.Shape, res)
        elif a.Length = 1 then
            let res = Array.zeroCreate<float32> b.Length
            Kernels.scale a.Data.[0] b.Data res
            Tensor(Array.copy b.Shape, res)
        elif b.Length = 1 then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.scale b.Data.[0] a.Data res
            Tensor(Array.copy a.Shape, res)
        else
            let sA = String.Join(", ", a.Shape)
            let sB = String.Join(", ", b.Shape)
            failwith (sprintf "Shape mismatch in multiplication: [%s] * [%s]" sA sB)

    static member (*) (a: Tensor, scalar: float32) : Tensor =
        let res = Array.zeroCreate<float32> a.Length
        Kernels.scale scalar a.Data res
        Tensor(Array.copy a.Shape, res)

    static member (*) (scalar: float32, a: Tensor) : Tensor =
        a * scalar

    static member (/) (a: Tensor, b: Tensor) : Tensor =
        if a.Shape = b.Shape then
            let res = Array.zeroCreate<float32> a.Length
            Kernels.div a.Data b.Data res
            Tensor(Array.copy a.Shape, res)
        elif b.Length = 1 then
            let inv = 1.0f / b.Data.[0]
            let res = Array.zeroCreate<float32> a.Length
            Kernels.scale inv a.Data res
            Tensor(Array.copy a.Shape, res)
        else
            let sA = String.Join(", ", a.Shape)
            let sB = String.Join(", ", b.Shape)
            failwith (sprintf "Shape mismatch in division: [%s] / [%s]" sA sB)

    static member (/) (a: Tensor, scalar: float32) : Tensor =
        let inv = 1.0f / scalar
        let res = Array.zeroCreate<float32> a.Length
        Kernels.scale inv a.Data res
        Tensor(Array.copy a.Shape, res)

    static member (~-) (a: Tensor) : Tensor =
        a * -1.0f

module Tensor =

    // --- Constructors ---

    let zeros (shape: int[]) : Tensor =
        let numEl = if shape.Length = 0 then 1 else shape |> Array.fold (*) 1
        Tensor(shape, Array.zeroCreate<float32> numEl)

    let ones (shape: int[]) : Tensor =
        let numEl = if shape.Length = 0 then 1 else shape |> Array.fold (*) 1
        let data = Array.create<float32> numEl 1.0f
        Tensor(shape, data)

    let scalar (v: float32) : Tensor =
        Tensor([| 1 |], [| v |])

    let fromArray (data: float32[]) : Tensor =
        Tensor([| data.Length |], Array.copy data)

    let from2DArray (data: float32[,]) : Tensor =
        let r = data.GetLength(0)
        let c = data.GetLength(1)
        let flat = Array.zeroCreate<float32> (r * c)
        for i in 0 .. r - 1 do
            for j in 0 .. c - 1 do
                flat.[i * c + j] <- data.[i, j]
        Tensor([| r; c |], flat)

    let create (shape: int[]) (data: float32[]) : Tensor =
        Tensor(shape, data)

    let init (shape: int[]) (f: int[] -> float32) : Tensor =
        let t = zeros shape
        let rank = shape.Length
        let indices = Array.zeroCreate<int> rank

        let rec fill dim =
            if dim = rank then
                t.[indices] <- f indices
            else
                for i in 0 .. shape.[dim] - 1 do
                    indices.[dim] <- i
                    fill (dim + 1)

        fill 0
        t

    let arange (startVal: float32) (endVal: float32) (step: float32) : Tensor =
        let count = int (MathF.Ceiling((endVal - startVal) / step))
        let data = Array.zeroCreate<float32> count
        for i in 0 .. count - 1 do
            data.[i] <- startVal + (float32 i * step)
        Tensor([| count |], data)

    let linspace (startVal: float32) (endVal: float32) (steps: int) : Tensor =
        if steps <= 1 then
            Tensor([| 1 |], [| startVal |])
        else
            let step = (endVal - startVal) / float32 (steps - 1)
            let data = Array.zeroCreate<float32> steps
            for i in 0 .. steps - 1 do
                data.[i] <- startVal + (float32 i * step)
            Tensor([| steps |], data)

    let randn (shape: int[]) (seed: int option) : Tensor =
        let rng = match seed with Some s -> Random(s) | None -> Random()
        let numEl = if shape.Length = 0 then 1 else shape |> Array.fold (*) 1
        let data = Array.zeroCreate<float32> numEl

        let mutable i = 0
        while i < numEl do
            let u1 = float32 (rng.NextDouble()) |> max 1e-7f
            let u2 = float32 (rng.NextDouble())
            let r = MathF.Sqrt(-2.0f * MathF.Log(u1))
            let theta = 2.0f * MathF.PI * u2
            let z0 = r * MathF.Cos(theta)
            let z1 = r * MathF.Sin(theta)
            data.[i] <- z0
            if i + 1 < numEl then
                data.[i + 1] <- z1
            i <- i + 2

        Tensor(shape, data)

    let rand (shape: int[]) (seed: int option) : Tensor =
        let rng = match seed with Some s -> Random(s) | None -> Random()
        let numEl = if shape.Length = 0 then 1 else shape |> Array.fold (*) 1
        let data = Array.init numEl (fun _ -> float32 (rng.NextDouble()))
        Tensor(shape, data)

    // --- Transformations & Shapes ---

    let reshape (newShape: int[]) (t: Tensor) : Tensor =
        Tensor(newShape, t.ToArray())

    let flatten (t: Tensor) : Tensor =
        Tensor([| t.Length |], t.ToArray())

    let transpose (t: Tensor) : Tensor =
        if t.Rank <> 2 then
            let s = String.Join(", ", t.Shape)
            failwith (sprintf "Transpose is currently implemented for 2D tensors, but got rank %d [%s]" t.Rank s)
        let r = t.Shape.[0]
        let c = t.Shape.[1]
        let res = Array.zeroCreate<float32> (r * c)
        Kernels.transpose r c t.Data res
        Tensor([| c; r |], res)

    // --- Linear Algebra & Matrix Math ---

    let matmul (a: Tensor) (b: Tensor) : Tensor =
        if a.Rank = 2 && b.Rank = 2 then
            let m, k1 = a.Shape.[0], a.Shape.[1]
            let k2, n = b.Shape.[0], b.Shape.[1]
            if k1 <> k2 then
                failwith (sprintf "Incompatible matrix dimensions for matmul: (%dx%d) and (%dx%d)" m k1 k2 n)
            let res = Array.zeroCreate<float32> (m * n)
            Kernels.matmul m k1 n a.Data b.Data res
            Tensor([| m; n |], res)
        elif a.Rank = 1 && b.Rank = 2 then
            let k1 = a.Shape.[0]
            let k2, n = b.Shape.[0], b.Shape.[1]
            if k1 <> k2 then
                failwith (sprintf "Incompatible dimensions: (%d) and (%dx%d)" k1 k2 n)
            let res = Array.zeroCreate<float32> n
            Kernels.matmul 1 k1 n a.Data b.Data res
            Tensor([| n |], res)
        elif a.Rank = 2 && b.Rank = 1 then
            let m, k1 = a.Shape.[0], a.Shape.[1]
            let k2 = b.Shape.[0]
            if k1 <> k2 then
                failwith (sprintf "Incompatible dimensions: (%dx%d) and (%d)" m k1 k2)
            let res = Array.zeroCreate<float32> m
            Kernels.matmul m k1 1 a.Data b.Data res
            Tensor([| m |], res)
        else
            let sA = String.Join(", ", a.Shape)
            let sB = String.Join(", ", b.Shape)
            failwith (sprintf "Matmul unsupported for shapes [%s] and [%s]" sA sB)

    let dot (a: Tensor) (b: Tensor) : float32 =
        if a.Length <> b.Length then
            failwith (sprintf "Dot product requires tensors of same length, got %d and %d" a.Length b.Length)
        Kernels.dot a.Data b.Data

    // --- Reductions ---

    let sum (t: Tensor) : float32 =
        Kernels.sum t.Data

    let sumTensor (t: Tensor) : Tensor =
        Tensor([| 1 |], [| Kernels.sum t.Data |])

    let mean (t: Tensor) : float32 =
        (Kernels.sum t.Data) / float32 t.Length

    let meanTensor (t: Tensor) : Tensor =
        Tensor([| 1 |], [| mean t |])

    let max (t: Tensor) : float32 =
        Kernels.maxElement t.Data

    let min (t: Tensor) : float32 =
        Kernels.minElement t.Data

    let sumDim2D (dim: int) (t: Tensor) : Tensor =
        if t.Rank <> 2 then failwith "sumDim2D requires 2D tensor"
        let r, c = t.Shape.[0], t.Shape.[1]
        if dim = 0 then
            let res = Array.zeroCreate<float32> c
            for i in 0 .. r - 1 do
                let offset = i * c
                for j in 0 .. c - 1 do
                    res.[j] <- res.[j] + t.Data.[offset + j]
            Tensor([| 1; c |], res)
        elif dim = 1 then
            let res = Array.zeroCreate<float32> r
            for i in 0 .. r - 1 do
                let offset = i * c
                let mutable rowSum = 0.0f
                for j in 0 .. c - 1 do
                    rowSum <- rowSum + t.Data.[offset + j]
                res.[i] <- rowSum
            Tensor([| r; 1 |], res)
        else
            failwith (sprintf "Invalid dim %d for 2D tensor" dim)

    let meanDim2D (dim: int) (t: Tensor) : Tensor =
        let s = sumDim2D dim t
        let divisor = if dim = 0 then float32 t.Shape.[0] else float32 t.Shape.[1]
        s * (1.0f / divisor)

    let argmax (dim: int) (t: Tensor) : int[] =
        if t.Rank <> 2 then failwith "argmax currently implemented for 2D tensors"
        let r, c = t.Shape.[0], t.Shape.[1]
        if dim = 1 then
            let result = Array.zeroCreate<int> r
            for i in 0 .. r - 1 do
                let offset = i * c
                let mutable maxIdx = 0
                let mutable maxVal = t.Data.[offset]
                for j in 1 .. c - 1 do
                    if t.Data.[offset + j] > maxVal then
                        maxVal <- t.Data.[offset + j]
                        maxIdx <- j
                result.[i] <- maxIdx
            result
        else
            failwith "Only dim=1 argmax supported for 2D tensors"

    // --- Non-Linear Activations ---

    let relu (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.relu t.Data res
        Tensor(Array.copy t.Shape, res)

    let leakyRelu (alpha: float32) (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.leakyRelu alpha t.Data res
        Tensor(Array.copy t.Shape, res)

    let sigmoid (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.sigmoid t.Data res
        Tensor(Array.copy t.Shape, res)

    let tanh (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.tanh t.Data res
        Tensor(Array.copy t.Shape, res)

    let gelu (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.gelu t.Data res
        Tensor(Array.copy t.Shape, res)

    let exp (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.exp t.Data res
        Tensor(Array.copy t.Shape, res)

    let log (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.log t.Data res
        Tensor(Array.copy t.Shape, res)

    let sqrt (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.sqrt t.Data res
        Tensor(Array.copy t.Shape, res)

    let abs (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.abs t.Data res
        Tensor(Array.copy t.Shape, res)

    let clip (minVal: float32) (maxVal: float32) (t: Tensor) : Tensor =
        let res = Array.zeroCreate<float32> t.Length
        Kernels.clip minVal maxVal t.Data res
        Tensor(Array.copy t.Shape, res)

    let softmax (t: Tensor) : Tensor =
        if t.Rank <> 2 then failwith "softmax currently implemented for 2D tensors"
        let r, c = t.Shape.[0], t.Shape.[1]
        let res = Array.zeroCreate<float32> (r * c)

        for i in 0 .. r - 1 do
            let offset = i * c
            let mutable maxV = t.Data.[offset]
            for j in 1 .. c - 1 do
                if t.Data.[offset + j] > maxV then maxV <- t.Data.[offset + j]

            let mutable expSum = 0.0f
            for j in 0 .. c - 1 do
                let ev = MathF.Exp(t.Data.[offset + j] - maxV)
                res.[offset + j] <- ev
                expSum <- expSum + ev

            let invSum = 1.0f / expSum
            for j in 0 .. c - 1 do
                res.[offset + j] <- res.[offset + j] * invSum

        Tensor([| r; c |], res)
