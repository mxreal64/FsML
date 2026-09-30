namespace FsML.Core

open System
open System.Numerics
open System.Threading.Tasks

module Kernels =

    // ==========================================
    // SIMD Vectorized Element-wise Math (Float32)
    // ==========================================

    let inline private vecSize< ^T when ^T : struct> = Vector<float32>.Count

    let add (a: float32[]) (b: float32[]) (dest: float32[]) : unit =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            let vb = Vector<float32>(b, i)
            (va + vb).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- a.[i] + b.[i]

    let sub (a: float32[]) (b: float32[]) (dest: float32[]) : unit =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            let vb = Vector<float32>(b, i)
            (va - vb).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- a.[i] - b.[i]

    let mul (a: float32[]) (b: float32[]) (dest: float32[]) : unit =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            let vb = Vector<float32>(b, i)
            (va * vb).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- a.[i] * b.[i]

    let div (a: float32[]) (b: float32[]) (dest: float32[]) : unit =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            let vb = Vector<float32>(b, i)
            (va / vb).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- a.[i] / b.[i]

    let scale (scalar: float32) (a: float32[]) (dest: float32[]) : unit =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)
        let vs = Vector<float32>(scalar)

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            (va * vs).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- a.[i] * scalar

    let addScalar (scalar: float32) (a: float32[]) (dest: float32[]) : unit =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)
        let vs = Vector<float32>(scalar)

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            (va + vs).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- a.[i] + scalar

    // ==========================================
    // Reductions & Aggregations
    // ==========================================

    let sum (a: float32[]) : float32 =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)
        let mutable acc = Vector<float32>.Zero

        for i in 0 .. vSize .. (simdLength - vSize) do
            acc <- acc + Vector<float32>(a, i)

        let mutable total = Vector.Dot(acc, Vector<float32>.One)
        for i in simdLength .. count - 1 do
            total <- total + a.[i]
        total

    let dot (a: float32[]) (b: float32[]) : float32 =
        let count = a.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)
        let mutable acc = Vector<float32>.Zero

        for i in 0 .. vSize .. (simdLength - vSize) do
            let va = Vector<float32>(a, i)
            let vb = Vector<float32>(b, i)
            acc <- acc + (va * vb)

        let mutable total = Vector.Dot(acc, Vector<float32>.One)
        for i in simdLength .. count - 1 do
            total <- total + (a.[i] * b.[i])
        total

    let maxElement (a: float32[]) : float32 =
        if a.Length = 0 then 0.0f
        else
            let mutable m = a.[0]
            for i in 1 .. a.Length - 1 do
                if a.[i] > m then m <- a.[i]
            m

    let minElement (a: float32[]) : float32 =
        if a.Length = 0 then 0.0f
        else
            let mutable m = a.[0]
            for i in 1 .. a.Length - 1 do
                if a.[i] < m then m <- a.[i]
            m

    // ==========================================
    // Non-Linear Activations & Mathematical Functions
    // ==========================================

    let relu (src: float32[]) (dest: float32[]) : unit =
        let count = src.Length
        let vSize = Vector<float32>.Count
        let simdLength = count - (count % vSize)
        let zeroVec = Vector<float32>.Zero

        for i in 0 .. vSize .. (simdLength - vSize) do
            let v = Vector<float32>(src, i)
            Vector.Max(v, zeroVec).CopyTo(dest, i)

        for i in simdLength .. count - 1 do
            dest.[i] <- if src.[i] > 0.0f then src.[i] else 0.0f

    let reluBackward (gradOut: float32[]) (src: float32[]) (gradDest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            gradDest.[i] <- if src.[i] > 0.0f then gradOut.[i] else 0.0f

    let leakyRelu (alpha: float32) (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- if src.[i] > 0.0f then src.[i] else alpha * src.[i]

    let leakyReluBackward (alpha: float32) (gradOut: float32[]) (src: float32[]) (gradDest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            gradDest.[i] <- if src.[i] > 0.0f then gradOut.[i] else alpha * gradOut.[i]

    let sigmoid (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- 1.0f / (1.0f + MathF.Exp(-src.[i]))

    let sigmoidBackward (gradOut: float32[]) (outVal: float32[]) (gradDest: float32[]) : unit =
        for i in 0 .. outVal.Length - 1 do
            let s = outVal.[i]
            gradDest.[i] <- gradOut.[i] * s * (1.0f - s)

    let tanh (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- MathF.Tanh(src.[i])

    let tanhBackward (gradOut: float32[]) (outVal: float32[]) (gradDest: float32[]) : unit =
        for i in 0 .. outVal.Length - 1 do
            let t = outVal.[i]
            gradDest.[i] <- gradOut.[i] * (1.0f - t * t)

    let gelu (src: float32[]) (dest: float32[]) : unit =
        let sqrt2OverPi = MathF.Sqrt(2.0f / MathF.PI)
        for i in 0 .. src.Length - 1 do
            let x = src.[i]
            let x3 = x * x * x
            let inner = sqrt2OverPi * (x + 0.044715f * x3)
            dest.[i] <- 0.5f * x * (1.0f + MathF.Tanh(inner))

    let exp (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- MathF.Exp(src.[i])

    let log (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- MathF.Log(MathF.Max(src.[i], 1e-12f))

    let pow (p: float32) (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- MathF.Pow(src.[i], p)

    let sqrt (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- MathF.Sqrt(src.[i])

    let abs (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- MathF.Abs(src.[i])

    let clip (minVal: float32) (maxVal: float32) (src: float32[]) (dest: float32[]) : unit =
        for i in 0 .. src.Length - 1 do
            dest.[i] <- Math.Clamp(src.[i], minVal, maxVal)

    // ==========================================
    // High-Performance Parallel Matrix Multiplication (GEMM)
    // ==========================================

    /// Matrix multiplication: C (M x N) = A (M x K) * B (K x N)
    /// Cache-friendly transposed inner loop with Parallel.For acceleration
    let matmul (m: int) (k: int) (n: int) (a: float32[]) (b: float32[]) (c: float32[]) : unit =
        // Pre-transpose B (K x N -> N x K) for contiguous linear memory reads in the inner loop
        let bT = Array.zeroCreate<float32> (k * n)
        for i in 0 .. k - 1 do
            let rowOffset = i * n
            for j in 0 .. n - 1 do
                bT.[j * k + i] <- b.[rowOffset + j]

        Parallel.For(0, m, fun i ->
            let aOffset = i * k
            let cOffset = i * n
            for j in 0 .. n - 1 do
                let bOffset = j * k
                let mutable sum = 0.0f
                let vSize = Vector<float32>.Count
                let simdLength = k - (k % vSize)
                let mutable acc = Vector<float32>.Zero

                for x in 0 .. vSize .. (simdLength - vSize) do
                    let va = Vector<float32>(a, aOffset + x)
                    let vb = Vector<float32>(bT, bOffset + x)
                    acc <- acc + (va * vb)

                sum <- Vector.Dot(acc, Vector<float32>.One)
                for x in simdLength .. k - 1 do
                    sum <- sum + (a.[aOffset + x] * bT.[bOffset + x])

                c.[cOffset + j] <- sum
        ) |> ignore

    /// Matrix multiplication accumulating to existing C: C += A * B
    let matmulAccumulate (m: int) (k: int) (n: int) (a: float32[]) (b: float32[]) (c: float32[]) : unit =
        let temp = Array.zeroCreate<float32> (m * n)
        matmul m k n a b temp
        for i in 0 .. (m * n) - 1 do
            c.[i] <- c.[i] + temp.[i]

    /// Matrix transpose: A (Rows x Cols) -> AT (Cols x Rows)
    let transpose (rows: int) (cols: int) (src: float32[]) (dest: float32[]) : unit =
        Parallel.For(0, rows, fun r ->
            let srcOffset = r * cols
            for c in 0 .. cols - 1 do
                dest.[c * rows + r] <- src.[srcOffset + c]
        ) |> ignore
