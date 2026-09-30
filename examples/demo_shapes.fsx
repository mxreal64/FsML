#r "../src/FsML/bin/Debug/net11.0/FsML.dll"

open System
open FsML
open FsML.Shapes

printfn "==============================================================="
printfn "   FsML Static Shape Analysis (Shapes as Types in F#)"
printfn "===============================================================\n"

// Define phantom dimension markers
type Batch = Batch
type SeqLen = SeqLen
type Hidden = Hidden
type Vocab = Vocab

printfn "[1] Instantiating strongly-typed tensors..."

// 1. Create a Batch x SeqLen input tensor: [32, 128]
let input : Tensor2D<float32, Batch, SeqLen> =
    Typed.init2D<Batch, SeqLen> (32, 128) (fun r c -> float32 (r + c))

printfn $"    Input Tensor : Tensor2D<float32, Batch, SeqLen> -> Shape [{input.Rows}, {input.Cols}]"

// 2. Create an Embedding / Projection Weights matrix: [128, 768] (SeqLen x Hidden)
let weights : Tensor2D<float32, SeqLen, Hidden> =
    Typed.init2D<SeqLen, Hidden> (128, 768) (fun r c -> 0.01f * float32 (r * c))

printfn $"    Weights Matrix: Tensor2D<float32, SeqLen, Hidden> -> Shape [{weights.Rows}, {weights.Cols}]"

// 3. Compile-Time Safe Matrix Multiplication:
// Signature: Tensor2D<'T, 'M, 'K> -> Tensor2D<'T, 'K, 'N> -> Tensor2D<'T, 'M, 'N>
// Notice the 'K' dimension (SeqLen) matches statically.
let projected : Tensor2D<float32, Batch, Hidden> =
    Typed.matmul input weights

printfn "\n[2] Compile-Time Guaranteed Matrix Multiplication:"
printfn $"    (Batch x SeqLen) * (SeqLen x Hidden) -> (Batch x Hidden)"
printfn $"    Output Type  : Tensor2D<float32, Batch, Hidden>"
printfn $"    Output Shape : [{projected.Rows}, {projected.Cols}]"

// 4. Transposition:
let transposed = Typed.transpose projected
printfn "\n[3] Static Transpose:"
printfn $"    (Batch x Hidden) -> (Hidden x Batch)"
printfn $"    Output Shape : [{transposed.Rows}, {transposed.Cols}]"

printfn "\n[4] Why This Beats Python/PyTorch:"
printfn "    In Python: multiplying [32, 128] by [32, 128] crashes at RUNTIME after expensive initialization."
printfn "    In FsML  : 'Typed.matmul input input' is a COMPILE-TIME TYPE ERROR (FS0001)."
printfn "    The compiler literally prevents dimension mismatch bugs before code ever runs!\n"
printfn "===============================================================\n"
