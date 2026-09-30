namespace FsML.Shapes

open System
open FsML

// ==========================================
// Dimension Phantom Types
// ==========================================

type Batch = Batch
type SeqLen = SeqLen
type Hidden = Hidden
type Vocab = Vocab
type Channels = Channels
type Height = Height
type Width = Width
type InFeatures = InFeatures
type OutFeatures = OutFeatures

type D1 = D1
type D2 = D2
type D3 = D3
type D4 = D4

type M = M
type N = N
type K = K
type P = P

// ==========================================
// Strongly-Typed Compile-Time Shape Tensors
// ==========================================

/// 1D Strongly Typed Tensor with phantom dimension 'Dim
type Tensor1D<'T, 'Dim> = {
    Data: 'T[]
    Dim: int
}

/// 2D Strongly Typed Tensor with phantom dimensions 'Rows and 'Cols
type Tensor2D<'T, 'Rows, 'Cols> = {
    Underlying: Tensor
    Rows: int
    Cols: int
}

/// 3D Strongly Typed Tensor with phantom dimensions 'D1, 'D2, 'D3
type Tensor3D<'T, 'D1, 'D2, 'D3> = {
    Underlying: Tensor
    Dim1: int
    Dim2: int
    Dim3: int
}

/// 4D Strongly Typed Vision Tensor with phantom dimensions 'N (Batch), 'C (Channels), 'H (Height), 'W (Width)
type Tensor4D<'T, 'N, 'C, 'H, 'W> = {
    Underlying: Tensor
    Batch: int
    Channels: int
    Height: int
    Width: int
}

module Typed =

    // --- 2D Operations ---

    let create2D<'Rows, 'Cols> (r: int, c: int) (data: float32[]) : Tensor2D<float32, 'Rows, 'Cols> =
        let t = Tensor([| r; c |], data)
        { Underlying = t; Rows = r; Cols = c }

    let init2D<'Rows, 'Cols> (r: int, c: int) (f: int -> int -> float32) : Tensor2D<float32, 'Rows, 'Cols> =
        let data = Array.zeroCreate<float32> (r * c)
        for i in 0 .. r - 1 do
            for j in 0 .. c - 1 do
                data.[i * c + j] <- f i j
        let t = Tensor([| r; c |], data)
        { Underlying = t; Rows = r; Cols = c }

    let fromTensor2D<'Rows, 'Cols> (t: Tensor) : Tensor2D<float32, 'Rows, 'Cols> =
        if t.Rank <> 2 then failwith $"Expected 2D tensor, got rank {t.Rank}"
        { Underlying = t; Rows = t.Shape.[0]; Cols = t.Shape.[1] }

    /// Compile-Time Safe Matrix Multiplication:
    /// Type signature: (Rows x Shared) * (Shared x Cols) -> (Rows x Cols)
    /// Matrix multiplication will NOT compile if 'Shared dimensions mismatch.
    let matmul (a: Tensor2D<float32, 'M, 'K>) (b: Tensor2D<float32, 'K, 'N>) : Tensor2D<float32, 'M, 'N> =
        let res = Tensor.matmul a.Underlying b.Underlying
        { Underlying = res; Rows = a.Rows; Cols = b.Cols }

    /// Transposition with compile-time swapped dimensions: (Rows x Cols) -> (Cols x Rows)
    let transpose (a: Tensor2D<float32, 'Rows, 'Cols>) : Tensor2D<float32, 'Cols, 'Rows> =
        let res = Tensor.transpose a.Underlying
        { Underlying = res; Rows = a.Cols; Cols = a.Rows }

    /// Compile-time safe element-wise addition (must have identical shapes)
    let add (a: Tensor2D<float32, 'R, 'C>) (b: Tensor2D<float32, 'R, 'C>) : Tensor2D<float32, 'R, 'C> =
        let res = a.Underlying + b.Underlying
        { Underlying = res; Rows = a.Rows; Cols = a.Cols }

    /// Compile-time safe element-wise multiplication
    let mul (a: Tensor2D<float32, 'R, 'C>) (b: Tensor2D<float32, 'R, 'C>) : Tensor2D<float32, 'R, 'C> =
        let res = a.Underlying * b.Underlying
        { Underlying = res; Rows = a.Rows; Cols = a.Cols }

    /// Applies ReLU activation preserving static shapes
    let relu (a: Tensor2D<float32, 'R, 'C>) : Tensor2D<float32, 'R, 'C> =
        let res = Tensor.relu a.Underlying
        { Underlying = res; Rows = a.Rows; Cols = a.Cols }

    // --- 4D Vision Operations ---

    let create4D<'N, 'C, 'H, 'W> (n: int, c: int, h: int, w: int) (data: float32[]) : Tensor4D<float32, 'N, 'C, 'H, 'W> =
        let t = Tensor([| n; c; h; w |], data)
        { Underlying = t; Batch = n; Channels = c; Height = h; Width = w }

    let unwrap (typed: Tensor2D<'T, 'R, 'C>) : Tensor =
        typed.Underlying
