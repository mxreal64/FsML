namespace FsML.Data

open System
open FsML

[<AutoOpen>]
module ActivePatterns =

    /// Active pattern to decompose a Tensor by dimensionality
    let (|Scalar|Vector|Matrix|Tensor3D|Tensor4D|HighDimensional|) (t: Tensor) =
        match t.Rank with
        | 0 -> Scalar t.[0]
        | 1 -> Vector t.Length
        | 2 -> Matrix (t.Shape.[0], t.Shape.[1])
        | 3 -> Tensor3D (t.Shape.[0], t.Shape.[1], t.Shape.[2])
        | 4 -> Tensor4D (t.Shape.[0], t.Shape.[1], t.Shape.[2], t.Shape.[3])
        | _ -> HighDimensional t.Shape

    /// Active pattern recognizing square matrices
    let (|SquareMatrix|_|) (t: Tensor) =
        if t.Rank = 2 && t.Shape.[0] = t.Shape.[1] then
            Some t.Shape.[0]
        else None

    /// Active pattern recognizing batch-sequence NLP shapes [Batch, SeqLen, Hidden]
    let (|BatchSequence|_|) (t: Tensor) =
        if t.Rank = 3 then
            Some (t.Shape.[0], t.Shape.[1], t.Shape.[2])
        else None

    /// Active pattern recognizing empty tensors
    let (|EmptyTensor|NonEmptyTensor|) (t: Tensor) =
        if t.Length = 0 then EmptyTensor else NonEmptyTensor t
