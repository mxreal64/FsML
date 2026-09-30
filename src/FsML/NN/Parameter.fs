namespace FsML.NN

open System
open FsML
open FsML.Autograd

/// Represents a trainable weight or bias tensor in a neural network model
type Parameter(value: Value) =
    member _.Value : Value = value
    member _.Data : Tensor = value.Data
    member _.Grad : Tensor option = value.Grad
    member _.Shape : int[] = value.Shape
    member _.ZeroGrad() = value.ZeroGrad()

module Init =

    /// Xavier / Glorot uniform initialization: [-limit, limit] where limit = sqrt(6 / (fanIn + fanOut))
    let xavierUniform (fanIn: int) (fanOut: int) (seed: int option) : Parameter =
        let limit = MathF.Sqrt(6.0f / float32 (fanIn + fanOut))
        let rng = match seed with Some s -> Random(s) | None -> Random()
        let count = fanIn * fanOut
        let data = Array.init count (fun _ -> (float32 (rng.NextDouble()) * 2.0f - 1.0f) * limit)
        let t = Tensor([| fanIn; fanOut |], data)
        Parameter(Value.param t)

    /// Kaiming / He normal initialization: N(0, sqrt(2 / fanIn))
    let kaimingNormal (fanIn: int) (fanOut: int) (seed: int option) : Parameter =
        let std = MathF.Sqrt(2.0f / float32 fanIn)
        let t = Tensor.randn [| fanIn; fanOut |] seed * std
        Parameter(Value.param t)

    /// Zero initialization (typically for biases)
    let zeros (shape: int[]) : Parameter =
        Parameter(Value.param (Tensor.zeros shape))

    /// Uniform initialization in [-bound, bound]
    let uniform (shape: int[]) (bound: float32) (seed: int option) : Parameter =
        let rng = match seed with Some s -> Random(s) | None -> Random()
        let count = shape |> Array.fold (*) 1
        let data = Array.init count (fun _ -> (float32 (rng.NextDouble()) * 2.0f - 1.0f) * bound)
        Parameter(Value.param (Tensor(shape, data)))
