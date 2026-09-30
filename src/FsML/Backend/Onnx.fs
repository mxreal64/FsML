namespace FsML.Backend

open System
open System.Collections.Generic
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open FsML
open FsML.Core

type OnnxError =
    | ModelLoadError of string
    | MissingOutputNode of string
    | NativeInferenceError of string
    | InvalidTensorType of expected: string * actual: string

type ExecutionProvider =
    | Cpu
    | Cuda of deviceId: int

type SessionConfig = {
    Provider: ExecutionProvider
    IntraOpNumThreads: int option
    EnableMemoryPattern: bool
}

module SessionConfig =
    let defaultConfig = {
        Provider = Cpu
        IntraOpNumThreads = None
        EnableMemoryPattern = true
    }

module OnnxSession =

    let private buildNativeOptions (config: SessionConfig) : SessionOptions =
        let opts = new SessionOptions()
        opts.EnableMemoryPattern <- config.EnableMemoryPattern
        config.IntraOpNumThreads |> Option.iter (fun t -> opts.IntraOpNumThreads <- t)
        match config.Provider with
        | Cpu -> ()
        | Cuda deviceId -> opts.AppendExecutionProvider_CUDA(deviceId) |> ignore
        opts

    /// Loads an ONNX model from file path using a specific SessionConfig
    let loadWithConfig (config: SessionConfig) (modelPath: string) : Result<InferenceSession, OnnxError> =
        try
            use nativeOpts = buildNativeOptions config
            Ok (new InferenceSession(modelPath, nativeOpts))
        with ex ->
            Error (ModelLoadError ex.Message)

    /// Loads an ONNX model using default CPU configuration
    let load (modelPath: string) : Result<InferenceSession, OnnxError> =
        loadWithConfig SessionConfig.defaultConfig modelPath

    /// Executes inference on an ONNX session with FsML tensors
    let run (session: InferenceSession) (inputs: (string * Tensor) list) : Result<Map<string, Tensor>, OnnxError> =
        try
            let nativeInputs =
                inputs
                |> List.map (fun (name, tensor) ->
                    let denseTensor = new DenseTensor<float32>(Memory<float32>(tensor.Data), tensor.Shape)
                    NamedOnnxValue.CreateFromTensor(name, denseTensor)
                )
                |> List.toArray

            use nativeResults = session.Run(nativeInputs)

            let mutable resultMap = Map.empty
            let mutable earlyError = None

            for v in nativeResults do
                if earlyError.IsNone then
                    match v.Value with
                    | :? DenseTensor<float32> as dt ->
                        let shape = dt.Dimensions.ToArray()
                        let data = dt.Buffer.ToArray()
                        resultMap <- Map.add v.Name (Tensor(shape, data)) resultMap
                    | null ->
                        earlyError <- Some (NativeInferenceError $"Node '{v.Name}' returned null data.")
                    | other ->
                        earlyError <- Some (InvalidTensorType (expected = "DenseTensor<float32>", actual = other.GetType().Name))

            match earlyError with
            | Some err -> Error err
            | None -> Ok resultMap

        with ex ->
            Error (NativeInferenceError ex.Message)
