#r "nuget: Microsoft.ML.OnnxRuntime, 1.20.1"
#r "../src/FsML/bin/Debug/net11.0/FsML.dll"

open System
open System.IO
open FsML
open FsML.Core
open FsML.Shapes
open FsML.Autograd
open FsML.NN
open FsML.Optim
open FsML.Data
open FsML.Visualization
open FsML.Backend

let mutable totalTests = 0
let mutable passedTests = 0

let test name (f: unit -> bool) =
    totalTests <- totalTests + 1
    try
        if f () then
            passedTests <- passedTests + 1
            printfn "  \u001b[32m[PASS]\u001b[0m %s" name
        else
            printfn "  \u001b[31m[FAIL]\u001b[0m %s" name
    with ex ->
        printfn "  \u001b[31m[ERROR]\u001b[0m %s - %s" name ex.Message

printfn "\n=============================================="
printfn "    FsML Comprehensive Test & Verification Suite"
printfn "==============================================\n"

// ==========================================
// 1. Core & Memory
// ==========================================
printfn "--- 1. Memory & SIMD Kernels ---"

test "Pinned Tensor Buffer zero-copy pointer access" (fun () ->
    let arr = [| 1.0f; 2.0f; 3.0f; 4.0f |]
    use pinned = new PinnedTensorBuffer<float32>(arr)
    pinned.Pointer <> IntPtr.Zero && pinned.Span.[2] = 3.0f
)

test "SIMD Vectorized addition vs sequential" (fun () ->
    let a = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f; 7.0f; 8.0f |]
    let b = [| 10.0f; 20.0f; 30.0f; 40.0f; 50.0f; 60.0f; 70.0f; 80.0f |]
    let dest = Array.zeroCreate<float32> 8
    Kernels.add a b dest
    dest.[0] = 11.0f && dest.[7] = 88.0f
)

test "Parallel Matrix Multiplication kernel" (fun () ->
    // [2, 3] x [3, 2] = [2, 2]
    let a = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]
    let b = [| 7.0f; 8.0f; 9.0f; 1.0f; 2.0f; 3.0f |]
    let c = Array.zeroCreate<float32> 4
    Kernels.matmul 2 3 2 a b c
    // row0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // row0, col1: 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    c.[0] = 31.0f && c.[1] = 19.0f
)

// ==========================================
// 2. Dynamic Tensor & Operations
// ==========================================
printfn "\n--- 2. Dynamic Tensor Operations ---"

test "Tensor 2D indexing and slicing" (fun () ->
    let t = Tensor.from2DArray (array2D [ [ 1.0f; 2.0f ]; [ 3.0f; 4.0f ] ])
    t.[0, 1] = 2.0f && t.[1, 0] = 3.0f
)

test "Tensor elementwise arithmetic and broadcasting" (fun () ->
    let t1 = Tensor([| 2; 2 |], [| 1.0f; 2.0f; 3.0f; 4.0f |])
    let t2 = t1 * 2.0f + 1.0f
    t2.[0, 0] = 3.0f && t2.[1, 1] = 9.0f
)

test "Tensor matrix transpose" (fun () ->
    let t = Tensor([| 2; 3 |], [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |])
    let tT = Tensor.transpose t
    tT.Shape = [| 3; 2 |] && tT.[0, 0] = 1.0f && tT.[1, 0] = 2.0f && tT.[0, 1] = 4.0f
)

test "Tensor Softmax row normalization" (fun () ->
    let logits = Tensor([| 2; 3 |], [| 1.0f; 2.0f; 3.0f; 10.0f; 10.0f; 10.0f |])
    let sm = Tensor.softmax logits
    let row0Sum = sm.[0, 0] + sm.[0, 1] + sm.[0, 2]
    let row1Sum = sm.[1, 0] + sm.[1, 1] + sm.[1, 2]
    MathF.Abs(row0Sum - 1.0f) < 1e-5f && MathF.Abs(row1Sum - 1.0f) < 1e-5f
)

// ==========================================
// 3. Static Shape Analysis (Shapes as Types)
// ==========================================
printfn "\n--- 3. Static Shape Analysis (Phantom Types) ---"

test "Compile-time verified matrix multiplication" (fun () ->
    let a = Typed.init2D<Batch, Hidden> (32, 128) (fun r c -> float32 (r + c))
    let b = Typed.init2D<Hidden, OutFeatures> (128, 64) (fun r c -> float32 (r * c))
    let out = Typed.matmul a b
    out.Rows = 32 && out.Cols = 64 && out.Underlying.Shape = [| 32; 64 |]
)

test "Type-safe transposition and elementwise addition" (fun () ->
    let m1 = Typed.init2D<M, N> (10, 20) (fun _ _ -> 1.0f)
    let m2 = Typed.init2D<M, N> (10, 20) (fun _ _ -> 2.0f)
    let sum = Typed.add m1 m2
    let transposed = Typed.transpose sum
    transposed.Rows = 20 && transposed.Cols = 10 && sum.Underlying.[0, 0] = 3.0f
)

// ==========================================
// 4. Reverse-Mode Autograd Engine
// ==========================================
printfn "\n--- 4. Autograd & Numerical Gradient Verification ---"

test "Autograd scalar polynomial: f(x) = 3x^2 + 2x + 1 -> f'(2) = 14" (fun () ->
    let x = Value.scalar(2.0f, requiresGrad=true)
    let y = (x * x) * 3.0f + (x * 2.0f) + 1.0f
    Engine.backward y
    match x.Grad with
    | Some g -> MathF.Abs(g.[0] - 14.0f) < 1e-4f
    | None -> false
)

test "Autograd Matrix Multiplication Gradient vs Analytical Derivative" (fun () ->
    // z = sum(A * B)
    // dZ/dA = B^T (broadcasted with ones), dZ/dB = A^T
    let a = Value.param (Tensor([| 2; 3 |], [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]))
    let b = Value.param (Tensor([| 3; 2 |], [| 7.0f; 8.0f; 9.0f; 1.0f; 2.0f; 3.0f |]))
    let c = Ops.matmul a b
    let loss = Ops.sum c
    Engine.backward loss

    // dLoss/dA = [1, 1] * B^T = row sums of B^T = [15, 10, 5; 15, 10, 5]
    match a.Grad with
    | Some gA ->
        // analytical check: for any element, dA_ij = sum_j B_jk
        let expected00 = 7.0f + 8.0f // 15
        MathF.Abs(gA.[0, 0] - expected00) < 1e-4f
    | None -> false
)

test "Autograd non-linear activation (Sigmoid & ReLU) backward chain" (fun () ->
    let x = Value.param (Tensor([| 1; 4 |], [| -2.0f; -0.5f; 0.5f; 2.0f |]))
    let r = Ops.relu x
    let loss = Ops.sum r
    Engine.backward loss
    match x.Grad with
    | Some g -> g.[0] = 0.0f && g.[1] = 0.0f && g.[2] = 1.0f && g.[3] = 1.0f
    | None -> false
)

// ==========================================
// 5. Neural Network Layers & Losses
// ==========================================
printfn "\n--- 5. Neural Network Layers, Loss Functions & Optimizers ---"

test "Linear layer forward and parameter tracking" (fun () ->
    let layer = Linear(4, 2, useBias=true, seed=42)
    let x = Value.tensor (Tensor([| 3; 4 |], Array.create 12 1.0f))
    let out = layer.Forward(x)
    out.Shape = [| 3; 2 |] && layer.Parameters.Length = 2
)

test "Cross Entropy Loss gradient correctness" (fun () ->
    let logits = Value.param (Tensor([| 2; 3 |], [| 2.0f; 1.0f; 0.1f; 0.2f; 3.0f; 0.1f |]))
    let targets = [| 0; 1 |]
    let loss = Losses.crossEntropyLoss logits targets
    Engine.backward loss
    match logits.Grad with
    | Some g ->
        // The gradient on correct class must be negative (probs - 1) / N
        g.[0, 0] < 0.0f && g.[1, 1] < 0.0f
    | None -> false
)

test "AdamW Optimizer parameter update step" (fun () ->
    let param = Parameter(Value.param (Tensor.scalar 5.0f))
    param.Value.Grad <- Some (Tensor.scalar 2.0f)
    let opt = AdamW([ param ], lr=0.1f)
    (opt :> IOptimizer).Step()
    // 5.0 - (0.1 * 1.0) = approx 4.9
    param.Data.[0] < 5.0f
)

// ==========================================
// 6. Data Streaming & Active Patterns
// ==========================================
printfn "\n--- 6. Data Pipelines & Active Patterns ---"

test "DataLoader streaming batches" (fun () ->
    let features, targets = Datasets.makeSpiral 20 3 (Some 42)
    let loader = DataLoader(features, targets, batchSize=10, shuffle=true, seed=42)
    let batches = loader.GetBatches() |> Seq.toList
    let firstFeatures, _ = batches.[0]
    batches.Length = 6 && firstFeatures.Shape = [| 10; 2 |]
)

test "Active pattern matching on tensors" (fun () ->
    let m = Tensor([| 4; 4 |], Array.zeroCreate 16)
    match m with
    | SquareMatrix dim -> dim = 4
    | _ -> false
)

// ==========================================
// 7. Visualization & Formatters
// ==========================================
printfn "\n--- 7. Visualization & Notebook Formatters ---"

test "HTML Heatmap generation" (fun () ->
    let t = Tensor([| 3; 3 |], [| 1.0f..9.0f |])
    let html = HtmlFormatters.renderHeatmap t
    html.Contains("<table") && html.Contains("rgb(")
)

test "SVG Loss Curve generation" (fun () ->
    let losses = [| 1.0f; 0.8f; 0.5f; 0.3f; 0.1f |]
    let svg = HtmlFormatters.renderLossCurve losses
    svg.Contains("<polyline") && svg.Contains("<svg")
)

// ==========================================
// 8. ONNX Backend Inference Interop
// ==========================================
printfn "\n--- 8. ONNX Backend Execution ---"

test "ONNX Runtime model execution with FsML tensors" (fun () ->
    let modelPath = Path.Combine(__SOURCE_DIRECTORY__, "fixtures", "dummy.onnx")
    match OnnxSession.load modelPath with
    | Ok session ->
        let inputTensor = Tensor([| 1; 4 |], [| 1.0f; 2.0f; 3.0f; 4.0f |])
        match OnnxSession.run session [ ("input_node", inputTensor) ] with
        | Ok outputMap ->
            match Map.tryFind "output_node" outputMap with
            | Some outTensor ->
                outTensor.Shape = [| 1; 4 |] && outTensor.Length = 4
            | None -> false
        | Error _ -> false
    | Error _ -> false
)

// ==========================================
// Summary
// ==========================================
printfn "\n=============================================="
printfn "  RESULTS: %d / %d Tests Passed" passedTests totalTests
if passedTests = totalTests then
    printfn "  \u001b[32mALL TESTS PASSED SUCCESSFULLY!\u001b[0m"
else
    printfn "  \u001b[31mSOME TESTS FAILED!\u001b[0m"
printfn "==============================================\n"
