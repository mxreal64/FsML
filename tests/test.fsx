#r "nuget: Microsoft.ML.OnnxRuntime, 1.20.1"
#r "../src/FsML/bin/Debug/net11.0/FsML.dll"

open System
open System.IO
open FsML
open FsML.Core
open FsML.Backend

let dummyModelPath = Path.Combine(__SOURCE_DIRECTORY__, "fixtures", "dummy.onnx")

/// Utility to print test results cleanly
let assertTest name result =
    match result with
    | Ok v -> printfn $"[PASS] %s{name}\n       Result: %A{v}"
    | Error e -> printfn $"[FAIL/ERROR] %s{name}\n       Message: %A{e}"

let assertExpectedError name expectedError result =
    match result with
    | Error e when e.GetType() = expectedError.GetType() ->
        printfn $"[PASS] %s{name} (Caught expected error: %A{e})"
    | Ok _ -> printfn $"[FAIL] %s{name} (Expected error, but succeeded)"
    | Error e -> printfn $"[FAIL] %s{name} (Wrong error type: %A{e})"

// ==========================================
// Test 1: The Happy Path (End-to-End Pipeline)
// ==========================================
let testHappyPath () =
    result {
        use! session = OnnxSession.load dummyModelPath

        let inputs = [
            ("input_node", Tensor([| 1; 4 |], [| 1.0f; 2.0f; 3.0f; 4.0f |]))
        ]

        let! outputs = OnnxSession.run session inputs
        match Map.tryFind "output_node" outputs with
        | Some outTensor -> return outTensor.ToArray()
        | None -> return! Error (MissingOutputNode "output_node")
    }

// ==========================================
// Test 2: Missing File Trapping
// ==========================================
let testMissingFile () =
    OnnxSession.load "non_existent_model.onnx"

// ==========================================
// Test 3: Missing Output Node Trapping
// ==========================================
let testMissingNode () =
    result {
        use! session = OnnxSession.load dummyModelPath

        let inputs = [
            ("input_node", Tensor([| 1; 4 |], [| 1.0f; 2.0f; 3.0f; 4.0f |]))
        ]

        let! outputs = OnnxSession.run session inputs
        match Map.tryFind "wrong_node_name" outputs with
        | Some _ -> return ()
        | None -> return! Error (MissingOutputNode "wrong_node_name")
    }

printfn "--- RUNNING FsML ONNX BACKEND TESTS ---\n"

// Run Happy Path
testHappyPath () |> assertTest "End-to-End Inference & Memory Extraction"

// Run Expected Errors
testMissingFile () |> assertExpectedError "Missing Model File" (ModelLoadError "")
testMissingNode () |> assertExpectedError "Missing Output Node" (MissingOutputNode "")

printfn "\n--- TESTS COMPLETE ---"
