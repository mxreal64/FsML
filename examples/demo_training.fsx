#r "../src/FsML/bin/Debug/net11.0/FsML.dll"

open System
open System.IO
open FsML
open FsML.Autograd
open FsML.NN
open FsML.Optim
open FsML.Data
open FsML.Visualization

printfn "==============================================================="
printfn "  FsML End-to-End Neural Network Training (Non-Linear Spiral)"
printfn "===============================================================\n"

// 1. Generate Non-Linear Spiral Dataset
let numSamplesPerClass = 100
let numClasses = 3
let features, targets = Datasets.makeSpiral numSamplesPerClass numClasses (Some 42)
printfn "[1] Dataset Created: %d samples, %d features, %d classes." features.Shape.[0] features.Shape.[1] numClasses

// 2. Define Multi-Layer Perceptron (MLP)
// Architecture: 2 -> 32 (ReLU) -> 32 (ReLU) -> 3 (Logits)
let l1 = Linear(2, 32, seed=1)
let l2 = Linear(32, 32, seed=2)
let l3 = Linear(32, 3, seed=3)
let model = Sequential([ l1; l2; l3 ])

// Functional forward pass with non-linear activations
let forward (x: Value) : Value =
    x
    |> l1.Forward
    |> Ops.relu
    |> l2.Forward
    |> Ops.relu
    |> l3.Forward

printfn "[2] Model Architecture: 2 -> 32 (ReLU) -> 32 (ReLU) -> 3"
printfn "    Total Parameters: %d weight & bias tensors" model.Parameters.Length

// 3. Setup Optimizer and Data Loader
let optimizer = AdamW(model.Parameters, lr=0.05f, weightDecay=0.001f) :> IOptimizer
let batchSize = 64
let dataLoader = DataLoader(features, targets, batchSize=batchSize, shuffle=true, seed=42)
let numEpochs = 60
let lossHistory = Array.zeroCreate<float32> numEpochs

printfn "    Training for %d epochs with AdamW (batch size = %d)...\n" numEpochs batchSize

// 4. Training Loop
for epoch in 1 .. numEpochs do
    let mutable epochLossSum = 0.0f
    let mutable numBatches = 0

    for (batchX, batchY) in dataLoader.GetBatches() do
        optimizer.ZeroGrad()

        let xVal = Value.tensor batchX
        let logits = forward xVal
        let loss = Losses.crossEntropyLoss logits batchY

        Engine.backward loss
        optimizer.Step()

        epochLossSum <- epochLossSum + loss.Data.[0]
        numBatches <- numBatches + 1

    let avgEpochLoss = epochLossSum / float32 numBatches
    lossHistory.[epoch - 1] <- avgEpochLoss

    if epoch % 10 = 0 || epoch = 1 then
        printfn "    Epoch %3d / %d | Mean Cross-Entropy Loss: %.4f" epoch numEpochs avgEpochLoss

// 5. Evaluate Full-Dataset Accuracy
let allX = Value.tensor features
let finalLogits = forward allX
let predictedClasses = Tensor.argmax 1 finalLogits.Data

let mutable correct = 0
for i in 0 .. targets.Length - 1 do
    if predictedClasses.[i] = targets.[i] then
        correct <- correct + 1

let accuracy = (float32 correct / float32 targets.Length) * 100.0f
printfn "\n[4] Training Complete!"
printfn "    Final Classification Accuracy: %.2f%% (%d/%d)" accuracy correct targets.Length

// 6. Generate Rich HTML Training Report
let lossPlotHtml = HtmlFormatters.renderLossCurve lossHistory
let statsCardHtml = HtmlFormatters.renderSummaryCard finalLogits.Data
let heatmapHtml = HtmlFormatters.renderHeatmap finalLogits.Data

let fullReport =
    sprintf """<!DOCTYPE html>
<html>
<head>
    <title>FsML Training Report</title>
    <style>body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }</style>
</head>
<body>
    <h1>FsML Training Report</h1>
    <p><b>Model:</b> 3-Layer MLP (2 &rarr; 32 &rarr; 32 &rarr; 3) with AdamW Optimizer</p>
    <p><b>Final Accuracy:</b> <span style="color: #0f9d58; font-size: 18px; font-weight: bold;">%.2f%%</span></p>

    <h2>Training Loss Curve</h2>
    %s

    <h2>Logits Summary Statistics</h2>
    %s

    <h2>Sample Predictions Heatmap</h2>
    %s
</body>
</html>""" accuracy lossPlotHtml statsCardHtml heatmapHtml

File.WriteAllText("training_report.html", fullReport)
let reportPath = Path.GetFullPath("training_report.html")
printfn "[5] Generated HTML Report: file://%s" reportPath
printfn "===============================================================\n"
