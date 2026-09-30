namespace FsML.Visualization

open System
open System.Text
open FsML
open FsML.Autograd

module HtmlFormatters =

    /// Renders a 2D Tensor as an interactive HTML Heatmap with a custom title
    let renderHeatmapWithTitle (title: string) (t: Tensor) : string =
        let r, c =
            if t.Rank = 2 then t.Shape.[0], t.Shape.[1]
            elif t.Rank = 1 then 1, t.Shape.[0]
            else 1, t.Length

        let minVal = Tensor.min t
        let maxVal = Tensor.max t
        let range = max (maxVal - minVal) 1e-6f

        let sb = StringBuilder()
        sb.AppendLine("<div style=\"font-family: system-ui, -apple-system, sans-serif; margin: 12px 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; background: #fafafa; box-shadow: 0 2px 4px rgba(0,0,0,0.05);\">") |> ignore
        sb.AppendLine(sprintf "<div style=\"font-weight: 600; font-size: 14px; margin-bottom: 8px; color: #202124;\">%s</div>" title) |> ignore
        sb.AppendLine("<div style=\"overflow-x: auto; max-height: 400px;\">") |> ignore
        sb.AppendLine("<table style=\"border-collapse: collapse; font-size: 11px; text-align: center;\">") |> ignore

        let maxRows = min (r - 1) 40
        let maxCols = min (c - 1) 40

        for i in 0 .. maxRows do
            sb.AppendLine("<tr>") |> ignore
            for j in 0 .. maxCols do
                let v = if t.Rank = 2 then t.[i, j] else t.[j]
                let normalized = (v - minVal) / range
                let red = int (normalized * 220.0f + 30.0f)
                let blue = int ((1.0f - normalized) * 220.0f + 30.0f)
                let bg = sprintf "rgb(%d, 70, %d)" red blue
                let textColor = if normalized > 0.4f && normalized < 0.8f then "#000" else "#fff"

                sb.AppendLine(sprintf "<td style=\"padding: 4px 6px; min-width: 38px; background: %s; color: %s; border: 1px solid rgba(255,255,255,0.2); font-mono: true;\" title=\"[%d,%d] = %f\">%.3f</td>" bg textColor i j v v) |> ignore
            sb.AppendLine("</tr>") |> ignore

        sb.AppendLine("</table>") |> ignore
        sb.AppendLine("</div>") |> ignore
        sb.AppendLine(sprintf "<div style=\"font-size: 11px; color: #5f6368; margin-top: 6px;\">Min: %.4f | Max: %.4f | Mean: %.4f</div>" minVal maxVal (Tensor.mean t)) |> ignore
        sb.AppendLine("</div>") |> ignore
        sb.ToString()

    /// Renders a 2D Tensor as an interactive HTML Heatmap with default title
    let renderHeatmap (t: Tensor) : string =
        let shapeStr = String.Join(", ", t.Shape)
        let title = sprintf "Tensor Heatmap (Shape: [%s])" shapeStr
        renderHeatmapWithTitle title t

    /// Renders summary statistics card for any Tensor
    let renderSummaryCard (t: Tensor) : string =
        let meanVal = Tensor.mean t
        let minVal = Tensor.min t
        let maxVal = Tensor.max t
        let zerosCount = t.Data |> Array.filter (fun v -> MathF.Abs(v) < 1e-7f) |> Array.length
        let sparsity = (float32 zerosCount / float32 t.Length) * 100.0f
        let shapeStr = String.Join(", ", t.Shape)

        let sb = StringBuilder()
        sb.AppendLine("<div style=\"display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; font-family: system-ui, sans-serif; padding: 12px; background: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px;\">") |> ignore
        sb.AppendLine(sprintf "<div style=\"padding: 8px; background: white; border-radius: 6px; text-align: center; border: 1px solid #eee;\"><div style=\"font-size: 11px; color: #70757a;\">Shape</div><div style=\"font-size: 15px; font-weight: bold; color: #1a73e8;\">[%s]</div></div>" shapeStr) |> ignore
        sb.AppendLine(sprintf "<div style=\"padding: 8px; background: white; border-radius: 6px; text-align: center; border: 1px solid #eee;\"><div style=\"font-size: 11px; color: #70757a;\">Mean</div><div style=\"font-size: 15px; font-weight: bold; color: #202124;\">%.4f</div></div>" meanVal) |> ignore
        sb.AppendLine(sprintf "<div style=\"padding: 8px; background: white; border-radius: 6px; text-align: center; border: 1px solid #eee;\"><div style=\"font-size: 11px; color: #70757a;\">Min / Max</div><div style=\"font-size: 14px; font-weight: bold; color: #202124;\">[%.2f, %.2f]</div></div>" minVal maxVal) |> ignore
        sb.AppendLine(sprintf "<div style=\"padding: 8px; background: white; border-radius: 6px; text-align: center; border: 1px solid #eee;\"><div style=\"font-size: 11px; color: #70757a;\">Sparsity</div><div style=\"font-size: 15px; font-weight: bold; color: #ea4335;\">%.1f%%</div></div>" sparsity) |> ignore
        sb.AppendLine("</div>") |> ignore
        sb.ToString()

    /// Renders training loss curve as a responsive standalone SVG
    let renderLossCurveCustom (width: int) (height: int) (title: string) (losses: float32[]) : string =
        let w = width
        let h = height
        if losses.Length < 2 then "<svg></svg>"
        else
            let minLoss = losses |> Array.min
            let maxLoss = losses |> Array.max
            let range = max (maxLoss - minLoss) 1e-6f

            let padding = 40
            let plotW = w - padding * 2
            let plotH = h - padding * 2

            let points = StringBuilder()
            for i in 0 .. losses.Length - 1 do
                let x = padding + int (float32 i / float32 (losses.Length - 1) * float32 plotW)
                let normY = (losses.[i] - minLoss) / range
                let y = h - padding - int (normY * float32 plotH)
                points.Append(sprintf "%d,%d " x y) |> ignore

            let sb = StringBuilder()
            sb.AppendLine("<div style=\"font-family: system-ui, sans-serif; background: #ffffff; padding: 12px; border: 1px solid #dadce0; border-radius: 8px;\">") |> ignore
            sb.AppendLine(sprintf "<div style=\"font-weight: 600; font-size: 13px; margin-bottom: 6px;\">%s</div>" title) |> ignore
            sb.AppendLine(sprintf "<svg width=\"%d\" height=\"%d\" style=\"overflow: visible;\">" w h) |> ignore
            sb.AppendLine(sprintf "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#9aa0a6\" stroke-width=\"1.5\" />" padding (h - padding) (w - padding) (h - padding)) |> ignore
            sb.AppendLine(sprintf "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"#9aa0a6\" stroke-width=\"1.5\" />" padding padding padding (h - padding)) |> ignore
            sb.AppendLine(sprintf "<polyline fill=\"none\" stroke=\"#1a73e8\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" points=\"%s\" />" (points.ToString().Trim())) |> ignore
            sb.AppendLine(sprintf "<text x=\"%d\" y=\"%d\" font-size=\"10\" fill=\"#5f6368\">%.4f</text>" padding (padding - 6) maxLoss) |> ignore
            sb.AppendLine(sprintf "<text x=\"%d\" y=\"%d\" font-size=\"10\" fill=\"#5f6368\">%.4f</text>" padding (h - padding + 15) minLoss) |> ignore
            sb.AppendLine(sprintf "<text x=\"%d\" y=\"%d\" font-size=\"10\" fill=\"#5f6368\" text-anchor=\"end\">Epoch %d</text>" (w - padding) (h - padding + 15) losses.Length) |> ignore
            sb.AppendLine("</svg>") |> ignore
            sb.AppendLine("</div>") |> ignore
            sb.ToString()

    /// Renders training loss curve with default dimensions (500x200)
    let renderLossCurve (losses: float32[]) : string =
        renderLossCurveCustom 500 200 "Training Loss Curve" losses
