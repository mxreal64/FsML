namespace FsML.Autograd

open System.Collections.Generic
open FsML

module Engine =

    /// Builds a reverse topological ordering of the computation graph starting from the root node
    let buildTopo (root: Value) : Value list =
        let topo = List<Value>()
        let visited = HashSet<int64>()

        let rec dfs (v: Value) =
            if not (visited.Contains(v.Id)) then
                visited.Add(v.Id) |> ignore
                for p in v.Parents do
                    dfs p
                topo.Add(v)

        dfs root
        topo |> Seq.toList

    /// Computes reverse-mode automatic differentiation (backpropagation) over the entire computation DAG
    let backward (root: Value) : unit =
        let topo = buildTopo root

        // Initialize root gradient to 1.0 (matching root tensor shape)
        root.Grad <- Some (Tensor.ones root.Shape)

        // Traverse in reverse topological order (from outputs to inputs)
        for i in topo.Length - 1 .. -1 .. 0 do
            let node = topo.[i]
            node.Backward()

    /// Resets gradients of all nodes connected in the subgraph starting from root
    let zeroGrad (root: Value) : unit =
        let topo = buildTopo root
        for node in topo do
            node.ZeroGrad()

    /// Extracts all leaf parameters requiring gradients from a computation graph
    let getParameters (root: Value) : Value list =
        let topo = buildTopo root
        topo |> List.filter (fun v -> v.RequiresGrad && v.OpName = "param")
