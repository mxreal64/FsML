namespace FsML.Core

open System
open System.Buffers
open System.Runtime.InteropServices

/// Represents a pinned memory buffer that enables zero-copy interop with native C++/ONNX runtimes
type PinnedTensorBuffer<'T when 'T : unmanaged>(data: 'T[]) =
    let handle = GCHandle.Alloc(data, GCHandleType.Pinned)
    let mutable disposed = false

    member _.Data : 'T[] = data
    member _.Length : int = data.Length
    member _.Pointer : IntPtr = handle.AddrOfPinnedObject()
    member _.Span : Span<'T> = Span<'T>(handle.AddrOfPinnedObject().ToPointer(), data.Length)
    member _.ReadOnlySpan : ReadOnlySpan<'T> = ReadOnlySpan<'T>(handle.AddrOfPinnedObject().ToPointer(), data.Length)

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                handle.Free()
                disposed <- true

module Memory =

    /// Pins a managed array to native memory and safely executes a function before freeing
    let withPinned (data: 'T[]) (action: IntPtr -> 'R) : 'R =
        use pinned = new PinnedTensorBuffer<'T>(data)
        action pinned.Pointer

    /// Memory pool for temporary tensor buffers to minimize GC allocations during tight loops
    type ArrayPoolBuffer<'T>(size: int) =
        let array = ArrayPool<'T>.Shared.Rent(size)
        let mutable disposed = false

        member _.Array : 'T[] = array
        member _.Span : Span<'T> = Span<'T>(array, 0, size)
        member _.Size : int = size

        interface IDisposable with
            member _.Dispose() =
                if not disposed then
                    ArrayPool<'T>.Shared.Return(array)
                    disposed <- true

    /// Rents a buffer from the shared ArrayPool and returns an IDisposable wrapper
    let rentBuffer<'T> (size: int) : ArrayPoolBuffer<'T> =
        new ArrayPoolBuffer<'T>(size)

/// Result computation expression with robust resource management
type ResultBuilder() =
    member _.Bind(m: Result<'T, 'E>, f: 'T -> Result<'U, 'E>) : Result<'U, 'E> =
        Result.bind f m

    member _.Return(x: 'T) : Result<'T, 'E> = Ok x
    member _.ReturnFrom(m: Result<'T, 'E>) : Result<'T, 'E> = m
    member _.Zero() : Result<unit, 'E> = Ok ()

    member _.Using(resource: #IDisposable, body: #IDisposable -> Result<'T, 'E>) : Result<'T, 'E> =
        use r = resource
        body r

    member _.Delay(f: unit -> Result<'T, 'E>) : unit -> Result<'T, 'E> = f
    member _.Run(f: unit -> Result<'T, 'E>) : Result<'T, 'E> = f ()

    member _.Combine(a: Result<unit, 'E>, b: unit -> Result<'T, 'E>) : Result<'T, 'E> =
        match a with
        | Ok () -> b ()
        | Error e -> Error e

    member _.TryWith(f: unit -> Result<'T, 'E>, handler: exn -> Result<'T, 'E>) : Result<'T, 'E> =
        try f ()
        with ex -> handler ex

    member _.TryFinally(f: unit -> Result<'T, 'E>, compensation: unit -> unit) : Result<'T, 'E> =
        try f ()
        finally compensation ()

/// Scoped memory execution builder for deterministic lifecycle of tensors and native allocations
type ScopeBuilder() =
    member _.Return(x) = x
    member _.ReturnFrom(x) = x
    member _.Zero() = ()
    member _.Using(resource: #IDisposable, body: #IDisposable -> 'a) =
        use r = resource
        body r
    member _.Delay(f: unit -> 'a) = f
    member _.Run(f: unit -> 'a) = f ()
    member _.Combine(a: unit, b: unit -> 'a) = b ()
    member _.TryWith(f: unit -> 'a, handler: exn -> 'a) =
        try f ()
        with ex -> handler ex
    member _.TryFinally(f: unit -> 'a, compensation: unit -> unit) =
        try f ()
        finally compensation ()

[<AutoOpen>]
module Builders =
    let result = ResultBuilder()
    let scope = ScopeBuilder()
