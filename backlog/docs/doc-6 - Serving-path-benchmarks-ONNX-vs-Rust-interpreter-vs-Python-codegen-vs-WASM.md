---
id: doc-6
title: 'Serving-path benchmarks: ONNX vs Rust interpreter vs Python codegen vs WASM'
type: guide
created_date: '2026-07-19 00:48'
tags:
  - benchmark
  - performance
  - wasm
  - onnx
  - codegen
  - serving
---
## Context — why these benchmarks exist

Two project goals: ergonomic SQL to transformer authoring, and **fast inference**. Open questions this
session set out to answer with hard numbers instead of intuition:

1. Where does inference cost actually live — compute, or the boundaries (pydantic/FFI)?
2. How do the candidate serving paths compare: the **Rust interpreter** (`InferFn`, pyo3), the
   **Python codegen** (`CodegenFn`), **ONNX Runtime**, and a **WASM** kernel?
3. Is WASM a legit *multi-language serving output* — one artifact across Python/Go/Java, no per-language FFI?

**Status: exploratory spike, not production measurement.** Everything below is a microbenchmark of one
tiny numeric transform on a laptop-class box with background noise. Treat the *order-of-magnitude gaps*
as the finding; do not quote the exact nanoseconds as SLAs. Run-to-run variance is real (see Caveats).

---

## TL;DR — the implications

- **Inference is boundary-bound, confirmed.** The pydantic in/out boundary dominates both the Rust
  interpreter and the Python codegen: **~4,000–4,500 ns/row batched**, vs **single-digit ns/row** for
  paths that take raw columnar arrays (ONNX/WASM/numpy). ~1,000x gap, and it is *not* the arithmetic.
- **Python codegen is about equal to the Rust interpreter** for the pydantic contract (codegen slightly
  ahead). Both are pinned by the same pydantic boundary, so swapping the interpreter for codegen barely
  moves batched throughput. The lever is owning the columnar path, not the engine internals.
- **ONNX and WASM both hit single-digit ns/row batched** when fed raw columns. ONNX wins in Python
  (numpy zero-copy); WASM matches or beats it in Go/Java with a **309-byte artifact and no FFI**.
- **n=1 is where they split hard.** No-FFI WASM runtimes: **Go ~48 ns, Java-AOT ~24 ns**. ONNX at n=1:
  **14–29 microseconds** — a ~300-800x gap, entirely ONNX's per-Run() session/binding tax.
- **Two traps, measured:** chicory *interpreter* is ~57x slower than its own AOT (always enable the AOT
  compiler); and the pydantic boundary (~4,000+ ns/row) does not amortize with batch size.

---

## The shared workload (every benchmark ran this)

SQL (post-fit, scaler/feature-math shaped — the smallest realistic numeric transform):

```
SELECT age / 271.0 AS age_r, amount / 100.0 AS amt_b,
       amount - age AS d, amount + age AS s,
       age * 2.0 AS a2, amount * 0.5 AS h
FROM t
```

2 f64 inputs (amount, age) to 6 f64 outputs. Inputs: 10,000 deterministic rows. **Parity gate:** every
engine's output is `allclose(rtol=1e-9)` to the hand-computed arithmetic *before* any timing is taken.

**WASM kernel** (`kernel.wat`, hand-written, 6 f64 ops over linear memory; assembled to a **309-byte**
`.wasm` via `wasmtime.wat2wasm`; the SAME .wasm is loaded by all three host runtimes):

```
(func (export "infer") (param $amount i32)(param $age i32)(param $n i32)(param $out i32)
  loop i in 0..n:
    a = f64.load amount+8i ;  g = f64.load age+8i
    out[0n+i]=g/271 ; out[1n+i]=a/100 ; out[2n+i]=a-g
    out[3n+i]=a+g   ; out[4n+i]=g*2   ; out[5n+i]=a*0.5 )
```

NOTE: the WAT was **hand-written**, not emitted by codegen. It proves the runtime/serving layer, NOT a
SQL to wasm compiler.

**ONNX model** (`model.onnx`, 367 bytes, same 6 ops as a graph): inputs amount,age (DOUBLE,[None]);
Div/Div/Sub/Add/Mul/Mul with constant initializers 271,100,2,0.5; 6 DOUBLE outputs. Same file loaded by
Python, Go, and Java ORT.

---

## Settings (common)

- **Machine:** Intel i5-12400F (6 P-cores / 12 threads), Windows 11 Pro build 26200. Not core-pinned,
  not isolated; laptop-class background load.
- **Sizes:** n=1 (online latency) and N=10,000 (batch). `batch/row` = per-call batch time / 10,000.
- **ONNX:** 1 intra-op and 1 inter-op thread (single-threaded, to compare kernels not thread pools).
- **Timing methods (they differ — this matters for n=1):**
  - Python `bench()`: warmup 50, then **median of single `perf_counter()` samples** (n=1: 3000 samples;
    batch: 300 samples).
  - Go/Java `bench()`: warmup 100–200, then **best-of-7 of amortized loops** (total wall / iters; n=1
    iters 200k, batch iters 3k).
  - Consequence: Python n=1 includes per-call timer overhead; Go/Java n=1 is amortized best. **n=1 is not
    strictly cross-host comparable; batch/row is** (both amortize over 10k rows).
- **Versions:** Python 3.14.0; onnxruntime 1.27.0; wasmtime 46.0.1; numpy 2.5.1; pydantic 2.13.4;
  sqlglot 30.12.0. Go 1.26.5; wazero 1.12.0; onnxruntime_go 1.31.0 (cgo, compiled with **zig 0.16.0 as
  `zig cc`** — no MSVC/mingw); onnxruntime.dll 1.27.x reused from the Python venv. Java Temurin 21.0.11;
  chicory 1.4.0; onnxruntime(java) 1.22.0; maven 3.9.16. JVM/Maven/zig installed user-local via **mise**.

---

## What each benchmark actually did (sketches)

**Python** (`bench_mech.py`): rows are 10k pydantic `Row(amount, age)`; a,g are 10k float64 numpy arrays.
- rust-interp: `fn.infer({"t": rows})` — `InferFn`, pydantic in, validated pydantic out
- codegen: `cg.infer({"t": rows})` — `CodegenFn`, identical contract
- onnx: `sess.run(cols, {"amount": a, "age": g})` — numpy arrays in/out, 1 thread
- wasm: write a,g into wasmtime linear memory, `infer(...)`, read 6 cols back (includes the copy)
- numpy: `np.stack([g/271, a/100, a-g, a+g, g*2, a*0.5])` — pure-kernel reference

**Go** (`gobench/`, `goonnx/`):
- wazero: `NewRuntimeConfigCompiler()`; write cols via `mem.WriteFloat64Le`; `infer.Call(...)` (incl copy).
  A hand-written pure-Go loop is the native ceiling.
- onnxruntime_go: `ort.SetSharedLibraryPath(venv onnxruntime.dll)`; `DynamicAdvancedSession(model.onnx)`;
  `NewTensor` per call; `sess.Run`. cgo, built with `zig cc`.

**Java** (`javabench/`, `javaonnx/`):
- chicory interpreter: `Instance.builder(module).build()`
- chicory AOT: `Instance.builder(module).withMachineFactory(MachineFactoryCompiler::compile).build()`
  (both write cols to `Memory` via little-endian `ByteBuffer`, then `infer.apply(...)`)
- onnxruntime(java): `OrtEnvironment`; `createSession(model.onnx)`; `OnnxTensor.createTensor` per call;
  `session.run`. Native libs bundled in the Maven jar (no DLL wrangling). 1 intra/inter-op thread.

---

## Results — batch (10k rows), per-row (cross-host comparable)

| Host | Rust interp (InferFn) | Python codegen | ONNX | WASM | native ceiling |
|---|---|---|---|---|---|
| Python | 4,525 ns (pydantic) | 3,997 ns (pydantic) | **3.8 ns** | 8.7 ns (wasmtime) | 3.9 ns (numpy) |
| Go | — | — | 10.3 ns | **8.6 ns** (wazero) | 2.3 ns (pure-Go) |
| Java | — | — | 14.7 ns | **8.3 ns** (chicory AOT) / 472.8 (interp) | — |

## Results — n=1 (see method caveat; not strictly cross-host)

| Host | Rust interp | codegen | ONNX | WASM |
|---|---|---|---|---|
| Python | 4.4 us | 3.8 us | 14.3 us | 39.7 us* |
| Go | — | — | 20.1 us | ~0.048 us |
| Java | — | — | 29.4 us | ~0.024 us (AOT) |

\* Python wasm n=1 is dominated by the per-call numpy to linear-memory copy in the harness, not the wasm kernel.

---

## Caveats — what these numbers are NOT

1. **Asymmetric boundaries by design.** Rust-interp and codegen include the pydantic in/out boundary;
   ONNX/WASM/numpy take raw arrays. That asymmetry IS the finding ("native is slow" = the boundary, not
   the kernel), but it means these are not same-boundary comparisons.
2. **n=1 methodology differs across hosts** (Python median-of-singles vs Go/Java amortized best-of-7).
   Batch/row is comparable; n=1 cross-host is not.
3. **Run-to-run variance ~2.6x on the pydantic-bound paths** (an earlier, more-loaded run measured
   native/codegen batch at ~11,700 ns/row vs ~4,000–4,500 here). Raw-array paths were stable (~3–10 ns/row).
   Order-of-magnitude conclusions hold; exact ns do not.
4. **Go pure-kernel n=1 was dead-code-eliminated** by the compiler (reads 0); only its batch/row (2.3 ns)
   is meaningful.
5. **This is 6 arithmetic ops on non-null f64.** It does NOT exercise strings, NULLs, casts, joins,
   lookups, or real fitted transforms — the WASM/columnar fast path only covers the typed numeric subset.

---

## Implications for engine strategy

- **Native/Rust `InferFn` stays as the parity oracle** (DataFusion is the oracle) regardless of serving
  path. It is not the thing to optimize for throughput — its batched cost is the pydantic boundary,
  shared with codegen.
- **Python codegen does not beat native meaningfully under the pydantic contract.** The real lever is
  owning the columnar path (eliminate the per-row pydantic boundary), consistent with the boundary-bound
  thesis.
- **WASM is a credible multi-language serving OUTPUT for the typed numeric subset:** one 309-byte
  artifact ran at 8.3–8.7 ns/row across Python/Go/Java with no FFI, and crushed ONNX at n=1 (no
  session/binding tax). The open risk is the compiler — a SQL to wasm codegen for the numeric subset,
  reusing the existing codegen front-end (IR to WAT string emit + `wat2wasm`). The full DataFusion
  surface in wasm (strings/nulls/joins) is a separate, larger bet, not required for the numeric win.
- **ONNX only wins batched-and-Python-hosted** (numpy zero-copy); its per-Run() floor makes it a poor
  fit for n=1 online serving.

---

## Reproduction

All sources live in the session scratchpad (not committed):
- `bench_mech.py` — Python: InferFn / CodegenFn / onnxruntime / wasmtime / numpy
- `kernel.wat`, `kernel.wasm` — hand-written 6-op WASM kernel (wat2wasm)
- `model.onnx` — identical 6-op ONNX graph
- `gobench/` — Go wazero + pure-Go reference
- `goonnx/` — Go onnxruntime_go (cgo via zig cc; loads the venv onnxruntime.dll)
- `javabench/` — Java chicory interpreter + AOT
- `javaonnx/` — Java onnxruntime

Tooling installed user-local via **mise**: java (Temurin 21.0.11), maven 3.9.16, zig 0.16.0. Python deps
(onnxruntime, wasmtime, onnx, numpy) added to the project `.venv`. The ONNX native lib for Go was the
`onnxruntime.dll` already bundled in the venv — no separate system install.
