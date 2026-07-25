# SQL Specializer for ML Model Serving — Design

**Status:** draft for review
**Replaces the framing of:** the "DuckDB native interpreter" stub (this is not an
interpreter; it is a partial evaluator that emits one)
**Relates to:** `2026-07-14-sqltransform-rust-backend-design.md` (the current
row-at-a-time engine), `2026-07-17-codegen-inferfn-design.md` (the Python codegen
engine), the boundary-cost benchmark (memory: inference is boundary-bound).

## 1. What this is

A prepare-once / run-millions engine:

```
prepare(sql, static_tables) -> f          # slow path, seconds are fine
f(batch) -> batch                         # hot path, ns/call budget, allocation-free
```

The first Futamura projection applied to a query: the query and every static
relation are inputs to a *specializer*, and what falls out is a native function
whose only remaining variable is the request payload (`__THIS__`, n ≈ 1–100
rows). This matches exactly what `SQLTransform.fit()` already produces
conceptually — frozen state + a callable — but replaces "interpret a plan
against the frozen state" with "the frozen state was compiled into the code".

### How it maps onto the existing Python API

Same surface as `SQLTransform` minus transformer refs (v0 explicitly excludes
the transformer callout):

```python
t = SpecializedTransform("SELECT age / mean_age AS z FROM __THIS__ JOIN stats ON ...")
t.fit(train_table)          # STAGE 0+1: extract state, prepare, compile
t.infer_batch(rows)         # STAGE 2: run(in, out, scratch)
```

## 2. Restatement of the architecture (compressed)

Binding-time separation is the invariant:

- **STAGE 0 (load):** static tables → immutable, indexed.
- **STAGE 1 (prepare):** SQL → bind → relational IR → rewrite → **binding-time
  analysis** → static subtrees evaluated NOW and replaced by constants (scalars,
  perfect-hash tables, packed arrays) → produce/consume lowering of the dynamic
  frontier → imperative IR → backend (interpreter or codegen).
- **STAGE 2 (execute):** `void run(const Batch* in, Batch* out, Arena* scratch)`
  — columnar, caller-owned, `|out|` known up front, zero malloc/syscalls.

Two IRs (relational: rewrites + BTA annotation; imperative: SSA, typed, explicit
null lane, `StaticRef(id)` handles, verifier + round-trippable text format). Two
backends behind one interface (closure-compiled interpreter as oracle/fallback;
codegen for production). Differential testing of codegen against the interpreter
on randomized inputs is what makes the codegen backend safe to write.

The load-bearing insight: for this workload, BTA collapses almost the whole plan.
A hash join against a static build side is a probe of a prepare-time structure;
with no pipeline breakers left, the query is a straight-line function over a
`.rodata` blob. **Most of the win is in BTA, before any backend choice.**

## 3. Disagreements and flags (requested — "disagreement over compliance")

1. **The engine is not the bottleneck until the boundary is fixed — and the
   boundary is row-major, so the prompt's "columnar in and out" ABI is wrong
   for this workload.** We measured the first half (2026-07: FFI/pydantic
   marshalling dominates; compute is noise at n≈1). The second half follows
   from how requests arrive: as row-major structures (dict / pydantic model /
   proto). At n≈1–100 a transpose into any columnar format is a per-call fixed
   cost — exactly the species of overhead this design exists to eliminate —
   and "zero-copy" was never on the table for row-born data. Layout is also
   computationally irrelevant at this scale: the batch is L1-resident either
   way, and produce/consume generates row-at-a-time code regardless.

   The fix is the Futamura move applied to the boundary itself: the row schema
   is static input, so **the marshaller is generated at prepare time** —
   fixed field order, interned attribute names, direct unbox into a packed row
   struct (no generic `Value` dispatch), output filled `model_construct`-style
   into fixed slots rather than run through `model_validate`. The pydantic
   classes remain the API; their generic marshalling path does not.

   Columnar retains two roles only: static tables at prepare time (already
   `pa.Table`, cold path), and a possible *alternate* large-n batch entry point
   — deferred until a consumer at n ≥ ~1k actually exists.

2. **Substrait is a bet, not a given.** `duckdb-substrait-extension` gets us
   parse+bind+typecheck+optimize for free and a serializable plan — the right
   default. But the extension has coverage gaps (window frames, some casts, DDL
   of temp state) and version drift vs. DuckDB releases. Mitigation: keep the
   frontend behind a `plan/` interface; the fallback frontend is our existing
   sqlparser wiring (already speaks the DuckDB dialect from the stub work). We
   validate substrait coverage against our v0 SQL subset in week one — it's a
   prep item, not an article of faith.

3. **Prepare-time static evaluation should be DuckDB itself.** The prompt says
   "eval NOW" for static subtrees. We don't need to implement that evaluator:
   at prepare time, run the static subtree *as SQL in DuckDB* and materialize
   the result. DuckDB is simultaneously the frontend, the static-subtree
   evaluator, and the differential oracle. We only ever implement the dynamic
   frontier — which is thin by construction.

4. **Two IRs + verifier + text format is the biggest cost center — and worth it
   here specifically** because this project will be built by a loop/workflow
   (see companion doc). Machine-checkable gates (verifier passes, text format
   round-trips, differential suite green) are what let an unattended loop make
   safe progress. The verifier is not engineering hygiene; it is the loop's
   review substitute between human checkpoints.

5. **Defer QuickScorer.** Branchless ensemble traversal is a codegen pattern for
   a workload we don't serve yet (no tree-encoded static tables exist in this
   repo today). It stays in the doc as the marquee example of "codegen pattern
   recognized during lowering", but it is milestone-last, behind a real model.

6. **Copy-and-patch is out for v0.** Research-grade in Rust, and our prepare
   happens at deploy time (fit), so we have milliseconds. LLVM ORC vs Cranelift
   is the real choice — resolved below.

7. **Cache/epoch lifecycle is mostly out of scope here.** In this repo the
   "static-table epoch" *is* the fit. Refit → re-prepare → new `f`; the old one
   keeps serving until swapped. Double-buffered rebuild is serving-infra work,
   not engine work; one paragraph in the ops doc, no design budget.

## 4. Open decisions — resolved

| decision | choice | why |
|---|---|---|
| Language | Rust, same crate/workspace as `_interpreter` | existing pyo3 wiring, team velocity, wasm door stays open |
| Codegen backend | **Interpreter first (oracle), Cranelift second, LLVM ORC only if measured gap matters** | Cranelift is a mature Rust-native dep (`cranelift-jit`); −10–30% vs LLVM is invisible next to the boundary win; no C++ toolchain in the build |
| Frontend | DuckDB + substrait extension; sqlparser fallback behind the same `plan/` interface | flag 2 |
| v0 SQL subset | projection + WHERE over `__THIS__`, LEFT/INNER equi-join to static tables, CASE, arithmetic/comparison/logic, the current builtin set (upper/lower/trim/substr/abs/round/coalesce/nullif/concat), CAST | this is precisely the surface the differential corpus already covers — the oracle tests exist |
| Batch layout | row-major packed AoS: per-schema `#[repr(C)]` row structs frozen at prepare; optional fields are `(u8 flag, T)` pairs — the IR's null lane laid flat | requests are born row-major (flag 1); at this n, columnar buys nothing and costs a transpose |
| Stage 2 ABI | `run(in: *const RowIn, n, out: *mut RowOut, scratch: *mut Arena)` | amends the prompt's columnar `Batch*`; `|out|` still known up front, still zero alloc |
| Null typing strength | strong enough that the verifier rejects 3VL mistakes statically | `T?` and `T` are distinct IR types; ops on `T?` must go through `null_check`/`unwrap_or`; no implicit coercion — see §6 |
| Multi-language future | imperative IR keeps a wasm-compatible profile (no host callbacks in the hot path) | the wasm spike showed one Rust→wasm artifact serves Go/Java near-native; Cranelift is wasmtime's backend — same lowering can target both eventually |

## 5. Stage 1 pipeline, concretely

```
sql text
  │  duckdb: PREPARE + get_substrait(sql)          (or sqlparser fallback)
  ▼
substrait plan ──► plan/: decode to Relational IR
  │
  ▼  rewrite rules (predicate pushdown, projection pruning — most of this
  │   already happened inside DuckDB's optimizer; we keep only what BTA needs)
  ▼
binding-time analysis:
    taint(__THIS__); propagate up.
    for each maximal static subtree S:
        result = duckdb.execute(sql_of(S))          # flag 3: DuckDB evals statics
        replace S with Const(materialize(result))   # scalar | array | perfect hash
  │
  ▼
dynamic frontier (probe → filter → project ribbon)
  │  produce/consume lowering (Neumann push model, fused, no materialization)
  ▼
Imperative IR  ──► verifier ──► { interpreter (oracle) | cranelift (prod) }
```

Static-structure selection at `Const` materialization:

```rust
enum StaticStruct {
    Scalar(Value),                    // 1×1 result (e.g. MEAN OVER ())
    DenseArray { base: i64, values: Column },   // dense int keys → direct index
    PerfectHash(PtHashMap),           // general equi-join build side
    Inline(SmallVec<Row>),            // tiny tables → unrolled compare chain
}
```

## 6. Imperative IR sketch

SSA, typed, no allocation vocabulary, explicit null lane. Text format is the
diagnostic surface and must round-trip (`parse(print(ir)) == ir`).

```
;; SELECT age / s.mean_age AS z FROM __THIS__ t LEFT JOIN stats s ON t.seg = s.seg
;; after BTA: stats collapsed into staticref @0 (perfect hash: seg -> mean_age)

fn run(in: batch{age: i64?, seg: str}, out: batch{z: f64?}) {
entry(row: idx):
  %age.f, %age.v = load.opt in.age, row        ; (i1 flag, i64 payload)
  %seg        = load     in.seg, row           ; NOT NULL lane: no flag
  %hit.f, %m  = probe.opt @0, %seg             ; miss -> flag=0 (LEFT JOIN)
  %num        = cast f64, %age.v
  %q          = fdiv %num, %m
  %z.f        = and %age.f, %hit.f             ; null iff either input null
  store.opt out.z, row, %z.f, %q
  next row
}
```

Verifier rules (initial set):

1. SSA: single def, defs dominate uses.
2. Type check every op; `load.opt`/`probe.opt`/`store.opt` are the only ops that
   touch flags; a `T?` value cannot flow into an arithmetic op — the verifier
   rejects it (this is the "3VL mistakes are statically impossible" property).
3. `StaticRef` ids must resolve against the plan's static-structure table, with
   matching key/value types.
4. No op allocates; `scratch` is reachable only from varlen `store` ops.
5. `out` column set and row count semantics must match the declared plan shape
   (`|out| = |in|` for pure projection; filter introduces the one allowed
   divergence and must declare it).

## 7. Backends

**Interpreter (oracle):** closure-compile the IR once — pre-traverse, build a
tree of `Box<dyn Fn(&mut Frame)>`; ~50 LOC of dispatch. Never optimized, always
correct, always available; also the fallback for ops Cranelift doesn't cover yet.

**Cranelift:** straight-line mapping from the IR above (it is deliberately shaped
like CLIF: SSA, explicit flags, no implicit control flow). `StaticRef` resolves
to absolute addresses of the prepare-time structures, which are owned by the
compiled artifact and dropped together with it.

Every prepared query is validated interpreter-vs-codegen on randomized inputs at
prepare time (cheap; prepare is cold) and in CI on the full corpus.

## 8. Testing

Three rings, outside in:

1. **DuckDB as end-to-end oracle** — the existing `tests/differential.py`
   harness pattern, new backend id `"specialized"`, oracle = `duckdb` (python
   package) instead of DataFusion. Same xfail-strict bug process as decision-1,
   with DuckDB as the semantics authority for this engine.
2. **Corpus mining** — `duckdb/test/sql/` (cloned in-repo) filtered to the v0
   subset: extract `query`/`statement ok` blocks over projections, filters, and
   joins-to-constant-tables; skip everything touching DDL/transactions/multi-
   statement state. A script materializes these as parametrized differential
   cases. This is fan-out work (see companion doc).
3. **IR-level** — verifier unit tests, text-format round-trip property tests,
   interpreter-vs-cranelift differential on random IR programs.

## 9. Module layout

Extend the existing crate (no workspace split until it hurts):

```
src/
  value.rs types.rs error.rs schema.rs lookup.rs    # shared (already exists)
  datafusion/                                       # existing engine, untouched
  specializer/
    catalog.rs      # static tables, prepared static structures
    frontend.rs     # duckdb/substrait (or sqlparser fallback) -> Relational IR
    plan.rs         # relational IR + rewrites + BTA
    lower.rs        # produce/consume -> imperative IR
    ir/             # imperative IR: defs, verifier, printer, parser
    exec/           # interp.rs, cranelift.rs
    runtime.rs      # arena, batch layout, perfect hash, string ops
```

The earlier `DuckDBInferFn` stub becomes the pyclass shell of this module
(`prepare` in `__init__`, `run` on Arrow batches); its NotImplementedError body
is replaced by the real Stage 2 entry point.

## 10. Measurement discipline

Before optimizing anything: baseline the boundary (row extraction, FFI crossing,
output construction) with a no-op `f` — both through the generic pydantic path
and through the generated marshaller, so the marshaller's win is itself a
measured number rather than an assumption. Then report p50/p99 ns/call at n ∈ {1, 8, 64,
1024}, always with the interpreter backend as control, and always next to the
current native engine and codegen engine numbers so the comparison is against
what we ship today, not against nothing.

## 11. Milestones (= the review boundaries in "How to proceed")

1. **M-restate** — this document, argued and amended. ✅ you are here
2. **M-ir** — imperative IR: grammar, types, verifier, text format; round-trip
   green.
3. **M-interp** — interpreter backend over the IR; hand-written IR programs pass.
4. **M-lower** — frontend + BTA + lowering for the v0 subset; differential suite
   vs DuckDB green; corpus-mined tests running.
5. **M-cranelift** — codegen backend; interpreter-vs-cranelift green; first
   ns/call numbers vs baseline.
6. **M-boundary** — generated row marshaller wired into the Python API (flag 1);
   end-to-end p50/p99 vs current engines.
7. (later, behind a real model) M-quickscorer.
