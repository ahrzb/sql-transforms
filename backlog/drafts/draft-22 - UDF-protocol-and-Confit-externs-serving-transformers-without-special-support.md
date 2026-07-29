---
id: DRAFT-22
title: "UDF protocol and Confit externs: serving transformers without special support"
status: Draft
type: spike
created: 2026-07-29
---

## Where this came from

Design dialogue with AmirHossein, 2026-07-29, right after loop 5
(transformers as UDAFs v0, PR #61). The question: **how do fitted
transformers — the ones Confit has no special support for — serve through
Confit?** Converged on a full API; parked until the go.

Two constraints fixed the shape early:

1. **The Python fallback is permanent infrastructure**, not a stopgap.
   There will always be a transformer we haven't optimized, and the
   fallback doubles as the oracle that optimized implementations gate
   against.
2. **Transformers can sit mid-expression** (`sc(age) OVER () + 1`, a CASE
   arm, a WHERE clause — once the v0 bare-final-item refusals lift). Any
   pre-SQL/python/post-SQL sandwich dies with that; the opaque call has to
   be an expression node the engine evaluates in place.

## Decision 1 — per-group weights are data, not closure state

A fitted per-group transformer is a *dict of estimators keyed by partition
key* — which is exactly what params tables already are. So the group lookup
does NOT hide inside the callable; the estimator is a value fetched by the
existing join, and the UDF is a **pure function of its arguments**:

```sql
-- SELECT sc(age) OVER (PARTITION BY country) + 1 AS z   =>   serving_sql:
SELECT __cf_tf0(p0.__cf_est, t.age) + 1 AS z
FROM __THIS__ AS t
LEFT JOIN __CF_PARAMS_0__ AS p0 ON (t.country IS NOT DISTINCT FROM p0.country)
```

- Params table gains an instance-id column (`__cf_est: i64`); ids index the
  fitted-clone registry. Both travel together: params stay Arrow, the
  instance registry is the one pickle payload, the id is the link.
- **One NULL story.** Unseen group -> LEFT JOIN misses -> NULL id -> NULL
  output. Same mechanism as every marginalized aggregate; no dict-miss
  convention.
- **No key semantics inside the extern.** The engine's join machinery
  already owns `IS NOT DISTINCT FROM`, NULL keys, key expressions. The
  extern receives one nullable i64 + typed features. (The rejected
  alternative — lookup-in-closure — would force Confit's Rust side to
  re-implement join-key equality.)
- **Desugaring becomes a column-width decision.** For a scaler, fit writes
  `mean_0, scale_0` into the same params table instead of an id, and the
  call site inlines to `(t.age - p0.mean_0) / p0.scale_0`. Same join, same
  NULL flow; the oracle-vs-optimized diff is confined to the projected
  expression.

## Decision 2 — the UDF protocol (sql_transform), transformer as subclass

The base concept is a *declared scalar UDF*; the fitted transformer is the
subclass that adds instance dispatch:

```python
class UDF:                                    # Confit's whole world
    name: str
    takes: tuple[str, ...]                    # Confit Ty vocabulary: "i1"|"i64"|"f64"|"str"
    returns: tuple[str, ...]
    def __call__(self, *args): ...            # scalar semantics — THE contract
    def apply_batch(self, *cols: pa.Array) -> tuple[pa.Array, ...]:
        ...                                   # default: loop __call__; override to vectorize

class PythonUDF(UDF):                         # any plain function
    def __init__(self, name, fn, takes, returns): ...

class PythonTransform(UDF):
    instances: dict[int, Any]                 # instance id -> fitted object with .transform
    # implicit leading arg: nullable i64 id, never written in `takes`
    # NULL id -> None (unseen group); id missing from instances -> TRAP
    #   (params table and instances from different fits = broken artifact, not NULL)
```

Rules:

- **Declared types, no probing.** `takes`/`returns` are written in Confit's
  own `Ty` vocabulary so build-time checking is literal. Everything is
  nullable, always. Deletes sklearn-attribute sniffing; duck-typed
  transforms are first-class. `SQLProjection` generates declarations at
  `_prepare` (it owns sklearn: `takes` from the resolved query schema,
  `returns` from the fitted estimator) so declaration/instance drift is
  impossible there; hand-rolled users are kept honest by the runtime check.
- **Two enforcement points.** Build: call-site argument expressions
  type-check against `takes`, each use of output slot j against
  `returns[j]` — named refusal before any row flows. Runtime: the
  trampoline checks each returned tuple against `returns` and traps on
  violation.
- **Inputs: scalars for semantics, Arrow for bulk, dicts nowhere.** Row
  level takes positional Python scalars (None = NULL) — the boundary has no
  names and the row path is the latency path. Batch level takes pyarrow
  arrays — already the currency on every side (Confit statics/infer_arrow,
  DuckDB vectorized UDFs zero-copy), NULLs ride the validity bitmap.
- **Default `apply_batch` loops the scalar protocol** — amortizes the
  boundary (one GIL crossing, one Arrow exchange per chunk) WITHOUT
  vectorizing the math. `est.transform` on an n-row block goes through
  different BLAS kernels than 1-row calls; looping keeps
  row ≡ batch ≡ oracle bit-exact. A `VectorizedTransform` override is the
  explicit opt-in (documented ulp caveat).
- **Determinism is the contract.** Fit-time DuckDB, serve-time Confit, and
  the oracle all call the same object and are compared bit-exact; a UDF
  with randomness breaks the round-trip invariant unlocalizably. Docstring
  contract, not enforcement code.
- **Scope fence: scalar UDFs only.** A user-defined aggregate over
  `__THIS__` is what transformers already are (fit/transform + window
  syntax). The protocol never grows an aggregation arm.

## Decision 3 — the InferFn API

```python
fn = InferFn(
    serving_sql,                              # contains __cf_tf0(p0.__cf_est, t.age) calls
    row_tables={"__THIS__": Row},
    static_tables=params,                     # id columns are ordinary Arrow data
    udfs=[PythonTransform("__cf_tf0", instances=..., takes=("f64",), returns=("f64",))],
    shape="map",
)
```

- Contract, restated with one parameter: **serve bit-for-bit identical to
  DuckDB *with these udfs registered* (`con.create_function(u.name, u)`
  each), or refuse.** Same two outcomes; the oracle stays runnable because
  the udfs list is exactly what you register in DuckDB.
- Undeclared unknown function still refuses at build, verbatim.
- Confit never sees sklearn: Arrow tables + declared callables only.
  Serialization stays on the Python side; the InferFn constructor args are
  the complete artifact description.
- k-output call at the boundary -> `list[float] | None` field (marshaller
  assembles k slots); mid-expression use must index down to a scalar.
  Confit's IR stays scalar-only (`i1/i64/f64/str`) — outputs are k slots
  via out-param array, the existing `len_out` pattern.

## Confit implementation route

The cranelift backend already routes every nontrivial op through
`extern "C"` helpers sharing code with the interpreter
(`packages/confit/src/specializer/exec/cranelift.rs`) — a `*mut Cx` context
pointer, trap flag checked after every fallible call. The UDF extern is
**one more helper family in an existing pattern**:

```rust
extern "C" fn h_extern(p: *mut Cx, slot: i64, /* args by ABI */ out: *mut f64) -> i64;
// Cx carries the slot table: boxed trampoline (GIL + __call__) or, later, a
// native kernel. Python exception -> trap flag, like every fallible helper.
```

Frontend: unknown function + matching udfs entry -> extern-call IR
instruction with k outputs, type-checked like any other op.

Engine bindings, one object, four targets:

| engine | binding |
|---|---|
| DuckDB (batch path + oracle) | `con.create_function(u.name, u.apply_batch, type="arrow")` |
| Confit interpreter | boxed closure calling `u.__call__` |
| Confit cranelift | C-ABI trampoline over `u.__call__` |
| future optimized | native kernel in the same slot (`NativeAffine(...)` etc.) |

The oracle gate for optimized impls is "same query, same statics, swap the
list entry." Bit-exact for scaler-class ops; matvec-class needs a
**declared per-estimator ulp tolerance** (BLAS summation order is
unpinnable) — written down per kind, not discovered per bug.

## What this deletes / changes in sql_transform

- `marginalize` emits the call **in place** (mid-expression works) instead
  of helper columns; the `TransformSpec` splicing block in
  `_projection.py::transform` deletes — batch transform becomes
  register-udfs-and-run.
- The v0 refusals (bare final-level items only) lift with no serving-side
  design change.
- Plain scalar UDFs in authoring SQL resolve exactly like transformers
  (registry, then caller scope, guarded by the `duckdb_functions()`
  catalog) — the only difference is scalar vs window call position.

## Sequencing (when greenlit)

1. SQL-call rewrite in `marginalize` + DuckDB UDF binding — batch path gets
   mid-expression transformers with zero Confit involvement.
2. Confit `udfs=` API + extern helper family + trampoline.
3. The params-join wiring Confit already needs regardless
   (`IS NOT DISTINCT FROM`, `ON ((1 = 1))`) — previously queued.

## Addendum (2026-07-29, follow-up dialogue — supersedes two points above)

AmirHossein's shape for the PCA/tree cases, adopted:

1. **Optimization = a typed udf entry, never a SQL rewrite.** "Desugaring
   becomes a column-width decision" above is superseded: fitted state does
   NOT move into wide params columns, and the call is NOT inlined into SQL
   arithmetic. Instead the udfs list carries typed entries —
   `confit.udfs.PCA("pca", n_components=2, instances={id: {"mean": [...],
   "components": [[...]]}})` — hyperparams on the entry, extracted state
   (plain arrays / tree tables, no pickle) per instance id, executed by a
   native kernel behind the same extern slot. The SQL text is identical
   across tiers, so the oracle gate is "swap the entry, rerun, compare" and
   a disagreement is always the kernel's fault. Full design: DRAFT-23.
2. **Keyless transformers inline the literal id** (`pca(0, feats...)`) —
   no one-row params table, no `ON ((1 = 1))` join for the global case.
   (Keyless *aggregate* params still join; that wiring stays needed.)
3. **Serving SQL keeps the author's function name** (`pca`, not
   `__cf_tf0`) — safe because the catalog guard proved no DuckDB collision;
   suffix (`pca_1`) when the same name is applied twice with different
   partitions.

## Rejected along the way

- **Compile-out as the primary path** (desugar/ONNX): a fallback is
  mandatory anyway (coverage + oracle); ONNX kernels aren't sklearn-exact.
- **The sandwich / staged serving plan**: mid-expression kills the single
  sandwich; the staged DAG generalization works but costs N boundary
  crossings, helper-schema soup, per-stage shape contracts. Named escape
  hatch if the extern loop stalls, nothing more.
- **Group lookup inside the closure**: re-implements join-key semantics in
  Rust; two NULL conventions.
- **Arity/type probing in Confit**: replaced by declared `takes`/`returns`.
- **Affine probing** (recover W,b numerically, serve as matvec): a third
  "approximately" mode — exactly what the two-outcome contract forbids.
