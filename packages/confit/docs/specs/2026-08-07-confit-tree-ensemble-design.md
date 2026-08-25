# Confit: native tree-ensemble scoring

Date: 2026-08-07
Status: **SUPERSEDED 2026-08-08 for everything above the kernel.**

> The user-facing surface in this document — a `models=` constructor kwarg, a
> `tree_predict('name', id, struct_pack(...))` builtin, and a user-visible
> `pack_trees` — was never approved and has been removed. A fitted tree
> ensemble is a `TreeBasedTransform`: constructed and passed in `udfs=` like
> every other transform, and called `name(id, feats...)`. See
> `docs/serving-fitted-models.md`.
>
> Scoping this document to "confit only, no `udfs=` surface" (below) is what
> produced the divergence: it ruled the registration question out of scope and
> then answered it anyway with a second, parallel mechanism. **The kernel, the
> node/model table layout, the f32 threshold rewrite and the parity argument
> all stand unchanged** — the migration did not touch them.

Original scope: **confit only**. No `udfs=` surface, no DuckDB registration,
no sql_transform family protocol — those live in
`2026-08-07-optimized-transforms-api-design.md` and DRAFT-22 and are out of
scope here. This document adds one built-in capability to the engine.

## What confit gains

A decision tree, random forest, or gradient-boosted ensemble becomes a
native operation in the specialized row function: no Python, no boundary
crossing, no `ecall` trampoline. Per row the engine probes the existing
params map for a model id and walks packed nodes over an f64 register file.

One kernel covers `DecisionTree*`, `RandomForest*`, `GradientBoosting*`, and
(via the same node layout) XGBoost and LightGBM.

**Open question, not settled here:** whether linear/logistic/PCA-class models
ever want a matvec kernel. `w0*x0 + ... + b` with a `sigmoid` lowers to a
branchless `fmul`/`fadd` chain with weights riding the params map, and for a
handful of features a kernel call would plainly cost more than it saves. Two
cases are unmeasured: **wide k** (k weights means k map value columns and an
O(k) program, so the pressure is program size, not arithmetic) and
**multi-output** (multinomial logistic, PCA — 20 features x 50 components is
1000 unrolled multiply-adds per row, lowered as scalar chains that nothing
vectorizes). Settle it by measuring the unrolled path at k = 8 / 64 / 512 and
at 20x50 — per-row latency and compile time — before writing any `mat_stack`
kernel. Scope of *this* document is unaffected either way: tree_ensemble
only.

## The shape: a model is a prepare-time structure

Confit compiles each query into a specialized per-row function. Anything
known *before rows arrive* is materialized once at prepare time and reached
through an opaque handle — that is what the IR calls a **static**, and it is
already how a params table works:

```text
static @2 : map(str) -> (i64)
%hit, %id = probe @2, %region      # this IS the LEFT JOIN; a miss is %hit = false
```

Three candidate homes for a fitted model, and only one survives contact with
the IR:

| candidate | verdict |
|---|---|
| a value in the params map | **impossible today**: map values are `Vec<ScalarVal>` — scalars only ([`exec/mod.rs`](../../../packages/confit/src/specializer/exec/mod.rs) `StaticData::Map`). List-typed IR values are a far bigger change than one opcode. |
| an `ecall` extern | that is the Python trampoline (DRAFT-22). Wrong tier: a built-in needs no GIL, no `ExternImpl` box, and should be type-checked and foldable like any other op. |
| **a third static kind** | chosen. `PreparedStatic` is already a heterogeneous table (`Scalar \| Map`) reached through `Cx::statics`, so a `Model` variant reuses materialization, `Cx` wiring, the compile-time kind check and lifetime verbatim. Layout inside it is a backend decision, exactly as the design doc already licenses for maps. |

A separate top-level `model @N` declaration (the regex precedent) was the
first sketch and is strictly more machinery: regexes need their own namespace
because they are not row-shaped data, whereas a model is precisely "a
prepare-time structure behind a handle", which is what `static` already
means.

The layout freedom matters: `predict` can become QuickScorer, or grow a
vectorized multi-tree walk, without touching the IR. A *data-driven*
traversal in the IR would need a loop, which the acyclic-CFG rule forbids
today ("lift when one needs a loop, e.g. QuickScorer" —
[`ir/mod.rs`](../../../packages/confit/src/specializer/ir/mod.rs)); putting
the loop in the kernel leaves that rule intact.

### Kernel vs lowering the tree into IR — measured 2026-08-07, kernel stays

The acyclic rule does **not** rule out the other design: a tree is itself
acyclic, so it can be lowered into nested `brif` blocks with no loop at all —
the model compiled to code rather than interpreted from data. Both are
viable; this document picked the kernel on structural grounds, and the
measurement below confirms it:

| | one instruction + kernel | lower the tree into IR branches |
|---|---|---|
| IR surface | 1 declaration + 1 opcode | a lowering pass emitting O(nodes) blocks |
| forest of 100 x 255 nodes | data; program size unchanged | ~25k blocks — cranelift compile time and code size |
| refit | swap the Arrow batch, re-prepare | re-JIT the whole program |
| interpreter backend | native speed for free | interpreted per node — slow |
| small tree (depth <= 6) | indirect loads, loop overhead | **confirmed** faster: 18.6 vs 36.6 ns/row at 31 nodes — and 2.8% of end-to-end |
| large ensembles | can adopt QuickScorer / vectorized walks with no IR churn | i-cache and branch-prediction pressure |
| parity | identical | identical |

**Measured 2026-08-07** — release build, i7-12700-class 12-core, 8 features,
`n_jobs=1`. The branch program is generated *from the packed tables*, so both
paths encode literally the same tree, and a parity gate (`==` on raw doubles,
against sklearn too) runs before any timing. Both assert
`fn.backend == "cranelift"`; the compile error is discarded and the
interpreter fallback is silent, so an unasserted run may time the wrong
engine.

Serving here is boundary-bound, so the table reports **compute** — p50 at
n=1024 minus the same-shape `SELECT f0` floor (668 µs / 1024 rows).

| point | nodes | kernel compute | branch compute | kernel build | branch build | branch program |
|---|---|---|---|---|---|---|
| 1 tree, depth 4 | 31 | 36.6 ns/row | **18.6 ns/row** | 1.0 ms | 1.7 ms | 1.1 KB |
| 10 trees, depth 6 | 1266 | 192 ns/row | 604 ns/row | 1.1 ms | 31.8 ms | 42 KB |
| 100 trees, depth 8 | 47828 | 4.4 µs/row | 20.2 µs/row | 2.6 ms | 2062 ms | 1.6 MB |
| 500 trees, depth 12 | 1301092 | 106 µs/row | — | 49.9 ms | — | 43 MB, not attempted |

**The kernel stays, and the hybrid is not worth building.** The predicted
small-tree win is real and repeatable — 6 of 6 alternating-order rounds, ~2x
on compute — but it is 18 ns/row against a ~650 ns/row boundary floor: **2.8%
end-to-end**, and at n=1 the two are identical to the resolution of the clock
(1.30 µs, both exactly the floor). The crossover sits between 31 and 1266
nodes, i.e. immediately: by 10 trees the branch program is already 3.1x the
kernel's compute and 29x its build, and by 100 trees it is 4.6x compute and
**783x** build (2.1 s of cranelift for one model). At 500 trees the branch
path does not exist — 45 MB of generated SQL.

So the hybrid the structural argument left open — lower small trees to
branches, emit `predict` for large ones — would buy under 3% on the only
shape where it wins, in exchange for a second code path and a lowering pass.
The reversibility still holds if that ever changes; nothing in the SQL surface
or the Arrow boundary depends on the choice.

One thing the numbers say that the table did not predict: the kernel's own
cost is **memory-bound at scale**, not compare-bound. Per node actually
visited it goes 3.2 ns (1266 nodes) → 5.5 ns (47.8k) → 19.4 ns (1.3M) — the
same instruction getting 6x slower as the working set leaves cache. That, not
branch prediction, is the axis a QuickScorer or vectorized walk would attack,
and it is reachable without touching the IR — which was the kernel's argument
in the first place.

## IR surface

No new declaration class — one more `static_ty`:

```text
static_ty := "scalar" "<" col_ty ">"
           | "map" "(" ty ("," ty)* ")" "->" "(" ty ("," ty)* ")"
           | "model" "<" "tree_ensemble" "(" INT ")" ">"      # INT = n_features
```

One instruction, variadic like `probe` and `ecall`:

```text
%d = predict @N, %id, %f0, .., %f{k-1}      # %id: i64, %fj: f64, %d: f64
```

Bare scalars in, bare scalar out: the null lane never enters the kernel and
stays ordinary `i1` algebra in the caller, per the IR's null-lane rule.

Verifier rules, all local (the declaration carries the arity, so nothing needs
prepare-time data to check):

- `@N` is a static declared `model<...>`; operand count is exactly
  `n_features + 1`.
- `%id` is `i64`; every feature operand is `f64`; the result is `f64`.
- `n_features >= 1`.
- Symmetrically, `probe`/`sload` against a `model<...>` static refuse — the
  same kind check the other two static kinds already get.

Round-trip closure (`parse(print(p)) == p`) extends to the new static kind and
instruction; `gen.rs` grows a case so the property tests cover them.

### Lowered example

```text
# SELECT tree_predict('trees', m.mid, price, sqft) AS pred
#   FROM row r LEFT JOIN models m ON (r.region IS NOT DISTINCT FROM m.region)

static @2 : map(str) -> (i64)
static @3 : model<tree_ensemble(2)>

b0:
  %hit, %id = probe @2, %region
  %pv, %p0  = load.opt in.price
  %sv, %s0  = load.opt in.sqft
  %nan = const.f64 nan
  %f0  = select %pv, %p0, %nan       # NULL feature -> NaN, per the missing rule below
  %f1  = select %sv, %s0, %nan
  brif %hit, b1(%id, %f0, %f1), b2
b1(%id: i64, %f0: f64, %f1: f64):
  %p = predict @3, %id, %f0, %f1
  store.opt out.pred, true, %p
  emit
b2:
  %z = const.f64 0.0
  store.opt out.pred, false, %z
  emit
```

Only the probe hit gates the output: a NULL *feature* is a value the model
knows how to handle, while a missing *model* is not. `load.opt` yields the
type default (`0.0`) on an invalid lane, so the NULL-to-NaN conversion is an
explicit `select`, not an implicit payload reading.

Branching on the hit rather than scoring-then-discarding is the lowering's
choice; both are correct (a miss leaves `%id` at its defined default), the
branch merely avoids a pointless traversal.

## SQL surface

```sql
tree_predict('trees', <id expr>, <f64 expr>, ...)
```

The first argument is a **string literal naming the model set**, hoisted at
build into a `model<...>` static — the same literal-to-prepared-structure
treatment regex patterns already get, so no new name-resolution machinery and
no ambiguity with a column called `trees`. Features are positional `f64` expressions; the count
must equal the declared `n_features` or the build refuses. The id expression
is any `i64`, usually a params-map probe result.

Deferred sugar: a struct call site (`tree_predict('trees', id,
struct_pack(price := ..., sqft := ...))`) flattened to lanes at build with
names checked against the model's feature names. Nice, not needed for the
first cut.

## The kernel

```rust
// exec/models/tree_ensemble.rs — sibling of strip_accents.rs: a large
// table-driven routine behind a scalar instruction.
pub struct TreeEnsemble {
    // struct-of-arrays; a tree's nodes are contiguous, a model's trees are contiguous
    feature: Vec<i32>,        // -1 = leaf
    threshold: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    missing_left: Vec<bool>,  // sklearn's tree_.missing_go_to_left, per node
    value: Vec<f64>,
    tree_span:  Vec<(u32, u32)>,   // tree  -> node range
    model_span: Vec<(u32, u32)>,   // model -> tree range   (id = dense index)
    base: Vec<f64>,
    agg:  Vec<Agg>,                // Sum | Mean
    link: Vec<Link>,               // Identity | Sigmoid
    n_features: u32,
}

impl TreeEnsemble {
    pub fn from_arrow(nodes: &RecordBatch, models: &RecordBatch)
        -> Result<Self, Refusal>;          // validates; refuses by name
    pub fn predict(&self, id: i64, feats: &[f64]) -> f64;
}
```

Both backends route through this one routine — the existing shared-code rule
that keeps the interpreter and cranelift from drifting.

### The cranelift binding

The `h_probe` pattern almost verbatim
([`cranelift.rs`](../../../packages/confit/src/specializer/exec/cranelift.rs)):
a per-site descriptor owned by the `CraneliftFn` whose absolute address is
baked in as an `iconst`, arguments marshalled through a stack slot, result
returned in a register.

```rust
struct PredictDesc { static_id: usize, n_features: usize }

extern "C" fn h_predict(p: *mut Cx, desc: *const PredictDesc, id: i64, feats: *const f64) -> f64 {
    let c = unsafe { cx(p) };
    let d = unsafe { &*desc };
    let feats = unsafe { std::slice::from_raw_parts(feats, d.n_features) };
    let statics = unsafe { std::slice::from_raw_parts(c.statics, c.statics_len) };
    let PreparedStatic::Model(m) = &statics[d.static_id] else {
        unreachable!("kind checked at compile")
    };
    match m.predict(id, feats) {          // the SAME routine the interpreter calls
        Ok(v)  => v,
        Err(t) => { c.set_trap(t.0); f64::NAN }
    }
}
```

```rust
// one slot per function, sized to the widest predict site
let slot = b.create_sized_stack_slot(StackSlotData::new(ExplicitSlot, 8 * max_feats, 3));
// at the site
for (j, v) in feats.iter().enumerate() { b.ins().stack_store(*v, slot, (8 * j) as i32); }
let fp = b.ins().stack_addr(types::I64, slot, 0);
let dp = b.ins().iconst(types::I64, desc_addr as i64);
let r  = b.inst_results(b.ins().call(h_predict_ref, &[cxp, dp, id, fp]))[0];
// standard trap_flag check after a fallible helper
```

One deliberate difference from `probe`/`ecall`: those marshal through 16-byte
`Cell` pairs because they carry validity lanes and string spans. `predict`
needs neither — every feature is a bare `f64` and NaN carries missingness — so
it uses a **contiguous f64 slot**: no validity stores, no unpacking loop in
the helper.

The trap guard is only reachable when the id comes from a row column. Where
the id is a static map's value, the build-time check makes the call infallible
and the guard can be elided — an optimization to take later, not in the first
cut.

### Which backend runs it

One backend runs a whole program; they are never mixed per instruction.
[`duckdb/mod.rs`](../../../packages/confit/src/duckdb/mod.rs) tries
`cranelift::compile_ext` first and falls back to the interpreter when
compilation fails, with `SPECIALIZER_FORCE_INTERP` pinning the interpreter
for benches and debugging. The kernel is native Rust either way: the
interpreter calls it directly, cranelift emits a call to an `extern "C"` shim
over the same routine, so tree-scoring cost is identical and only the
surrounding row code differs.

**The fallback is silent** — the compile error is discarded. A `predict`
implemented in the interpreter but not yet in cranelift therefore drops every
query that uses it to the interpreter *entirely*, with correct results and
misleading numbers. Two consequences:

- land both bindings before drawing any performance conclusion;
- the cranelift binding needs a test that fails when it is missing, which
  means calling `cranelift::compile_ext` directly (as `exec/tests.rs` already
  does) rather than going through the fallback-guarded path.

## The boundary

Two Arrow record batches per model set, materialized into a
`StaticData::Model` entry in the same statics vector `compile` already takes,
and type-checked against the declaration (`n_features` must agree):

```text
nodes:  model_id i64 | tree_id i64 | node_id i64 | feature i32 (-1 = leaf)
                     | threshold f64 | left i32 | right i32 | missing_left bool
                     | value f64
models: model_id i64 | base f64 | agg str ('sum'|'mean') | link str ('identity'|'sigmoid')
```

Nothing but Arrow crosses. The Python side is a builder that walks
`tree_.__getstate__()` and emits those two batches; no estimator object, no
pickle, no live model reference reaches the engine.

`model_id` values are dense indices assigned by the builder; the params map's
value column holds them.

**As built**, one model set is a four-key mapping, not a bare pair:

```python
DuckDBInferFn(sql, ..., models={"trees": {
    "nodes": nodes_table, "models": header_table, "features": ["price", "sqft"],
    "compare_grid": "float32",
}})
```

`features` is a key because the names describe the SET, not a row of
either table — and the struct call site resolves against them by name, so
they have to be somewhere. `compare_grid` is there for the same reason and is
REQUIRED: it says which floating-point grid the thresholds were fitted on, so
the engine knows whether an INTEGER feature must narrow through float32 on its
way to the comparison (sklearn) or reach it exactly (a float64 library). It
cannot live per-model, because the model id is a runtime value while the
conversion is chosen once at lowering (TASK-77). `sql_transform._trees.pack(estimators, features)`
produces exactly this mapping from fitted sklearn regressors.

Decoding reuses the pyarrow buffer walk in `duckdb/arrow.rs` rather than
`to_pylist()`: a 100k-node forest would otherwise allocate 100k Python dicts
to build a structure that is pure numbers. int32 and int64 are both accepted
for the id, child and feature columns — `pa.array([1, 2])` defaults to int64,
and refusing that would be a papercut with no safety value.

The `models=` entries are materialized into `StaticData::Model` values
appended AFTER every map static, which is what keeps existing probes' `@N`
from shifting. Both materialization paths (the cranelift attempt consumes the
static data, so the interpreter fallback rebuilds it) go through one
`materialize_statics`, and its zip over `statics[n_join..]` asserts the
ordering rather than assuming it.

## RESOLVED: the float32 comparison gap (found and fixed 2026-08-07)

The parity claim above was **false for quantized features**, for a structural
reason rather than a rounding detail.

sklearn narrows `X` to float32 (`DTYPE`) before traversal and keeps
`tree_.threshold` in float64, so it evaluates `float32(x) <= threshold`. The
kernel compares raw f64. The threshold is *exactly* the float32 midpoint of
the two neighbouring training values, which proves the splitter itself only
ever saw float32 — the narrowing is baked into where the split SITS, not
applied incidentally at predict time.

```text
threshold        = 0.15000000223517418   # == mean(f32(0.1), f32(0.2))
float64 midpoint = 0.15000000000000002   # not this
x = 0.15   ->  kernel -1.0,  sklearn +1.0
```

Measured: 2-decimal grid, `RandomForestRegressor(30)`, 157/3000 rows differ,
max delta 0.43 on a −7.9..19.4 range. Pre-narrowing inputs to float32 gives
0/3000. Continuous float64 draws hide it (the band is ~1 f32 ULP wide), which
is why the original gate passed — it was passing by luck, not immunity.

So the f64 compare is not a more accurate evaluation of the same model. It is
a different model.

### The decision: rewrite the thresholds in `pack_trees`

Three candidates were on the table: rewrite thresholds at pack time, narrow
each feature to f32 inside the kernel, or declare an f32 contract at the
boundary and refuse anything else. The deferral note argued that none of them
was right on its own, because the same two-table layout is documented as the
path for other libraries and **XGBoost is float32 while LightGBM is float64**
— so the narrowing looked like a property of the model ENTRY, needing a
per-entry flag in the header and therefore a layout change.

**That premise was wrong, and measurement is what showed it.** The rewrite is
not an approximation of the f32 compare — it *is* the f32 compare. Rounding to
float32 is monotone, so `float32(x) <= t` is still a single cutpoint over the
doubles; `t'` is the largest double that still narrows to the largest float32
at or below `t`, i.e. their midpoint, minus one ulp where ties-to-even would
round the midpoint back up. Verified by walking f64 ULPs across the boundary
of 18012 thresholds — sklearn-shaped midpoints, exactly-f32 values,
subnormals, and both overflow ends — for **zero** disagreements in ~4.8M
probes.

Exactness is what collapses the layout question. Because the rewrite is
lossless, it belongs wholly to whoever packs: a LightGBM packer simply does
not call it, and the wire format keeps meaning "compare this double" for every
library. No header flag, no kernel change, no per-row cast, no float32
anywhere in the engine.

What it costs: the packed `threshold` column no longer equals
`tree_.threshold`. That is documented rather than hidden — the column means
"the double this split compares against", not "sklearn's stored number".

Two boundaries the implementation has to respect, both found by running it
rather than by reading it:

- **`t = +inf` is a real fitted threshold**, not a sentinel — sklearn writes
  it for "every non-missing value goes left". It already admits every non-NaN
  double, so it passes through untouched. A first revision *refused*
  out-of-f32-range thresholds and broke all three missing-value tests.
- **The overflow ends have no finite neighbour** to take a midpoint against;
  ±inf stands in at ±2**128, where float32 rounding actually tips.

Pinned by three tests. The load-bearing one is
`test_threshold_rewrite_reproduces_the_f32_comparison`, which walks ULPs
rather than sampling: mutating the rewrite by a **single ULP** still passes
the 1500-row end-to-end parity test, measured. The end-to-end test is not a
sufficient gate for this property, and now it does not have to be.

Independently swept after the fix: 4 estimator families × 6 quantisation grids
(integers, 1–3 decimals, wide scale, percentages) × both backends × 3000 rows
= 144000 rows, **0 mismatches**.

## Build-time refusals (P7)

Every one of these names the offending row or field, before any data flows:

- a child index out of range, or one that does not strictly follow its parent
  — that ordering is what makes traversal provably terminate, and it rules
  out cycles by construction;
- a non-root node with anything other than exactly one parent: none (it is
  unreachable from its tree's root) or two (a shared child — a decision DAG,
  not a tree);
- a leaf (`feature = -1`) with children, or an internal node without both;
- `feature >= n_features`;
- unknown `agg` or `link` spelling;
- a `model_id` present in `nodes` but absent from `models`, or vice versa;
- non-dense or duplicated `model_id`;
- call-site arity that disagrees with the declared `n_features`;
- a model id in a *static* map's value column that names no model — the
  static case is fully known at build, so it refuses instead of trapping.

The exactly-one-parent rule is worth stating precisely, because an earlier
draft of this list got it wrong in both directions (TASK-76, adjudicated
2026-08-08).

It called a shared child "a cycle". It is not one — with children forced to
strictly follow their parent, cycles are impossible by construction, and a
shared child is a decision DAG that traverses in exactly one path, terminates,
and yields the value the table names. It was never a wrong-answer bug.

It is refused anyway, because *"every non-root node has exactly one parent"* is
a **complete** characterisation of tree-ness here: one parent each makes the
parent function total, and the forward ordering makes walking parents strictly
decrease, so every node has a unique path back to node 0. The validator kept a
saturating parent count already and rejected only the zero end. Rejecting zero
while allowing two was an arbitrary place to stop, and the other end is the
same array — so the full check costs one line and no extra pass.

Every refusal above is exercised by construction in
`known_divergences/` (then a single `test_known_divergences.py`), not assumed.

## Pinned runtime semantics

These are the correctness surface; each is a test, not a comment.

- **Comparison**: `x <= threshold` goes left, on the raw double. Reproducing
  a library that compares on a narrower grid is the packer's job, not this
  instruction's — see the float32 resolution above.
- **Missing values follow sklearn, per node, not a house rule.** Measured on
  sklearn 1.9.0: `predict` never raises on NaN. Every tree carries
  `tree_.missing_go_to_left`, and a NaN feature takes that node's declared
  direction — right by default (`0`), left where the tree learned a missing
  branch. A tree trained without any missing values still has the array, so
  the flag is always available and the layout always carries it. XGBoost's
  `missing=` direction lands in the same column.
- **NULL is NaN**: a SQL NULL feature is presented to the model as NaN and
  follows the same per-node direction. This is deliberately *not* the usual
  NULL-in-NULL-out rule, because the model has a defined answer for missing —
  and it matches the engine's existing convention, where `_as_feature`
  already maps NULL to NaN as "the estimator's own missing-value convention"
  ([`_udf.py`](../../../packages/sql-transform/sql_transform/_udf.py)).
  Consequence for the IR: NULL features are *not* folded into the caller's
  validity `and`-chain; only the probe miss is. The kernel therefore does
  receive NaN, and `predict` needs a NaN-carrying feature operand rather than
  a bare validity split.
- **Probe miss** (unseen group, no model) still yields NULL — that is a
  missing *model*, not a missing feature.
- **NaN out of the reference UDF**: register it with `type="arrow"`. Measured
  on DuckDB 1.5.5 both modes preserve NaN and NULL distinctly on the way *in*,
  but the native return conversion maps a NaN prediction to NULL, which would
  make the oracle disagree with the kernel for a reason that has nothing to do
  with the kernel.
- **Aggregation**: `base + sum(leaf values)` for `sum`, `base + mean(...)` for
  `mean`, then the link. Summation follows `tree_span` order, so the result is
  reproducible bit-for-bit across backends and runs.
- **Bad id at runtime**: a row-supplied id outside `model_span` traps with a
  named message. (Reachable only when the id is not a static-map value; see
  the refusal above.) It **traps rather than yielding NULL** — a NULL feature
  is data the model handles, but an id naming no model is a bug in the
  caller's params table, and quietly nulling it would hide a broken join.

## What it touches

| file | change |
|---|---|
| `ir/mod.rs` | grammar + doc for the `model<...>` static kind and `predict` |
| `ir/parse.rs`, `ir/print.rs` | round-trip for both — **check first** whether `const.f64 nan` already round-trips; the NULL-to-NaN lowering needs a NaN literal, and `canon_f64_bits` suggests NaN awareness exists but does not prove the text format carries it |
| `ir/verify.rs` | the local rules above |
| `ir/gen.rs` | generator case, so property tests cover it |
| `exec/mod.rs` | `StaticTy::Model` + `StaticData::Model` + `PreparedStatic::Model`; kind check at compile |
| `exec/models/tree_ensemble.rs` | new: layout, `from_arrow`, `predict` |
| `exec/interp.rs`, `exec/cranelift.rs` | one call site each into the shared routine |
| `frontend.rs` | resolve `tree_predict`, hoist the literal, arity/type check |
| `plan.rs` | prepare a model set from the two Arrow batches |
| `confit/_engine.pyi` | the model-set argument on the compile entry point |

## Gate

The contract was never "match DuckDB bare" — it is **match DuckDB with the
reference function registered** (DRAFT-22, Decision 3). DuckDB lacking a
`tree_predict` builtin therefore costs nothing: register one for the test and
the standing differential harness applies unchanged.

The distinction that matters: a Python *reimplementation of our node layout*
would be a twin — it can drift with the kernel and proves nothing. The
**fitted estimator itself** is an oracle: it is the ground truth the kernel
exists to reproduce.

Four gates, strongest first.

1. **Differential against sklearn, via `check`.** Same SQL text both sides;
   the DuckDB side registers `tree_predict` as a scalar UDF over the original
   fitted estimator (`est.predict`), the confit side runs the `predict`
   instruction over batches extracted from that same estimator. Random
   ensembles x random rows. This is the only gate that covers the whole
   chain at once — extraction, packing, traversal, aggregation, link — and it
   subsumes the layout check I had listed as a separate item.
2. **Differential inside confit: `ecall` vs `predict`.** The same query
   lowered twice — once with the estimator behind the Python trampoline
   (`ExternImpl`, already in the IR), once through the kernel. No DuckDB, no
   SQL-level differences; isolates the kernel from everything else, so a
   disagreement here is unambiguously the kernel's fault.
3. **interp ≡ cranelift** — the standing backend-equality harness, extended
   with programs containing `predict`.
4. **Pins** for the cases a random generator will not reliably produce and
   whose expected value should be frozen regardless: NaN feature, NULL
   feature, probe miss, single-node (root-is-leaf) tree, empty ensemble.

**As built** ([`_trees_test.py`](../../../packages/sql-transform/sql_transform/_trees_test.py),
[`test_tree_predict.py`](../../../packages/confit/tests/test_tree_predict.py)):

- Gate 1 compares against `est.predict` **directly** rather than through a
  DuckDB-registered UDF. The SQL on both sides would have been
  `SELECT tree_predict(...) FROM __THIS__` with no other construct in it, so
  DuckDB contributes only transport — and transport with a known hazard, the
  NaN-return conversion this document already flags. The estimator is the
  oracle either way; routing it through DuckDB would have added a failure
  mode without adding coverage.
- Gate 3 is the existing `fuzz_cranelift_agrees_with_interpreter`, which now
  generates `predict` sites; the Python gates additionally assert
  `fn.backend`, because the cranelift compile error is discarded and the
  fallback is silent.
- Gate 2 (`ecall` vs `predict` inside confit) **is built**: the same fitted
  estimator behind a `PythonTransform` trampoline on one side and the kernel
  on the other, same rows, `==` on the answers. Covered: plain forest,
  missing values (the two paths reach NaN by different routes — `_as_feature`
  vs the lowering's `select`), a boosted model (base and learning-rate
  scaling exist only on the kernel side), and a NULL id. The ecall side
  counts its trampoline invocations and asserts one per scored row, so a
  regression that made both sides take the SAME path fails loudly rather than
  passing green. This is the gate the swap-in needs: replacing the trampoline
  with the kernel must not move a bit, or every model fit against the old
  serving path silently shifts.

**Float tolerance — SETTLED BY MEASUREMENT (2026-08-07), and the premise was
wrong.** sklearn 1.9.0 / numpy 2.5.1, 2000 rows at 10 / 100 / 500 trees:

- **Neither estimator uses numpy's pairwise summation.** `ForestRegressor`
  accumulates `out[0] += prediction` per tree into a zeroed array;
  `GradientBoosting` runs `raw_predictions[k] += learning_rate *
  tree.predict(X)` stage by stage. Both are strictly sequential in tree
  order. `arr.sum(axis=1)` agrees with neither — it diverges on 509 to 1853
  of 2000 rows in every configuration, up to 320 ULP, and the divergence is
  already there at 10 trees (numpy's 8-way unrolled path engages at n >= 8).
  So a sequential `tree_span` walk is what makes us bit-exact; the plan to
  replicate pairwise order would have *introduced* the divergence it was
  meant to avoid.
- **Where the base term enters is not cosmetic, and it differs by mode.**
  A boosted model SEEDS its accumulator with the init prediction
  (`_raw_predict_init`) and adds each stage into it. Summing the trees and
  adding the base afterwards diverges on up to 1365/2000 rows, 632 ULP. A
  forest starts at 0.0, accumulates, divides by `n`, and has no base at all.
  The kernel therefore seeds for `sum` and adds after the divide for `mean`,
  with a pin using values whose addition is order-sensitive.
- **`n_jobs=1` is the reference configuration, and gate 1 fits with it.**
  Not a workaround — a serving decision (AmirHossein, 2026-08-07):
  per-request parallelism is not on the table inside our latency budget, so
  the single-threaded estimator is the one we are reproducing. It also
  happens to be the only configuration where bit-exactness is a coherent
  goal: with `n_jobs != 1` the lock-serialized `+=` interleaves
  nondeterministically, and two `.predict` calls on the same fitted model
  and the same X in one process differ on up to 1187/2000 rows (19 ULP). No
  fixed-order kernel can be bit-exact against a target that moves against
  itself, so a forest fitted for throughput rather than serving needs a
  declared tolerance — a property of how it was configured, not of the
  kernel.

A single tree still returns its leaf value verbatim, so bit-exactness there
was never in question.

Extraction fidelity has one more independent check available if wanted:
compare against the estimator's own `decision_path`/`apply` (which leaf did
each row reach) rather than only the final value, localizing a traversal bug
to a node instead of a number.

## Decisions taken before implementation (2026-08-07)

- **confit stays sklearn-free.** The engine accepts only the two Arrow
  batches. The sklearn walk — `tree_.__getstate__()`, `missing_go_to_left`,
  ensemble unrolling — lives in sql_transform; confit's own tests build
  batches by hand. confit never imports sklearn.
- **Coverage in the first cut**: regressors (`DecisionTreeRegressor`,
  `RandomForestRegressor`, `GradientBoosting*`) plus **binary** classifiers
  through the single probability lane. Multiclass refuses by name at
  extraction — its `value` is `n_nodes x n_outputs x n_classes` and would
  change `predict`'s return from one `f64` to k.
- **Float policy**: measure the real divergence between numpy's pairwise
  summation and a sequential `tree_span` walk on realistic ensembles first. If
  it is nonzero, replicate pairwise order in the kernel so the differential
  gate stays bit-exact, per C2's match-or-refuse rule. A declared tolerance is
  the fallback, not the default.
- **Landing**: specs and implementation land together in one PR on
  `claude/optimized-transforms-499a5a`. No stacked PRs.

## What the implementation changed (2026-08-07, commit `fd761d4`)

Three places where the design above was wrong or heavier than needed. All
three are simplifications; none changes the SQL surface or the Arrow
boundary.

- **`from_arrow(&RecordBatch)` does not exist, and should not.** confit has
  no `arrow-rs` dependency and that is deliberate:
  [`duckdb/arrow.rs`](../../../packages/confit/src/duckdb/arrow.rs) walks
  pyarrow buffers by address through the Python buffer API. The kernel
  therefore takes plain Rust slices (`NodeRows` / `ModelRows`), and the
  pyarrow decode will live beside the existing ingest. Arrow is still what
  crosses from Python; only the Rust-side type changed. This also keeps
  confit's own tests free of any Arrow construction.
- **`predict` needs no per-site descriptor.** `probe` and `ecall` own a
  boxed `ProbeDesc`/`ExternDesc` because they carry `Vec<Ty>`. Everything
  `predict` needs at a site is two integers, which ride in as `iconst`
  immediates — no heap descriptor, no ownership vector on `CraneliftFn`, no
  extra parameter threaded through `translate_inst`.
- **Features share `slot_vals` rather than getting their own slot**, packed
  at 8-byte stride (a model contributes `ceil(k/2)` cells to the slot
  sizing). `slot_keys` is already shared between probe keys and ecall args,
  so this is the house pattern, not a new one.

Two notes for whoever picks this up:

- **The "acyclic CFG in v0" premise is stale.** `MULTI_EXPAND`
  ([`ir/fixtures.rs`](../../../packages/confit/src/specializer/ir/fixtures.rs))
  already contains a legal cycle via `emit.to`. The kernel-versus-lowering
  argument above leaned on that rule; it should lean on the size and refit
  arguments instead, which are unaffected.
- **The verifier had two silent holes** the compiler could not catch:
  `sload @model` and `sload.opt @model` fell through `Some(_) => {}` arms
  and would have verified clean, then panicked in both backends. Closed.

## Sequencing

1. IR: declaration + instruction + verify + round-trip (no execution yet).
2. Kernel: layout, `from_arrow` with its refusals, `predict`.
3. Interpreter binding + pins.
4. Cranelift binding + backend-equality harness, tested through
   `cranelift::compile_ext` directly so a missing binding fails loudly instead
   of falling back.
5. Frontend: `tree_predict` resolution, literal hoisting, arity/type checks.

Steps 1–2 are independent and can land separately; nothing before step 5 is
reachable from SQL, so each lands green on its own. Steps 3 and 4 are one
unit for benchmarking purposes — see the silent-fallback note above.

## Where this grows (sketch, not built)

The point of this section is the **ceiling**: two kernel operations, forever.
Neither the number of outputs nor the sklearn method name ever becomes part of
a function name — `multi_tree_predict_proba` and its siblings must not exist.

**Shape rides the declaration.** Measured on sklearn 1.9.0, `tree_.value` is
always `(n_nodes, n_outputs, max_n_classes)`; today's one-f64-per-node layout
is the `(1,1)` corner. Two extra layout fields cover every case, because the
libraries use two different mechanisms:

```rust
value_stride: u32,      // leaf width: 1 today; n_outputs * max_n_classes for vector leaves
tree_output:  Vec<u32>, // which output lane a tree feeds (all 0 for vector leaves)
```

- sklearn trees and forests use **vector leaves**: `value_stride = k`, every
  tree feeds all lanes (multi-output regressor `(15, 2, 1)`, multiclass
  `(5, 1, 3)`).
- GBM uses **one tree per class**: `estimators_` is `(n_stages, n_classes)`
  with scalar leaves, so `value_stride = 1` and `tree_output[t] = t %
  n_classes`. LightGBM and classic XGBoost `multi:softprob` do the same;
  XGBoost 2.x added vector leaves (unverified here). GBM refuses multi-output
  regression outright.

The instruction then declares its width, which is idiomatic — `probe` and
`ecall` are already multi-result:

```text
static @4 : model<tree_ensemble(2, 3)>       # n_features, n_outputs
%a, %b, %c = predict @4, %id, %f0, %f1
```

At the SQL level a k-wide call is a struct-valued call, which already landed
(single-eval field access, the shared ecall site).

**Methods compose in SQL.** The seven output methods across the target classes
reduce to two primitives:

| primitive | serves |
|---|---|
| aggregated leaf value | `decision_function` (raw), `predict_proba` (link), `predict_log_proba` (`ln`), `predict` (score, or `argmax` / `> 0.5`) |
| leaf id (`apply`) | `RandomTreesEmbedding.transform` (one-hot of leaf ids — the shape tier), and a gate that localizes a traversal bug to a node instead of a number |

`decision_path` is a sparse node-membership matrix, not a serving shape.

**Open fork this exposes in the current design.** `link` lives in the models
batch, so the kernel applies it and the pre-link score is unreachable — one
model cannot serve both `decision_function` and `predict_proba`. Returning the
raw score and applying the link in SQL is strictly more composable. What
decides it is bit-exactness: `sigmoid`/`softmax` must match numpy, and
DuckDB's `exp` need not. If it diverges, the answer is not to fold the link
back into the tree kernel but to ship `sigmoid`/`softmax` as small confit
builtins implementing scipy's `expit`, keeping one traversal entry.

Two wrinkles to handle when this is built, both of which produce wrong numbers
rather than errors: **softmax is cross-lane** (it normalizes across outputs,
unlike `sigmoid`), and **class counts can be ragged** (`n_classes_ = [2 3]`
padded to `max_n_classes = 3`, so the padding lane is not a probability) —
mask per output, or refuse the ragged case at extraction.

## Deferred

- `mat_stack` (MLP) as a second model kind — same instruction, new header
  spelling.
- Struct call sites with name-checked features.
- `HistGradientBoosting*`: binned thresholds and a different tree
  representation — the missing direction is covered by `missing_left`, the
  binning is not.
- QuickScorer or a vectorized multi-tree walk — a pure layout change behind
  the same instruction, with one constraint the measurement above pins: it
  must keep the **accumulation** sequential in `tree_span` order. Traversal
  can be reordered freely; the reduction cannot. A blocked or SIMD sum
  across trees is exactly the mistake `n_jobs != 1` makes, and it would cost
  bit-exactness for a reduction that is not where the time goes.
- kNN and kernel SVM: they need the training set as state, which is a
  different structure, not a bigger kernel.
