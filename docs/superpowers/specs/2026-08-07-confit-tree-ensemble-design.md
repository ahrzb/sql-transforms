# Confit: native tree-ensemble scoring

Date: 2026-08-07
Status: design, from the 2026-08-07 session with AmirHossein
Scope: **confit only**. No `udfs=` surface, no DuckDB registration, no
sql_transform family protocol — those live in
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

### Kernel vs lowering the tree into IR — unresolved, and not blocking

The acyclic rule does **not** rule out the other design: a tree is itself
acyclic, so it can be lowered into nested `brif` blocks with no loop at all —
the model compiled to code rather than interpreted from data. Both are
viable; this document picks the kernel, and the reasons are structural, not
measured:

| | one instruction + kernel | lower the tree into IR branches |
|---|---|---|
| IR surface | 1 declaration + 1 opcode | a lowering pass emitting O(nodes) blocks |
| forest of 100 x 255 nodes | data; program size unchanged | ~25k blocks — cranelift compile time and code size |
| refit | swap the Arrow batch, re-prepare | re-JIT the whole program |
| interpreter backend | native speed for free | interpreted per node — slow |
| small tree (depth <= 6) | indirect loads, loop overhead | likely faster: inlined compares, no indirection |
| large ensembles | can adopt QuickScorer / vectorized walks with no IR churn | i-cache and branch-prediction pressure |
| parity | identical | identical |

**Unmeasured.** The experiment that settles it: per-row latency for (1 tree,
depth 4), (100 trees, depth 8) and (500 trees, depth 12) — kernel versus a
hand-written branch program — plus cranelift compile time and code size at
each point.

The choice is also reversible and invisible to users: the two can coexist,
with the frontend lowering small trees to branches and emitting `predict` for
large ones. Nothing in the SQL surface or the Arrow boundary changes if that
happens, which is why the first cut does not wait on the measurement.

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

## Build-time refusals (P7)

Every one of these names the offending row or field, before any data flows:

- a child index out of range, or a cycle: a node reachable from two parents,
  or unreachable from its tree's root;
- a leaf (`feature = -1`) with children, or an internal node without both;
- `feature >= n_features`;
- unknown `agg` or `link` spelling;
- a `model_id` present in `nodes` but absent from `models`, or vice versa;
- non-dense or duplicated `model_id`;
- call-site arity that disagrees with the declared `n_features`;
- a model id in a *static* map's value column that names no model — the
  static case is fully known at build, so it refuses instead of trapping.

## Pinned runtime semantics

These are the correctness surface; each is a test, not a comment.

- **Comparison**: `x <= threshold` goes left.
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
  the refusal above.)

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

**Float tolerance, to be settled before writing gate 1, not discovered from a
red bar:** a single tree returns a leaf value verbatim, so bit-exact holds.
Forests and boosted models sum across trees, and numpy's `sum` uses pairwise
summation while `tree_span` order is sequential — the two need not agree in
the last ulp. Options: replicate numpy's pairwise order in the kernel, or
declare a per-aggregation-mode tolerance. Decide from a measurement of the
actual divergence on realistic ensemble sizes.

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

## Deferred

- `mat_stack` (MLP) as a second model kind — same instruction, new header
  spelling.
- Struct call sites with name-checked features.
- `HistGradientBoosting*`: binned thresholds and a different tree
  representation — the missing direction is covered by `missing_left`, the
  binning is not.
- QuickScorer or a vectorized multi-tree walk — a pure layout change behind
  the same instruction.
- kNN and kernel SVM: they need the training set as state, which is a
  different structure, not a bigger kernel.
