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

Three candidate homes for a fitted model, and only one survives contact with
the IR:

| candidate | verdict |
|---|---|
| a value in the params map | **impossible today**: map values are `Vec<ScalarVal>` — scalars only ([`exec/mod.rs`](../../../packages/confit/src/specializer/exec/mod.rs) `StaticData::Map`). List-typed IR values are a far bigger change than one opcode. |
| an `ecall` extern | that is the Python trampoline (DRAFT-22). Wrong tier: a built-in needs no GIL, no `ExternImpl` box, and should be type-checked and foldable like any other op. |
| **a prepare-time declaration** | chosen. Mirrors `regex`, which is already hoisted from a literal at build. Materialization layout is a backend decision, exactly as the design doc already licenses for maps. |

The layout freedom matters: `predict` can become QuickScorer, or grow a
vectorized multi-tree walk, without touching the IR. The CFG stays acyclic —
the traversal loop lives inside the kernel, so the "lift when one needs a
loop, e.g. QuickScorer" caveat in
[`ir/mod.rs`](../../../packages/confit/src/specializer/ir/mod.rs) stays
deferred rather than spent.

## IR surface

```text
program   := static* regex* model* extern* func
model     := "model" "@" INT ":" "tree_ensemble" "(" INT ")"    # INT = n_features
```

One instruction, variadic like `probe` and `ecall`:

```text
%d = predict @N, %id, %f0, .., %f{k-1}      # %id: i64, %fj: f64, %d: f64
```

Bare scalars in, bare scalar out: the null lane never enters the kernel and
stays ordinary `i1` algebra in the caller, per the IR's null-lane rule.

Verifier rules, all local (the header carries the arity, so nothing needs
prepare-time data to check):

- `@N` names a declared model; operand count is exactly `n_features + 1`.
- `%id` is `i64`; every feature operand is `f64`; the result is `f64`.
- `n_features >= 1`.

Round-trip closure (`parse(print(p)) == p`) extends to the new declaration
and instruction; `gen.rs` grows a case so the property tests cover them.

### Lowered example

```text
# SELECT tree_predict('trees', m.mid, price, sqft) AS pred
#   FROM row r LEFT JOIN models m ON (r.region IS NOT DISTINCT FROM m.region)

static @2 : map(str) -> (i64)
model  @3 : tree_ensemble(2)

b0:
  %hit, %id = probe @2, %region
  %pv, %f0  = load.opt in.price
  %sv, %f1  = load.opt in.sqft
  %a  = and %hit, %pv
  %ok = and %a, %sv
  brif %ok, b1(%id, %f0, %f1), b2
b1(%id: i64, %f0: f64, %f1: f64):
  %p = predict @3, %id, %f0, %f1
  store.opt out.pred, true, %p
  emit
b2:
  %z = const.f64 0.0
  store.opt out.pred, false, %z
  emit
```

Branching on validity rather than scoring-then-discarding is the lowering's
choice; both are correct (a miss leaves `%id` at its defined default), the
branch merely avoids a pointless traversal.

## SQL surface

```sql
tree_predict('trees', <id expr>, <f64 expr>, ...)
```

The first argument is a **string literal naming the model set**, hoisted at
build into a `model @N` declaration — the same treatment regex patterns
already get, so no new name-resolution machinery and no ambiguity with a
column called `trees`. Features are positional `f64` expressions; the count
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

## The boundary

Two Arrow record batches per model set, supplied to `compile` beside
`StaticData` as a `ModelData` payload and type-checked against the
declaration (`n_features` must agree):

```text
nodes:  model_id i64 | tree_id i64 | node_id i64 | feature i32 (-1 = leaf)
                     | threshold f64 | left i32 | right i32 | value f64
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

- **Comparison**: `x <= threshold` goes left. NaN compares false, so a NaN
  feature goes right — sklearn's own convention.
- **NULL**: a NULL feature or a probe miss yields a NULL result; the kernel is
  never handed a NULL. Validity is `i1` algebra in the caller. No
  missing-value direction in v0 — a model trained with one (HistGB) is out of
  scope until the layout carries it.
- **Aggregation**: `base + sum(leaf values)` for `sum`, `base + mean(...)` for
  `mean`, then the link. Summation follows `tree_span` order, so the result is
  reproducible bit-for-bit across backends and runs.
- **Bad id at runtime**: a row-supplied id outside `model_span` traps with a
  named message. (Reachable only when the id is not a static-map value; see
  the refusal above.)

## What it touches

| file | change |
|---|---|
| `ir/mod.rs` | grammar + doc for `model` declaration and `predict` |
| `ir/parse.rs`, `ir/print.rs` | round-trip for both |
| `ir/verify.rs` | the local rules above |
| `ir/gen.rs` | generator case, so property tests cover it |
| `exec/mod.rs` | `ModelData` beside `StaticData`; declaration type-check |
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

## Sequencing

1. IR: declaration + instruction + verify + round-trip (no execution yet).
2. Kernel: layout, `from_arrow` with its refusals, `predict`, layout check.
3. Interpreter binding + pins.
4. Cranelift binding + backend-equality harness.
5. Frontend: `tree_predict` resolution, literal hoisting, arity/type checks.

Steps 1–2 are independent and can land separately; nothing before step 5 is
reachable from SQL, so each lands green on its own.

## Deferred

- `mat_stack` (MLP) as a second model kind — same instruction, new header
  spelling.
- Struct call sites with name-checked features.
- Missing-value direction in the node layout (HistGB, XGBoost `missing=`).
- QuickScorer or a vectorized multi-tree walk — a pure layout change behind
  the same instruction.
- kNN and kernel SVM: they need the training set as state, which is a
  different structure, not a bigger kernel.
