# Serving fitted models

A fitted tree ensemble becomes a **static**, like a params table — prepared
once, then scored by a native instruction with no Python on the row path.
Nothing but Arrow crosses the boundary: no estimator object, no pickle, no
live model reference reaches the engine.

## Quick start

```python
from confit import DuckDBInferFn
from sklearn.ensemble import RandomForestRegressor
from sql_transform import pack_trees

FEATURES = ["price", "sqft"]
est = RandomForestRegressor(n_estimators=200, n_jobs=1).fit(X, y)

fn = DuckDBInferFn(
    "SELECT tree_predict('price_model', 0, struct_pack(price := price, sqft := sqft)) AS p "
    "FROM __THIS__",
    row_tables={"__THIS__": Row},
    static_tables={},
    models={"price_model": pack_trees([est], FEATURES)},
)
fn.infer_rows(rows)
```

## The SQL surface

```sql
tree_predict(<model set name>, <id expr>, struct_pack(<name> := <expr>, ...))
```

- The **model set name** is a string literal, hoisted at build into a
  `model<tree_ensemble(k)>` static. Not a column reference — no ambiguity with
  a column called `price_model`.
- The **id** is any `BIGINT` expression: which model in the set to score.
  Usually a params-map probe result.
- The **features** are a `struct_pack`, resolved **by name** against the
  entry's `features` list. Call-site order is free; a name that is not a
  declared feature, a missing one, or a duplicate is a build error. `INTEGER`
  features promote to `DOUBLE`.

## One model per group

The real serving shape: fit per group, join the group's model id, score.

```python
params = pa.table({"country": ["de", "fr"], "est": [0, 1]})   # dense ids
models = {"m": pack_trees([fit_de, fit_fr], FEATURES)}              # same order

sql = """
SELECT tree_predict('m', p.est, struct_pack(price := t.price, sqft := t.sqft)) AS p
FROM __THIS__ AS t
LEFT JOIN params AS p ON t.country = p.country
"""
```

An unseen country misses the `LEFT JOIN`, so `p.est` is NULL and the output is
NULL. A model set's ids are dense from 0 and follow the sequence you passed to
`pack_trees`, so the params table indexes straight into it.

## Which estimators pack

| estimator | aggregation | notes |
|---|---|---|
| `DecisionTreeRegressor` | single tree | |
| `RandomForestRegressor` | mean | |
| `ExtraTreesRegressor` | mean | |
| `GradientBoostingRegressor` | sum, seeded with the init | `init="zero"` and the default `DummyRegressor` only |

Everything else refuses by name at `pack_trees` time. Two refusals are worth knowing
because the objects *look* packable:

- **Classifiers** have a `tree_` exactly like a regressor, but their leaf
  `values` are per-class scores rather than the number `predict` returns.
  Packing one would score class-0 fractions and look entirely plausible.
- **`BaggingRegressor`** gives each tree a feature *subset*
  (`estimators_features_`), so that tree's `feature` ids are subset-local.
  Reading them as global indices scores the wrong columns.

`HistGradientBoostingRegressor`, XGBoost and LightGBM use different tree
representations; they would need their own packer emitting the same two
tables. The engine side is already general — see below.

## Parity with sklearn — read this before trusting a number

### The float32 grid, and why the packed thresholds are not sklearn's

sklearn narrows `X` to **float32** before traversal and keeps thresholds in
float64, so it splits on `float32(x) <= threshold`. The threshold *is* the
float32 midpoint of two neighbouring training values, because the split was
**learned** on that grid — so comparing the raw double is not a more precise
evaluation of that model, it is a different model.

Left alone, that cost 157 of 3000 rows on a 2-decimal price grid with
`RandomForestRegressor(30)`, max delta 0.43 against a −7.9..19.4 range: a
whole-leaf jump, not a rounding wobble. Continuous float64 draws hide it —
the mismatch band is about one float32 ULP wide — so it is quantized
features, prices and percentages and any decimal grid, that hit it.

`pack_trees` closes it at **build time**. Rounding to float32 is monotone, so
`float32(x) <= t` is still a single cutpoint in the doubles, and moving the
cutpoint reproduces it exactly:

```python
threshold        = 0.15000000223517418   # sklearn's, == mean(f32(0.1), f32(0.2))
packed threshold = 0.14999999850988385   # ours: last double still narrowing to <= it
x = 0.15   ->  sklearn right, packed right   (raw f64 went left)
```

So **the `threshold` column does not match `tree_.threshold`, deliberately**.
The one threshold left untouched is `+inf`, which sklearn writes for "every
non-missing value goes left" and which already admits every non-NaN double.
A float64-trained library's packer would simply skip this step.

### The rewrite is exact for a DOUBLE feature; an INTEGER one needs one more step

The rewritten cutpoint answers `float32(x) <= t` for whatever double `x` it is
handed. For a `DOUBLE` feature that is the whole story. For an **integer**
feature it is not, because the value handed over is `float64(n)`, and above
`2**53` that has already rounded once — so the comparison was
`float32(float64(n))` where sklearn, which narrows an int64 array to float32
in a single step, computes `float32(n)`. A whole float32 ULP apart.

An integer feature therefore converts with the IR's `itof.f32` rather than the
ordinary promotion: `n as f32 as f64`, one rounding. Below `2**53`,
`float64(n)` is exact and the two are identical, so this is a no-op for every
ordinary integer feature rather than a trade (TASK-77).

That opcode adds no float32 TYPE — its lane is f64 out, the same way
`ftoi.nearest` is a rounding mode and not an integer type. The engine still
computes in exactly `i64` / `f64` / string / bool, and no cast lands on the
row path.

`test_threshold_rewrite_reproduces_the_f32_comparison` walks float64 ULPs
across each rewritten boundary rather than sampling, because a rewrite that is
off by a single ULP still passes an end-to-end parity test on 1500 rows —
measured.

### Summation order

With the branch decisions identical, what is left is arithmetic — and it is
why parity is asserted at `==` on raw doubles rather than at a tolerance: the
packing
mirrors each family's own summation order rather than a tidier one:

- a forest sums its trees in order, then divides;
- a boosted model **seeds** its accumulator with the init prediction and adds
  `learning_rate * value` per stage — the scaling is folded into the leaf
  values at pack time so the engine performs the same multiply-then-add on the
  same double.

Measured 2026-08-07: applying the base *after* the sum instead diverges on up
to 1365 of 2000 rows (632 ULP), and `arr.sum(axis=1)`'s pairwise order on up
to 1853 of 2000 (320 ULP) — differences invisible at repr precision.

`n_jobs=1` is the reference serving configuration, not a workaround. At a
serving latency budget, per-request parallelism is not on the table;
parallelism goes across rows, never inside one prediction. A forest run with
`n_jobs != 1` does not reproduce *itself* run to run.

## Missing values

Follows sklearn per node, not a house rule.

| input | behaviour |
|---|---|
| `NaN` feature | takes that node's `missing_left` direction |
| `NULL` feature | presented to the model as `NaN` — same path, same answer |
| unseen group (probe miss / NULL id) | output is `NULL` |
| id naming no model | **raises** |

A NULL feature is deliberately *not* NULL-in-NULL-out: the model has a defined
answer for missing. A bad id is different — that is a broken join in your
params table, and nulling it would hide the break.

## Building the tables yourself

For a library `pack_trees` does not know, emit these two tables directly. The engine
never sees sklearn.

```python
models={"m": {"nodes": nodes_table, "models": header_table, "features": [...]}}
```

**`nodes`** — one row per node, grouped by model then tree:

| column | type | |
|---|---|---|
| `model_id` | int64 | dense from 0; a model's rows are contiguous |
| `tree_id` | int64 | a tree's rows are contiguous |
| `node_id` | int64 | dense from 0 **within each tree** |
| `feature` | int32 | `-1` marks a leaf |
| `threshold` | float64 | `feature <= threshold` goes left |
| `left`, `right` | int32 | tree-local node ids, `-1` on a leaf |
| `missing_left` | bool | where `NaN` goes, per node |
| `value` | float64 | the leaf's contribution |

**`models`** — one row per model:

| column | type | |
|---|---|---|
| `model_id` | int64 | dense from 0 |
| `base` | float64 | seeds the accumulator for `sum`, added after for `mean` |
| `agg` | string | `"sum"` or `"mean"` |
| `link` | string | `"identity"` or `"sigmoid"` |

int32 and int64 are both accepted for the id, child and feature columns.
A NULL anywhere in either table is a build error naming the row.

Decoding walks the pyarrow buffers directly — a 100k-node forest costs no
Python objects.

## Build-time refusals

Each names the offending row or field, before any data flows:

- a child index out of range, or a child that does not follow its parent
  (that ordering is what makes traversal provably terminate);
- a node unreachable from its tree's root;
- a leaf with children, or a split node missing one;
- `feature` beyond the declared width;
- a node id out of dense order;
- an unknown `agg` or `link` spelling;
- non-dense or non-contiguous `model_id`, a model with no nodes, an empty
  model set;
- a call-site feature name that is not declared, missing, or duplicated;
- a model set name that was not passed to the constructor.

## Which backend runs it

The kernel is native Rust either way: the interpreter calls it directly,
Cranelift emits a call to an `extern "C"` shim over the same routine. Scoring
cost is identical; only the surrounding row code differs.

Note that `DuckDBInferFn` **discards the Cranelift compile error and falls
back to the interpreter silently**. If you are measuring, assert the engine:

```python
assert fn.backend == "cranelift"
```

`SPECIALIZER_FORCE_INTERP=1` pins the interpreter.
