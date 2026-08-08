# Serving fitted models

`TreeBasedTransform` is `PythonTransform`'s native sibling: the same
construction, the same registration, the same call site. The difference is
invisible from SQL — the engine scores it with a native kernel instead of
calling back into Python, so there is no GIL on the row path.

You hand over the **fitted estimator**. Turning it into the tables the engine
walks happens inside the class, at construction; nothing but Arrow crosses the
boundary, and no estimator object, pickle or live model reference reaches the
engine.

## Quick start

```python
from confit import DuckDBInferFn
from sklearn.ensemble import RandomForestRegressor
from sql_transform import TreeBasedTransform

est = RandomForestRegressor(n_estimators=200, n_jobs=1).fit(X, y)

fn = DuckDBInferFn(
    "SELECT price_model(0, price, sqft) AS p FROM __THIS__",
    row_tables={"__THIS__": Row},
    static_tables={},
    udfs=[
        TreeBasedTransform(
            "price_model",
            instances={0: est},
            takes=pa.schema([("price", pa.float64()), ("sqft", pa.float64())]),
        )
    ],
)
fn.infer_rows(rows)
```

## The SQL surface

```sql
<name>(<id expr>, <feature expr>, ...)
```

Exactly the shape every other declared transform has:

- The **name** is the transform's own `name`, resolved case-insensitively in
  the same namespace as the ecall UDFs. A tree transform and an ecall UDF
  cannot share one — that refuses at construction.
- The **id** is the implicit leading argument, any `BIGINT` expression: which
  instance to score. Never written in `takes`. Usually a params-map probe.
- The **features** follow, bound **by position** in `takes` order. A call
  passing the wrong number refuses by name; so does a non-numeric argument.
  `INTEGER` features convert per the compare grid (below).

The schema is arrow: `takes` is a `pa.Schema` — names and types in one
declaration — and `returns` is the SQL return type, `pa.float64()` for a tree
(scored one number per row), which is the default.

Transposing two features is the caller's mistake to avoid, the same as for any
positional call. The schema's NAMES do not bind the call site; they are
checked against the estimator's `feature_names_in_` at construction, so a
DataFrame-fitted model whose columns you name in the wrong order refuses
instead of scoring plausibly (TASK-78).

## One model per group

The real serving shape: fit per group, join the group's instance id, score.

```python
params = pa.table({"country": ["de", "fr"], "est": [0, 1]})   # dense ids

sql = """
SELECT score(p.est, t.price, t.sqft) AS p
FROM __THIS__ AS t
LEFT JOIN params AS p ON t.country = p.country
"""
udfs = [TreeBasedTransform("score", instances={0: fit_de, 1: fit_fr},
                           takes=pa.schema([("price", pa.float64()),
                                            ("sqft", pa.float64())]))]
```

An unseen country misses the `LEFT JOIN`, so `p.est` is NULL and the output is
NULL. Instance ids must be dense from 0 — that is what the engine indexes on,
and it is what a fitted params table's `__cf_est` column already produces.

## Which estimators pack

| estimator | aggregation | notes |
|---|---|---|
| `DecisionTreeRegressor` | single tree | |
| `RandomForestRegressor` | mean | |
| `ExtraTreesRegressor` | mean | |
| `GradientBoostingRegressor` | sum, seeded with the init | `init="zero"` and the default `DummyRegressor` only |

Everything else refuses by name when the transform is constructed. Two refusals
are worth knowing
because the objects *look* packable:

- **Classifiers** have a `tree_` exactly like a regressor, but their leaf
  `values` are per-class scores rather than the number `predict` returns.
  Packing one would score class-0 fractions and look entirely plausible.
- **`BaggingRegressor`** gives each tree a feature *subset*
  (`estimators_features_`), so that tree's `feature` ids are subset-local.
  Reading them as global indices scores the wrong columns.

`HistGradientBoostingRegressor`, XGBoost and LightGBM use different tree
representations; they would need their own class emitting the same two
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

`TreeBasedTransform` closes it at **build time**. Rounding to float32 is
monotone, so
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

A feature **declared** `pa.int64()` therefore converts with the IR's
`itof.f32` rather than the ordinary promotion: `n as f32 as f64`, one
rounding. Below `2**53`, `float64(n)` is exact and the two are identical, so
this is a no-op for every ordinary integer feature rather than a trade
(TASK-77).

**The declaration decides, not the column.** A BIGINT column passed into a
lane declared `pa.float64()` is cast to a double by DuckDB before the call, so
the model sees `float64(n)` and narrows from there — a different leaf above
`2**53`, and the right one for that declaration. The class's own `__call__`
follows the same rule, which is what makes it a usable oracle: it builds an
int64 array and narrows it in one step for a declared BIGINT lane, and passes
the double through for a declared DOUBLE one. (`np.float32(n)` is *not* that
conversion — the scalar constructor rounds via float64. Only the array
`astype` reproduces it.)

That opcode adds no float32 TYPE — its lane is f64 out, the same way
`ftoi.nearest` is a rounding mode and not an integer type. The engine still
computes in exactly `i64` / `f64` / string / bool, and no cast lands on the
row path.

### Which grid you are on is DECLARED, not assumed

Both of the above are sklearn's semantics, and neither is universal — so a
transform says which floating-point grid its comparisons live on, as the third
member of `tree_tables()`:

```python
def tree_tables(self):
    return nodes, models, "float32"
```

`TreeBasedTransform` rewrites its thresholds and declares `"float32"`. A class
wrapping a library that compares in float64 skips the threshold rewrite and
declares `"float64"`, and the engine then converts its integer features
exactly rather than narrowing them.

Without the field the narrowing would fire for every model, which would make
the wire format quietly sklearn-specific: a float64-grid packer would get its
integer features narrowed anyway, silently losing precision above `2**24` that
it had every right to keep. The threshold rewrite is skippable by a packer;
the conversion is not, so the engine has to be told.

**It is required, not defaulted.** The packer that would get this wrong is
exactly the one that never thought about it, and a default would be the same
trap with an extra step.

**It belongs to the TRANSFORM, not to an instance inside it.** `score(id, ..)`
takes the instance id from a row, so it is a runtime value, while the
conversion is chosen once when the query is lowered. A per-instance grid could
only be honoured with a per-row branch.

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

## Plugging in another library

For a library `TreeBasedTransform` does not know, write your own transform
class. The whole protocol is four attributes and one method — the engine never
sees sklearn, and never calls `__call__` on a class that has `tree_tables`.

```python
class XGBTransform:
    name = "score"
    # names + types in one declaration; the features bind by position
    takes = pa.schema([("price", pa.float64()), ("sqft", pa.float64())])
    returns = pa.float64()
    instances = {0: booster}    # presence is what adds the leading id argument

    def tree_tables(self):
        return nodes_table, header_table, "float64"   # or "float32" if your
        # thresholds were rewritten onto sklearn's grid — see above
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
  model table;
- instance ids that are not dense from 0;
- a call passing the wrong number of arguments, or a non-numeric one;
- two declared UDFs whose names collide case-insensitively, whichever kinds
  they are;
- a declared UDF whose name is a builtin (`least`, `round`, `upper`, …): the
  binder matches the builtin catalogue before it consults the declared UDFs,
  while DuckDB lets a registered function shadow its own builtin, so the two
  engines would answer the same SQL differently;
- a `tree_tables()` that raises or does not return
  `(nodes, models, compare_grid)`.

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
