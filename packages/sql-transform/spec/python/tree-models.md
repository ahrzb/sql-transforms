# Tree models

**claim: tree-transform-construction.** `TreeBasedTransform` is `PythonTransform`'s native sibling: the same
construction, the same registration, the same call site. The difference is
invisible from SQL — the engine scores it with a native kernel instead of
calling back into Python, so there is no GIL on the row path.

You hand over the **fitted estimator**. Turning it into the tables the engine
walks happens inside the class, at construction; nothing but Arrow crosses the
boundary, and no estimator object, pickle or live model reference reaches the
engine.

*Evidence:* `_trees_test.py::test_a_tree_based_transform_registers_like_every_other_transform`.

Confit owns the tree tables, their build checks and the backends: see its
[serving contract](../../../confit/docs/specs/serving-contract.md#tree-tables).

## Example

```pycon
>>> import numpy as np
>>> import pyarrow as pa
>>> from confit import DuckDBInferFn
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sql_transform import TreeBasedTransform
>>> X = np.array([[100.0, 50.0], [200.0, 80.0], [150.0, 60.0], [300.0, 120.0]])
>>> y = np.array([1.0, 2.0, 1.5, 3.0])
>>> fit_de = RandomForestRegressor(n_estimators=5, random_state=0, n_jobs=1).fit(X, y)
>>> fit_fr = RandomForestRegressor(n_estimators=5, random_state=1, n_jobs=1).fit(X, 2 * y)
>>> score = TreeBasedTransform("score", instances={0: fit_de, 1: fit_fr},
...                            takes=pa.schema([("price", pa.float64()), ("sqft", pa.float64())]))
>>> params = pa.table({"country": ["de", "fr"], "est": [0, 1]})
>>> row = pa.schema([("country", pa.string()), ("price", pa.float64()), ("sqft", pa.float64())])
>>> fn = DuckDBInferFn(
...     "SELECT score(p.est, t.price, t.sqft) AS p FROM __THIS__ AS t "
...     "LEFT JOIN params AS p ON t.country = p.country",
...     row_tables={"__THIS__": row}, static_tables={"params": params}, udfs=[score])
>>> rows = [{"country": c, "price": 100.0, "sqft": 50.0} for c in ("de", "fr", "xx")]
>>> answer = fn.infer_rows(rows)
>>> assert answer[0]["p"] == fit_de.predict(X[:1])[0]
>>> assert answer[1]["p"] == fit_fr.predict(X[:1])[0]
>>> assert answer[2]["p"] is None

```

## The SQL call

**claim: tree-call-shape.**

```sql
<name>(<id expr>, <feature expr>, ...)
```

Exactly the shape every other declared transform has:

- The **name** is the transform's own `name`, resolved case-insensitively in
  the same namespace as the declared UDFs. A tree transform and an ecall UDF
  cannot share one — that refuses at construction.
- The **id** is the implicit leading argument, any `BIGINT` expression: which
  instance to score. Never written in `takes`. Usually a params-map probe.
- The **features** follow, bound **by position** in `takes` order. A call
  passing the wrong number refuses by name; so does a non-numeric argument.
  `INTEGER` features convert per the compare grid (below).

The schema is arrow: `takes` is a `pa.Schema` — names and types in one
declaration — and `returns` is the SQL return type, `pa.float64()` for a tree
(scored one number per row), which is the default.

*Evidence:* `_trees_test.py::test_a_tree_based_transform_registers_like_every_other_transform`, `packages/confit/tests/test_tree_predict.py::test_features_bind_by_position`, `packages/confit/tests/test_tree_predict.py::test_feature_count_mismatch_refuses`.

**claim: feature-names-checked.** Transposing two features is the caller's mistake to avoid, the same as for any
positional call. The schema's NAMES do not bind the call site; they are
checked against the estimator's `feature_names_in_` at construction, so a
DataFrame-fitted model whose columns you name in the wrong order refuses
instead of scoring plausibly.

*Evidence:* `_trees_test.py::test_dataframe_fitted_model_refuses_mismatched_schema_names`, `_trees_test.py::test_schema_names_do_not_bind_the_call_site`.

## One model per group

**claim: one-model-per-group.** The real serving shape: fit per group, join the group's instance id, score.

An unseen country misses the `LEFT JOIN`, so `p.est` is NULL and the output is
NULL. Instance ids must be dense from 0, because the engine indexes on them.

*Evidence:* `_trees_test.py::test_per_group_models_score_by_id`, `packages/confit/tests/test_tree_predict.py::test_two_models_score_independently`.

## Which estimators pack

**claim: packable-estimators.**

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
tables. Confit's [tree tables](../../../confit/docs/specs/serving-contract.md#tree-tables) are already general.

*Evidence:* `_trees_test.py::test_matches_sklearn_bit_exactly`, `_trees_test.py::test_unfitted_family_refuses`, `_trees_test.py::test_hist_gradient_boosting_refuses`.

## Parity with scikit-learn

### The float32 threshold grid

**claim: float32-threshold-rewrite.** sklearn narrows `X` to **float32** before traversal and keeps thresholds in
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

```text
threshold        = 0.15000000223517418   # sklearn's, == mean(f32(0.1), f32(0.2))
packed threshold = 0.14999999850988385   # ours: last double still narrowing to <= it
x = 0.15   ->  sklearn right, packed right   (raw f64 went left)
```

So **the `threshold` column does not match `tree_.threshold`, deliberately**.
The one threshold left untouched is `+inf`, which sklearn writes for "every
non-missing value goes left" and which already admits every non-NaN double.
A float64-trained library's packer would simply skip this step.

`test_threshold_rewrite_reproduces_the_f32_comparison` walks float64 ULPs
across each rewritten boundary rather than sampling, because a rewrite that is
off by a single ULP still passes an end-to-end parity test on 1500 rows —
measured.

*Evidence:* `_trees_test.py::test_threshold_rewrite_reproduces_the_f32_comparison`, `_trees_test.py::test_threshold_rewrite_flips_the_ticketed_witness`, `_trees_test.py::test_quantised_features_match_sklearn`.

### Integer features

**claim: integer-feature-narrowing.** The rewritten cutpoint answers `float32(x) <= t` for whatever double `x` it is
handed. For a `DOUBLE` feature that is the whole story. For an **integer**
feature it is not, because the value handed over is `float64(n)`, and above
`2**53` that has already rounded once — so the comparison was
`float32(float64(n))` where sklearn, which narrows an int64 array to float32
in a single step, computes `float32(n)`. A whole float32 ULP apart.

A feature **declared** `pa.int64()` therefore converts with the IR's
`itof.f32` rather than the ordinary promotion: `n as f32 as f64`, one
rounding. Below `2**53`, `float64(n)` is exact and the two are identical, so
this is a no-op for every ordinary integer feature rather than a trade.

**The declaration decides, not the column.** A BIGINT column passed into a
lane declared `pa.float64()` is cast to a double by DuckDB before the call, so
the model sees `float64(n)` and narrows from there — a different leaf above
`2**53`, and the right one for that declaration. The class's own `__call__`
follows the same rule, which is what makes it a usable oracle: it builds an
int64 array and narrows it in one step for a declared BIGINT lane, and passes
the double through for a declared DOUBLE one. (`np.float32(n)` is *not* that
conversion — the scalar constructor rounds via float64. Only the array
`astype` reproduces it.)

*Evidence:* `_trees_test.py::test_integer_feature_above_2_53_matches_sklearn`, `_trees_test.py::test_call_and_kernel_agree_on_an_integer_feature_above_2_53`, `packages/confit/tests/test_tree_predict.py::test_compare_grid_decides_whether_an_integer_feature_narrows`.

### The declared grid

**claim: declared-float32-grid.** `TreeBasedTransform` rewrites its thresholds and declares `"float32"` as the compare grid of `tree_tables()`.

*Evidence:* `_trees_test.py::test_integer_feature_above_2_53_matches_sklearn`, `packages/confit/tests/test_tree_predict.py::test_compare_grid_is_required`.

Confit's [compare grid](../../../confit/docs/specs/serving-contract.md#tree-tables) rules apply.

### Summation order

**claim: summation-order.** With the branch decisions identical, what is left is arithmetic — and it is
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

*Evidence:* `_trees_test.py::test_matches_sklearn_bit_exactly`, `_trees_test.py::test_ecall_and_predict_agree_on_a_boosted_model`, `packages/confit/tests/test_tree_predict.py::test_forest_averages_and_boosted_sums_the_base`.

## Missing values

**claim: tree-missing-values.** Follows sklearn per node, not a house rule.

| input | behaviour |
|---|---|
| `NaN` feature | takes that node's `missing_left` direction |
| `NULL` feature | presented to the model as `NaN` — same path, same answer |
| unseen group (probe miss / NULL id) | output is `NULL` |
| id naming no model | **raises** |

A NULL feature is deliberately *not* NULL-in-NULL-out: the model has a defined
answer for missing. A bad id is different — that is a broken join in your
params table, and nulling it would hide the break.

*Evidence:* `_trees_test.py::test_nan_features_match_sklearn`, `_trees_test.py::test_null_and_nan_are_the_same_input`, `packages/confit/tests/test_tree_predict.py::test_null_feature_takes_the_missing_branch`.
