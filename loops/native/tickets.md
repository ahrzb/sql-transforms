# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T6 | `SplineTransformer`, dense output | `claude/native-spline` | — | `catalog_test.py`, `__init__.py`, PLANS | `session_014t6cVSh3kwJS3HYM64qom4` | | in progress |
| T8 | `ColumnTransformer` and `FeatureUnion` of catalog entries | `claude/native-compose` | — | `compose.py`, `catalog_test.py`, PLANS | `session_01YVd4KRieYvTaH5SkrQPYX4` | | in progress |
| T9 | `IsotonicRegression` as a transformer | `claude/native-isotonic` | — | `catalog_test.py`, `__init__.py`, PLANS | `session_0189CHkotRvVeto6vtDu3kkh` | | in progress |
| T10 | `FeatureAgglomeration` (pooling) | `claude/native-agglomeration` | — | `catalog_test.py`, `__init__.py`, PLANS | `session_01GQwHcGiJBqxR32mtsBQf4r` | | in progress |

Next up, once a slot frees: `AdditiveChi2Sampler` (its `log`, `cos` and
`sin` are numpy's own kernels, so it waits on a bound per configuration,
PLANS "Next" item 5). Inline: served compositions in coverage.md, and
`QuantileTransformer` as one search tree per feature.

## T6: `SplineTransformer`, dense output

**Goal.** `sklearn.preprocessing.SplineTransformer` serves natively,
bit-exact: per feature, the B-spline basis values at `x`, with each of the
five extrapolations.

**The twin.** sklearn 1.9 `SplineTransformer.transform`
(`sklearn/preprocessing/_polynomial.py`), which calls scipy 1.18
`BSpline.__call__` on each fitted `bsplines_[i]` (`t`, `c`, `k`,
`extrapolate`). `__call__` runs `_dierckx.evaluate_spline`, compiled C++:
read de Boor's recurrence (`_deBoor_D`) and the interval search in scipy
1.18's source (`scipy/interpolate/src/__fitpack.h`/`.cc` on GitHub). What
`transform` does per feature:
- `continue`, `error`, `periodic`: `spl(x)`. `periodic` first maps
  `x` to `t[k] + np.remainder(x - t[k], t[n] - t[k])` (zeros when the
  period is 0). NaN rows become 0 (`handle_missing="zeros"`).
- `constant`, `linear`: `spl(x)` inside `[t[k], t[-k-1]]`, else 0. Then,
  below or above that range, the first or last `degree` lanes take
  `spl(xmin)`/`spl(xmax)` (`constant`) or `f + (x - xmin) * fp` with
  `fp = spl(xmin, nu=1)` (`linear`; `degree + 1` lanes when
  `degree <= 1`). NaN fails every comparison, so it gives 0.
- `error`: raises when any output is NaN (`x` past the knots).
- `include_bias=False` drops each feature's last lane.

**Pointers.**
- `polynomial.py` is the closest family. A new module `spline.py` is
  fine.
- The boundary constants (`spl(xmin)`, `spl(xmin, nu=1)`, ...) are the
  fitted instance's own numbers. Compute them with the twin's function at
  translation time and embed them as literals.

**Traps.**
- The interval search must pick the same interval as scipy's, including
  `x` equal to a knot and `x == t[n]` (the last interval is closed).
  Fixture rows exactly at the knots.
- De Boor's recurrence in scipy's exact operation order. Each lane is
  `sum_a c[l + a - k, j] * h[a]`, with `c` the identity, so a lane is one
  basis value plus exact zeros. Show that, and that `-0.0` cannot leak.
  Under `periodic`, `c` folds the last `degree` bases onto the first, so a
  lane is a sum of two terms: keep the order.
- `np.remainder` is not DuckDB's `%`/`fmod`: numpy takes `fmod`, then
  adds the divisor when the signs differ, and copies the divisor's sign
  onto a zero. Spell exactly that.
- Expression size grows with `n_knots`, the lanes and `degree^2`.
  `MAX_LANES` applies. Measure build and serve times for the widest
  configuration and put them in the PR.

**Rules.**
- `sparse_output=True`: `NotNative`
  (`decisions/open/sparse-outputs.md`).
- `extrapolation="error"`: raise with `error()` on the rows where the twin
  would produce NaN, or answer there. goal.md says the entry may answer
  where the twin raises; pick one and say which in the PR.
- `knots` (`uniform`, `quantile`, an array), `order` and `degree` only
  change fit or memory layout: serve them all.

**Fixtures** (your own `FIXTURES[SplineTransformer]` block):
- degrees 0 to 3 and one higher, `n_knots` 2 to about 8, and each
  `knots` kind;
- the five extrapolations, `include_bias` both ways, `handle_missing` both
  ways (`__sklearn_tags__` turns `allow_nan` on for `"zeros"`);
- rows at the knots, outside the fitted range, and NaN.

**confit today.** Nothing the family needs is known to be missing. On
master: a repeated pure subexpression is computed once per row where it
costs nothing (#363, #377); `round`, guarded `ln`/`sqrt`/`sin`/`cos` and
the total unaries count as unable to trap (#362, #375); `greatest` and
`least` lower as a running extreme (#374). The spline is constants, CASE
arms and DOUBLE arithmetic. If a width is slow, measure it and tell the
supervisor rather than capping it silently.

**Acceptance.** Bit-exact (8 seeds in the gate; run `NATIVE_SEEDS=200`
once over your configurations and report it). Every refusal named and
tested. Gate green, coverage regenerated. The PR states scipy's operation
sequence as you read it, and the timings.

**Branch:** `claude/native-spline` (create it from `origin/master`). This is native T6 on the board, `loops/native/tickets.md`.

## T8: `ColumnTransformer` and `FeatureUnion` of catalog entries

**Goal.** A step whose instances are fitted `ColumnTransformer`s or
`FeatureUnion`s of catalog entries serves natively and bit-exact: each
part's translation over its own columns, concatenated in the twin's order,
with weights applied. `Pipeline` already composes this way (T1, #368).
Together they cover the deployed shape "impute and scale the numbers,
encode the strings, concatenate".

**The twins.**
- sklearn 1.9 `ColumnTransformer.transform`
  (`sklearn/compose/_column_transformer.py`).
- sklearn 1.9 `FeatureUnion.transform` (`sklearn/pipeline.py`).

Both call `_transform_one`, which returns `res * weight` when a weight is
set, and then `_hstack`. Read in the installed sklearn:
- the order of the fitted parts (`transformers_`, `remainder` included);
- how columns are selected on a numpy row: indices, slices, boolean masks
  and callables, resolved at fit. Names need a DataFrame, which the step
  never passes;
- `"drop"` and `"passthrough"`, and `transformer_weights`;
- when `_hstack` returns a sparse matrix (`sparse_output_`,
  `sparse_threshold`).

**Pointers.** Extend `compose.py`, which composes a `Pipeline` through
`catalog()`. A part gets its own columns' expressions and declared types,
and the lanes it returns are DOUBLE.

**Rules.**
- Every part must be a catalog entry registered at 0 ulps (a nested
  `Pipeline` counts), or `"drop"` or `"passthrough"`. Anything else raises
  `NotNative` naming the part. Register both classes at 0.
- Also `NotNative`, naming the cause:
  - a sparse output (`sparse_output_`);
  - a string column passed through (the step reads lanes with `float()`);
  - column specs that need a DataFrame.
- A weight multiplies its part's output. Spell `x * w` with the weight as
  a DOUBLE literal: that is the twin's double.
- Instance dispatch, `null_when` and width work as for any entry.
  `to_native`'s trial build refuses what confit cannot build.

**Fixtures.** Your own `FIXTURES[ColumnTransformer]` and
`FIXTURES[FeatureUnion]` blocks.
- Column specs must work at every width the generator draws, 1 to 32
  features: use slices and callables, or index lists guarded for width.
- `ColumnTransformer`:
  - two and three parts;
  - `remainder` both `"drop"` and `"passthrough"`;
  - one weighted part;
  - an encoder part over string columns beside a scaler over numeric
    ones;
  - a nested `Pipeline` part.
- `FeatureUnion`:
  - two and three parts;
  - one weighted part;
  - a `"drop"` part and a `"passthrough"` part.
- Check what `ColumnTransformer.__sklearn_tags__` and
  `FeatureUnion.__sklearn_tags__` report for input tags (strings, NaN).
- Extend the generator as little as possible (as T1 did with `_runs`), so
  that an encoder part gets string columns and an imputer part gets
  holes, while every other entry's draws stay unchanged. Say what you did.

**confit today.** Nothing the family needs is known to be missing. On
master:
- a repeated pure subexpression is computed once per row where it costs
  nothing (#363, #377);
- the total unaries, `round`, and guarded `ln`/`sqrt`/`sin`/`cos` count as
  unable to trap (#362, #375);
- `greatest` and `least` lower as a running extreme (#374).

If a width is slow, measure it and tell the supervisor rather than
capping it silently.

**Acceptance.**
- Bit-exact: 8 seeds in the gate. Run `NATIVE_SEEDS=200` once over your
  configurations and report it.
- Every refusal named and tested.
- Gate green, coverage regenerated. Both classes are "composition" rows in
  coverage.md: leave `coverage.py`'s categories as they are. PLANS "Next"
  item 3 (showing served compositions) is a separate ticket; say what you
  would show.
- The PR states the twins' operation sequence as you read it, and the
  build and serve times of the widest fixture.

**Branch:** `claude/native-compose` (create it from `origin/master`). This
is native T8 on the board, `loops/native/tickets.md`.

## T9: `IsotonicRegression` as a transformer

**Goal.** `sklearn.isotonic.IsotonicRegression` serves natively and
bit-exact: the fitted step function interpolated linearly, as its
`transform` answers. It is a one-feature, one-lane step.

**The twin.** sklearn 1.9 `IsotonicRegression.transform` → `_transform`
(`sklearn/isotonic.py`):
- `check_array` on `T` with the thresholds' dtype. NaN and infinity raise,
  so those rows are not compared.
- with `out_of_bounds="clip"`, `np.clip(T, X_min_, X_max_)`;
- then `self.f_(T)`. `f_` is built in `_build_f`: a constant when one
  threshold is left, else scipy 1.18's `interp1d(X_thresholds_,
  y_thresholds_, kind="linear", bounds_error=out_of_bounds == "raise")`.

Read scipy's `interp1d._call_linear` and `_check_bounds`
(`scipy/interpolate/_interpolate.py`, installed). In order:
- the index is `searchsorted(x, x_new)` (side left), clipped to
  `[1, len(x) - 1]`;
- the value is `((x_new - x_lo) / (x_hi - x_lo)) * y_hi + ((x_hi - x_new)
  / (x_hi - x_lo)) * y_lo`, in exactly that order;
- below `x[0]` or above `x[-1]` the answer is `fill_value`, NaN by
  default. `"raise"` raises instead.

**Pointers.**
- `quantile.py` already spells a piecewise-linear search (`_pieces`,
  `_tree`), but with `np.interp`'s semantics, not `interp1d`'s. The search
  side, the lack of an exact arm at a knot, and the arithmetic all differ.
  Reuse the tree shape, not the leaf arithmetic.
- Precompute what the twin computes from fitted numbers alone (`x_hi -
  x_lo` per interval) only where the twin computes it the same way.
  Spell the rest in the twin's order.

**Rules.**
- `out_of_bounds`:
  - `"nan"`: NaN outside the range;
  - `"clip"`: clip first, as the twin does;
  - `"raise"`: the entry may answer or raise where the twin raises
    (goal.md, "tolerated differences"). Say which in the PR.
- Thresholds with a dtype other than float64: `NotNative` (the twin casts
  to that dtype).
- One threshold left (a constant prediction) serves as the constant.
- Register the class at 0 ulps. A configuration you cannot make bit-exact
  is `NotNative` with its reason, never a bound.

**Fixtures** (your own `FIXTURES[IsotonicRegression]` block):
- `increasing` True, False and `"auto"`;
- each `out_of_bounds`;
- `y_min`/`y_max` set;
- fits with ties in X;
- a fit that leaves one threshold;
- rows exactly at thresholds, between them, and outside the range.

The step has exactly one feature. Check whether the generator draws one
feature for this class: its `__sklearn_tags__` and the 1-D input
requirement. Say what you did to get one.

**confit today.** Nothing the family needs is known to be missing (CASE
trees, DOUBLE division and arithmetic). PLANS "Needs from confit" has one
entry, about two CASE trees in one expression. A single tree builds
linearly.

**Acceptance.**
- Bit-exact: 8 seeds in the gate. Run `NATIVE_SEEDS=200` once over your
  configurations and report it.
- Every refusal named and tested.
- Gate green, coverage regenerated (`IsotonicRegression` leaves "not
  yet").
- Move it out of PLANS "Later", where it sits with the transformers that
  read their fit samples. It reads only its fitted thresholds.
- The PR states scipy's operation sequence as you read it, and the build
  and serve times of the widest fixture: the most thresholds.

**Branch:** `claude/native-isotonic` (create it from `origin/master`). This
is native T9 on the board, `loops/native/tickets.md`.

## T10: `FeatureAgglomeration` (pooling)

**Goal.** `sklearn.cluster.FeatureAgglomeration` serves natively and
bit-exact: each output lane pools the fitted cluster's features of the
row.

**The twin.** sklearn 1.9 `FeatureAgglomeration.transform`
(`sklearn/cluster/_feature_agglomeration.py`, installed). It runs
`validate_data` (NaN and infinity raise, so those rows are not compared),
then:
- **`pooling_func == np.mean`, dense input** (the default): each row is
  `np.bincount(labels_, X[i, :]) / size`, where `size =
  np.bincount(labels_)`. Read numpy's `bincount` with weights. Each bin is
  a sequential sum from 0.0, adding the row's values in feature order,
  then divided by the bin's count. Check that order in numpy 2.5's source
  or by probe, and say which.
- **any other `pooling_func`:** `pooling_func(X[:, labels_ == l],
  axis=1)` for each label in `np.unique(labels_)`.

**Pointers.**
- `_helpers.py` has the row reductions the catalog already uses
  (`row_sum` is numpy's pairwise sum). `bincount` is not pairwise, so do
  not reuse it blindly.
- `scalers.py` (`Normalizer`) is the closest family shape: one reduction
  per lane.
- confit computes a repeated pure subexpression once per row (#363,
  #377). Every lane reads several features, but none is repeated.

**Rules.**
- Serve `np.mean`, the default.
- `np.max` and `np.min` are exact on the finite rows the twin answers
  (`greatest`/`least`, which confit lowers as a running extreme, #374).
  Serve them if you verify that numpy's `max(axis=1)` answers the same
  double, signed zeros included.
- Any other `pooling_func` (`np.median`, a lambda, ...): `NotNative`,
  naming it.
- Register the exact class only, at 0 ulps.

**Fixtures** (your own `FIXTURES[FeatureAgglomeration]` block):
- `n_clusters` from 1 up to the number of features;
- the linkages (`ward`, `complete`, `average`, `single`);
- `pooling_func` mean, and max/min if served;
- clusters of one feature and of many.

The generator draws the feature count, so `n_clusters` must fit every
width: use a factory that caps it, or say how you handled it. Check the
twin on rows with signed zeros. Mean pooling starts its sum from 0.0, so a
cluster of -0.0 values pools to 0.0.

**confit today.** Nothing the family needs is known to be missing.

**Acceptance.**
- Bit-exact: 8 seeds in the gate. Run `NATIVE_SEEDS=200` once over your
  configurations and report it.
- Every refusal named and tested.
- Gate green, coverage regenerated (`FeatureAgglomeration` leaves "not
  yet").
- Add the family's lines to PLANS. It is not in PLANS today.
- The PR states the twin's operation sequence as you read it (numpy's
  bincount order included), and the build and serve times of the widest
  fixture.

**Branch:** `claude/native-agglomeration` (create it from
`origin/master`). This is native T10 on the board,
`loops/native/tickets.md`.
