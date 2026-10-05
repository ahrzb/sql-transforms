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
| T7 | adopt confit #374, #375 and #377: re-measure the max norm, `sin`/`cos` and the multi-instance `QuantileTransformer` | `claude/native-adopt` | — | `scalers.py`, `function.py`, PLANS | inline | #378 | in review |

Next up, once a slot frees: `ColumnTransformer` and `FeatureUnion`, then
`AdditiveChi2Sampler`. Inline: served compositions in coverage.md, and
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
