# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T6 | `SplineTransformer`, dense output | `claude/native-spline` | — | `catalog_test.py`, `__init__.py`, PLANS | `session_014t6cVSh3kwJS3HYM64qom4` | #384 | in review |
| T11 | A bound per configuration (`FunctionTransformer`'s transcendentals) | `claude/native-bounds` | — | `_registry.py`, `compose.py`, `coverage.py`, `function.py`, `catalog_test.py`, PLANS | `session_01Xdhd9vA5E9cnTuGCUZcskZ` | | in progress |
| T12 | `QuantileTransformer` as one search tree per feature | `claude/native-quantile-tree` | — | `quantile.py`, `catalog_test.py`, PLANS | `session_01U2RFd5WLRad6pyPaHCugdS` | | in progress |

Next up, once a slot frees: `AdditiveChi2Sampler`, after T11. Its `log`
feeds `cos` and `sin`, and an argument one rounding apart moves them by
about the argument's size times as many ulps, so it may need a record in
`decisions/open/` rather than a bound. Inline: the milestone reading at
the end of wave 3 (after T6).

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

## T11: A bound per configuration

**Goal.** An entry's parity bound can depend on the fitted estimator, not
only on its class. Then `FunctionTransformer` can serve numpy's `exp`,
`log`, `log2` and `tan` (1 ulp from DuckDB's), `log10` (2) and `cbrt` (3)
within their measured bounds, while the identity and the exact functions
stay bit-exact.

**Today.**
- `translates(cls, ulps=n)` (`_registry.py`) stores `Entry(translate,
  ulps)`.
- `bound(step)` is the loosest of its instances' *classes*, and `check()`
  (`_check.py`) defaults to it.
- `compose.py` refuses a step or part whose entry has `ulps`.
- `coverage.py` prints "bit-exact" or "within n ulps" per class.
- `function.py` refuses `_NOT_EXACT` because the class is registered
  bit-exact. Those distances were measured with numpy 2.5.1 against DuckDB
  1.5.5 on x86-64 with AVX-512, over 1,600,000 draws.

**The change.** This is shared machinery (`_registry.py`, `compose.py`,
`coverage.py`, and `_check.py` if needed); say so at the top of the PR.
- A translator can declare a bound per estimator. One shape: `ulps` stays
  the class's ceiling, and the entry gains `bound(est) -> int`, which
  defaults to the ceiling. Pick the shape, and say which and why.
- `bound(step)`: the loosest of its instances' own bounds.
- `Pipeline`, `ColumnTransformer` and `FeatureUnion` compose only parts
  whose own bound is 0. They read the part's bound, not its class's, so
  `FunctionTransformer()` stays composable while `FunctionTransformer(np.exp)`
  is refused there, with its bound named.
- `coverage.py`: a class whose bound varies says so, for example
  "bit-exact; within 3 ulps for some configurations". Keep the KPI
  `nonzero_ulp_bounds` meaningful (`loops/native/report-format.md`). If its
  definition changes, change the format doc in the same PR.
- A class registered with `ulps=n` and no per-estimator bound reads exactly
  as it does today.

**FunctionTransformer.**
- Serve `np.exp`, `np.log`, `np.log2`, `np.tan`, `np.log10` and `np.cbrt`
  at their measured bounds, each from its DuckDB function (`exp`, `ln`,
  `log2`, `tan`, `log10`, `cbrt`).
- The twin answers numpy's IEEE values where DuckDB may raise: `log` of 0
  is -inf and of a negative is NaN, `exp` can overflow, `tan` of an
  infinity is NaN, and NaN passes through. Check each of these in DuckDB.
  Guard them as `_trig` guards `sin`/`cos` (`function.py`), in the guard
  shapes confit reads as non-trapping (#362, #375), so a struct field read
  stays linear in the width.
- numpy picks its SIMD kernels by CPU, so a bound measured here holds
  here. Probe at first use, as `kernel_is_confits` does: measure each
  function on a fixed sample against confit, and refuse (`NotNative`)
  where the distance exceeds the declared bound.
- `log1p`, `expm1` and every other function stay refused, as today.

**Fixtures.** Each function, validated and not, over the generator's rows
plus the domain edges above.

**Rules.**
- A bound is measured over at least 200 seeds and cited where it is
  declared.
- A bound that is not small is a record in `decisions/open/`, not code.

**confit today.** The guards may need `ln`/`log2`/`log10` under a CASE to
count as non-trapping, as #362 made `ln` and `sqrt`. If confit counts one
of them as trapping, measure the width at which a struct read turns
quadratic. Then send a reproduction to the supervisor for the confit loop,
and cap the width meanwhile.

**Acceptance.**
- Gate green. Report `NATIVE_SEEDS=200` over the FunctionTransformer
  configurations, with the largest distance measured per function.
- Every refusal named and tested.
- Coverage regenerated.
- In PLANS: the "Next" item "A bound per configuration" removed, the
  `FunctionTransformer` "Left Python" line updated, and the "Non-linear
  maps" item updated with what `AdditiveChi2Sampler` waits on now.
- The PR states the machinery change and its shape, and the build and
  serve times of the widest fixture.

**Branch:** `claude/native-bounds` (create it from `origin/master`). This is
native T11 on the board, `loops/native/tickets.md`.

## T12: `QuantileTransformer` as one search tree per feature

**Goal.** Each feature's two `np.interp` searches become one balanced CASE
tree, so a feature builds in about half today's time or less and the caps
(`MAX_QUANTILES`, `MAX_SQUARES` in `quantile.py`) can rise. The entry stays
bit-exact.

**Today.** `quantile.py` answers a feature strictly inside `(q[0],
q[-1])` with `0.5 * (_tree(x, up) - _tree(-x, down))`. That is two CASE
trees over the same breakpoints: `down` holds the pieces of
`np.interp(-x, -q[::-1], -r[::-1])`. Two trees in one expression build in
about 3.8 times one tree's time, growing a little faster than the quantile
count (PLANS "Needs from confit", "Two CASE trees"; confit #383 puts the
superlinear term in Cranelift's `define_function`). Measured on a release
build, one feature: 1.3–1.4 s at 1,000 quantiles and 4.4–4.7 s at 2,000
once warm; the first build in a process is about 1.6 times slower.

**The idea** (PLANS "Next", "`QuantileTransformer` as one search tree per
feature"): bisect once, on `x`.
- Strictly between two neighbouring distinct breakpoints, both searches
  land on the same interval. A leaf computes both lines, each in the
  twin's own arithmetic (`_line`, with the slopes precomputed as today).
- At a breakpoint `x == q[j]`, the ascending search takes the interval
  that starts at `q[j]`, the last of a run of equal quantiles. The
  mirrored search takes the interval that ends there, the first of the
  run. So a leaf over `[q[j], q[j+1])` needs an `x == q[j]` arm for the
  mirrored side.
- Derive which piece each search lands on, exactly, from numpy's
  `binary_search_with_guess` (cited in `quantile.py`'s docstring),
  including runs of equal quantiles and the infinite-slope exact arms.
  Write the derivation into the module docstring.

**Rules.**
- 0 ulps; `interp_is_numpys` stays.
- Re-measure the build time: release build, warm and first-in-process, at
  1,000, 2,000 and 4,000 quantiles over one feature, and at the slowest
  shapes the caps allow.
- Raise `MAX_QUANTILES` and `MAX_SQUARES` to what builds in about today's
  worst case (about 7 s warm), and cite the measurement in the comment.
- Keep the old two-tree spelling only if a configuration needs it, and say
  why.

**Fixtures.** The existing `FIXTURES[QuantileTransformer]` plus your own
tests:
- runs of equal quantiles, and quantiles at signed zeros;
- rows exactly at each breakpoint and at its `nextafter` neighbours;
- fits whose slope overflows (the exact arms).

**confit today.** One tree's build time grows a little faster than its
size (T9 measured 0.46 s at 1,000 thresholds and 5.9 s at 8,000), far
below two trees in one expression. Nothing known is missing.

**Acceptance.**
- Bit-exact: 8 seeds in the gate. Run `NATIVE_SEEDS=200` over the
  `QuantileTransformer` configurations and report it.
- Gate green. Coverage is unchanged; regenerate it anyway.
- In PLANS: remove that "Next" item, and update the "Two CASE trees" entry
  under "Needs from confit" to say whether the catalog still needs it.
- The PR gives the old and new build times side by side, and the new caps.

**Branch:** `claude/native-quantile-tree` (create it from `origin/master`).
This is native T12 on the board, `loops/native/tickets.md`.
