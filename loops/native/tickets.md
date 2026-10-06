# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T13 | Re-fit the spline build estimate after confit #387 | `claude/native-spline-refit` | — | `spline.py`, `catalog_test.py`, PLANS | `session_014t6cVSh3kwJS3HYM64qom4` | | in progress |
| T14 | Adopt confit #390: `FunctionTransformer(np.cbrt)` | `claude/native-cbrt` | — | `function.py`, `catalog_test.py`, PLANS | `session_01Xdhd9vA5E9cnTuGCUZcskZ` | | in progress |

Next up, once a slot frees: boolean features in the fixture generator (no
entry is tested on a boolean column yet; PLANS "Left Python"), as its own
ticket, since every class's draws change. `AdditiveChi2Sampler` waits on
the owner (`decisions/open/additive-chi2-parity-bound.md`). Inline: the
milestone reading at the end of wave 3.

## T13: re-fit the spline build estimate after confit #387

**Why.** confit #387 ("a shared value is computed just before its first
reader") landed on master after T6 fitted `_build_estimate`, and spline
builds got about 4x faster. In an A/B on one container, alternating confit
at `7702ae1` (before #387) and at master, three runs each, a 16-feature,
degree-3, 8-knot `continue` spline builds in 4.4–4.6 s before and
0.98–1.08 s after. QuantileTransformer is unchanged within noise. So the
estimate now refuses steps that build well under 7 s. On master `81c3344`
(release, cap lifted, warm):

| step | estimate | actual |
|---|---|---|
| 32 features, degree 3, 8 knots, `continue` | 22 s | 2.3 s |
| same, `error` | 11 s | 1.1 s |
| same, `periodic` | 17 s | 2.1 s |
| 16 features, degree 5, 7 knots, `continue` | 20 s | 4.1 s |

**Do.**
- Re-fit the estimate on current master: warm, release, the same grid as
  before, extended to the wider and higher-degree steps now in reach.
- Cite the fit in the comment, as before: the range of the ratio, the
  slowest step accepted and the fastest refused.
- Keep `MAX_BUILD_S` at 7 s, and check that the 4,000,000-token expansion
  cap still refuses where it must (e.g. degree 5 at 32 features).
- If the gate's time allows, relax `narrow`. Report the gate's spline
  share before and after.
- In PLANS, update "A value bound once" under "Needs from confit": #387
  cut the build time, but the expression still doubles per degree, so the
  need stands for size. Update the spline line under "Left Python" with
  the new widths.

**Acceptance.**
- Gate green. Run `NATIVE_SEEDS=200` over the spline fixtures and report
  it.
- The refusal test still passes, past the new estimate.
- The PR gives the old estimate, the new estimate and the actual build
  side by side for the widths above.

**Branch:** `claude/native-spline-refit` (create it from `origin/master`).
This is native T13 on the board, `loops/native/tickets.md`.

## T14: adopt confit #390, serve `FunctionTransformer(np.cbrt)`

**Why.** confit #390 ("cbrt is glibc's, as DuckDB's is") fixed T11's
finding, so confit's `cbrt` should now equal DuckDB's. numpy's is within 3
ulps of DuckDB's over T11's 1,600,000 draws.

**Do.**
- Rebuild confit in release on master, and re-run T11's cbrt reproduction.
  confit against DuckDB 1.5.5 must be 0 apart. If it is not, stop and tell
  the supervisor: that goes back to the confit loop.
- Move `np.cbrt` from `_WAITS` to `_BOUNDS` at its measured bound.
  Re-measure numpy against confit over the same draws and 200 seeds of
  fixtures, and cite both. If the maximum is 3, the class ceiling becomes
  3, and coverage reads "within 3 ulps for some configurations".
- DuckDB's `cbrt` is total, so no guard is needed. Still check ±0, ±inf,
  NaN and the subnormals, in the probe and in SPECIALS.
- Add `np.cbrt` to `BOUNDED` (fixtures, validated and not) and to the
  probe tests. Drop the cbrt refusal tests, or turn them into served
  cases.
- In PLANS, remove the cbrt entry under "Needs from confit", add #390 to
  the list of what confit has served this catalog, and update the
  `FunctionTransformer` line under "Left Python".

**Acceptance.**
- Gate green. Run `NATIVE_SEEDS=200` over the FunctionTransformer
  configurations and report it.
- The KPI `nonzero_ulp_bounds` is unchanged at 2 (FunctionTransformer
  already counts).
- Coverage regenerated.

**Branch:** `claude/native-cbrt` (create it from `origin/master`). This is
native T14 on the board, `loops/native/tickets.md`.
