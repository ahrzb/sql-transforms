# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T18 | Periodic `SplineTransformer` with `handle_missing="zeros"` answers ±inf as its twin (NaN) | `claude/native-spline-inf` | — | `spline.py`, `catalog_test.py` | inline (previous supervisor) | #406 | in review |

T16 (#403) and T17 (#402) merged on 2026-10-06, and coverage is 31 of 68.
The loop is moving from session `session_01RPafXSAhursBFF5evqW3JD` to
the "Transforms Loop" project thread. The previous supervisor finishes
T18, which it found while it reviewed #404, and then hands over.

The next ticket is ±inf in the fixture generator's edge values. The row
generator draws no ±inf, which is why the gate missed T18; #404
recommends the change. It changes every class's draws, so it runs alone,
after T18. Every other "not yet" row waits on the owner (the five records
in `decisions/open/`; draft #404 holds research and a recommendation for
each) or needs a design first (PLANS "Later"). Later tickets come from
the partly native classes' "Left Python" lines. Inline: reading 4 after
T18, covering waves 4 to 6.

## T18: periodic `SplineTransformer` at ±inf

**Why.** With `extrapolation="periodic"` and `handle_missing="zeros"`,
the twin answers NaN for a feature at ±inf (`np.remainder` of ±inf is
NaN), and the entry answered 0.0. Over the 398 served spline
configurations, 22 parted from the twin, all periodic with "zeros":
degree 0 at any knot count, and from `n_knots = degree + 3` up.

**Do.** In `spline.py`, send `abs(x) = inf` to NaN when the period is
positive. Add a test that fails without the fix, and a periodic
`handle_missing="zeros"` fixture. Re-run the grid and the spline
fixtures at 200 seeds. The PR (#406) holds the evidence.
