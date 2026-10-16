# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Reading 7 merged on 2026-10-16
([`reports/2026-10-16-reading-7.md`](reports/2026-10-16-reading-7.md)), on
master f54ddab, after T38 to T41: coverage 41 of 68, with
`sklearn.cross_decomposition` and `sklearn.discriminant_analysis` wholly
native. On one machine, master serves a native call of the bench in a
median of 0.85 times the time of reading 6's commit (confit #459), and the
widest `Pipeline` builds in 0.49 s, against 0.93 s at reading 6 (confit
#459 and #461). This session's machine started on 2026-10-16 and is a
little slower than reading 6's: a twin call takes a median of 1.11 times
as long at the same commit, and T40's builds near the caps take 1.05 to
1.36 times T40's own times, so a build that a cap takes can take up to
about 9 s. In the parity run, 236 `FastICA` steps stay Python: 133 fits
broke down (finding 78), and 103 whiten a component of rounding noise.
T42 (#495), merged after reading 7's commit, serves option 1 of the ruling
on bounded steps in compositions
(`decisions/closed/bounded-steps-in-compositions.md`): with
`allow_bound=True`, a step with a parity bound above 0 serves as the last
step of a `Pipeline`, or as a part of a `ColumnTransformer` or
`FeatureUnion` that has no weight, and `native.check` holds each output
field to the bound of the step that gives it. T43 (#497) fixes findings 81
and 82. At one feature and one output field, the twin of `FastICA` and of
the PLS family multiplies a 1×1 row by a 1×1 matrix with `np.dot`, which
answers the product itself, -0.0 included, so their entries no longer add
0.0 there: over seeds 0 to 199 of the parity run, no output field of the
ten projections differs from the twin in its sign alone, against 54 of
`FastICA` before. The ARPACK configuration of `TruncatedSVD` in FIXTURES
has a `random_state`, and a test draws every fixture configuration under
two global random states. T44 (#498) hands each translator a number as
`x - 0.0`, the same double, which confit computes once a row where a step
reads it often; it computed the input guard's `coalesce(x, NaN)` again at
each read (finding 85). In one process against master 9f74bc0's reads, a
1,024-row call of `SplineTransformer` takes 0.23 to 0.67 times the time,
of `QuantileTransformer` over one feature 0.45 to 0.61, of
`IsotonicRegression` 0.53 to 0.57 (its two ends are tested in one
condition), of `PCA` up to 256 features 0.59 to 0.81 and of a number
`OrdinalEncoder` 0.34; no other step measured is slower in every round. A
projection reads `x - 0.0` only where its build stays within the cap
(`project._reads_apart`). T45 (#499) counts the edges of
`KBinsDiscretizer`'s ordinal bins by bisection, inside a first condition
that tests both ends, so that confit computes the feature's `x - 0.0` once
a row where it computed it again in each arm of a chain of one test per
edge. In one process against that chain, a 1,024-row call takes 0.29 times
the time at 32 float64 features of 31 bins and 0.04 at 8 features of 255
bins (float32: 0.82 and 0.17). T46 (#500) has a translator's tests for the
input guard read each number as the guard does (`_registry._plain`): read
by the guard, ahead of the body, T44's `x - 0.0` rode through the whole
body. In one process, a cubic `SplineTransformer` of 128 features under
`extrapolation="error"` built in 9.2 s with T44's reads, against 4.3 s
before them and 3.6 s with T46; an `OrdinalEncoder` of 256 number
features, with the test of both ends below, in 11.2 s, against 2.7 s and
1.7 s. A number field of an ordinal encoder tests both ends of its classes
first, so that confit computes the feature once a row: at 64 features of
64 classes, a row of a 1,024-row call takes 37 µs, against 32 µs with
T44's reads alone and 83 µs before them. The forest, isotonic, quantile
and spline build estimates are fitted again on T46 (on master 12aca98, on
the machine the session restarted on). The caps reach about 26,800
isotonic thresholds (19,700 at T40), 15,000 quantile leaves (13,400), 170
to 230 spline features of degree 3 and 8 knots (unchanged), and 10,000
forest output fields at 8 features (10,300), but 5,200 at 128 (6,800):
with T44's reads, a forest over 128 features builds in 1.22 times the
time. The loop runs from the "Transforms Loop" project thread. The Confit
Loop is paused since 2026-10-16: a confit need waits, and the loop works
around it.

Next: the rest of PLANS "Ruled 2026-10-06, to build", item 1
(`FactorAnalysis` first), then the rest of item 2.
