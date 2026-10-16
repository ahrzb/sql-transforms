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
time. T47 (#501) serves the distances to fitted centres of `KMeans`,
`MiniBatchKMeans`, `BisectingKMeans` and `Birch` (distance.py), with
`allow_bound=True`, within K = n + 5 on the squares: near a centre the
distance itself has no small bound, so the family compares squares
(`ErrorScale.g`), and each message that states the bound says so
(`ErrorScale.on`). Coverage is 45 of 68. Over seeds 0 to 199 of the
parity run (6,386 steps of 1 to 64 features), 53% of 13.6 million output
fields part from the twin, none at one feature; the largest |entry² −
twin²| / (eps·S) is 7.39 (`Birch`, 64 features, K = 69), and the least
room 2.79 against K = 7 at 2 features. The entry sums ‖x‖² as a balanced
tree: in numpy's order, a chain n/2 deep, confit refused it at 1,024
features and raised `RecursionError` at 2,048, which `Normalizer(l2)`
still meets (finding 86). The build cap reads its own estimate, fitted
on 66 shapes (0.85 to 1.27 times it): a struct of 2,000 centres over 2
features builds in 73 s, and `PCA`'s estimate misses a term in n² at wide
rows (finding 87). T48 (#502) fixes both findings. A step whose SQL nests
past Python's recursion limit stays Python with a reason
(`_registry._translate`), where `to_native` raised `RecursionError`:
`Normalizer(l2)` over 2,048 features. The projections' two build estimates
are fitted again with a term in n², on 512 builds of 2 to 6,144 features
and 1 to 2,000 components (master af89dae). A list over shared values
builds about as their number squared, where a struct and a single
component do not: `PCA` of 2 components over 4,096 features, read
`x - 0.0`, built in 11.3 s as a list, 5.9 s as a struct, against 5.9 s by
the old `_centered_build_estimate`, which took it. That estimate now has a
term in n² for a list, and each of 177 shapes built within 0.74 to 1.27
times it (0.96 to 2.04 times the old one); `_build_estimate`, for the
reading `coalesce(x, NaN)`, within 0.90 to 1.18 times (1.09 to 1.89).
Neither cap took a build past 7.11 s. So a `PCA` of 2 components reads
`x - 0.0` up to about 3,000 features (4,600 before), and the cap takes it
up to about 4,600 (6,500); of one component, up to about 5,000 (8,000).
T49 (#503) serves `AdditiveChi2Sampler` with `sample_steps=1` (kernel.py),
bit-exact: one output field per feature, 0 where the feature is 0, else
`sqrt(x * s)` with s the sample interval, a product and a root that both
sides round correctly. Coverage is 46 of 68. The entry traps where the
twin raises: on a negative number, NaN, NULL or an infinity. Its cos and
sin fields (`sample_steps` of 2 or more, the default 2 included) stay
Python until the error scale of the ruling. Over boolean features only,
the twin keeps the array boolean, so True answers 1.0: the entry stays
Python there unless sqrt(s) is 0 or 1. A step builds about as fast as a
`StandardScaler` of its width (1.1 s at 1,024 features, as a list).
The loop runs from the "Transforms Loop" project thread. The Confit Loop
is paused since 2026-10-16: a confit need waits, and the loop works around
it.

Next: PLANS "Next" (the rest of "Ruled 2026-10-06, to build", item 1,
`AdditiveChi2Sampler`'s cos and sin fields first, one PR each).
