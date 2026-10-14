# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Reading 5 merged on 2026-10-14
([`reports/2026-10-14-reading-5.md`](reports/2026-10-14-reading-5.md)), on
master 44052cf: coverage 31 of 68, no new class. On one machine, the
default forest serves a row 3.8 times as fast as reading 4's code, and the
input guard costs a median 8% of a 64-row call. Its new findings 68
(`QuantileTransformer`'s cap builds past 7 s there) and 69 (a `Pipeline`'s
guard costs 4.4 times its build) closed with T34 and T35. T30
(#446) dropped the forests' paths spelling: one CASE per tree is their one
spelling, and a forest estimated past 7 s stays Python. T31 (#447) probes
an encoder's categories in about four calls of `transform` a feature, not
one a category: `to_native` of `OrdinalEncoder` over 32 string features of
125 categories takes 1.3 s, 40.8 s before. T32 (#448) names a value
of a step's output an output field, not a lane, in the catalog's code and
in `_udf.py`, as GLOSSARY.md now defines it. T33 (#450) finds a number
among a run of consecutive integer categories by a range in the encoders'
tests: `OrdinalEncoder` over 32 number features of 125 categories builds
in 0.74 s, 4.6 s before. T34 (#452) caps `QuantileTransformer` and
`IsotonicRegression` at an estimated 7 s build, as the forests and splines
are capped. 8,000 quantiles over one feature build in 6.9 to 7.4 s and
stay served, 10 over 800 features (11.4 s) now stay Python, and integer
fits of up to 500 quantiles over 32 features (5.5 s) are now served.
`IsotonicRegression` takes about 11,700 thresholds, 8,000 before.
Finding 68's 8.5 to 9.9 s timed `to_native` as a whole, which also spends
1.7 to 1.9 s in confit's `SqlFunction` at that step (PLANS, "Needs from
confit"). T35 (#454) spells the input guard as one test for each kind of
rejected value, a `greatest` or `least` over the features that reject it,
not one test a feature joined by OR. The widest bench `Pipeline` builds in
0.9 to 1.2 s, 2.4 to 2.7 s before, and serves 64 rows in 5.4 ms, 6.4 ms
before. The OR of its tests cost most of the build: with 240 bound values
in place of 1,069, it built as slowly. The native bench now fails where a
step serves more slowly than its twin. T36 (#455) reads a sparse output
dense, as `.toarray()` makes it, at the serve call, the fit probe and the
catalog's probes (`_udf._dense`, decisions/closed/sparse-outputs.md).
The defaults of `OneHotEncoder`, `KBinsDiscretizer` and
`RandomTreesEmbedding`, `MissingIndicator(sparse=True)`, a sparse
`SplineTransformer`, and a `Pipeline`, `ColumnTransformer` or
`FeatureUnion` with a sparse output are served, under both of sklearn's
sparse containers. A step handed a sparse matrix by an earlier one runs
its own sparse branch: it is served where measured to answer the 0/1
matrix as dense (an imputer filling a stored 1 with 49.0, then scaled,
parted in 36 of 40 rows). A sparse spline stays Python at degree 0 under
`constant` or `periodic`, under `handle_missing="zeros"` over two or more
features with `continue`, `error` or `periodic`, and over NaN knots; no
other of 1,440 sparse spline configurations parted from the step. T37
(#458) serves the first family on the parity bound, `PCA`,
`IncrementalPCA`, `TruncatedSVD` and `GaussianRandomProjection`
(`project.py`), within (n + 3)·eps·S + τ of the twin, as the matvec ruling
derives it, and only with `allow_bound=True`. The entry sums each output
field's n products as a balanced tree. Over 200 seeds of 1 to 64
features, the largest |entry − twin| / (eps·S) was 3.90 (`TruncatedSVD` at
17 features), against K = 20. A whitened component of
rounding noise and a `PCA` or `IncrementalPCA` fitted in float32 stay
Python, as does a step estimated past a 7 s build (about 80 components over
256 features). Up to 16 features the entry serves 64 rows 50 to 170 times
as fast as the twin, but at 256 features of 64 components it takes twice
the twin's time: confit serves each read of a feature in about 60 ns
(PLANS, "Needs from confit"). Coverage is 35 of 68, past half. The loop
runs from the "Transforms Loop" project thread.

Next: the milestone report, since coverage crossed 34 of 68 (reporting.md);
then PLANS "Ruled 2026-10-06, to build": the rest of the linear
projections, then the other families on the parity bound.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
