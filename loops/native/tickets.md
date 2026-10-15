# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Reading 6 merged on 2026-10-15
([`reports/2026-10-15-reading-6.md`](reports/2026-10-15-reading-6.md)), on
master 98033e8, after T30 to T37: coverage 35 of 68, past half. Since
confit #456, a struct read field by field builds in about the square of
its fields: forests build up to 7 times as long as at reading 5, past
their estimates (finding 76; confit #461, merged after the reading,
addresses it). `TruncatedSVD` and `GaussianRandomProjection` answer -0.0
where the twin answers 0.0 in 403 output fields of the parity run, within
their bound (finding 77). Reading 5's search for finding 69 ran 8 of its
commits on reading 4's confit (finding 74): with each commit's own confit,
the widest `Pipeline` never served more slowly than its twin. T38 (#462)
adds 0.0 to each sum of the projections, because the twin answers 0.0
where every product is a zero: over seeds 0 to 199 of the parity run, no
output field then differs from the twin in its sign alone, against 403
before. T39 (#463) fits the projections' build estimate again on confit
#459 and #461: `PCA` over 256 features of 64 components builds in 1.1 to
1.5 s (5.2 to 5.9 s before) and serves 64 rows in 3.7 to 4.3 ms, 8 times
as fast as its twin (77 to 83 ms before, twice the twin's time). The cap
now takes about 280 components over 256 features, against 80. T40
(#T40_PR) fits the build estimates of the forests, `IsotonicRegression`,
`QuantileTransformer` and the splines again on confit #461: the caps now
take about 10,300 output fields of a forest over 8 features (5,990
before), 19,700 isotonic thresholds (11,700), 13,400 quantile leaves
(8,200) and SPLINE_BOARD. The loop ran from the "Transforms Loop" project
thread. On 2026-10-15 the owner asked the loops to wind down: finish the
work in flight and pick no new ticket. After T40 and a short reading, the
loop stops.

Next, on a restart: PLANS "Ruled 2026-10-06, to build", item 1, the rest
of the linear projections, then the other families on the parity bound.
A draft of the centered projections (T41: `FastICA`,
`LinearDiscriminantAnalysis` and the PLS family), written while T40 ran
and never run, is in the project's shared files at
`transforms-loop/t41-draft/`.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
