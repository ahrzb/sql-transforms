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
Open: at one feature, `FastICA` answers 0.0 where its twin answers -0.0,
in 54 output fields (finding 81), and the ARPACK configuration of
`TruncatedSVD` in FIXTURES has no `random_state`, so the same seed can
give a different step in each process (finding 82). T42 (#495), merged
after reading 7's commit, serves option 1 of the ruling on bounded steps
in compositions (`decisions/closed/bounded-steps-in-compositions.md`):
with `allow_bound=True`, a step with a parity bound above 0 serves as the
last step of a `Pipeline`, or as a part of a `ColumnTransformer` or
`FeatureUnion` that has no weight, and `native.check` holds each output
field to the bound of the step that gives it. The loop runs from the
"Transforms Loop" project thread. The Confit Loop is paused since
2026-10-16: a confit need waits, and the loop works around it.

Next: T43, PLANS "Next" (findings 81 and 82). Then the rest of PLANS
"Ruled 2026-10-06, to build", item 1, then the rest of item 2.
