# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Reading 8 merged on 2026-10-17
([`reports/2026-10-17-reading-8.md`](reports/2026-10-17-reading-8.md)), on
master 612831d, after T42 to T52: coverage 48 of 68, with `sklearn.cluster`
wholly native. In seven measurements of the PRs, run again on master, each
value is within its K, with room of 2.39 or more. This session's machine
(started 2026-10-16 18:11 UTC) is about as fast as reading 7's: there,
f54ddab serves a native call of the bench in a median of 0.98 times reading
7's time. In the bench, 14 classes serve their widest step faster at master
than in both runs of f54ddab, most of them from T44's reads on (a search
over the commits of T43 to T46). Builds near the caps take 0.71 to 1.34
times their estimates. Two findings are open. `AdditiveChi2Sampler` has no
build estimate, so no cap applies to its builds (finding 91: 28.5 s with
`sample_steps=3` over 256 features, nearly all of it the trial build). T46's
four estimates time the trial build alone, where those of T47, T48, T51 and
T52 time the whole warm `to_native`, probe included (finding 92). Need 5 is
new: confit shares no `ln` (a value that can raise), so `SkewedChi2Sampler`
computes n·m logarithms a row (finding 90). The needs are in PLANS "Needs
from confit".

The loop runs from the "Transforms Loop" project thread. The Confit Loop
is paused since 2026-10-16: a confit need waits, and the loop works around
it.

Next: PLANS "Next" (finding 91's estimate and cap for
`AdditiveChi2Sampler`, then the rest of "Ruled 2026-10-06, to build", item
1, `Nystroem` next, one PR each).
