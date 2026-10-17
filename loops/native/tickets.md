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
T53 (#508) gives `AdditiveChi2Sampler` a build estimate of the whole warm
`to_native` (`kernel._additive_chi2_build_estimate`), and the entry refuses
a step estimated past 7 s (finding 91). Over 117 shapes of 1 to 1,024
features and 1 to 400 sample steps, each build over 0.3 s is within 0.84 to
1.22 times the estimate (scripts and builds in
`/mnt/project-files/transforms-loop/t53/`). The cap takes about 900
features of one step, 148 of three or 12 of 100. Each build estimate's
docstring now says which time it estimates: the trial build alone (T46's
forest, isotonic, quantile and spline estimates) or the whole warm
`to_native` (finding 92).
T54 serves `Nystroem` of the rbf kernel (`kernel._nystroem`) within
K = max(n + 5, m + 3), the bound that the owner ruled, with n features and
m components. Its S reads each kernel at the least squared distance within
the rounding of the distance (k̃_j): the research's S, with the kernel
itself, fails at 11.1 million fields where the rows and the fit sit far
from 0. Over 354 million fields of 16 seeds (1 to 64 features of 8 and 64
components, numpy's `exp` of AVX-512 and glibc's), no field is past K: the
largest |entry - twin| / (eps·S) is 3.19 against K = 67, and the least
room is 1.99 against K = 11. The entry refuses a step whose build
`kernel._nystroem_build_estimate` estimates past 7 s (110 shapes, each
within 0.86 to 1.13 times the estimate). `check` runs a definition that
grows as m²·n, so the tests keep to few components (a new need in PLANS).
Scripts and data are in `/mnt/project-files/transforms-loop/t54/`.

The loop runs from the "Transforms Loop" project thread. The Confit Loop
is paused since 2026-10-16: a confit need waits, and the loop works around
it.

Next: PLANS "Next" (the rest of "Ruled 2026-10-06, to build", item 1,
`PowerTransformer` Yeo-Johnson and `standardize=True` next, one PR each).
