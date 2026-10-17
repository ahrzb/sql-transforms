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
T55 serves `PowerTransformer(standardize=False)` on the error scales that
the owner ruled (power.py): Box-Cox within K = 5, S = |t|, and Yeo-Johnson,
newly served, within K = 10, S = (1 + max(w, 0))·|t|, with t the twin's
output and w the argument of its `expm1`. DuckDB has neither `log1p` nor
`expm1`, so the entry spells them from `ln` and `exp`. It serves only where
the kernel probe finds confit's `ln` and `exp` the C library's and, for
Yeo-Johnson, numpy's `log1p` and `expm1` within 1 ulp of it. The twin's
Yeo-Johnson overflows in `expm1` before it divides, so the entry answers
±inf from the twin's own overflow edges (`power._edges`). Over 153.6
million fields of 32 seeds (numpy's kernels of AVX-512 and glibc's), no
field is past K: the largest |entry - twin| / (eps·S) is 3.38 against
K = 10 and 2.50 against K = 5. The entry refuses a step whose build
`power._build_estimate` estimates past 7 s (32 widths, each within 0.88 to
1.14 times the estimate): it takes about 347 features of Yeo-Johnson and
702 of Box-Cox. `standardize=True` stays Python, as the twin's scaler
raises on an infinite t (PLANS). Coverage stays 49 of 68. Scripts and data
are in `/mnt/project-files/transforms-loop/t55/`.
T56 serves `PowerTransformer(standardize=True)` (power.py) within K = 12
for Yeo-Johnson and K = 7 for Box-Cox, S = (S_t + |mean_|) / scale_, the
bound that the owner ruled, with S_t T55's scale of t. The twin's scaler
raises on a row whose t is infinite, so the entry traps from each of the
twin's own overflow edges on, Box-Cox's too, and clamps t to ±DBL_MAX
below them. It computes `(t - mean_) / scale_` with the fitted doubles,
and it refuses a `mean_` of 2**970 or more, where `t - mean_` can overflow
on one side only. The input guard's probe row is now where t is 0
(`power._at_zero`), so a Box-Cox of a boolean feature serves too, and a
standardized Yeo-Johnson with a lambda above about 1,024. Over 137.6
million fields of 32 seeds (numpy's kernels of AVX-512 and glibc's, four
kinds of scalers), the entry's edges trap each of the 2.0 million rows
where the twin raises and no other row, and no field is past K: the
largest |entry - twin| / (eps·S) is 3.94 against K = 12 and 3.00 against
K = 7. The cap reads a new estimate of the
standardized step (24 widths, each within 0.90 to 1.20 times it): it takes
about 313 features of Yeo-Johnson and 682 of Box-Cox. Coverage stays 49 of
68. Scripts and data are in `/mnt/project-files/transforms-loop/t56/`.
T57 serves `SparseRandomProjection` bit-exactly over the CSR components
that its fit makes (project.py, `_sparse_project`): scipy sums each output
field from 0.0, adding the stored products in stored order, unfused, and
the entry spells the same sum, where the kernel probe finds scipy's sums
so (`csr_sum_is_scipys`). Of density 1 the components are dense, and the
step is `GaussianRandomProjection`'s, within the matvec bound. Over 567
million fields of 36 seeds (1 to 512 features, 1 to 128 components, five
kinds of densities, float32 fits, seven kinds of rows), each field equals
the twin's, in calls of 20,000 rows and of one row, and the entry traps on
each of the 30,045 rows where the twin raises and on no other. A sum in
stored order nests two levels of SQL an entry, and confit parses 493
entries, so the entry refuses a component of more than 480. The cap reads
a new estimate (`_sparse_build_estimate`): a struct read field by field
builds in about 17 µs a field and sum that two fields repeat (PLANS "Needs
from confit"). 116 shapes are each within 0.66 to 2.57 times it, and 50
new shapes within 0.80 to 1.26. Coverage 50 of 68. Scripts and data are
in `/mnt/project-files/transforms-loop/t57/`.

The loop runs from the "Transforms Loop" project thread. The Confit Loop
is paused since 2026-10-16: a confit need waits, and the loop works around
it. The owner paused this loop after T55 and resumed it at 07:01 UTC the
same day (2026-10-17).

Next: PLANS "Next" (the rest of "Ruled 2026-10-06, to build", item 1,
`FactorAnalysis`, then the rest of item 2).
