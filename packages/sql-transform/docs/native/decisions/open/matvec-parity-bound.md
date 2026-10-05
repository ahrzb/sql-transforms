# How close a matvec entry must be

**Question.** The parity ruling (confit's
`docs/decisions/closed/native-transform-parity-bounds.md`) holds a matvec
family "within a small per-family ulp bound", measured. Measured, no small
bound in ulps of the result exists for one. What bound does a matvec entry
declare?

The families it decides: the linear projections (`PCA`, `IncrementalPCA`,
`TruncatedSVD`, `FactorAnalysis`, `FastICA`, the random projections, the PLS
family, `LinearDiscriminantAnalysis`), distances to fitted centres
(`KMeans` and kin), and the kernel samplers (`RBFSampler`, ...). The exact
families are unaffected.

**Why the twin's order cannot be followed.** sklearn's matvec is BLAS
(`X @ components_.T`). OpenBLAS picks its kernel for the CPU at run time
(SkylakeX on the measuring machine, likely another on CI), and its kernels
use fused multiply-add, which SQL cannot spell. So the entry sums in its own
order, left to right, and differs from the twin by rounding. Everything
around the matvec can be exact: PCA's `mean_ @ components_.T` is a constant
the translation computes with numpy as the twin does, and whitening is one
division.

**Why the result's ulps are not small.** sklearn scores PCA as
`x·c_k − m·c_k`: project, then subtract the projected mean. For a row near
the mean the two products cancel, and a rounding in `x·c_k` is many ulps of
their small difference. Measured: `PCA(whiten=True)` (every component) on
the catalog's fixtures, the entry summing left to right against the twin's
BLAS, master b851298 (2026-10-05):

| fixture seeds | lanes | max, ulps of the result | p99, eps of the term scale | max, eps of the term scale |
|---|---|---|---|---|
| 0–39 | 3,855 | 1,306 | 0.99 | 1.76 |
| 0–199 | 23,050 | 9,877,709,850 | 1.10 | 2.99 |

The term scale of a lane is the sum of the magnitudes of what it adds,
`(Σ|x_i·c_ki| + |m·c_k|) / scale_k`. The lane worst in ulps is 0.33 eps of
its term scale: the dot products agree to a rounding, and the result is ten
billion times smaller than its terms. Any order other than the twin's own
has this tail, and the maximum over N seeds grows with N. With one or two
components the tail is shorter (9 and 32 ulps at 40 seeds), because a lane
of a leading component rarely cancels, but no component is immune.

**Options.**

1. *A term-scale bound for the matvec families:*
   `|native − twin| ≤ K · eps · S` per lane, S its term scale, K declared
   per family from the measurement (PCA: 3). It bounds what the two sums
   can differ by, whatever the data: a dot product of n terms is within
   about n·eps·S of exact on either side, and measured within 3 eps·S at
   n ≤ 32. Equal values, NaN and NULL compare as now; `check` gains the
   term scale per lane.
2. *A ulp bound measured over N seeds:* about 10¹⁰ at N = 200. Not small,
   and not a bound: the next seed can exceed it.
3. *Matvec families stay Python* until BLAS's order is reproducible, which
   it is not portably (CPU-dispatched kernels with fused multiply-add).

**Provisional choice.** None for the entries: the matvec families wait in
PLANS, and the loop goes on with the exact families. Option 1 is the
loop's proposal.

**What would close it.** An owner ruling on the form of a matvec bound.
