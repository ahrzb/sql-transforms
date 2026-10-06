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

**Methodology (2026-10-06).** The notes are in
[research/2026-10-06/](../research/2026-10-06/README.md): matvec.md,
distances.md and framework.md, each re-run by an adversarial verifier, plus
critique.md and inference.md (inference.md's verification is pending).

The setup:
- **Environment.** x86-64 with AVX-512; numpy 2.5.1, scikit-learn 1.9.0,
  OpenBLAS 0.3.33 (`DYNAMIC_ARCH`), DuckDB 1.5.5; master 113fba7.
- **The twin under other conditions.** Other CPUs' BLAS kernels are selected
  with `OPENBLAS_CORETYPE`. Each configuration runs in its own process,
  against one pickled model and the same rows.
- **The term scale.** `S_full = (Σ|x_i·c_ki| + Σ|m_i·c_ki|) / scale_k`. K is
  `|a − b| / (eps·S_full)`.

1. **What the twin computes** (sourced from sklearn, numpy `matmul.c.src`
   v2.5.1 and OpenBLAS v0.3.33 kernels, then measured).
   - One row is `gemv` for k ≥ 2 components, and `ddot` for k = 1.
   - Its order depends on:
     - the CPU's kernel;
     - k, and where the lane sits among the k;
     - the layout of `components_`: the `covariance_eigh` solver gives
       F-order, so `dgemv_n`;
     - row against batch;
     - the thread count, once m·n ≥ 460,800.
   - An emulation of the SkylakeX and Haswell `dgemv_t` order matched every
     lane tested bit for bit: 10,640 random lanes per kernel, and 6,970 more
     on the verifier's shapes. It uses FMA, which DuckDB lacks.
2. **The twin against itself.** Catalog `PCA(whiten=True)` fixtures, seeds
   0–199, 25,208 lanes:

   | pair | lanes differing | max ulps | max K |
   |---|---|---|---|
   | entry vs twin (SkylakeX, one row) | 4,819 | 9,877,709,850 | 2.99 |
   | twin, batch vs one row, same machine | 4,696 | 9,877,709,850 | 2.99 |
   | twin, Sandybridge vs SkylakeX kernel | 8,418 | 2,026 | 2.86 |
   | twin, Haswell vs SkylakeX kernel | 680 | 568 | 1.76 |
   | twin, 1 thread vs 4 threads (fixtures are below 460,800) | 0 | 0 | 0 |

   At 1,000 seeds, entry vs twin reaches K = 4.12.
3. **A bound that holds whatever the data** (derived; re-derived by two
   verifiers).
   - Any summation tree over n products, fused or not, errs by at most
     γ_n·Σ|x_i c_ki| per side (Higham, *Accuracy and Stability*, §3.1).
   - The twin recomputes `M = mean_ @ components_.T` on every call, on the
     host that serves it. M errs by at most γ_n·Σ|m_i c_ki|.
   - The subtraction and the division add about 2·eps·S.
   - Hence `|entry − twin| ≤ (n + 2)·eps·S_full` to first order. K = n + 3
     absorbs the second-order terms and check's own rounding of S.
4. **K against the width n** (measured on synthetic PCA-like data from
   n = 4 to 8,192, plus the verifiers' own datasets).
   - When the terms share a sign, K grows roughly as n^0.4: about 12 at
     n = 2,048
     and 16–17 at n = 8,192.
   - With mixed signs it stays near 1.5–3.
   - A constructed row reaches about (n − 2)/4, which is 7.25 at n = 32.
5. **Alternatives measured.**
   - *A correctly rounded dot product* (Ogita–Rump–Oishi Dot2, spelled in
     DuckDB SQL) was correctly rounded on 5,000 of 5,000 rows, at 11× the
     cost. It only halves the provable K, because the twin itself is up to
     2.84 eps·S from exact.
   - *Pairwise summation* has the same flop count and the error bound
     γ_{⌈log2 n⌉+1}. At n = 2,048 it cut measured K from 13.1 to 3.5. A
     left-to-right nested sum of 999 terms exceeds DuckDB's default
     `max_expression_depth` (1000); 500 terms parse.
   - *An allclose test*: its atol depends on the data's units. It is 3.5e-5 on
     these fixtures, and 2.6e283 once the ±1e300 edge rows are included.
6. **The other families this record decides** (distances.md).
   - *KMeans and kin* compute `sqrt(max(((−2·x·c) + ‖x‖²) + ‖c‖², 0))`.
     - On the distance itself, K reaches about 9e7 near a centre.
     - On its square, with `S = ‖x‖² + ‖c‖² + 2Σ|x_i c_i|`, K ≤ 2.4. The
       derived bound is n + 4.
   - *RBF and SkewedChi2 samplers* hold with S = c·(Σ|L_i W_ij| + |b_j| + 1),
     at measured K ≤ 2.8. The derived bounds are n + 2 and n + 3.
   - *Nystroem* needs an S that accounts for its kernel.
   - *PolynomialCountSketch* is an FFT convolution, not a matvec. Its proposed
     per-row S can be exactly 0 while the twin returns noise, so it needs a
     normwise S.
7. **Downstream predictions** (inference.md, verification pending).
   - The setup: PCA, KMeans, Yeo-Johnson and chi2, each in front of
     LogisticRegression, a linear regressor, a decision tree, a random forest,
     HistGradientBoosting and KMeans argmin. About 1.6M served rows, against
     14 twin variants (CPU kernel, numpy SIMD, threads, batch).
   - On new rows, neither the entry nor any twin variant changed a label or a
     tree leaf.
   - The left-to-right entry equals the twin on the Sandybridge kernel bit for
     bit, in 11 of 12 PCA cases.
8. **Verifier corrections taken.**
   - The thread count does change the twin above m·n = 460,800.
   - The record's |m·c| scale never fails inside `check`, because one process
     shares M. S_full matters when the twin is served on another host.
   - Measured envelopes such as "K ≤ 0.7·√n" are false.
   - The bit-exact entry-vs-DuckDB half of `check` cannot catch a wrong
     constant, since both sides share it.
9. **Not reproduced.**
   - The record's lane counts: 3,855 and 23,050. Every run gives 4,618 and
     25,208.
   - Its commit: its numbers match the fixture generator at HEAD, not b851298.

**Recommendation.** Option 1, amended into one general form that also closes
the power and additive chi2 records.

1. **The contract.** Per lane, `|g(entry) − g(twin)| ≤ K·eps·S + τ`.
   - S is declared per family, as a formula over the fitted state and the row.
   - K is derived from per-operation error bounds, and may depend on the
     width n.
   - g is the identity, except where a family names another comparison map.
   - τ is a small floor for underflow.
   - The measured maximum is kept as evidence, never as K.
   - Today's ulp bounds are the special case S = |lane|, so the exact
     families are unchanged.
2. **Matvec:** S = S_full and K = n + 3.
3. **KMeans and kin:** g squares the distance; S = ‖x‖² + ‖c‖² + 2Σ|x_i c_i|;
   K = n + 5. The extra 1 covers check's own rounding of the difference of
   squares.
4. **Samplers:** RBF n + 3, SkewedChi2 n + 4, Nystroem max(n + 5, m + 3), each
   one over its derivation to cover check's rounding of S.
   PolynomialCountSketch is not served until a normwise S and K are derived.
5. **The entry sums pairwise.** It costs the same as left to right, halves the
   provable K, cuts measured K about 4× at n = 2,048, and keeps SQL depth at
   log2 n.
6. **check computes S overflow-safely.**
   - NaN matches only NaN.
   - A lane with exactly one infinite side passes only if the other side is
     finite, has the same sign, and lies within K·eps·S of DBL_MAX.
7. **Refuse noise lanes:** a `whiten=True` component whose scale was clipped
   to eps, or whose variance is ≲ eps × the largest. Such a lane is rounding
   noise of size O(1) for the twin too. On rank-deficient data the twin flips
   labels against itself (batch vs row: 109 LR and 594 HGB flips per 70k
   rows) about as often as the entry does (114 and 645).

Why, judged against the goal that inference does not change when the entry
replaces the twin:

- **The twin does not give one answer.** The same model and row differ by
  9.9e9 ulps between a one-row and a batch call on one machine, and by 2,026
  ulps between CPU kernels. A ulp bound would fail sklearn against itself.
- **The term scale is what stays invariant.** Every pair measured (entry or
  twin, any kernel, row or batch) stays within K ≤ 4.12 of S_full.
- **Predictions behave the same way.**
  - On new rows, nothing changed a prediction.
  - Where predictions do change, the twin changes them against itself at the
    same rate. That happens where HistGradientBoosting thresholds equal
    training values and a training row is re-served, and on noise lanes.
  - The bound therefore promises what the twin can promise about itself.
- **A measured K is not a bound.** K = 3 already breaks at 1,000 seeds and on
  wider inputs. n + 3 holds for every order.
- **The looser K costs almost no detection power.** A structural bug (a
  dropped term, a wrong sign or index) moves a lane by about S/n, far above
  (n + 3)·eps·S for any n below about 10^7.
- **Why S_full and not the record's S:**
  - Both agree inside `check`.
  - The twin recomputes M on whatever host serves it. With |M|, a twin on
    another CPU kernel reaches K = 1,731.

Still for the owner (critique.md §2):
- **Same-host or cross-host?** Is the contract "on the host that translated
  the step", which is all `check` can test? That decides S_full, and whether
  per-call recomputations count as shared constants: PCA's M, KMeans's ‖c‖²,
  chi2's `cosh(πjs)`.
- **May a bounded step sit in a composition?** `compose.py` refuses any
  nonzero bound today. Option 1 alone therefore leaves
  `make_pipeline(StandardScaler(), PCA())` NotNative. A bounded step last in a
  Pipeline needs no new theory. One that feeds a discontinuous step (a
  binner, a threshold) has no bound.
- **check's API.** It must move from an integer ulp bound to a per-lane S that
  reads the fitted estimator, with K(n).
