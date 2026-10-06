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

**Methodology (2026-10-06).** The research notes are in
[research/2026-10-06/](../research/2026-10-06/README.md). This record uses
five of them. An adversarial verifier re-ran four of the five notes. In this
record, a verifier is a second agent that tried to refute the key claims of a
note.

- matvec.md is the note on the matvec (matrix-vector product) families. A
  verifier re-ran it.
- distances.md is the note on two other groups of families. A verifier
  re-ran it too.
  - The first group computes the distance from a row to each fitted centre,
    for example `KMeans`. A fitted centre is the centre point of one cluster
    of the fitted model.
  - The second group is the kernel samplers, for example `RBFSampler`. A
    kernel sampler is a sklearn transformer that makes new features. The new
    features approximate a kernel function, which measures the similarity of
    two rows.
- framework.md is the note on the general form of a parity bound. A verifier
  re-ran it too.
- critique.md is the note that settles the contradictions between the
  notes.
- inference.md is the note that tests if a prediction changes when the
  catalog entry replaces its twin. A verifier re-ran it too.

The research used this setup and these terms:

- **Entry.** A catalog entry is one entry of the native catalog. It
  translates one fitted sklearn transformer into a transform that confit
  serves. This record calls the catalog entry of a step "the entry".
- **Environment.** The host is x86-64 with AVX-512 (Advanced Vector
  Extensions, 512-bit). The software is numpy 2.5.1, scikit-learn (sklearn)
  1.9.0, OpenBLAS 0.3.33 and DuckDB 1.5.5. The code is commit 113fba7 of the
  master branch.
- **OpenBLAS.** OpenBLAS is the library of BLAS (Basic Linear Algebra
  Subprograms) kernels that the twin uses. An OpenBLAS kernel is the code of
  one operation for one type of CPU (central processing unit). It is not a
  kernel function. The build option `DYNAMIC_ARCH` makes OpenBLAS select a
  kernel for the CPU at run time. OpenBLAS names its kernels after CPU
  generations, for example SkylakeX, Haswell and Sandybridge.
- **The twin under other conditions.** The research selected the BLAS
  kernels of other CPUs with the environment variable `OPENBLAS_CORETYPE`.
  Each configuration ran in its own process. All configurations used one
  fitted model, saved with Python's `pickle`, and the same rows.
- **Fixtures.** The catalog's fixtures are the test cases that the test file
  of the catalog (`catalog_test.py`) generates. For each seed, a generator in
  that file draws fitted steps and their input rows.
- **The term scale.** The term scale is the error scale S that this record
  proposes for a field of a matvec family. It is the sum of the magnitudes of
  the terms that the field adds.
- **S_full.** This record uses the full term scale of an output field k,
  `S_full = (Σ|x_i·c_ki| + Σ|m_i·c_ki|) / scale_k`. Here x is the input row
  of n features, c_k is component k and m is the fitted mean `mean_`.
  `scale_k` is the number by which whitening divides component k.
- **The record's S.** The record's S is the term scale in the sections
  above the Methodology. It uses |m·c_k| as its second term. S_full uses Σ|m_i·c_ki| in its
  place.
- **K.** For two results a and b of one field, K is
  `|a − b| / (eps·S_full)`, with eps = 2^-52.
- **check.** `native.check` is the test that serves the same query with the
  twin and with the entry and compares the results. It also runs the entry's
  own definition on the oracle (DuckDB with the optimizer off). It compares
  that answer with confit's answer for the entry, bit-exact. This record
  calls the test `check`.

In the counts below, a field is one output field of one row. A sourced claim
cites the code of a package or a web address (URL). A derived claim gives its
derivation.

1. **What the twin computes** (sourced, then measured). The sources are
   sklearn, numpy's matrix product code `matmul.c.src` v2.5.1 and the
   OpenBLAS v0.3.33 kernels.
   - If there are k ≥ 2 components, the twin computes one row with `gemv`,
     the BLAS matrix-vector product. If k = 1, it uses `ddot`, the BLAS dot
     product.
   - The order of its sum depends on these conditions:
     - the CPU kernel
     - k, and the position of the field among the k components
     - the memory layout of `components_`, the fitted matrix of components
     - a call on one row or a call on a batch
     - the thread count, once the matrix of the product has
       m·n ≥ 460,800 elements
   - In PCA (principal component analysis), the `covariance_eigh` solver
     gives `components_` in F-order (column-major). So OpenBLAS uses
     `dgemv_n`, its gemv kernel for a matrix in F-order.
   - The research emulated the order of `dgemv_t`, the OpenBLAS kernel for
     `components_` in C-order (row-major). The emulation of the SkylakeX and
     Haswell kernels was bit-exact on every field tested.
   - That is 10,640 random fields for each kernel, and 6,970 more on the
     matrix shapes of the verifier.
   - The emulation uses fused multiply-add (FMA), and DuckDB does not have
     FMA.
2. **The twin against itself.** The data is the catalog's fixtures for
   `PCA(whiten=True)`, which is PCA with whitening. The seeds are 0–199,
   with 25,208 fields in all:

   | pair | fields differing | max ulps | max K |
   |---|---|---|---|
   | entry vs twin (SkylakeX, one row) | 4,819 | 9,877,709,850 | 2.99 |
   | twin, batch vs one row, same machine | 4,696 | 9,877,709,850 | 2.99 |
   | twin, Sandybridge vs SkylakeX kernel | 8,418 | 2,026 | 2.86 |
   | twin, Haswell vs SkylakeX kernel | 680 | 568 | 1.76 |
   | twin, 1 thread vs 4 threads (fixtures are below 460,800) | 0 | 0 | 0 |

   At 1,000 seeds, the entry against the twin reaches K = 4.12.
3. **A bound that holds whatever the data** (derived, and re-derived by two
   verifiers).
   - On each side, any summation tree over n products, fused or not, has an
     error of at most γ_n·Σ|x_i c_ki|. The source is Higham, *Accuracy and
     Stability of Numerical Algorithms*, §3.1.
   - Here γ_n = n·u/(1 − n·u), and u = 2^-53 is the unit roundoff.
   - The twin recomputes the projected mean `M = mean_ @ components_.T` on
     every call, on the host that serves it. The error of M is at most
     γ_n·Σ|m_i c_ki|.
   - The subtraction and the division add about 2·eps·S.
   - So `|entry − twin| ≤ (n + 2)·eps·S_full` to first order.
   - K = n + 3 also covers the second-order terms and the rounding of S in
     `check` itself.
4. **K against the width n** (measured on synthetic PCA-like data from
   n = 4 to 8,192, and on the datasets of the verifiers).
   - If the terms share a sign, K grows roughly as n^0.4. K is about 12 at
     n = 2,048, and 16–17 at n = 8,192.
   - If the signs are mixed, K stays near 1.5–3.
   - At n = 32, a constructed row reaches a K of about 7.25. That is about
     (n − 2)/4.
5. **Alternatives measured.**
   - *A correctly rounded dot product.* The research wrote the Dot2
     algorithm of Ogita, Rump and Oishi in DuckDB SQL. It gave the correctly
     rounded result on 5,000 of 5,000 rows, at 11× the cost of the
     left-to-right sum.
   - Dot2 only halves the provable K, because the twin itself is up to
     2.84 eps·S from the exact result.
   - *Pairwise summation* adds the terms as a balanced tree of pairs. It has
     the same number of floating-point operations as the left-to-right sum.
     Its error bound is γ_{⌈log2 n⌉+1}. At n = 2,048, it cut the measured K
     from 13.1 to 3.5.
   - A nested left-to-right sum of 999 terms exceeds the maximum depth of an
     expression in DuckDB, `max_expression_depth` (1000 by default). A
     nested sum of 500 terms parses.
   - *An allclose test.* The numpy test `allclose` passes if
     |a − b| ≤ atol + rtol·|b|. Its absolute term atol depends on the units
     of the data.
   - On these fixtures, an atol of 3.5e-5 is necessary to pass every field.
     If the fixtures include the edge rows at ±1e300, the necessary atol is
     2.6e283.
6. **The other families that this record decides** (distances.md).
   - *KMeans and kin* compute the distance to a centre c as
     `sqrt(max(((−2·x·c) + ‖x‖²) + ‖c‖², 0))`.
     - On the distance itself, K reaches about 9e7 near a centre.
     - On the square of the distance, with
       `S = ‖x‖² + ‖c‖² + 2Σ|x_i c_i|`, K ≤ 2.4. The derived bound is
       n + 4.
   - *The RBF (radial basis function) and SkewedChi2 samplers* hold with
     S = c·(Σ|L_i W_ij| + |b_j| + 1), at a measured K ≤ 2.8. The derived
     bounds are n + 2 and n + 3.
     - In this S, W and b are the random weights and offsets of the
       sampler, and c is its output factor.
     - L_i is x_i for RBF. For SkewedChi2, L_i is ln(x_i + s), where s is
       its `skewedness` parameter.
   - *Nystroem* needs an S that accounts for its kernel function.
   - *PolynomialCountSketch* is a convolution through the fast Fourier
     transform (FFT), not a matvec. The S that the research proposed for
     each row can be exactly 0 while the twin returns noise.
   - So PolynomialCountSketch needs a normwise S. A normwise S uses the
     norm (the length) of each whole vector, not the size of each term.
7. **Downstream predictions** (inference.md, verified, and one independent
   measurement of Box-Cox).
   - The research put PCA, KMeans, Yeo-Johnson and chi2 (the additive chi2
     sampler) each in front of six models. The models are
     LogisticRegression, a linear regressor, a decision tree, a random
     forest, HistGradientBoosting and KMeans argmin (the nearest centre).
   - It served about 1.6 million rows, against 14 variants of the twin. The
     variants change the CPU kernel, numpy's SIMD (single instruction,
     multiple data) kernels, the thread count or the batch.
   - On new rows, neither the entry nor any variant of the twin changed a
     label or a tree leaf of these five models: LogisticRegression, the
     linear regressor, the decision tree, the random forest and KMeans
     argmin.
   - HistGradientBoosting puts its thresholds at training values of a
     feature. If a served row repeats a training value, the row lands on
     such a threshold. Then one rounding decides the branch.
     - *Matvec.* The research served the training rows again. Out of 1.63
       million rows, the entry changed the branch of 6,569 rows and the
       label of 19.
     - `fit_transform` is the sklearn method that fits a step and
       transforms the training data in one call. Against one-row serving,
       the twin's own `fit_transform` changed the branch of 3,349 rows and
       the label of 7.
     - The twin in batch calls changed the branch of 3,899 rows and the
       label of 7.
     - *KMeans.* The entry was 2–5× worse than every variant of the twin.
       For example, the branch changed for 313 rows, against 161.
     - *Elementwise families* transform each feature on its own, for
       example Yeo-Johnson. On one host, the twin's outputs for a row, for a
       batch and from `fit_transform` are bit-exact with each other.
     - So the twin never flips (changes) a predicted label there.
     - The Box-Cox entry that the catalog serves today has an ulp bound
       of 4. It flipped 53–80 labels per 200,000 new rows with values on a
       grid (the verifier, 4 seeds).
     - An independent measurement of Box-Cox found 13–24 flips (2 seeds).
     - The Yeo-Johnson entry flipped 86–111 labels per 100,000–200,000
       rows. The twin without numpy's AVX-512 kernels flipped 17–98.
   - In 11 of 12 PCA cases, the left-to-right entry is bit-exact with the
     twin on the Sandybridge kernel. If `mean_` is not zero, this holds
     only if the translation also sums the constant M left to right.
8. **The record takes these corrections from the verifiers.**
   - The thread count does change the twin above m·n = 460,800.
   - The record's S, with |m·c_k|, never fails inside `check`, because one
     process shares M. If another host serves the twin, S_full matters.
   - Upper limits on K that come from measurements, such as "K ≤ 0.7·√n",
     are false.
   - `check` compares confit's answer for the entry with the answer of the
     oracle, bit-exact. This comparison cannot find a wrong constant,
     because both sides share the constant.
9. **Not reproduced.**
   - The first table of the record counts 3,855 and 23,050 fields. Every
     run gives 4,618 and 25,208.
   - The text above that table names commit b851298 of the master branch.
     The record's numbers match the fixture generator at HEAD, the commit
     that the research had checked out. They do not match the generator at
     b851298.

**Recommendation.** Take Option 1, amended into one general form. Option 1
is the term-scale bound for the matvec families, in **Options** above. The
general form also closes two other records: the power record
(`power-parity-bound.md`) and the additive chi2 record
(`additive-chi2-parity-bound.md`).

1. **The contract.** For each output field,
   `|g(entry) − g(twin)| ≤ K·eps·S + τ`.
   - S is the error scale. Each family declares S as a formula over the
     fitted state and the row.
   - Each family derives K from the error bounds of its operations. K may
     depend on the width n.
   - g is a comparison map, a function that `check` applies to both sides
     before it compares them. It is the identity, unless a family names
     another map.
   - τ is a small floor for underflow.
   - Keep the measured maximum as evidence. Never use it as K.
   - Today's ulp bounds are the special case S = |y|, where y is the value
     of the field. So the exact families do not change.
2. **Matvec:** S = S_full and K = n + 3.
3. **KMeans and kin:** g squares the distance. The error scale is
   S = ‖x‖² + ‖c‖² + 2Σ|x_i c_i|, and K = n + 5. The extra 1 covers the
   rounding of the difference of squares in `check` itself.
4. **Samplers:** K is n + 3 for RBF, n + 4 for SkewedChi2 and
   max(n + 5, m + 3) for Nystroem. Here m is the number of components of
   Nystroem.
   - Each K is 1 more than its derivation, to cover the rounding of S in
     `check`.
   - Do not serve PolynomialCountSketch until a derivation gives a normwise
     S and its K.
5. **The entry sums pairwise.** Pairwise summation has these effects:
   - It costs the same as the left-to-right sum.
   - It halves the provable K.
   - It cuts the measured K about 4× at n = 2,048.
   - It keeps the depth of the SQL expression at log2 n.
6. **check computes S in a way that is safe from overflow.**
   - NaN (not a number) matches only NaN.
   - If exactly one side of a field is infinite, the field passes only under
     three conditions:
     - The other side is finite.
     - The other side has the same sign as the infinite side.
     - The other side lies within K·eps·S of DBL_MAX, the largest finite
       double.
7. **Do not serve noise fields.** A noise field is a field of a
   `whiten=True` component with one of these two properties:
   - sklearn clipped its scale to eps.
   - Its variance is at most about eps times the largest variance.

   Such a field is rounding noise of size O(1), that is, of order 1, for
   the twin too. Rank-deficient data is data whose matrix does not have
   full rank. On such data, the twin flips labels against itself about as
   often as the entry does:

   | comparison, per 70,000 rows | LogisticRegression flips | HistGradientBoosting flips |
   |---|---|---|
   | the twin, batch vs row | 109 | 594 |
   | the entry | 114 | 645 |

These are the reasons, judged against the goal that inference does not
change when the entry replaces the twin:

- **The twin does not give one answer.** For the same model and row, the
  twin differs by 9.9e9 ulps between a one-row call and a batch call on one
  machine. It differs by 2,026 ulps between CPU kernels. A ulp bound would
  fail sklearn against itself.
- **The term scale is the quantity that stays invariant.** Every pair that
  the research measured stays within K ≤ 4.12 of S_full. The pairs include
  the entry and the twin, any kernel, and row or batch calls.
- **Predictions mostly behave the same way, but not always.**
  - Linear models, trees, forests and KMeans argmin changed no prediction.
  - On repeated training values, the predictions of HistGradientBoosting do
    change. For matvec, the twin changes them against itself about half as
    often as the entry.
  - For elementwise families on one host, the twin never changes them.
  - So a parity bound promises that the entry stays inside the range of the
    twin's own results on different hosts. It does not promise that the
    predictions of the entry are bit-exact with those of the twin on one
    host. For this reason, the owner's ruling below makes bit-exact the
    default.
- **A measured K is not a bound.** K = 3 already fails at 1,000 seeds and
  on wider inputs. K = n + 3 holds for every order.
- **The looser K costs almost no power to detect defects.** A structural
  defect, such as a dropped term, a wrong sign or a wrong index, moves a
  field by about S/n. That is far above (n + 3)·eps·S for any n below about
  10^7.
- **Why S_full and not the record's S:**
  - Both agree inside `check`.
  - The twin recomputes M on whatever host serves it. With |M| in S, a twin
    on another CPU kernel reaches K = 1,731.

The critique raised three more questions (critique.md §2). The ruling below
answers or places each of them:

- **Same host or another host?** Is the contract "on the host that
  translated the step"? That is all that `check` can test. The answer
  decides two things:
  - It decides if S must be S_full.
  - It decides if values that the twin recomputes on each call count as
    shared constants. These values are PCA's M, KMeans's ‖c‖² and chi2's
    `cosh(πjs)`.
- **May a bounded step sit in a composition?** A composition combines
  several steps, for example a sklearn Pipeline, which runs its steps one
  after the other. A bounded step is a step whose parity bound is not 0.
  Today, the catalog module for compositions (`compose.py`) raises
  `NotNative` for any bounded step.
  - So with Option 1 alone, the translation of
    `make_pipeline(StandardScaler(), PCA())` still raises `NotNative`.
  - If the bounded step is the last step of a Pipeline, that case needs no
    new theory.
  - If the bounded step feeds a discontinuous step, such as a binner or a
    threshold, that case has no bound.
- **The interface of check.** The application programming interface (API)
  of `check` must change from an integer ulp bound to an S for each field.
  That S reads the fitted estimator, and K is a function K(n) of the width.

**Ruling (owner, 2026-10-06).** The owner approved the recommendation, with
one amendment after the downstream verification.

- The parity bound has the general form above. Matvec uses S_full and
  K = n + 3. The other families use the scales and the values of K above.
- Bit-exact is the default. An entry with a parity bound above 0 serves only
  when the caller asks for it, with `to_native(step, allow_bound=True)`.
- The reason for the amendment: on repeated training values,
  HistGradientBoosting flips labels for such an entry. On one host, the twin
  of an elementwise family flips none.
- Until `allow_bound` exists, the Box-Cox entry and the FunctionTransformer
  functions with a bound above 0 still serve by default. PLANS.md lists this
  change first.
- **Same host or another host:** the approved forms hold on another host. S
  is S_full, and the chi2 record counts its `cosh` constant as not shared.
- **The interface of check:** this is work for the loop, not a question.
  PLANS.md lists it.
- **A bounded step in a composition:** this question stays open, in
  [open/bounded-steps-in-compositions.md](../open/bounded-steps-in-compositions.md).
