# How close a power transform must be

**Question.** The parity ruling (confit's
`docs/decisions/closed/native-transform-parity-bounds.md`) holds an entry
whose order cannot be the twin's "within a small per-family ulp bound",
measured. `PowerTransformer` has no small bound in ulps of the result in
two of its configurations. What bound do they declare? The same question as
[matvec-parity-bound.md](matvec-parity-bound.md), for another family.

**What is served.** `PowerTransformer(method="box-cox", standardize=False)`,
within 4 ulps (`native/power.py`): scipy's Box-Cox is C on glibc's `log`,
`exp` and `expm1`, the entry has glibc's `log` and `exp` (DuckDB's and
confit's), and only `expm1` is spelled, Kahan's `(u - 1) * (w / ln(u))`,
within 2 ulps of glibc's.

**Why the rest is not small.** DuckDB 1.5.5 has no `log1p` or `expm1`
(confit cannot add one: the definition must run on DuckDB), so the entry
spells them from `ln` and `exp`, a rounding or two from the twin's.

1. *Yeo-Johnson* is `expm1(lambda * log1p(x)) / lambda` (and its mirror
   for x < 0) through numpy's `log1p` and `expm1`, which on x86-64 with
   AVX-512 are numpy's own SIMD kernels, not glibc's (they differ from
   glibc on about 1% and 5% of draws). An ulp apart in `log1p(x)` is about
   `|lambda * log1p(x)|` ulps apart after `expm1`, up to about 700 before
   it overflows, and at the overflow edge one side is infinite.
2. *`standardize=True`* follows with the fitted `(t - mean_) / scale_`:
   for a `t` near `mean_` an ulp apart in `t` is any number of ulps of the
   difference, as in PCA's projected mean.

Measured on the catalog's fixtures, seeds 0-199, the entry spelling
`log1p(z)` as Goldberg's `z * (ln(u) / (u - 1))`, `u = 1 + z`, and `expm1`
as above (2026-10-05, x86-64 with AVX-512, glibc 2.39, numpy 2.5.1):

| configuration | lanes | max, ulps of the result |
|---|---|---|
| Yeo-Johnson, `standardize=False` | 18,173 | 1,004 |
| Yeo-Johnson, lambdas pinned at 0, eps, 2, 2 + 2 eps, ... | 16,429 | 16 |
| Yeo-Johnson, `standardize=True` | 14,464 | 1,650 |
| Yeo-Johnson, pinned lambdas, `standardize=True` | 16,930 | 4.4e18 (the twin's 0.0 against 8.9e-16) |
| Box-Cox, `standardize=True` | 9,304 | 362 |
| Box-Cox, `standardize=False` (served) | 9,042 | 2 |

**Options.**

1. *A term-scale bound,* as matvec-parity-bound.md proposes: per lane
   `|native - twin| <= K * eps * S`, with S the lane's condition: for
   Yeo-Johnson `(1 + |lambda * log1p(x)|) * |t|`, with `standardize`
   `(S_t + |mean_|) / scale_`. `check` gains the per-lane scale.
2. *A ulp bound measured over N seeds:* not small (above), and not a
   bound: the next seed can exceed it.
3. *They stay Python* until DuckDB has `log1p` and `expm1` and numpy's
   kernels are reproducible, which they are not portably.

**Provisional choice.** Option 3: Yeo-Johnson and Box-Cox with
`standardize=True` raise `NotNative` naming this record. Option 1 is the
loop's proposal, with the matvec families'.

**What would close it.** The owner's ruling on matvec-parity-bound.md, if
it covers a per-lane condition other than a dot product's term scale, or a
ruling here.

**Methodology (2026-10-06).** The research notes are in
[research/2026-10-06/](../research/2026-10-06/README.md). This record uses
four of them:

- power.md is the note on this family. An adversarial verifier re-ran it.
  The verifier is a second agent that tried to refute each claim of the note.
- framework.md is the note on the general form of a parity bound.
- critique.md is the note that settles the contradictions between the notes.
- inference.md is the note that tests if the entry changes a prediction. An
  adversarial verifier re-ran it too.

The environment is an x86-64 host with AVX-512 (Advanced Vector Extensions,
512-bit). The software is numpy 2.5.1, scipy 1.18.0, scikit-learn (sklearn)
1.9.0, glibc 2.39 and DuckDB 1.5.5. Here glibc is the GNU C library. The code
is master at commit 113fba7.

In the counts below, a field is one output field of one row. Each item says
how the research found its claims. A sourced claim cites the code of a package
or a URL.

The error scale is `S = (1 + max(w, 0))·|t|`. Here t is the output of the
power transform, x is its input and λ is its fitted parameter. The variable w
is λ·log1p(x), or (2 − λ)·log1p(−x) on the branch for x < 0. The function
`log1p(x)` computes ln(1 + x), and `expm1(w)` computes e^w − 1.

With `standardize=True`, S is `(S_t + |mean_|)/scale_`. S_t is the error scale
of t, and `mean_` and `scale_` are the fitted mean and scale. For two results a
and b of one field, K is `|a − b| / (eps·S)`.

1. **What the twin computes** (sourced).
   - sklearn's Yeo-Johnson is scipy's function `_yeojohnson_transform`.
   - That function compares λ with eps = 2^-52 by strict tests. With
     λ = 2 − eps, it takes the `-log1p(-x)` branch. With λ = 2 + 2eps, it
     takes `expm1` with the divisor −2eps.
   - `standardize=True` is sklearn's class `StandardScaler`. It subtracts
     `mean_`, then it divides by `scale_`.
   - Box-Cox is scipy's function `scipy.special.boxcox`, on glibc's `log` and
     `expm1`.
2. **Which kernels the twin runs** (sourced from numpy v2.5.1, then
   measured).
   - On Linux x86-64 with AVX-512, numpy's float64 `log1p`, `expm1`, `log`
     and `exp` are high-accuracy kernels from Intel's Short Vector Math
     Library (SVML). On other hosts, they are the C math library of the
     platform (libm).
   - The research measured `log1p` and `expm1` against MPFR over
     800,000 + 300,000 draws. MPFR is the GNU Multiple Precision
     Floating-Point Reliable library, and it rounds correctly. The research
     used it through gmpy2, its Python binding. The results:
     - SVML is within 0.60 ulp of the correctly rounded result.
     - glibc is within 0.78 ulp.
     - The two are at most 1 ulp apart. They differ on 1–6% of draws,
       depending on the mix of draws.
   - The environment variable `NPY_DISABLE_CPU_FEATURES=X86_V4` turns off
     numpy's AVX-512 kernels. With it, numpy is bit-exact with glibc.
3. **The twin against itself.** The research ran the same fitted steps on
   seeds 0–199, under SVML and under glibc. The results:
   - Without `standardize`, 11.4% of the fields differ. With it, 7.7% differ.
   - The largest difference is 886 ulps, and the largest K is ≈ 1.9.
   - One field with `standardize` differs by 1e19 ulps on its own (0.0
     against a nonzero value).
4. **The entry's spellings** (derived from glibc's documented accuracies,
   0.52 ulp for `log` and 0.51 ulp for `exp`). In this item, z is the
   argument of `log1p`, and u is the unit roundoff, 2^-53. This u is not the
   u = 1 + z of the spelling above.
   - Goldberg's `log1p` is within 4.04u. Yeo-Johnson calls it only on z ≥ 0.
     If 1 + z exceeds 2^53, the bound grows to 5.04u.
   - Kahan's `expm1` is within 4.06u. If e^w < ½ or e^w > 2^53, the bound is
     5.06u.
   - Measured over 400,000 draws each, Goldberg's spelling is within 2.01
     ulps and Kahan's spelling is within 2.30 ulps.
   - DuckDB's results for both spellings are bit-exact with a Python model of
     the entry.
5. **Conditioning** (derived, and re-derived by the verifier).
   - The relative condition of t with respect to `log1p` is
     c(w) = w·e^w/(e^w − 1) ≤ 1 + max(w, 0).
   - For w < 0, the record's scale `(1 + |w|)·|t|` is valid, but it is up to
     (1 + |w|)× loose.
   - The first-order analysis gives this worst-case K:

     | configuration | K |
     |---|---|
     | Yeo-Johnson | 9.05 |
     | Yeo-Johnson, `standardize` | 11.05 |
     | Box-Cox | 4.53 |
     | Box-Cox, `standardize` | 6.53 |
6. **Measured K.**
   - On the catalog's fixtures, K ≤ 1.99 in every configuration.
   - On 2M fields of a stress test, the largest K is 2.84.
   - The adversarial search of the verifier reached K = 3.9.
   - No run found a field where the two sides disagree on infinity (inf) or
     on not a number (NaN).
7. **The record's field at 4.4e18 ulps** has K = 0.112. It comes from a fit
   column c that holds the same int64 value in every row. There,
   `mean_ = t(c)` exactly, and the served x is c.
8. **The served Box-Cox bound** (critique.md, 400M targeted fields). In that
   run, 692 fields are at exactly 4 ulps, and none are at 5. The bound holds,
   but it has no headroom. The first-order worst case is about 9 ulps.
9. **DuckDB** has no `log1p` or `expm1`, in 1.5.5 or on its main branch
   (2026-10-06).
10. **Downstream** (inference.md, verified, and one independent check).
    - Linear models, trees and forests changed no prediction on new rows.
    - HistGradientBoosting puts its thresholds at training values. On
      grid-valued data, such as prices in cents, new rows repeat those
      values, and one rounding decides the branch.
    - On one host, the twin's row, batch and `fit_transform` outputs are
      bit-identical. So the twin never flips a label there.
    - The shipped Box-Cox entry (ulp bound 4) flipped 53–80 labels per
      200,000 fresh rows in the verifier's 4 seeds. An independent check
      found 13–24 in 2 seeds, with 1,150 rows on another branch.
    - The Yeo-Johnson entry flipped 86–111 labels per 100,000–200,000 rows.
      The twin without AVX-512 numpy flipped 17–98.
11. **The record takes these corrections from the verifier.**
    - The rule for overflow must test the output, not w, because
      `standardize` moves the point of overflow.
    - In DuckDB, `NaN >= 0` is TRUE.
    - The constants assume `ln` and `exp` with the accuracy of glibc's. With
      1-ulp functions, K becomes 10.5 and 12.5 (Yeo-Johnson without and with
      `standardize`).

**Recommendation.** Option 1, a parity bound with an error scale for each
output field.

- **Scales:**
  - Yeo-Johnson: `S_t = (1 + max(w, 0))·|t|`.
  - Box-Cox: `S_t = |t|`. Both sides round its `log` and its branch for
    large w the same way.
  - With `standardize`: `(S_t + |mean_|)/scale_`.
- **K, from the first-order analysis, rounded up:**

  | configuration | K |
  |---|---|
  | Yeo-Johnson | 10 |
  | Yeo-Johnson, `standardize` | 12 |
  | Box-Cox, `standardize` | 7 |
  | Box-Cox, as served today | 5, with S = \|t\| |

  - The served Box-Cox entry moves from 4 ulps to K = 5. The reason is that
    its 4 ulps is a measurement with no headroom.
  - These constants assume that the entry's `ln` and `exp` have the accuracy
    of glibc's.
  - The kernel probe (`function.kernel_distance`) measures, on the host, how
    far numpy's kernel for a function is from the entry's spelling of it.
  - Gate the constants on the kernel probe, as the family for function
    transformers (`native/function.py`) already gates its bounds.
  - On a host that does not pass the probe, make the entry raise `NotNative`.
  - On a host with 1-ulp functions, K would be 11 and 13 for Yeo-Johnson. For
    Box-Cox, K would be 6 and 8.
- **The entry:**
  - It copies the strict tests with which scipy chooses a branch.
  - It has an explicit branch for NaN.
  - It adds a branch for infinity (`isinf`) to Kahan's `expm1`. Today that
    spelling returns NaN at overflow (inf·0).
- **check:** `native.check` is the test that serves the same query with the
  twin and with the entry and compares the results. If exactly one side of a
  field is infinite, the field passes only under three conditions:
  - The other side is finite.
  - It has the same sign.
  - It lies within K·eps·S of DBL_MAX, the largest finite double.

These are the reasons, judged against the goal that inference does not
change:

- **The exit from Option 3 never comes.**
  - On Linux with AVX-512, the twin uses SVML. On every other host, it uses
    libm. Between the two, the twin differs from itself by 886 ulps.
  - Even if DuckDB had `log1p` and `expm1`, the entry would equal only the
    twin on a host without AVX-512.
- **The bound covers both distances with headroom.** At worst, the entry
  differs from the twin by K ≤ 3.9. The twin differs from itself by K ≈ 1.9.
  Both sit inside the derived 10 and 12, with about 3× headroom.
- **The open case of this record closes.** Under this form, its field at
  4.4e18 ulps has K = 0.11.
- **Predictions can change on one host.** HistGradientBoosting flips some
  labels for the entry where the twin flips none. The owner's ruling below
  therefore makes bit-exact the default, and puts this bound behind an
  explicit choice of the caller.
