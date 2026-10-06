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

**Methodology (2026-10-06).** The research notes are in the folder
[research/2026-10-06/](../research/2026-10-06/README.md). In this record, a
verifier is a second agent that tried to refute the key claims of a note.
This record uses four of the notes:

- [power.md](../research/2026-10-06/power.md) is the note on this family. A
  verifier re-ran power.md.
- [framework.md](../research/2026-10-06/framework.md) is the note on the
  general form of a parity bound.
- [critique.md](../research/2026-10-06/critique.md) is the note that settles
  the contradictions between the notes.
- [inference.md](../research/2026-10-06/inference.md) is the note that tests
  if the catalog entry changes a prediction. A verifier checked inference.md
  too.

The research used this host and this software:

- **Host.** The host is x86-64, the 64-bit processor architecture of Intel
  and AMD. It has AVX-512 (Advanced Vector Extensions, 512-bit), a set of
  processor instructions that work on 512 bits at a time.
- **Software:**
  - numpy 2.5.1, the array library of Python
  - scipy 1.18.0, a library of scientific functions for Python
  - scikit-learn (sklearn) 1.9.0, the machine-learning library whose
    transformers the native catalog translates
  - glibc 2.39, the GNU C library, which holds the C math functions
  - DuckDB 1.5.5, the SQL database
- **Code.** The code of this repository is the master branch at commit
  113fba7.

This record uses these terms:

- **Field.** In the counts below, a field is one output field of one row.
- **Power transformer.** This family holds sklearn's power transformer,
  `PowerTransformer`. It passes each input feature x through a power
  function with a fitted exponent λ. It has two methods, Yeo-Johnson and
  Box-Cox.
- **t.** t is the output of the power transformer before `standardize`. With
  the option `standardize=True`, the step then standardizes t with the
  fitted mean `mean_` and the fitted scale `scale_`.
- **Functions.** `log` and `ln` both name the natural logarithm, and
  `exp(x)` is e^x. `log1p(x)` computes ln(1 + x), and `expm1(w)` computes
  e^w − 1.
- **Spellings.** The entry's spelling of a function is the formula from `ln`
  and `exp` that computes the function in SQL. The entry spells `log1p` with
  Goldberg's formula and `expm1` with Kahan's formula. Both formulas are in
  the sections above.
- **Ulp.** An ulp (unit in the last place) is the gap between a double and
  the next double.
- **Error scale.** The error scale is `S = (1 + max(w, 0))·|t|`. The
  variable w is λ·log1p(x), or (2 − λ)·log1p(−x) on the branch for x < 0.
  With `standardize=True`, S is `(S_t + |mean_|)/scale_`, where S_t is the
  error scale of t.
- **K.** For two results a and b of one field, K is `|a − b| / (eps·S)`,
  with eps = 2^-52.
- **check.** `native.check` is the test that serves the same query with the
  twin and with the entry, and compares the results. This record calls it
  `check`.
- **Labels.** The words in parentheses after the title of an item say how
  the research found its claims. A measured claim comes from a run. A
  sourced claim cites the code of a package or a URL. A derived claim comes
  from a derivation.

1. **What the twin computes** (sourced).
   - sklearn's Yeo-Johnson is scipy's function `_yeojohnson_transform`.
   - That function tests λ against eps = 2^-52 with strict comparisons. With
     λ = 2 − eps, it takes the `-log1p(-x)` branch. With λ = 2 + 2eps, it
     takes `expm1` with the divisor −2eps.
   - With `standardize=True`, sklearn applies its standard scaler
     (`StandardScaler`). The scaler subtracts `mean_`, then it divides by
     `scale_`.
   - Box-Cox is scipy's function `scipy.special.boxcox`. It runs on glibc's
     `log` and `expm1`.
2. **Which kernels the twin runs** (sourced from numpy v2.5.1, then
   measured). A kernel here is the compiled code of a math function.
   - On Linux x86-64 with AVX-512, numpy's float64 `log1p`, `expm1`, `log`
     and `exp` are high-accuracy kernels from Intel's Short Vector Math
     Library (SVML). On other hosts, they are the C math library of the
     platform (libm).
   - The research measured `log1p` and `expm1` against MPFR over
     800,000 + 300,000 draws. MPFR is the GNU Multiple Precision
     Floating-Point Reliable library, and it rounds correctly. The research
     used it through gmpy2, the Python interface to MPFR. The results:
     - SVML is within 0.60 ulp of the correctly rounded result.
     - glibc is within 0.78 ulp.
     - The two are at most 1 ulp apart. They differ on 1–6% of draws,
       depending on the mix of draws.
   - The environment variable `NPY_DISABLE_CPU_FEATURES=X86_V4` turns off
     numpy's AVX-512 kernels. With it, numpy is bit-exact with glibc.
3. **The twin against itself.** The research ran the same fitted steps on
   seeds 0–199, once under SVML and once under glibc. The results:
   - Without `standardize`, 11.4% of the fields differ. With it, 7.7% differ.
   - The largest difference is 886 ulps, and the largest K is ≈ 1.9.
   - One field with `standardize` differs by 1e19 ulps on its own (0.0
     against a nonzero value).
4. **The entry's spellings** (derived from glibc's documented accuracies,
   0.52 ulp for `log` and 0.51 ulp for `exp`). In this item, z is the
   argument of `log1p`, and u is the unit roundoff, 2^-53. This u is not the
   u of the formulas above.
   - Goldberg's `log1p` is within 4.04u. Yeo-Johnson calls it only on z ≥ 0.
     If 1 + z exceeds 2^53, the bound grows to 5.04u.
   - Kahan's `expm1` is within 4.06u. If e^w < ½ or e^w > 2^53, the bound is
     5.06u.
   - Measured over 400,000 draws each, Goldberg's spelling is within 2.01
     ulps and Kahan's spelling is within 2.30 ulps.
   - For both spellings, DuckDB's results are bit-exact with the results of
     a Python model of the entry.
5. **Conditioning** (derived, and re-derived by the verifier of power.md).
   - The relative condition of t with respect to `log1p` is
     c(w) = w·e^w/(e^w − 1) ≤ 1 + max(w, 0). It is the factor by which a
     relative error in `log1p(x)` grows in t.
   - Option 1 above uses the error scale `(1 + |w|)·|t|`. This error scale
     is valid. For w < 0, it can be up to (1 + |w|) times larger than
     necessary.
   - The first-order analysis gives this worst-case K:

     | configuration | K |
     |---|---|
     | Yeo-Johnson | 9.05 |
     | Yeo-Johnson, `standardize` | 11.05 |
     | Box-Cox | 4.53 |
     | Box-Cox, `standardize` | 6.53 |
6. **Measured K.**
   - On the catalog's fixtures, K ≤ 1.99 in every configuration.
   - On 2M (2 million) fields of a stress test, the largest K is 2.84.
   - The verifier of power.md searched for inputs with a large K. Its search
     reached K = 3.9.
   - No run found a field where the entry and the twin disagree on infinity
     (inf) or on not a number (NaN).
7. **The field at 4.4e18 ulps** in the table above has K = 0.112. It comes
   from a column of the data that the step was fitted on. That column holds
   the same int64 (64-bit integer) value c in every row. There,
   `mean_ = t(c)` exactly, and the served x is c.
8. **The ulp bound of the served Box-Cox entry, 4 ulps** (critique.md, 400
   million targeted fields). In that run, 692 fields are at exactly 4 ulps,
   and none are at 5. The bound holds, but it has no headroom. The
   first-order worst case is about 9 ulps.
9. **DuckDB** has no `log1p` or `expm1`, in 1.5.5 or on its main branch
   (2026-10-06).
10. **Downstream predictions** (inference.md, verified, and one independent
    measurement). A downstream model is a model that reads the output of
    the step. A label flip is a row whose predicted label changes between
    two ways to compute the step, for example the entry and the twin.
    - Linear models, trees and forests changed no prediction on new rows.
    - The sklearn model HistGradientBoosting puts its thresholds at
      training values. Rows with values on a grid, such as prices in cents,
      repeat those values.
    - `fit_transform` is the sklearn method that fits a step on data and
      transforms that data in one call. On one host, the twin's outputs for
      one row, for a batch and from `fit_transform` are bit-exact with each
      other. So the twin never flips a label there.
    - The Box-Cox entry that the catalog serves today has an ulp bound of 4.
      It flipped 53–80 labels per 200,000 new rows in 4 seeds of the
      verifier of inference.md.
    - An independent measurement found 13–24 flips in 2 seeds, with 1,150
      rows on another branch.
    - The Yeo-Johnson entry flipped 86–111 labels per 100,000–200,000 rows.
      The twin without numpy's AVX-512 kernels flipped 17–98.
11. **The record takes these corrections from the verifier of power.md.**
    - A result overflows when it is too large for a double. Then it is
      infinite. The rule of `check` for overflow must test the output, not
      w. The reason is that `standardize` moves the point of overflow.
    - In DuckDB, `NaN >= 0` is TRUE.
    - The constants assume `ln` and `exp` with the accuracy of glibc's. With
      `ln` and `exp` that are only within 1 ulp, K becomes 10.5 and 12.5
      (Yeo-Johnson without and with `standardize`).

**Recommendation.** Option 1 above: a parity bound with an error scale for
each output field.

- **Error scales:**
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
  - The kernel probe is the function `kernel_distance` in
    `native/function.py`, the family for function transformers. On the
    host, it measures how far numpy's kernel for a function is from the
    entry's spelling of it.
  - Use the constants only on a host that passes the kernel probe, as
    `native/function.py` already does for its bounds.
  - On a host that does not pass the probe, make the entry raise
    `NotNative`.
  - On a host whose `ln` and `exp` are only within 1 ulp, K would be 11 and
    13 for Yeo-Johnson. For Box-Cox, K would be 6 and 8.
- **The entry:**
  - It copies the strict tests with which scipy chooses a branch.
  - It has an explicit branch for NaN.
  - It adds a branch that tests for infinity (`isinf`) to Kahan's `expm1`.
    Today that spelling returns NaN at overflow, because it computes inf·0.
- **check:** If exactly one side of a field is infinite, the field passes
  only under three conditions:
  - The other side is finite.
  - It has the same sign.
  - It lies within K·eps·S of DBL_MAX, the largest finite double.

These are the reasons, judged against the goal that inference does not
change. Here, inference means the predictions of a downstream model, as in
item 10.

- **The condition that ends Option 3 never becomes true.**
  - Under Option 3, Yeo-Johnson and Box-Cox with `standardize=True` stay
    with the twin until DuckDB has `log1p` and `expm1` and numpy's kernels
    are reproducible.
  - On Linux with AVX-512, the twin uses SVML. On every other host, it uses
    libm. The twin under SVML and the twin under libm differ by 886 ulps.
  - Even if DuckDB had `log1p` and `expm1`, the entry would equal only the
    twin on a host without AVX-512.
- **The bound covers both distances with headroom.**
  - At worst, the entry differs from the twin by K ≤ 3.9.
  - The twin under SVML and the twin under glibc differ by K ≈ 1.9.
  - Both values are below the derived K for Yeo-Johnson, which is 10
    without `standardize` and 12 with it. The derived K is about 3 times
    the larger value.
- **The open case of this record closes.** Under this form, its field at
  4.4e18 ulps has K = 0.11.
- **Predictions can change on one host.** HistGradientBoosting flips some
  labels for the entry where the twin flips none. The owner's ruling makes
  bit-exact the default, and it puts this bound behind an explicit choice.

**Ruling (owner, 2026-10-06).** The owner approved the recommendation:
option 1, with the scales and the values of K above. The amendment in
[matvec-parity-bound.md](matvec-parity-bound.md) applies. So these entries
serve only when the caller asks for a parity bound above 0. That includes
the Box-Cox entry, which serves by default today.
