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

**Methodology (2026-10-06).** The notes are in
[research/2026-10-06/](../research/2026-10-06/README.md): power.md, re-run by
an adversarial verifier, plus framework.md, critique.md and inference.md
(inference.md's verification is pending). The environment is x86-64 with
AVX-512; numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, glibc 2.39,
DuckDB 1.5.5; master 113fba7. The scale is
`S = (1 + max(w, 0))·|t|`, with w = λ·log1p(x) (or (2 − λ)·log1p(−x)), and
with `standardize` `(S_t + |mean_|)/scale_`. K is `|a − b| / (eps·S)`.

1. **What the twin computes** (sourced).
   - sklearn's Yeo-Johnson is scipy's `_yeojohnson_transform`. It tests λ
     strictly against eps = 2^-52: λ = 2 − eps takes the `-log1p(-x)` branch,
     and λ = 2 + 2eps takes `expm1` with divisor −2eps.
   - `standardize` is a `StandardScaler`: subtract, then divide.
   - Box-Cox is `scipy.special.boxcox`, on glibc's `log` and `expm1`.
2. **Which kernels the twin runs** (sourced from numpy v2.5.1, then
   measured).
   - On Linux x86-64 with AVX-512, numpy's float64 `log1p`, `expm1`, `log`
     and `exp` are Intel SVML's high-accuracy kernels. Elsewhere they are
     the platform libm.
   - Against MPFR (gmpy2) over 800,000 + 300,000 draws, for `log1p` and
     `expm1`:
     - SVML is within 0.60 ulp of the correctly rounded result;
     - glibc is within 0.78 ulp;
     - the two are at most 1 ulp apart, and differ on 1–6% of draws,
       depending on the mix.
   - `NPY_DISABLE_CPU_FEATURES=X86_V4` makes numpy equal glibc bit for bit.
3. **The twin against itself.** The same fitted steps, seeds 0–199, under
   SVML and under glibc:
   - 11.4% of lanes differ without `standardize`, and 7.7% with it;
   - up to 886 ulps, at K ≈ 1.9;
   - one `standardize` lane differs by 1e19 ulps on its own (0.0 against
     nonzero).
4. **The entry's spellings** (derived from glibc's documented accuracies:
   0.52 ulp for `log` and 0.51 ulp for `exp`).
   - Goldberg's `log1p` is within 4.04u: Yeo-Johnson only calls it on z ≥ 0,
     and the bound grows to 5.04u once 1 + z exceeds 2^53.
   - Kahan's `expm1` is within 4.06u (5.06u when e^w < ½ or e^w > 2^53).
   - Measured: within 2.01 and 2.30 ulps over 400,000 draws each.
   - DuckDB evaluates both spellings bit-identically to the Python model.
5. **Conditioning** (derived; re-derived by the verifier).
   - The relative condition of t with respect to `log1p` is
     c(w) = w·e^w/(e^w − 1) ≤ 1 + max(w, 0). The record's `(1 + |w|)·|t|` is
     valid but up to (1 + |w|)× loose for w < 0.
   - The first-order worst-case K:

     | configuration | K |
     |---|---|
     | Yeo-Johnson | 9.05 |
     | Yeo-Johnson, `standardize` | 11.05 |
     | Box-Cox | 4.53 |
     | Box-Cox, `standardize` | 6.53 |
6. **Measured K.**
   - Fixtures: ≤ 1.99 in every configuration.
   - 2M stress lanes: 2.84.
   - The verifier's adversarial search: 3.9.
   - No inf/NaN mismatches anywhere.
7. **The record's 4.4e18-ulp lane** is at K = 0.112. It comes from an int64
   constant fit column c, where `mean_ = t(c)` exactly and the served x is c.
8. **The served Box-Cox bound** (critique, 400M targeted lanes): 692 lanes
   sit at exactly 4 ulps and none at 5. It holds, with no headroom; the
   first-order worst case is about 9 ulps.
9. **DuckDB** has no `log1p` or `expm1`, in 1.5.5 or on main (2026-10-06).
10. **Downstream** (inference.md, verification pending): Yeo-Johnson with
    `standardize`, on 6 datasets, changed no prediction on new rows.
11. **Verifier corrections taken.**
    - The overflow rule must be stated on the output, not on w, because
      `standardize` moves the point of overflow.
    - In DuckDB `NaN >= 0` is TRUE.
    - The constants assume glibc-grade `ln` and `exp`. With 1-ulp functions,
      K becomes 10.5 and 12.5.

**Recommendation.** Option 1.

- **Scales:**
  - Yeo-Johnson: `S_t = (1 + max(w, 0))·|t|`.
  - Box-Cox: `S_t = |t|`. Its `log` and its large-w branch are rounded
    identically on both sides.
  - With `standardize`: `(S_t + |mean_|)/scale_`.
- **K, from the first-order analysis, rounded up:**

  | configuration | K |
  |---|---|
  | Yeo-Johnson | 10 |
  | Yeo-Johnson, `standardize` | 12 |
  | Box-Cox, `standardize` | 7 |
  | Box-Cox, as served today | 5, with S = \|t\| |

  - The served Box-Cox moves from 4 ulps to K = 5 because its 4 ulps is a
    measurement with no headroom.
  - These constants assume glibc-grade `ln` and `exp` on the entry's side.
    Gate them on the kernel probe, as `function.py` already gates its
    bounds, and refuse elsewhere. On a host with 1-ulp functions, K would
    be 11 and 13 for Yeo-Johnson, and 6 and 8 for Box-Cox.
- **The entry:**
  - copies scipy's strict branch tests;
  - has an explicit NaN arm;
  - adds an `isinf` arm to Kahan's `expm1`, which returns NaN at overflow
    today (inf·0).
- **check:** a lane with exactly one infinite side passes only if the other
  side is finite, has the same sign, and lies within K·eps·S of DBL_MAX.

Why, judged against the goal that inference does not change:

- **Option 3's exit never arrives.**
  - The twin is SVML on Linux AVX-512 and libm everywhere else, and it
    differs from itself by 886 ulps between the two.
  - Even with a DuckDB `log1p`/`expm1`, the entry would equal only the
    non-AVX-512 twin.
- **The bound covers both distances with headroom.** The entry differs from
  the twin by K ≤ 3.9 at worst, and the twin differs from itself by K ≈ 1.9.
  Both sit inside the derived 10 and 12 with about 3× headroom.
- **The record's open case closes.** Its 4.4e18-ulp lane is K = 0.11 under
  this form.
- **No prediction changed** downstream on new rows.
