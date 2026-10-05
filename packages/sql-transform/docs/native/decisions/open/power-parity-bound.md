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
