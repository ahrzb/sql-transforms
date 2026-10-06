# PowerTransformer: log1p/expm1, numpy SIMD kernels, Yeo-Johnson conditioning

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


All scripts and outputs are in `R=scripts/power/` (outputs in `R/out/`). Each claim is tagged **[M]** measured, **[S]** sourced or **[D]** derived. Notation: u = 2^-53, eps = 2^-52 = 2u, w = λ·log1p(x) (or (2−λ)·log1p(−x) on the mirror branch), t = the transformed value before standardizing.

Environment note: the repo's tests cannot import under this venv. Python 3.14.0rc2 lacks `typing._eval_type(prefer_fwd_module=)`, which pydantic passes. My scripts add a 2-line shim before they import `sql_transform`. I edited nothing in the repo.

## 1. What the twin computes (sklearn 1.9.0, scipy 1.18.0), and corrections to the record

| step | code | numerics |
|---|---|---|
| Yeo-Johnson | `sklearn/preprocessing/_data.py:3471,3475` → `scipy.stats.yeojohnson` → `scipy/stats/_morestats.py:1728-1759` `_yeojohnson_transform` | `eps = finfo(float64).eps = 2^-52` (l.1734). `x>=0`: `abs(λ)<eps` → `log1p(x)`, else `expm1(λ*log1p(x))/λ` (1746-1750). `x<0`: `abs(λ-2)>eps` → `-expm1((2-λ)*log1p(-x))/(2-λ)`, else `-log1p(-x)` (1753-1757) |
| Box-Cox | `_data.py:10` `scipy.special.boxcox` → xsf `include/xsf/boxcox.h:17-22` | `abs(λ)<1e-19` → `log x`; `λ log x < 709.78` → `expm1(λ log x)/λ`; else `copysign(1,λ)·exp(λ log x − log|λ|) − 1/λ`. This matches `power.py` **[S]** |
| standardize | `_data.py:3445-3449` (`StandardScaler(copy=False)` fitted on the transformed fit data), applied at 3478 → `StandardScaler.transform`: `X -= mean_; X /= scale_` (`_data.py:1135-1138`) | two IEEE operations; a zero-variance column gets `scale_ = 1` |
| constant features | `_data.py:3434-3436` | YJ sets `λ = 1` |

Corrections and precisions:
- **The Yeo-Johnson code is scipy's, not sklearn's.** The branch tests are strict, with eps = 2^-52 [S]. So:
  - λ = ±eps takes the `expm1` branch;
  - λ = eps/2 takes `log1p`;
  - λ = 2 − eps takes the `-log1p(-x)` branch, because |λ−2| = eps is not > eps;
  - λ = 2 + 2eps takes `expm1` with divisor 2−λ = −2eps.

  NaN fails `x>=0` and goes to the mirror branch, giving NaN. −0.0 takes the positive branch and gives −0.0. The entry must copy these exact predicates. Including signed zeros, subnormals and ±1e300 over 14 edge λ values, 166 of 168 special lanes are bit-identical and the other two are within 2 ulps (`R/specials.py`) [M].
- **"numpy's own SIMD kernels" are Intel SVML, high-accuracy variant.** At numpy v2.5.1, `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src` l.8-57 and 110-133 call `__svml_log1p8_ha` / `__svml_expm18_ha`. `loops_exponent_log.dispatch.c.src` (around l.666-697 and 1316-1330) calls `__svml_exp8_ha` / `__svml_log8_ha` for float64 exp/log (https://raw.githubusercontent.com/numpy/numpy/v2.5.1/...) [S]. The float64 suffix was `8` (low-accuracy, ≤4 ulp) in numpy v1.22–v1.26 and `8_ha` from v2.0.0 on (I fetched the file at each tag) [S]. In the numpy PR #19478 thread, an Intel contributor gives SVML's tiers as "high (maximum errors up to 1 ulp), medium (up to 4 ulp)" (https://github.com/numpy/numpy/pull/19478) [S].
- **The amplification claim needs a sign condition.** "An ulp apart in log1p is about |λ log1p x| ulps after expm1, up to about 700" holds only for w > 0. For w < 0 the factor is |w|/(e^|w|−1) < 1, so there is no amplification (§4).

## 2. The twin depends on the CPU, the OS and the numpy version

**Dispatch** [M]. numpy 2.5.1 `__cpu_targets_info__` on this machine:
- log1p, expm1 (float64): current `X86_V4`; available `X86_V4 baseline(X86_V2)`.
- log, exp (float64): current `X86_V4`; available `X86_V4 X86_V3 baseline`.

SVML is linked only for Linux x86-64 builds that dispatch X86_V4 (`numpy/_core/meson.build:91-96`, option `disable-svml`) [S]. Everywhere else (non-AVX-512 CPUs, macOS, Windows, arm64), these calls go to the platform libm.

**Pinning** [M]. Pinning works but is process-wide, read at import, and changes every numpy ufunc.
- `NPY_DISABLE_CPU_FEATURES=X86_V4` sends log1p/expm1 to baseline (libm) and log/exp to X86_V3, which for float64 is also the scalar libm loop.
- `AVX512_SKX` and `AVX512F` are rejected with an `ImportWarning` ("not part of the dispatched optimizations (X86_V3 X86_V4 AVX512_ICL AVX512_SPR)") and have no effect.

**Kernel accuracy.** 800,000 draws per function against MPFR correctly rounded (gmpy2); errors in ulps of the correctly rounded result. Script: `R/kernels.py`, output `R/out/d_*.txt` [M]:

| fn | numpy SVML-HA max ulp | not correctly rounded | glibc 2.39 max ulp | not correctly rounded | numpy ≠ glibc | max distance |
|---|---|---|---|---|---|---|
| log1p | 0.602 | 0.12 % | 0.744 | 0.94 % | **1.04 %** | 1 ulp |
| expm1 | 0.517 | 0.011 % | 0.781 | 4.02 % | **4.02 %** | 1 ulp |
| log | 0.597 | 0.10 % | 0.513 | 0.034 % | 0.11 % | 1 ulp |
| exp | 0.721 | 4.56 % | 0.506 | 0.058 % | 4.56 % | 1 ulp |

- This reproduces the record's "about 1% and 5%". The fraction depends on the draw mix.
- With `X86_V4` disabled, numpy equals glibc bit for bit on all 4 × 800k draws.
- 1-element arrays (the twin's per-row path) equal batch calls: 0 of 20k mismatches per function.

**Twin against twin.** The same fitted steps and rows, fixture seeds 0–199, transformed with SVML and with `X86_V4` disabled (`R/twin_cpu.py`) [M]:

| configuration | lanes | lanes differing | max ulps of result | K under the record's S |
|---|---|---|---|---|
| YJ, `standardize=False` | 16,722 | 11.4 % | 886 | 1.90 |
| YJ, `standardize=True` | 16,045 | 7.7 % | 886 | 1.63 |

The twin is as far from itself across CPUs as the entry is from the twin (§5). Re-fitting under the two dispatches also gives different λ in 48.9 % of 2,400 fits, with relative differences up to 1.6e-5, because the fit's likelihood uses numpy log [M].

**glibc is CPU-dependent too.** x86-64 glibc picks FMA/FMA4/SSE2 variants of exp, log, log1p and expm1 at load time (`sysdeps/x86_64/fpu/multiarch/e_log.c`, `ifunc-avx-fma4.h`) [S]. Under `GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA`, 400k draws each differ in 35 (log), 219 (exp), 11 (log1p) and 328 (expm1) [M] (`R/glibc_fma.py`). On any one machine the twin's Box-Cox (scipy) and DuckDB's `ln`/`exp` use the same variant, so the shared roundings stay bit-equal. DuckDB imports `exp@GLIBC_2.2.5` and `log@GLIBC_2.2.5` dynamically (`nm -D`) [M].

## 3. The entry's spellings: bounds and measurements

**glibc accuracy** [S]:
- `libm-test-ulps` (x86_64, 2.39): log 1, exp 1, log1p 1, expm1 1 (double).
- Source comments:
  - `e_log.c:128`: "0.519 ULP (0.520 ULP without fma)";
  - `e_exp.c:150`: "< 0.5+1.11/N+poly" with N = 128 (`math_config.h:192`), about 0.509;
  - `s_log1p.c:62-63` and `s_expm1.c:97-98`: "error is always less than 1 ulp".

**Goldberg log1p, z ≥ 0** (Yeo-Johnson only ever calls log1p on a value ≥ 0) [D]:
- Let v̂ = fl(1+z). Then v̂−1 is exact for v̂ ≤ 2^53.
- f(v) = ln v/(v−1) has relative condition κ = 1/ln v − v/(v−1) ∈ [−1, −½] for v ≥ 1, so the rounding of 1+z costs at most u.
- Relative error ≤ u (sum) + 1.04u (ln at ≤ 0.52 ulp) + u (division) + u (multiplication) = **4.04u**, or **5.04u** when 1+z > 2^53.
- Goldberg's Theorem 4 gives 5ε, with ε = u, for 0 ≤ x < ¾ and ln within ½ ulp (Goldberg 1991; https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html, read via a search-result quote because the proxy blocks that host) [S].

**Kahan expm1** [D]:
- Let û = e^w(1+θ) and y = ln û ≈ w + θ. The computation returns g(y)·w exactly in the (û−1) factor, with g(y) = (e^y−1)/y.
- g(y)/g(w) − 1 ≈ θ(c(w)−1)/w, and |(c(w)−1)/w| < 1 for all w. So the exp rounding costs at most |θ| ≤ 1.02u.
- Total ≤ 1.02 + 1.04 + 2 = **4.06u**, or **5.06u** when û < ½ or û > 2^53 (û−1 inexact).
- This matches the "errors cancel in the division" result of Higham, *Accuracy and Stability*, §1.14.1 (I did not re-read the text).

**Measured, 400k draws each** (`R/spellings.py`) [M]:
- Goldberg: max error 2.01 ulp (p99.9 1.47); 18 % not correctly rounded; distance to glibc and to SVML log1p ≤ 2 ulps.
- Kahan: max error 2.30 ulp (p99.9 1.56); distance to both expm1 kernels ≤ 2 ulps. This confirms power.py's "within 2 ulps of glibc".
- Run on DuckDB 1.5.5, both spellings are bit-identical to the ctypes-glibc reimplementation: 0 of 400k mismatches each. My Python entry therefore is the DuckDB entry.

## 4. Conditioning: is the proposed S right?

For x ≥ 0 and λ ≠ 0, t = expm1(λL)/λ with L = log1p x. The relative condition with respect to L is [D]:

c(w) = |L·∂t/∂L / t| = w e^w/(e^w − 1),  c(0) = 1.

- For w > 0: c = w/(1−e^{−w}) and c − w = w/(e^w−1) ≤ 1, so c ≤ 1+w.
- For w < 0: c = |w|/(e^{|w|}−1) ≤ 1.

So **c(w) ≤ 1 + max(w,0) ≤ 1 + |w|**. The record's S_t = (1+|w|)|t| is a valid first-order bound, but up to (1+|w|)× too loose when w < 0, where t → −1/λ saturates. A tighter, equally valid choice is S_t = (1+max(w,0))|t|. It costs nothing measured: K moves from 1.36 to 1.59 on the worst configuration.

**First-order distance between the two sides** [D]. Let α₁, β₁ be the twin's log1p and expm1 relative errors, α₂, β₂ the entry's, and count one rounding each for λ·L and the division on each side:

|t̂₂ − t̂₁| ≤ |t|[c(w)(|α₁|+|α₂|+2u) + |β₁|+|β₂|+2u] ≤ eps·(1+|w|)|t| · K, with K = (|α₁|+|α₂|+|β₁|+|β₂|)/(2u) + 2.

| configuration | S | analytic K (twin kernels ≤ 1 ulp, i.e. ≤ 2u relative; entry 5.04u / 5.06u) | measured max K |
|---|---|---|---|
| YJ | (1+max(w,0))·\|t\| | **9.05** | 1.87 fixtures, 2.84 stress |
| YJ + standardize | (S_t+\|mean_\|)/scale_ | **11.05** (K_t + 2: the subtraction and division on each side add ≤ 2eps·\|t−m\|/s) | 1.72 |
| Box-Cox (log and big-w branch are shared roundings) | \|t\| | **4.53**, i.e. ≤ 9 ulps of t | 1.90 |
| Box-Cox + standardize | (\|t\|+\|mean_\|)/scale_ | **6.53** | 1.95 |

Side finding: the served Box-Cox bound of 4 ulps is a measurement. The first-order worst case is about 9 ulps. Measured max is 2–3 ulps on fixtures (3 under a glibc-only twin) and 4 in the record's 2.4M draws.

**The record's 4.4e18-ulp row is within the bound** [M] (`R/weird.py`).
- Setup: an int64 fit column holding a single integer c (generator kind 2, rounded), so `mean_ = t(c)` exactly (497 of 684 constant fits) and `scale_ = 1`, served x = c.
- Reproduced at c = 3, λ = 2 + 2eps: twin 0.0 against 8.88e-16 = 4.38e18 ulps, with **K = 0.112**.
- Worst over 684 such lanes: a sign flip (±5.55e-17, 8.7e18 ulps) at K = 0.371.
- With a glibc-only twin it also appears in the plain fixtures: seed 110, x = −2, pinned λ = 0, twin 0.0 against −2.22e-15, **K = 0.596**.
- In general, K = d·s/(eps(S_t+|m|)) ≈ 4/((2+|w|)|t|) for |t| ≈ |m| ≳ 1.

**Overflow edge** [M] (`R/overflow.py`). SVML expm1, glibc expm1 and Kahan-on-glibc all overflow first at the same w: 709.7827128933841 (ln DBL_MAX = 709.782712893384). An inf against a finite value therefore needs the two sides' w to straddle that double. I saw zero such lanes in more than 2.1M lanes. But the bound must say what to do there, because |inf − finite| exceeds any K·eps·S. Proposal: a lane with exactly one infinite side passes iff `|w − ln(DBL_MAX)| ≤ K·eps·(1+|w|)`, with w recomputed by `check` from x and λ.

`power.py`'s `expm1` returns NaN at overflow: (inf−1)·(w/ln inf) = inf·0, confirmed on DuckDB. A Yeo-Johnson entry must add an `exp(w) = inf → inf` arm; Box-Cox avoids it through its w < 709.78 guard.

## 5. Measured K on the catalog generator

Seeds 0–199 (`catalog_test._step/_rows`, variant 0), entry spelled as in §3, twin = `est.transform([row])`. Script: `R/kmeasure.py`, outputs `R/out/k200.json` (SVML twin) and `R/out/k200_glibc.json` (`NPY_DISABLE_CPU_FEATURES=X86_V4`) [M].

| configuration | lanes | max ulps (mine / record) | K max, SVML twin | K max, glibc twin | K p99.9 | inf/NaN mismatches |
|---|---|---|---|---|---|---|
| YJ std=False | 16,722 | 1,011 / 1,004 | 1.36 (tight S: 1.59) | 1.99 | 1.02 | 0 |
| YJ pinned λ, std=False | 16,722 | 62 / 16 | 1.87 | 1.87 | 1.33 | 0 |
| YJ std=True | 16,045 | **1,650 / 1,650** | 1.23 | 1.80 | 0.97 | 0 |
| YJ pinned, std=True (scaler refit) | 16,235 | 1,142 (4.39e18 with glibc twin) / 4.4e18 | 1.72 | 1.71 | 1.37 | 0 |
| Box-Cox std=True | 9,201 | 664 / 362 | 1.79 | 1.95 | 1.44 | 0 |
| Box-Cox std=False (served) | 9,228 | 2 / 2 | 1.90 | 1.90 | 1.45 | 0 |

- The 1,011-ulp lane has a fitted λ = 76.7 and w = 515, so c(w) ≈ 515, at K = 0.99. The 1,650-ulp lane is t − mean cancellation (306.69 − 306.04), at K = 0.43.
- Stress test, 2M Yeo-Johnson lanes (λ uniform in [−5, 7] or at branch edges; x = ±loguniform[1e-300, 1e300] or uniform(−1e3, 1e3); `R/stress.py`): 6.9 % of lanes differ, **K max 2.84**, p99.9 1.37, 0 inf/NaN mismatches. The worst lanes have small |w| (x ≈ 1e-11..1e-4), where Goldberg's tiny-z relative error dominates.
- Measured K (≤ 2.84) sits well under the analytic K (9–11) because the 6–8 error sources rarely align; Higham & Mary's probabilistic analysis describes this typical-versus-worst gap.

## 6. DuckDB

- 1.5.5 `duckdb_functions()` has `exp, ln, log, log2, log10` and no `log1p`/`expm1` [M].
- `extension/core_functions/scalar/math/functions.json` lacks both at v1.4.0, at v1.5.5 and on **main as fetched 2026-10-06** [S] (https://raw.githubusercontent.com/duckdb/duckdb/main/extension/core_functions/scalar/math/functions.json).
- `ln(0)` and `ln(<0)` raise "Out of Range"; `exp(710)` = inf [M]. Goldberg on z ≥ 0 never calls `ln` below 1.

**Option 3's exit condition does not end the problem.** Suppose DuckDB gained glibc `log1p`/`expm1`. The entry would then be bit-exact only against a non-AVX-512 (or `X86_V4`-disabled) twin. Against an AVX-512 Linux twin it would still differ by up to 886 ulps (K 1.9), as the twin does from itself.

## 7. What I could not confirm

- The "≤ 1 ulp" for SVML-HA is an Intel engineer's statement in a numpy PR, not Intel documentation I read. Measured ≤ 0.72 ulp.
- I did not check the log1p/expm1 accuracy of macOS libm or the Windows UCRT. The analytic K assumes ≤ 1 ulp there too.
- I did not re-read Higham §1.14.1, and the Goldberg text came via a search-result quote (the proxy blocks docs.oracle.com).
- My lane counts differ from the record's because I used variant 0 for every configuration. The 1,650 and 2 maxima match exactly; 1,011 against 1,004 is close; 664 against 362 differs. The record's pinned λ set and scaler handling are unknown; mine refits the scaler on the pinned transform.
- On aarch64 the compiler may fuse `λ*log x − log|λ|` in xsf's big Box-Cox branch into an FMA, which would break the "same operations" assumption there. Not measured.



## The researcher's recommendation

I recommend option 1 (a per-lane term-scale bound) for all four PowerTransformer configurations, rather than option 3 (keep them in Python).

- **Why not option 3:** its exit condition never arrives portably. Even with a DuckDB log1p/expm1, the twin is Intel SVML on Linux AVX-512 and libm everywhere else. The twin already differs from itself across CPUs by 886 ulps of the result (K ≈ 1.9), as far as the entry is from it.
- **Scale:** for Yeo-Johnson use S_t = (1 + max(w,0))·|t|, with w = λ·log1p(x) or (2−λ)·log1p(−x). The record's (1+|w|)·|t| is valid but too loose when w < 0. For Box-Cox use S_t = |t|, since its log and its large-w branch are rounded identically on both sides. With standardize, use S = (S_t + |mean_|)/scale_.
- **K:** declare it from the first-order analysis, not from a seed sweep. That gives 10 for Yeo-Johnson, 12 for Yeo-Johnson with standardize, 5 and 7 for Box-Cox, or simply 12 for the whole family. This holds whatever the data, on any platform whose log1p/expm1 are within 1 ulp. The measured maximum is 1.9 on fixtures and 2.84 over 2M stress lanes. A measured K of 3–4, like PCA's, would have little headroom for another libm.
- **Overflow edge:** `check` needs one rule. A lane with exactly one infinite side passes iff |w − ln(DBL_MAX)| ≤ K·eps·(1+|w|).
- **Entry requirements:** the entry must copy scipy's strict branch tests with eps = 2^-52 exactly, and add an inf arm to Kahan's expm1, which today returns NaN on overflow.
- **The record's 4.4e18-ulp lane** has K = 0.11 under this rule, so the form resolves the record's open case. It is the same per-lane-condition form as matvec-parity-bound.md, with a different S.


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| sklearn 1.9 YJ = scipy _yeojohnson_transform with eps=2^-52 and strict predicates (λ=±eps→expm1, λ=2−eps→log branch, λ=2+2eps→expm1 with divisor −2eps); standardize = StandardScaler(copy=False): X -= mean_; X /= scale_. | **CONFIRMED** | Read .venv/.../scipy/stats/_morestats.py:1728-1759: eps = xp.finfo(dtype).eps; `abs(lmbda) < eps` → log1p; `abs(lmbda - 2) > eps` → -expm1((2-λ)log1p(-x))/(2-λ). Checked the arithmetic: 2−2^-52 is representable and \|λ−2\| = eps is not > eps. 2+2^-51 is the next double above 2, so it takes expm1 with divisor −2eps. sklearn _data.py:3409-3479 dispatches to stats.yeojohnson and sets λ=1 for constant features. StandardScaler.transform (_data.py:1135-1138) does `X -= mean_; X /= scale_`. | One caveat for the entry: in DuckDB `NaN >= 0` is TRUE (checked on 1.5.5), so the entry cannot copy scipy's `x >= 0` test literally. NaN still comes out as NaN through Goldberg/Kahan, but the entry needs an explicit NaN arm. The proposed inf arm must test `isinf(u)` or `u = inf`, not `u >= …`, because in DuckDB NaN sorts above inf. |
| numpy 2.5.1 float64 log1p/expm1/log/exp are SVML __svml_*8_ha on this AVX-512 machine (v1.22–1.26 used low-accuracy `8`); SVML is linked only on Linux x86-64; NPY_DISABLE_CPU_FEATURES=X86_V4 pins them to glibc bit-identically; AVX512_SKX is rejected with an ImportWarning and does nothing. | **CONFIRMED** | Fetched numpy v2.5.1 loops_umath_fp.dispatch.c.src (lines 8-57: `func_suffix = f16, 8_ha`) and loops_exponent_log.dispatch.c.src (666-697, 1316-1337: `__svml_@func@8_ha`). The suffix is `8` at v1.22.0 and v1.26.4 and `8_ha` at v2.0.0 and v2.1.0. meson.build:91-96 sets use_svml only for linux, x86_64, X86_V4 and not disable-svml. nm on _multiarray_umath shows the __svml_*_ha symbols. __cpu_targets_info__: with X86_V4 disabled, log1p/expm1 → baseline and log/exp → X86_V3. With AVX512_SKX all four stay X86_V4 and the ImportWarning appears. My own 4×300k draws with X86_V4 disabled: numpy == libm with 0 mismatches. 1-element calls equal batch calls (0/20k). | A small inaccuracy in report §2: 'everywhere else these calls go to platform libm' is wrong for float64 exp/log on non-SVML, non-MSVC x86 builds with AVX512F. The #else branch at loops_exponent_log:1327-1332 uses numpy's own AVX512F_exp/log_DOUBLE kernels there. That affects only the λ fit, not the YJ transform, which uses log1p/expm1. |
| SVML and glibc differ on 1.04% (log1p) and 4.02% (expm1) of draws, always by ≤1 ulp; SVML max error 0.60/0.52 ulp, glibc 0.74/0.78 ulp vs MPFR. | **PARTLY** | My own draws (300k per function, gmpy2 at 200 bits, vk.py): max distance is always 1 ulp (confirmed). SVML max is 0.589 (log1p) / 0.513 (expm1), glibc 0.772 / 0.769. The two differ on 2.54% (log1p) and 6.19% (expm1) of draws. exp: SVML 0.687 ulp, 4.7% differ. | The ≤1 ulp distance and the SVML-HA accuracy (~0.6 ulp) hold up. The percentages and the glibc maxima depend on the draw mix (glibc log1p reached 0.772 > 0.744 on my draws), so they should not be quoted as properties of the kernels. Intel's compiler docs (via SPEC flag pages) put 'high' at max-error 0.6 ulp, which is tighter than the PR comment's 'up to 1 ulp'. |
| The YJ twin is CPU-dependent: the same fitted steps (seeds 0-199) under SVML vs glibc differ on 11.4% (std=False) / 7.7% (std=True) of lanes, by up to 886 ulps, at K = 1.90 / 1.63. | **CONFIRMED** | Independent pipeline: vgen.py pickles fitted steps, vtwin.py runs them under each dispatch, vk2.py compares. Variant 0: 11.37% / 7.68% of lanes, 886 ulps, K 1.901 / 1.632 under the record's S, which is the S the report's table uses (the claim's 'proposed S' wording is loose). Variant 1: 11.4% / 7.65%, 909 ulps, K 1.97 / 1.50. One std=True lane differs across CPUs by 1.0e19 ulps, the same zero/sign-flip pattern as the record's 4.4e18 row, arising naturally in the twin alone. |  |
| c(w) = w e^w/(e^w−1) ≤ 1+max(w,0); record S valid but up to (1+\|w\|)× loose for w<0; first-order K: YJ 9.05, YJ+std 11.05, Box-Cox 4.53, Box-Cox+std 6.53, using glibc e_log.c:128 (0.519 ULP), e_exp.c:150, s_log1p/s_expm1 '<1 ulp'. | **CONFIRMED** | Re-derived line by line. c−w = w/(e^w−1) ≤ 1 for w>0, and c = \|w\|/(e^\|w\|−1) ≤ 1 for w<0. Goldberg: κ = 1/ln v − v/(v−1) ∈ [−1, −½], giving u + 1.04u + u + u = 4.04u, plus u when 1+z > 2^53. Kahan: h(w) = (c−1)/w ∈ (0,1), giving 1.02u + 1.04u + 2u = 4.06u, plus u when û < ½ or û > 2^53. K = (2+5.04+2+5.06)/2 + 2 = 9.05; +2 for the subtraction and division gives 11.05; Box-Cox (2+5.06+2)/2 = 4.53. Fetched glibc-2.39 sources: e_log.c:127 'Worst case error if \|y\| > 0x1p-4: 0.519 ULP', and the near-1 path (line 75) '~0.507 ULP', so 0.52 covers both. e_exp.c:150 '< 0.5+1.11/N+poly' with EXP_TABLE_BITS 7. s_log1p.c:62-63 and s_expm1.c:97-98 say '<1 ulp'. Goldberg Theorem 4 (5ε, 0≤x<3/4, ln within ½ ulp) confirmed via search result. | The constants hold only for the entry's ln/exp at glibc accuracy (≤0.52/0.51 ulp). If DuckDB's ln/exp are 1-ulp functions (macOS/Windows libm, unverified), Goldberg becomes 6u and Kahan 7u. Then K = 10.5 for YJ and 12.5 for YJ+std, which exceeds the recommended 10 and 12. |
| Measured K on the catalog generator, seeds 0-199: ≤1.90 in every config with the SVML twin, ≤1.99 with a glibc twin, 2.84 over 2M stress lanes; no inf/NaN mismatches; max ulps 1,650 (YJ std) and 1,011 vs 1,004 (YJ). | **PARTLY** | Variant 0 reproduces exactly with my own entry code: YJ K 1.357 (1.594 under the tight S), YJ+std 1.229, Box-Cox+std 1.792, Box-Cox 1.904, max ulps 1011 / 1650 / 664 / 2, no inf/NaN mismatches. Variant 1: pinned Box-Cox std K = 1.934, above 1.90. Entry vs glibc twin with the same λ: YJ K 2.48, above 1.99 (the report's glibc run re-fitted λ, so it is a different fixture set). My adversarial stress, 4M + 8M lanes with small \|w\| and λ near 0 and 2 (vstress.py, vstress2.py): K max 3.91 (x = −3.7e-12, λ = −0.378), and a local search (vsearch.py) also tops out at 3.90. Variant 1 YJ std=False max ulps is exactly 1004, the record's number. | The measurements are sample maxima, not ceilings. The empirical tail is about 3.9, not 2.84. The ordering 'measured ≪ analytic 9.05' still holds. |
| The record's 4.4e18-ulp lane is within the bound: K=0.112 (int64 constant column, mean_ = t(c) exactly, x=c=3, λ=2+2eps); worst over 684 such lanes 0.371; natural occurrence under a glibc twin K=0.596. | **CONFIRMED** | vweird.py: a constant column of 3s with λ = 2+2^-51 gives twin 0.0 vs native 8.88e-16 (4.38e18 ulps), K = 0.112. Over 7 constants × 12 λ × 6 n, the worst is K = 0.361 (c=1, λ=eps). The glibc-run seed 110 entry (x=−2, λ=0, K 0.5956) is present in the report's k200_glibc.json. My pinned-λ fixtures show 4.38e18 / 8.76e18-ulp lanes at K ≤ 1.73. |  |
| Goldberg log1p and Kahan expm1 on glibc ln/exp are within 2.01 / 2.30 ulp of exact and within 2 ulps of glibc and SVML; on DuckDB 1.5.5 they are bit-identical to the Python reimplementation. | **CONFIRMED** | vspell.py, 400k draws each over different ranges: Goldberg max 1.57 ulp, Kahan max 2.13 ulp, both ≤ 2 ulps from glibc and from SVML. My own DuckDB SQL spelling matched the math-module version on 200k/200k values each. | These are sample figures. The analytic worst case allows about 4 ulps from exact (5.04u relative), so 'within 2 ulps of glibc' is a measurement, not a bound. |
| DuckDB 1.5.5 and main have no log1p/expm1; power.py's Kahan expm1 returns NaN on overflow; SVML, glibc and Kahan expm1 all first overflow at w = 709.7827128933841. | **CONFIRMED** | duckdb_functions() on 1.5.5 returns exp, ln, log, log2, log10 only. The fetched main extension/core_functions/scalar/math/functions.json has no log1p/expm1. On DuckDB: exp(709.782712893384) = 1.7976931348622732e308 for all three; at the next double, np.expm1 = inf, libm expm1 = inf, and DuckDB (u−1)*(w/ln u) = NaN (ln(inf) = inf, inf·0). |  |
| glibc exp/log/log1p/expm1 have FMA ifunc variants; disabling FMA/AVX2 via GLIBC_TUNABLES changes 35/219/11/328 of 400k results; the Box-Cox twin and DuckDB stay mutually bit-equal on one machine. | **CONFIRMED** | vfma.py, my own 400k draws: GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA (or just -FMA) changes exp 243, log 25, log1p 177, expm1 337, all by 1 ulp. nm -D: _duckdb imports exp/log@GLIBC_2.2.5, and scipy special _ufuncs imports exp/expm1/log/log1p@GLIBC_2.2.5, so both use the same dynamically resolved ifunc variant. | The counts depend on the draws. The claim text labels them 'exp/log' in the opposite order from report §2, which says 35 = log and 219 = exp; my run agrees with §2 (exp changes far more often than log). |

### What the report missed or got wrong

1. The proposed overflow rule is incomplete when standardize=True. The rule keys on |w − ln DBL_MAX|, but with scale_ < 1 the final (t − m)/s overflows at a different w. Demo (voverflow.py): x = 1.78e268, λ = 1.0175, w = 628.47; twin and entry t differ by 1e-13 relative. With s = 4.75e-36, the twin gives 1.7976931348622133e308 and the entry gives inf. The rule rejects this legitimate one-rounding lane (|w − ln DBL_MAX| = 81). An ordinary s = 0.5 with λ = 1.5 moves the straddle to w ≈ 709.49, which the rule also rejects. Box-Cox+std has the same problem. The rule should be stated on the output: one side ±inf, the other side finite with the same sign and within K·eps·S of ±DBL_MAX. `check` must also compute K·eps·S without overflow, multiplying K·eps·(1+w) before |t| and taking care with /scale_. Otherwise S = inf for |t| ≳ 2.5e305, and a naive `d <= K*eps*S` accepts any finite answer.

2. 'This holds on any platform whose log1p/expm1 are within 1 ulp' overclaims. The constants also assume the entry's ln/exp are at glibc's 0.52/0.51 ulp. With 1-ulp ln/exp, YJ needs K = 10.5 and YJ+std 12.5, so the recommended 10 / 12, and '12 for the whole family', are not guaranteed off glibc. Either declare 11 / 13 or scope the claim to glibc.

3. The measured tail is understated. My targeted stress and local search reach K ≈ 3.9, not 2.84. Variant-1 fixtures exceed the quoted '≤1.90 SVML / ≤1.99 glibc' (1.93, and 2.48 against a glibc twin with the same λ). This actually strengthens the report's argument against a measured K of 3–4.

4. DuckDB NaN semantics: `NaN >= 0` is TRUE and `NaN = NaN` is TRUE in DuckDB. 'Copy scipy's exact predicates' is therefore not literally possible; the entry needs an explicit NaN arm, as power.py has for Box-Cox. The new inf arm must not be written as an ordering comparison, because NaN sorts above inf.

5. Minor:
- Kernel-disagreement percentages and the glibc FMA counts are draw-specific. The claim text swaps the exp/log labels.
- On non-SVML AVX512F builds, numpy's float64 exp/log are numpy's own kernels, not libm. This affects only the λ fit.
- Moving the served Box-Cox entry from 4 ulps to K = 12·eps·|t| (up to about 24 ulps) loosens a served contract. K = 5 for Box-Cox is the analytic value and should be kept separate from the family-wide 12.


### Does the recommendation follow? Yes

The core recommendation follows from the evidence. Use option 1 (a per-lane bound scaled by the lane's condition), with S_t = (1+max(w,0))|t| for YJ, |t| for Box-Cox, (S_t+|m|)/s with standardize, and K declared from first-order analysis rather than a seed sweep.

The supporting points hold up:
- I reproduced the twin's own cross-CPU spread (886–909 ulps, K ≈ 1.9–2.0, and natural 1e19-ulp zero/sign lanes). That defeats option 3's exit condition.
- The conditioning derivation and the 9.05 / 11.05 / 4.53 / 6.53 constants check out.
- The record's 4.4e18-ulp lane does sit at K = 0.11.
- My adversarial search tops out around K ≈ 3.9. That vindicates declaring K analytically rather than from a sweep: a measured K of 3 would already be exceeded.

Three corrections are needed before it is adopted:
1. The overflow rule must be restated on the final output, not on w. Otherwise it wrongly fails lanes where standardize overflows at a different w. S must also be computed so it cannot overflow to inf, which would make the check pass anything.
2. The 'any platform' guarantee needs either glibc-accurate ln/exp on the entry side or slightly larger constants: 11 for YJ, 13 for YJ+std.
3. The entry requirements need an explicit NaN arm because of DuckDB's NaN ordering, and an `isinf`-style overflow arm.

Box-Cox should keep its own K = 5 rather than inherit a family-wide 12.

Scripts: scripts/verify-power/{vk.py, vfma.py, vgen.py, vtwin.py, vk2.py, vlib.py, vstress.py, vstress2.py, vsearch.py, vspell.py, vweird.py, voverflow.py}. Outputs are in its out/ directory (vk2_0.txt and vk2_1.txt are the fixture K tables).
