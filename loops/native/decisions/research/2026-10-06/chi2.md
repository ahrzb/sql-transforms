# AdditiveChi2Sampler: numpy log kernel and the cos/sin condition

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


Scripts and outputs are in `R=scripts/chi2/`. Every run is under 30 s. Sources: a sparse clone of numpy **v2.5.1** (`$R/src_numpy/np`, commit 5e1d03f), numpy's vendored SVML at the submodule commit 3a713b1 (`$R/src_svml/s`), and glibc 2.39 files (`$R/src_glibc`). Notation: u = 2^-53, eps = 2^-52, L = ln x, a = j·s·L, X = |a|, f = factor.

## TL;DR
- **The record's K = 1 is too tight.** With the real kernels, sklearn's own `sample_steps=3` default (s = 0.4) reaches **K = 1.241**, and s = 0.8 reaches **K = 1.211**. In general K approaches A(s,j). A = 2/σ(s), where σ is the significand of s: it is 1 only when s is a power of two (s = 0.5, the `sample_steps=2` default) and 1.25 at s = 0.4 or 0.8. The derived bound is **K ≤ max(d·A, κ) ≤ 3** for every (s, j), with d = 1 and κ ≤ 2.1.
- numpy's float64 `log` here is **Intel SVML `__svml_log8_ha`**, not numpy's own AVX512F (Tang) kernel. It is 1 ulp from glibc on 1.9e-4 of U(0,1e3) draws and about 2.7% of draws with |ln x| < 1/16. numpy's `sin`/`cos` are glibc's (0 differences in 1.4M).
- **Pinning works.** With `NPY_DISABLE_CPU_FEATURES=X86_V4`, numpy's log is glibc's, and the twin equals the native lanes on **0 of 23.4M** lanes. So the twin itself changes bits by CPU and numpy build.
- **The record's log-uniform row is biased.** x = exp(t) puts ln x within u of a double, so for |ln x| ≥ 2 the two kernels never disagree (0 of 10M). With random mantissas they do (rate 2.9e-4 down to 1.6e-6).
- **Correction to the record:** the default (s, j) list is off by one. `sample_steps=1` has no cos/sin lanes, so it is bit-exact today.
- **Spelling SVML's log in SQL** is feasible in principle but not in practice. It needs about 9 emulated FMAs, a 17-step rcp14 table, and a let-binding confit lacks; without binding, the SQL text grows exponentially.

## 1. The twin (sklearn 1.9, `.venv/.../sklearn/kernel_approximation.py`)
| item | source | verdict |
|---|---|---|
| `non_zero = X != 0.0`; lane 0 `sqrt(X_nz*s)` | :784, :788 | as recorded |
| `log_step = s*np.log(X_nz)`; `step = 2*X_nz*s` = (2x)·s | :792-793 | as recorded |
| `factor = sqrt(step/np.cosh(np.pi*j*s))`; lanes `factor*cos(j*log_step)`, `sin` | :796, :799, :803 | as recorded |
| `j in range(1, sample_steps)` | :795 | **steps=1: no j; steps=2 (default): s=0.5, j=1; steps=3: s=0.4, j=1,2** (:732-736). The record's "s=0.8, j=1 … s=0.4, j=1,2,3" is off by one. |
| `validate_data(..., ensure_non_negative=True)` in `transform` | :722; `check_non_negative` validation.py:1763-1788 | Negative inputs raise. NaN and inf raise (`ensure_all_finite` default). `-0.0` passes (min < 0 is false) and gives **+0.0** on every lane. |
| `sample_interval ∈ [0, ∞)` | :666 | s = 0 is legal: sin lanes give `-0.0` for x < 1. |

MEASURED (`k9_sklearn_edges.py`):
- x = 5e-324: factor underflows to 0, so lanes are ±0. With s = 0.5, x·s rounds to 0.
- x = 1e308: (2x)s overflows, so the twin answers ±inf. DuckDB 1.5.5 and confit also return inf and do not raise.
- The step's call `est.transform([row])` equals the vectorized transform bit-for-bit on 20,000 × 5 rows, for steps 2 and 3. SVML results do not depend on lane position.

## 2. Which `log` numpy runs (SOURCED) and how accurate (MEASURED)
**Dispatch chain:**
- `generate_umath.py:933-940`: `log` 'fd' goes to `loops_exponent_log`. `meson.build:1020-1024` compiles it for targets `[X86_V4, X86_V3]`.
- `loops_exponent_log.dispatch.c.src:1313-1340`: with `NPY_HAVE_AVX512_SKX && NPY_CAN_LINK_SVML`, `DOUBLE_log` calls `simd_log_f64`, which calls `__svml_log8_ha` (:691).
- The Tang AVX512F kernel (:921-1100) sits in the `#else` branch, so it is dead code on Linux SVML builds. That kernel would have deferred to glibc on (1−2^-4, 1+0x1.09p-4).
- On X86_V3, `log` is scalar `npy_log`, which is libm.
- SVML is linked iff Linux, x86_64, and X86_V4 is dispatched (`meson.build:91-101`). X86_V4 includes AVX512_SKX (`meson_cpu/x86/meson.build:28-33`).
- The installed `_multiarray_umath*.so` has `T __svml_log8_ha` (`nm`).
- The near-1 mismatches below are a second fingerprint: they are SVML's, not Tang's.

**SVML log8_ha algorithm** (`svml_z0_log_d_ha.s`):
- `vgetmantpd`/`vgetexppd` for mantissa and exponent.
- `vrcp14pd` → `vrndscalepd $0x58`, giving 17 values. MEASURED (`k10_rcp.c`, 2^26 m): 17 distinct values; the result differs from RN(1/m) on the same grid for 2.9e-4 of m.
- A 16-entry hi/lo table and **11 FMAs**; one is exact (R = Rcp·m − 1).

**float64 sin/cos** are libm: `loops_trigonometric.dispatch.cpp:215-231` ("Disable SIMD code sin/cos f64 and revert to libm").

**Documented accuracy:**
- numpy's test tolerance is 1 ulp on 147 float64 vectors (`tests/data/umath-validation-set-log.csv`).
- glibc `e_log.c:128`: "0.519 ULP (0.520 ULP without fma)"; near 1 (:73), 0.507.
- glibc `s_sin.c:29`: "~0.55 ULP".
- `libm-test-ulps` x86_64: log, cos and sin all at 1 (:1712, :923, :1930).
- Intel's "HA ≤ 1 ulp" claim is from a search snippet only; intel.com is blocked here, so this is **unverified**.

**Kernel accuracy, MEASURED** (`k1_kernels.py`; reference is MPFR via gmpy2):

| draws (n) | numpy≠glibc | max dist | numpy≠CR | glibc≠CR | max\|err\| numpy | max\|err\| glibc |
|---|---|---|---|---|---|---|
| U(0,1e3) 200k | 1.85e-4 | 1 | 1.45e-4 | 9e-5 | 0.5007 | 0.5002 |
| exp(U) 1e-300..1e300 200k | 2e-5 | 1 | 1e-5 | 1e-5 | 0.5008 | 0.5087 |
| 1+N(0,1e-3) 200k | 1.59e-2 | 1 | 1.59e-2 | 5e-6 | 0.586 | 0.5000 |
| U(0.9,1.1) 200k | 1.97e-2 | 1 | 1.92e-2 | 2.0e-3 | **0.620** | 0.517 |
| subnormal 50k | 0 | 0 | 0 | 0 | 0.49999 | 0.49999 |

Neither kernel is correctly rounded. A correctly rounded SQL log would still differ from numpy on about 2% of draws near 1.

**Mismatch rate by |ln x|** (`k11_rate.py`, 2M draws per bin):

| \|ln x\| | [0,1/16) | [1/16,1/4) | [1/4,1) | [1,2) | [2,4) | [4,16) | [16,64) | [64,256) | [256,745) |
|---|---|---|---|---|---|---|---|---|---|
| random mantissa | 2.7e-2 | 5.1e-3 | 1.2e-3 | 4.1e-4 | 2.9e-4 | 1.1e-4 | 2.3e-5 | 7.0e-6 | 1.6e-6 |
| x = exp(t) (record's construction) | 2.7e-2 | 5.2e-3 | 1.4e-3 | 3.8e-5 | **0** | **0** | **0** | **0** | **0** |

Why exp(t) draws are biased (DERIVED): if x = fl(e^t), then ln x = t + δ with |δ| ≤ u. For |t| ≥ 2, ulp(t) ≥ 4u, so ln x lies within ulp(t)/4 of the double t. That is never near a rounding midpoint, so the two kernels never disagree there.

**Other measurements:**
- cos and sin: numpy vs `math` (glibc) differ on **0 of 1.4M** arguments, including j·s·ln x values.
- Pinning (`k6_pin.py`, `k7_pinned_lanes.py`): `NPY_DISABLE_CPU_FEATURES=X86_V4` takes log mismatches from 4,933 to **0 of 1M**. Full sklearn lanes vs confit: **0 of 23.4M** (steps 2, 3, 4), against 2,796 / 5,157 / 7,597 lanes differing unpinned.
- The legacy names `AVX512F` and `AVX512_SKX` are **silently ignored** in numpy 2.5; only `X86_V4` works.
- So the twin's bits depend on CPU and build. On any non-AVX-512 x86 host, or a non-SVML build, numpy's log is libm's and this entry is bit-exact.
- glibc is CPU-dispatched too (`k8_glibc_variant.py`, `GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-FMA4`). Its FMA and SSE2 ifunc variants differ on log 1 of 1.5M, cos 342 of 500k and sin 382 of 500k. On one machine, numpy, DuckDB and confit share the same variant, so they agree: 0 lanes differed where the logs agreed, in every run of `k4`.
- numpy's float64 `cosh` is SVML on X86_V4 (`loops_umath_fp`, X86_V4 only). The baked constant c equals the twin's only on a host with the same dispatch.

## 3. Error analysis (DERIVED), checked against measurements
Both sides compute f, the products, cos and sin identically (same IEEE operations, same libm). The only difference is L' = numpy log vs L = glibc log, with |L'−L| ≤ d·ulp(L) ≤ d·eps·|L|.

- **d = 1:** the measured errors are below 0.62 and 0.52 ulp. Their sum is below 2 ulp, so the two doubles are at most 1 apart. Measured maximum distance: 1.
- **Rounding of s·L:** with t = fl(sL), |t'−t| ≤ s·ulp(L) + ½ulp(sL) + ½ulp(sL') < 2·eps·s|L|·(1+O(eps)).
- **Rounding of j·t:** a = fl(j t) adds ≤ eps·j|t| unless j is a power of two (then the product is exact).
- **Argument amplification:** define A := |a'−a| / (eps·j·s|L|). Then **A < 2 for j ∈ {1,2}, A < 3 for any j.**
- **Tight value of A, same-binade case:** with g = ulp(sL), t'−t ≤ g·⌈s·ulp(L)/g⌉. This gives A_sup = 2/σ(s) for σ ∈ (1,2), reached with probability σ−1 per mismatch, and A = 1 for s a power of two. Examples: s = 0.4 and 0.8 give 1.25; s = 0.26 gives 1.923. j = 3 multiplies by up to 4/3 (s = 0.4 gives 5/3).
- **Lane difference:**
  - |y'−y| ≤ f·|sin ξ|·|a'−a| + f·e_C·(ulp c + ulp c') + ½ulp(fc) + ½ulp(fc'), with e_C ≈ 0.55 for glibc cos/sin.
  - In units of eps·f this is ≤ d·A·X·|sin ξ| + κ|cos a|, with κ = 2e_C + 1 ≈ 2.1.
- **Scale and K:** K = |Δ| / (eps·f·(1+X)) ≤ (dA·X + κ)/(1+X) ≤ **max(dA, κ)**. It is a convex combination, so K tends to dA as X grows.
  - S = f(1+|a|) majorizes both first-order terms: the absolute condition number f|a sin a| with respect to a relative error in L, and the result's own rounding f|cos a|.
  - The relative condition number, |a tan a|, is unbounded at zeros of cos. That is why "ulps of the result" grows without limit (ADV draws: up to 8.8e11 ulps).

Synthetic check, any kernel 1 ulp off (`k5_synthetic.py`): L' = nextafter(L, ±∞) on 1.5M draws, using glibc cos and sin.

| (s, j) | A derived | A meas. | sup K meas. | K for X<0.1 | K for X≥100 |
|---|---|---|---|---|---|
| 0.5, 1 (steps=2 default) | 1 | 1.0000 | 0.996 | 0.80 | 0.996 |
| 0.4, 1 / 0.4, 2 (steps=3 default) | 1.25 | 1.2500 | 1.246 | 0.79 | 1.246 |
| 0.8, 1 | 1.25 | 1.2500 | 1.246 | 0.98 | 1.246 |
| 0.4, 3 | 1.667 | 1.6667 | 1.660 | 0.65 | 1.656 |
| 0.26, 1 / 0.26, 3 | 1.923 / – | 1.923 / 2.279 | 1.903 / 2.273 | ≤0.88 | 1.90 / 2.27 |
| 60 random s ∈ (0.05, 2), j ≤ 2 | < 2 | max 1.977 | – | – | – |

The constant term is far below its rigorous κ: at small X, measured K ≤ 0.98.

Real kernels (`k4_K.py`): twin = sklearn; native = confit serving the SQL spelling, equal to DuckDB on every lane. ENR = candidate draws kept only where numpy's log ≠ glibc's.

| steps, s | draws: R1 U(0,1e3) / R2b random-mantissa log-uniform / R3 near 1 / ADV near zeros / ENR | max ulps | max K | lanes K>1 |
|---|---|---|---|---|
| 2, 0.5 (default) | 200k / 200k / 200k / 236k / 268 | 5.3e10 | **0.984** | 0 |
| 3, 0.4 (default) | … / 545k / 1,937 | 8.1e10 | **1.241** | 83 of 7,748 ENR lanes |
| 2, 0.8 | … / 291k / 1,080 | 2.5e11 | 1.211 (ADV 1.077) | 29 ENR; 2 of 582k ADV |
| 3, 0.5 | … / 509k / 779 | 2.4e11 | 0.986 | 0 |
| 4, 0.4 | … / 873k / 3,396 | 8.8e11 | 1.633 (R1 1.031) | 288 ENR; 2 of 1.2M R1 |
| 4, 0.26 | … / 727k / 1,225 | 4.7e11 | 2.209 | 299 |

The sqrt lane is 0 ulps everywhere.

**Reproduction of the record's numbers:**
- R3: 2–4 ulps, K ≤ 0.005 (record: 4 ulps, 0.01).
- exp(U) row: 1–5 ulps (record: 2).
- R1: 817–7,482 ulps (record: 339 for its draws; T11 reported 8,192).

**So:**
- K < 1 holds only for power-of-two s (A = 1, and sup K → 1⁻ with about 0.4% margin).
- The record's "max K 0.91" reflects its draws: |ln x| ≤ 6.9 caps X/(1+X), and the log-uniform row exercised no mismatches at large |L|.
- d enters linearly. If a future numpy build were 2 ulps from glibc, K would double.

## 4. A bit-exact spelling of numpy's log: feasibility
Every step of `__svml_log8_ha` is deterministic IEEE arithmetic:
- `vgetexp` can be replaced by a comparison tree plus exact scaling.
- `vrcp14`+`vrndscale` is a 17-valued step function of m. Its 16 breakpoints can be found on hardware by bisection, since it differs from RN(1/m) on 2.9e-4 of m.
- R = Rcp·m − 1 is exact without FMA via a 47/6-bit split.
- The table is 16 hi/lo constants.
- About **9 FMAs are inexact**. Each needs correctly rounded emulation: Dekker TwoProduct plus a round-to-odd sum (Boldo & Melquiond, IEEE TC 2008; cited from memory). DuckDB and confit have `nextafter`, which makes round-to-odd expressible. That is roughly 30–40 operations per FMA, about 450 SQL nodes per feature.

Blockers:
1. Each emulated FMA reads its inputs 4–8 times, over a chain depth of about 6. Spelled as text, that is 4^6–8^6 copies, past the 4M-token cap. This needs PLANS "Needs from confit" #3, a value bound once.
2. The spelling is valid only where numpy dispatches X86_V4 with SVML linked, and only for vendored SVML commit 3a713b1. Elsewhere numpy's log is libm's. The entry would have to choose by probe.
3. A correctly rounded log does not help: SVML itself is not correctly rounded.

**Verdict: not feasible now; feasible but poor value later.** The probe-gated bound in §5 gets bit-exactness wherever it is cheap.

## 5. Bearing on the options
- **Option 1 with K = 1 fails** with the real kernels at a sklearn default (steps=3). Expect flaky gates.
- **Option 1 with K = 3** is a derived bound: d = 1, A < 3, κ ≤ 2.1. It does not drift with the number of draws. Its margin is 2.4× over the defaults' measured sup and 1.3× over the worst exotic configuration tried.
- **Option 2** is confirmed unbounded: up to 8.8e11 ulps.
- **Option 3's premise** ("numpy picks its kernel by CPU") is true, but it cuts both ways. The twin itself is not portable, and on non-SVML hosts the entry is exactly equal.
- **Independent of the ruling:** `sample_steps=1` (sqrt lane only) and every lane 0 are bit-exact.

**Unconfirmed:**
- Intel's documented HA bound (intel.com blocked).
- That `vrcp14` is identical across microarchitectures (only this CPU was tested).
- Hosts other than this x86-64 AVX-512 machine were not run.


## The researcher's recommendation

Take Option 1, the term-scale bound with S = factor·(1+|j·s·ln x|), but declare K = 3 rather than K = 1. K = 1 already fails with the real kernels at sklearn's sample_steps=3 default: rounding s·ln x amplifies a 1-ulp difference in log by up to 2/σ(s) = 1.25 (measured max K 1.241). Explicit sample_interval values push K toward 2–2.3. K = 3 comes from the derivation, K ≤ max(d·A, κ) with d = 1, A < 3 and κ ≈ 2.1, so it does not grow with more draws the way a seed maximum does. Gate the bound on the existing `kernel_distance(np.log)` probe. Where numpy's log is glibc's (non-AVX-512 hosts, non-SVML builds, or NPY_DISABLE_CPU_FEATURES=X86_V4), declare 0: there the entry matched on all 23.4M lanes. Ship sample_steps=1 now, since it is sqrt-only and bit-exact. Drop the 'spell numpy's log' path: it is SVML log8_ha, needing about 9 emulated FMAs and a let-binding confit does not have, and it is pinned to one vendored SVML commit. In the record, fix the off-by-one default (s, j) list, and replace the exp(uniform) row with random-mantissa draws.


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| numpy 2.5.1's float64 log on this host is SVML __svml_log8_ha, not the Tang AVX512F kernel; Tang is only in the #else branch; X86_V3 is scalar npy_log; float64 sin/cos are libm. | **CONFIRMED** | I fetched v2.5.1 from raw.githubusercontent.com myself. loops_exponent_log.dispatch.c.src:666 and :1316 have `#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)`, :691 calls `__svml_@func@8_ha`, and AVX512F_log_DOUBLE (:921) sits in the #else (:703-1102). core meson.build:91-101 has use_svml = linux && x86_64 && X86_V4; :1020-1024 dispatches [X86_V4, X86_V3]. The X86_V4 group includes AVX512_SKX (meson_cpu/x86/meson.build). The trig dispatch file has DISPATCH_DOUBLE_FUNC(sin/cos) calling npy_sin/npy_cos (libm). On X86_V3, SIMD_AVX2_FMA3 is used only by FLOAT_exp/log. `nm` shows `T __svml_log8_ha` in the installed .so. show_runtime finds X86_V4/AVX512_ICL/AVX512_SPR. numpy cos/sin vs DuckDB: 0 of 3M differ, including arguments up to 1e308. | Detail: the SVML branch is taken only for non-overlapping, loadable-stride arrays. Otherwise the same binary falls through to npy_log (glibc). sklearn's fresh X_nz always takes SVML; I measured 0 differences between vectorized, 1-element, scalar and strided calls. |
| numpy log and glibc log are at most 1 ulp apart; differ on 1.85e-4 of U(0,1e3) and 1.6-2.7% near 1; max err numpy 0.620, glibc 0.517 ulp; cos/sin 0 of 1.4M differ. | **CONFIRMED** | Own script v_acc.py (MPFR 256-bit via gmpy2; glibc via DuckDB ln, which equals math.log on 300k). Results: U(0,1e3) 1.43e-4 (57/400k), maxdist 1. U(0.9,1.1) 1.99e-2. 1+N(0,1e-3) 1.63e-2. U(0.94,1.07) 2.58e-2. Max error numpy 0.6177 ulp, glibc 0.5168. 2^20 points 1±k·2^-52 had 0 differences. The rate differences are within Poisson noise. A secondary source supports d<=1: a comment in numpy PR #19478 says SVML 'high (maximum errors up to 1 ulp)'. With glibc's <=0.52, the two results differ by less than 2 ulps, so they are at most 1 double apart. |  |
| K=1 is breached by the real kernels at sample_steps=3 (s=0.4): max K 1.241; s=0.8 1.211; s=0.5 stays <1 (0.984); explicit configs reach 1.633 (s=0.4,j=3) and 2.209 (s=0.26). | **CONFIRMED** | Independent harness v_k.py: twin is sklearn fit_transform; native is a DuckDB SQL spelling with s, c and j passed as DOUBLE columns, and confit equals DuckDB on 1.5M lanes. Mismatch-enriched draws: steps=3,s=0.4 max K 1.163 (3 lanes >1). A targeted run at \|ln x\| in [32,40)∪[64,80)∪[128,160) gave 1.2219, with 50-84 of 549 mismatched lanes per lane-type above 1. s=0.8 1.163. s=0.5 0.897. s=0.4 j=3 1.530. s=0.26 j=3 1.454. s=0.3 j=1 1.579. Unconditioned U(0,1e3) at steps=4 s=0.4 j=3: 1.143. The exact maxima 1.241/2.209 were not hit, but they agree with the synthetic sup (1.246, 2.27). |  |
| K <= max(d*A, kappa), d=1, kappa=2e_C+1~2.1; A<2 for j in {1,2}, A<3 any j; A_sup=2/sigma(s); synthetic A=1,1.25,1.667,1.923 and sup K 0.996,1.246,1.660,1.903. | **PARTLY** | Line-by-line check: \|t'-t\| <= s·ulp(L)+u·s\|L\|+u·s\|L'\| < (d+1)eps·s\|L\|, and \|a'-a\| < (d+2)eps·js\|L\|, so A<3 for d=1. A<2 when j is a power of two (exact product). The same-binade case gives ceil(sigma_s)=2 when sigma_s·sigma_L<2, so A_sup=2/sigma(s). kappa=2e_C+1 follows from the ulp(c)<=eps\|c\| and ½ulp(fc) terms. My synthetic run (v_synth.py, 2M draws) reproduces 1.0000/0.996, 1.25/1.246, 1.6667/1.660 and 2.279/2.271. | Three corrections. (1) The sup of A over legal configurations comes close to 3: s=0.2501,j=3 (sample_steps=4) gives A 2.636, K 2.571; j=9 gives 2.665; j=17 gives A 2.822, K 2.798. So K=3 holds with about 1.07-1.17× margin in those cases, not the '1.3× over worst tried'. (2) e_C=0.55 comes from s_sin.c's '~0.55 ULP' comment, but libm-test-ulps lists cos/sin double at 1, so kappa can be up to 3. K<3 still follows. (3) 'd enters linearly / K would double' is loose: the rigorous bound is A<d+2, i.e. 4 at d=2. The bound also assumes f, cos and sin are identical on both sides; the cosh constant is not automatically identical (see missed items). |
| NPY_DISABLE_CPU_FEATURES=X86_V4 makes numpy log equal glibc's and the twin equal native on all lanes; unpinned lanes differ; numpy 2.5 silently ignores AVX512F/AVX512_SKX. | **PARTLY** | Pinned: 0 of 14M log candidates differ from DuckDB ln, and 0 differing lanes across 16 (steps,s) configs × ~2M rows each. np.cosh is also libm when pinned. Unpinned: 3,907 of 200k near-1 log draws differ. Setting AVX512F or AVX512_SKX leaves dispatch unchanged (3,907 mismatches persist). | Not silent. numpy emits an ImportWarning ('You cannot disable CPU features (AVX512F), since they are not part of the dispatched optimizations (X86_V3 X86_V4 AVX512_ICL AVX512_SPR)'). Python's default filters hide it, and -W always shows it. |
| The record's exp(uniform) log-uniform row is biased: for \|ln x\|>=2 kernels never disagree (0 of 10M); random mantissas disagree at 2.9e-4 down to 1.6e-6. | **PARTLY** | Mechanism confirmed. x=np.exp(t) with \|t\| in [2,709) gives 0 mismatches in 15M draws, and glibc ln(exp t)==t for 100% of them. With the low 32 mantissa bits randomized: [2,4) 2.5e-4, [4,16) 8.0e-5, [16,64) 1.8e-5, [64,256) 2.3e-6, [256,709) 6.7e-7. | The record does not document how its row was built, so calling it 'the record's construction' is an inference, though it fits the record's tiny K=0.02. The derivation's \|delta\|<=u assumes a correctly rounded exp, while np.exp here is SVML; this is harmless at measured error. The report missed that the repo's own kernel_distance probe (function.py:214) and the _BOUNDS measurements use the same ±exp(uniform(-700,700)) construction. That quarter of the probe contributed 0 mismatches at the fixed seed. |
| Record's default (s,j) list is off by one; X>=0 validated, NaN/inf raise, -0.0 -> +0.0, s=0 legal. | **CONFIRMED** | kernel_approximation.py:666 (Interval [0,inf)), :722 (ensure_non_negative), :732/:736 (0.8/0.4), :795 range(1, sample_steps); validation.py:1763 check_non_negative. Ran: -0.0 gives all +0.0 lanes (signbit False). s=0 with x=0.5 gives [0,0,-0.0]. -1, NaN and inf raise ValueError. x=1e308 gives [7.07e153,-inf,inf], and DuckDB returns inf without raising. | 'x=5e-324: factor underflows to 0' depends on s. At s=0.4, j=1 the factor does not underflow (cos1 lane = -1.73e-162), while lane 0 and the j=2 lanes are 0. |
| Bit-exact SQL spelling of SVML log8_ha is not feasible: 11 FMAs (~9 inexact), 17-valued vrcp14+roundscale step (differs from RN(1/m) on 2.9e-4), needs let-binding confit lacks. | **PARTLY** | SVML commit 3a713b1 svml_z0_log_d_ha.s, fetched myself and identical to the local copy. The main path has 10 vfmadd + 1 vfmsub (R=Rcp·m-1, exact), so 11 FMA-class ops. Own AVX-512 C test (2^26 m): 17 distinct values; 2.861e-4 differ from RN on the 1/32 grid. PLANS.md 'Needs from confit' item 3 (a value bound once) exists as described. | `grep -c vfmadd` returns 10, not 11 (the 11th is a vfmsub). The ~450-node and text-blowup figures are estimates. The Boldo-Melquiond citation (IEEE TC 2008) is right but was cited from memory. vrcp14 equality across vendors (e.g. AMD Zen4) is untested, which would also make the twin vendor-dependent. |
| glibc log/cos/sin are ifunc-dispatched: FMA vs SSE2 differ on log 1/1.5M, cos 342/500k, sin 382/500k; within one machine numpy/DuckDB/confit agree. | **PARTLY** | glibc-2.39 multiarch/e_log.c is an ifunc (ifunc-avx-fma4.h), and dbl-64/e_log.c:128 reads '0.519 ULP (0.520 ULP without fma)' (verified from the bminor/glibc mirror). My own ctypes-libm dump with GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA,-FMA4 gave log 292/2M (278 in U(0.9,1.1), 14 in U(0,1e3)), cos 664/1M, sin 704/1M. Same-machine agreement holds: the pinned run had 0 differing lanes, and confit equals DuckDB on 1.5M×2 lanes. | The log rate is draw-dependent and understated. Their '1 of 1.5M' reproduces only with their draws; near 1 the variants differ at about 2.8e-4. |
| (implicit in TL;DR/§2/§5 and the recommendation) On a non-SVML build numpy's log is libm's, so the entry is bit-exact there. | **REFUTED** | The same source the report cites (lines 22-40 and :1331) defines SIMD_AVX512F when !_MSC_VER && NPY_HAVE_AVX512F. So on an AVX-512 host with a GCC/clang build that does not link SVML (macOS x86-64 on AVX-512 Macs, Linux builds with -Ddisable-svml), DOUBLE_log runs numpy's own Tang AVX512F_log_DOUBLE kernel. That kernel defers to npy_log only on (1-2^-4, 1+0x1.09p-4). | Only non-AVX-512 hosts, MSVC builds, or X86_V4-disabled processes get libm. On Tang hosts the probe must decide, and d<=1 is not established for Tang. |

### What the report missed or got wrong

1. Non-SVML builds are not libm on AVX-512 hardware. The report says 'non-SVML build -> libm -> declare 0'. The cited source shows GCC/clang builds without SVML run the Tang AVX512F kernel; it matches glibc only near 1. The probe-gated design probably still catches this: I estimated ~10 probe mismatches outside the near-1 interval at SVML-like rates, with 0 of 301 seeds missing. But the Tang kernel's accuracy relative to glibc (d<=1) was never measured, so K=3 there is unproven.

2. K=3 is valid but tight for legal configurations. With the report's own error model, A's sup approaches 3:
   - s=0.2501, j=3 (sample_steps=4): A 2.636, K 2.571.
   - j=17: A 2.822, K 2.798.
   So the '1.3x margin over the worst exotic configuration' only reflects the configurations tried. Any error source outside the model (d=2 somewhere unprobed, a cosh constant from a different dispatch) would break K=3 for such configurations. The repo already supports per-configuration bounds (#388, `bound=` callable); the report did not consider a K(s,j) derived from the formula.

3. The cosh constant is a second kernel difference. np.cosh here is SVML (loops_umath_fp). It differs from libm cosh on 23.6% of U(0,5) arguments (max 1 ulp), including sklearn's steps=3 constants cosh(0.4π) and cosh(0.8π) (np != math.cosh). The 'fitted constant' must therefore be computed by np.cosh on the twin's host and process. If it came from math.cosh or DuckDB, the 'declare 0' branch would fail, and κ would grow by about 1 (K up to about 3.1 in the bound). The recommendation gates only on kernel_distance(np.log). The derivation also assumes identical cos/sin, so the gate should also require the existing sin/cos probes to read 0, and should pin the constant's provenance.

4. The repo's own probe and _BOUNDS measurements use the biased construction. kernel_distance (function.py:214) and the _BOUNDS comment both use ±exp(uniform(-700,700)), the construction the report faults in the record. That quarter of the probe found 0 mismatches. Probe power on SVML hosts is still fine: 23 mismatches at the fixed seed, all from the N(0,1) and U(0,50) draws.

5. Smaller errors:
   - The legacy feature names are not 'silently ignored'; they raise an ImportWarning that the default filters hide.
   - The glibc FMA/SSE2 log difference rate is understated (2.8e-4 near 1, not 1 in 1.5M).
   - `grep -c vfmadd` gives 10; the 11th FMA is a vfmsub.
   - The 5e-324 factor underflow depends on s.
   - 'd enters linearly, K doubles' is loose (rigorous A < d+2).
   - The SVML path also needs non-overlapping, loadable-stride arrays; otherwise the same numpy falls back to glibc's npy_log.

6. Practical frequency was not quantified. K>1 needs a log mismatch with |j·s·ln x| above about 4. On U(0,1e3) at steps=3 I saw 0 lanes above 1 in 1M draws; the catalog fixtures (normal × 10^±8) can reach about 1.18. So K=1 would fail rarely but eventually, which supports 'flaky' rather than routine failure.


### Does the recommendation follow? Yes

The core recommendation follows from the evidence, which I reproduced independently:
- K=1 fails at a sklearn-supported default (sample_steps=3, s=0.4). My independent DuckDB-native harness reached K=1.22.
- K=3 is a derived bound, not a seed maximum. It holds given SVML HA <=1 ulp (now supported by a numpy PR #19478 comment), glibc ~0.52 ulp, e_C<=1, and identical factor, cos and sin on both sides.
- sample_steps=1 (sqrt only) is bit-exact.
- Pinning X86_V4 makes everything bit-exact (0 differing lanes in my run).
- Dropping the SVML-in-SQL path is reasonable: 11 FMA-class ops, a 17-step rcp table that depends on the CPU vendor's vrcp14, and no let-binding in confit.

Caveats that should change the details:
(a) Say plainly that K=3 is tight. Legal configurations reach K≈2.57 (steps=4, s≈0.25+) and ≈2.8 (large j). Either restrict which (s, j) the entry serves, or declare a per-configuration K from the formula, which the codebase already supports.
(b) Gate on more than the log probe. Also require the sin/cos probes to read 0, and require the cosh constant to be computed by np.cosh on the twin's host; SVML cosh differs from libm at exactly the steps=3 default constants.
(c) Drop 'non-SVML builds -> declare 0'. AVX-512 hosts with GCC/clang non-SVML builds run numpy's Tang kernel, not libm, so only the probe result should decide 0 vs 3. d<=1 is unverified for that kernel.
(d) When fixing the record's exp(uniform) row, also fix the identical construction in kernel_distance and the _BOUNDS measurements.

With those amendments, Option 1 at K=3 (0 where the probes read 0), plus shipping sample_steps=1 now, is a sound ruling.
