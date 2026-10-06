# Distances to centres and kernel samplers

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


**Key `distances`** · sklearn 1.9.0, numpy 2.5.1, scipy 1.18.0, OpenBLAS 0.3.33 (SkylakeX), glibc 2.39, x86-64 with AVX-512 · 2026-10-06

All scripts and outputs are in `R=scripts/distances/`. Every script runs as `cd $R && OPENBLAS_NUM_THREADS=1 .venv/bin/python expN_*.py`, which is the repo's venv interpreter.

**Twin and native in these experiments:**
- **Twin:** `est.transform([row])[0]`, a single-row call exactly as `PythonTransform.__call__` makes it (`_udf.py:313-314`).
- **Native-like:** a model of the SQL entry in Python floats. Dots run left to right with no fused multiply-add. `ln`/`exp`/`cos` are glibc's through `math`. Constants are computed as sklearn computes them.
- **Copied helper:** `import sql_transform` raises a pydantic `TypeError` (`prefer_fwd_module`) under this Python 3.14rc2. So `_helpers.row_sumsq` is copied verbatim into `$R/rowsumsq.py`, and its probe `row_sumsq_is_numpys()` returns True.

Notation: u = 2⁻⁵³, eps = 2⁻⁵², γₙ = nu/(1−nu).

---

## 1. KMeans, MiniBatchKMeans, BisectingKMeans, Birch

### 1.1 What the twin computes (SOURCED)

**Which function each `transform` calls:**
- `_BaseKMeans.transform` → `_transform` → `euclidean_distances(X, self.cluster_centers_)` (`sklearn/cluster/_kmeans.py:1133-1157`). KMeans, MiniBatchKMeans and BisectingKMeans all inherit it.
- `Birch.transform` → `euclidean_distances(X, self.subcluster_centers_)` under `assume_finite=True` (`_birch.py:667-687`). Birch has one lane per subcluster (230 in my run).

**`_euclidean_distances` for float64** (`sklearn/metrics/pairwise.py`):
- `XX = row_norms(X, squared=True)`, which is `np.einsum("ij,ij->i")` (`:389`, `utils/extmath.py:89`). `YY = row_norms(C)` (`:399`).
- `distances = -2 * safe_sparse_dot(X, C.T)`, then `distances += XX`, then `distances += YY` (`:409-411`). The order is **q = ((−2·D) + ‖x‖²) + ‖c‖²**.
- **Clipped:** `np.maximum(distances, 0)` (`:413-415`).
- Then `np.sqrt` (`:426`), and the result is not squared.
- The docstring itself warns of "catastrophic cancellation" (`:284-285`).

**Float32 and chunking:** the upcast/chunked path runs only if X or C is float32 (`:403-406`). `PythonTransform` always hands float64, and `_return_float_dtype` (`:51-71`) then upcasts float32 centres exactly. Measured in `exp7`: a float32-fitted KMeans gives a float64 twin equal to the float64-upcast path.

**The dot is BLAS gemv.**
- `safe_sparse_dot` runs `a @ b` (`extmath.py:229`). numpy's matmul turns a one-row `x @ C.T` into `gemv` (`special_case`/`vector_matrix`, numpy `matmul.c.src:464,479,545-547`, https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/matmul.c.src).
- On SkylakeX, OpenBLAS `dgemv_t` uses the Haswell FMA microkernel (`kernel/x86_64/dgemv_t_4.c:31-32`; `dgemv_t_microk_haswell-4.c:49-52` `vfmadd231pd`).
- `dgemv_n` (RBFSampler's `x @ W`) has a separate SkylakeX microkernel (`dgemv_n_4.c:34-37`).
- Source: https://github.com/OpenMathLib/OpenBLAS/tree/develop/kernel/x86_64

**Consequence for the entry:**
- ‖x‖² can be bit-exact through `row_sumsq`: it matched on every one of 9,345 rows in `exp1`.
- ‖c‖² is a numpy constant.
- My numpy re-implementation `sqrt(maximum((-2*(x@C.T)+XX)+CC, 0))` equals the twin bit for bit on every row, for all four estimators.
- **So the left-to-right dot is the only operation that differs.** The entry should spell `sqrt(greatest(((-2.0*dot_k) + XX) + CC_k, 0.0))`.

### 1.2 Condition (DERIVED)

Let S2ₖ = ‖x‖² + ‖cₖ‖² + 2Σᵢ|xᵢcₖᵢ|. Note that q ≤ S2 and every intermediate is ≤ S2.

**Squared lane:**
- Any summation order, fused or not, gives |D̂ − x·c| ≤ γₙ Σ|xᵢcᵢ| (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1, eq. 3.5).
- Hence 2|D̂ₙ − D̂ₜ| ≤ 2γₙ S2 ≈ n·eps·S2.
- The two additions round at most 2u·S2 per side.
- `max(·,0)` is 1-Lipschitz.
- `sqrt` followed by squaring adds at most eps·q per side.
- Result: **|dₙ² − dₜ²| ≤ (n+4)·eps·S2ₖ + O(eps²).**

**The lane itself:**
- dₙ − dₜ = (dₙ² − dₜ²)/(dₙ + dₜ).
- So |Δd| ≤ (n+4)·eps·S2/(dₙ + dₜ), and also ≤ √((n+4)·eps·S2).
- Near a centre the error is ~√(eps·S2): half the digits, any number of ulps, and ~1/√eps ≈ 6.7e7 in units of eps·√S2.
- Away from centres, ulps ≈ (n+4)·S2/(2d²). For uncentred data (‖x‖ ≫ cluster radius) S2/d² is large **for every row**, not just rows near centres.

**What S must include:** S must contain ‖x‖² + ‖c‖², not only the dot's term scale. Measured in `exp7`: with centres near the origin, S = 2Σ|xc| gives K = 2,450 (n = 4), while S2 gives K = 1.05.

**Clipping:**
- The twin answers 0.0 wherever its rounded q < 0, while the entry may answer ~1e-8, and the reverse also happens.
- Both values are within K·eps·S2 of the true d² (the twin's own error against exact is shown below).
- Comparing squares absorbs this automatically, because max(·,0) is a contraction.

### 1.3 Measured (`exp1_kmeans.py`)

`exp1` covers 136,725 lanes:
- n ∈ {2, 8, 32}
- data centred, offset by 1e3, or with feature scales 10^(j mod 5)
- rows far, near (δ = 1e-2 … 1e-11 relative), or exactly at a centre
- plus MiniBatchKMeans, BisectingKMeans and Birch

K2 = |dₙ² − dₜ²|/(eps·S2), computed exactly with `Fraction`. K1 = |Δd|/(eps·√S2).

| rows | max ulps | K2 max | K1 max | twin=0 while native>0 / the reverse |
|---|---|---|---|---|
| far, centred | 1.6e3 | 1.94 | 16 | 0 / 0 |
| far, offset 1e3 | **1.4e8** | 0.53 | 3.1e3 | 0 / 0 |
| near δ=1e-5 | 3.6e15 | 1.86 | 6.4e5 | 2 / 6 |
| near δ≤1e-8 | 4.6e18 | 2.40 | **6.4e7** | 9–37 / 3–33 per 750 lanes |
| x = centre | 4.6e18 | 1.05 | 4.6e7 | — |
| **pooled** | 4.6e18 | **2.40** (p99.9 1.38) | 6.4e7 | — |

**Reference comparisons:**
- **Twin against exact** (`Fraction` d², Decimal sqrt): K2 ≤ 1.85 and up to 4.5e18 ulps. The twin is no closer to the truth than the entry.
- **The accurate formula** √Σ(xᵢ−cᵢ)² as the entry: K2 ≤ 3.93 against the twin. The bound admits it, but the expansion tracks the twin more tightly.

**K2 against n** (`exp5_scaling.py`):

| n | 2 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| K2 max | 0.57 | 1.01 | 1.60 | 2.34 | 4.36 |

This grows roughly like 0.2·√n, the probabilistic rate (Higham & Mary 2019, SIAM J. Sci. Comput. 41(5):A2815, doi:10.1137/18M1226312).

### 1.4 The twin is not one function across CPUs (`exp2_coretype.py`)

Same fitted state and rows, with `OPENBLAS_CORETYPE` set per subprocess. Each cell is the share of lanes that differ from SkylakeX, then the largest difference.

| SkylakeX vs | KMeans lanes differing, max | RBFSampler lanes differing, max |
|---|---|---|
| Haswell / Zen | 0 differ | 26–49 %, 1.5e5 ulps, K ≤ 1.21 |
| Sandybridge / Prescott (generic) | 27–51 %, **4.5e18 ulps**, K2 ≤ 1.27 | 32–72 %, 2.4e6 ulps, K ≤ 2.35 |
| Cooperlake | 0 | 0 |

A ulp bound is not even defined between two machines both running the twin. The eps·S bound holds between them with K ≤ 2.35.

---

## 2. Kernel samplers

### 2.1 Formulas (SOURCED, `sklearn/kernel_approximation.py`)

**RBFSampler (`:412-415`):**
- θ = x @ W (gemv), then += b, then `np.cos`, then ×(2/D)^½.
- W ~ √(2γ)·N(0,1), b ~ U(0, 2π).

**SkewedChi2Sampler (`:571-576`):**
- x += s, then `np.log`, then @ W, then += b, then `np.cos`, then ×√2/√D.
- numpy's float64 `log`/`exp` on this CPU are its own AVX512F kernels (`AVX512F_exp_DOUBLE`/`AVX512F_log_DOUBLE`, `loops_exponent_log.dispatch.c.src:720,921`, https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/loops_exponent_log.dispatch.c.src).

**Nystroem (`:1098-1108`):** `pairwise_kernels(x, components_)`, then `@ normalization_.T`.
- With the default `kernel="rbf"`, the kernel is exp(−γ·q) with q from the **same clipped expansion** as §1 (`pairwise.py:1618-1621`).
- γ = 1/n when no gamma is set.

**PolynomialCountSketch (`:202-243`): not a matvec.**
- `X_gamma = sqrt(γ)·x` (plus a √coef0 column).
- Count sketches aₐ[h] += ±x_j in feature order. This is exactly reproducible.
- Then scipy `fft` (ducc0 backend: `scipy/fft/_basic_backend.py:77-86`, `_duccfft/basic.py:11-36`), `np.prod` over degree, `ifft`, and the real part.
- Mathematically y = a₁ ⊛ … ⊛ a_p, a circular convolution.
- An entry can follow the count sketch exactly, then convolve directly (D² products per degree step). Spelling ducc0's FFT to the bit is unverified: it uses SIMD across transforms and the compiler may contract to FMA.
- loops/native/PLANS.md lists it among "samplers that project through a matrix". That is inaccurate.

### 2.2 Scales (DERIVED)

**RBF lane j.** Let T = Σ|xᵢWᵢⱼ| + |bⱼ|.
- |Δθ| ≤ 2γₙ₊₁·T.
- cos is 1-Lipschitz and the same function on both sides (numpy `cos` = glibc: **0 mismatches in 3.6M calls**).
- cos and ×c roundings add about eps·c.
- **S = c·(T + 1), K_worst ≈ n + 2.**
- A scale of c·(|θ|+1) is not enough: measured K up to 302 when the dot cancels.

**SkewedChi2:** the same with xᵢ replaced by Lᵢ = ln(xᵢ+s).
- numpy's log against glibc: ≤ 1 ulp apart and ≤ 1 ulp from correctly rounded. They differ on 19 of 200k draws (`exp6`) and on 600 of 504k calls (`exp3`).
- That adds ≤ eps·T, so **K_worst ≈ n + 3.**

**Nystroem lane l.**
- Δkⱼ/kⱼ ≤ γ·|Δqⱼ| + O(eps). numpy's exp differs from glibc on 4.6% of draws, by ≤ 1 ulp.
- The matvec over m = n_components adds 2γₘΣ|N kⱼ|.
- **S_l = Σⱼ |N_lj|·kⱼ·(1 + γ·S2ⱼ), K_worst ≈ max(n+4, m+2).**

**PolynomialCountSketch.**
- FFT error is normwise, of order log₂D·u·‖·‖₂ (Higham §24.1, Thm 24.2; I recalled this statement and did not re-read it).
- So no per-lane term scale works.
- Use the row-wide **S = (1/D)·Σₘ Πₐ |FFT(aₐ)ₘ|**. It is the term scale of the final inverse DFT, and S ≥ |y_l| for every l.
- I did not derive K in closed form; I expect O(p·log₂D).

### 2.3 Measured (`exp3_samplers*.py`, `exp4_nystroem_pcs.py`)

| family | lanes | max ulps | K max (scale above) | failing alternative scale |
|---|---|---|---|---|
| RBFSampler (n 4–64, γ 0.1/1/scale, centred or offset 100) | 1.8M | 1.1e8 | **2.38** | c(\|θ\|+1): 302 |
| SkewedChi2Sampler (s 1 or 0.01; uniform, lognormal, Poisson) | 1.8M (+216k) | 9.5e6 | **2.12** (2.51 in the smaller run) | 27.6 |
| Nystroem rbf (n 4–64, m 50–100) | 140,000 | 9.0e7 | **0.83** | plain matvec Σ\|N k\|: 216 |
| PolynomialCountSketch (D 64/97/100/128, degree 2 or 3, coef0 0 or 1) | 101,400 | 4.4e18 | **1.32** | per-lane (\|a₁\|⊛\|a₂\|)_l: **∞** |

**PolynomialCountSketch detail.** At n = 4, D = 100, 16,800 lanes are structurally zero.
- On 16,200 of them the twin answers FFT noise around 1e-17 where the exact value (and a direct convolution) is 0.0. This mirrors §1's clip.
- The twin against exact on those lanes has per-lane K = ∞, and row-wide K_fft ≤ 0.79.

**Dot product alone** (`exp8_dot_vs_n.py`): S = Σ|xc| (the record's PCA scale), all-positive terms.

| n | 8 | 32 | 128 | 512 | 2048 |
|---|---|---|---|---|---|
| K max (dgemv_t and dgemv_n) | 2.2 | **4.06** | 6.0 | 11.5 | 22.9 |

The record's "within 3 eps·S at n ≤ 32" is exceeded at n = 32 on biased data. A constant K does not survive wide inputs. I did not reproduce the record's PCA fixture numbers; they need the catalog fixtures.

---

## 3. Per-family declaration for one general ruling

General form: per lane, |g(native) − g(twin)| ≤ K·eps·S. Equal values, NaN and NULL compare as now.

| family | g | S (check computes it from fitted state + row) | K worst (derived) | K measured |
|---|---|---|---|---|
| KMeans, MiniBatchKMeans, BisectingKMeans, Birch | **y²** | ‖x‖²+‖cₖ‖²+2Σ\|xᵢcₖᵢ\| | n+4 | 2.40 (n≤32), 4.36 (n=512) |
| RBFSampler | id | c(Σ\|xᵢWᵢⱼ\|+\|bⱼ\|+1) | n+2 | 2.38 |
| SkewedChi2Sampler | id | c(Σ\|ln(xᵢ+s)Wᵢⱼ\|+\|bⱼ\|+1) | n+3 | 2.51 |
| Nystroem (rbf) | id | Σⱼ\|N_lj\|kⱼ(1+γS2ⱼ) | max(n+4, m+2) | 0.83 |
| PolynomialCountSketch | id | (1/D)Σₘ Πₐ\|FFT(aₐ)ₘ\|, the same for every lane of a row | O(p·log₂D), not derived | 1.32 |

Every S is computable by check from the fitted state and the input row, plus, for the distance family, the two outputs if written as |Δd| ≤ K·eps·S2/(dₙ+dₜ).

**What the ruling has to allow:**
- **(a) A comparison map g, or an S that reads the outputs.** Without it the distance family has no small K: measured K1 = 6.4e7 ≈ 1/√eps. The alternative |Δd| ≤ √(K·eps·S2) is about 1e8 looser on ordinary rows and would pass a float32 bug.
- **(b) S that is row-wide, not a per-lane term scale.** PolynomialCountSketch needs it.
- **(c) K as a function of the reduction lengths (n, m, D).** Measured K grows like √n.

**Why the derived worst-case K is safe to adopt:**
- It is proven for any summation order, with or without FMA. "The next seed exceeds it" cannot happen, apart from the underflow and overflow caveat below.
- It is still about 10⁶–10⁸ times tighter than a real defect: float32 rounding is ~1e-8 relative, and a dropped term is O(S).

**Not covered here:**
- **Underflow and overflow.** The γ bounds assume no underflow. S2 overflows for ‖x‖ ≳ 1e154. Check should add a tiny absolute floor and compute S with scaling or exactly.
- **Assumptions.** DuckDB/confit do not contract `a*b+c` into an FMA; the catalog's bit-exact entries already rely on this. numpy's `cos` equals glibc's on this host and was only probed here. The FFT constant was not derived.
- **Scope.** All of these are "not yet" and in scope (coverage.md). Nystroem sits in PLANS "Later" (it reads fit samples), and goal.md puts it in scope.



## The researcher's recommendation

One general ruling covers distances to centres and all the kernel samplers, but not in the record's form of "a dot product's term scale and one measured constant K". The ruling should read: per lane, |g(native) − g(twin)| ≤ K·eps·S. Each family declares three things: S, computable by check from the fitted state and the input row (the table in report section 3); g, the identity except for the KMeans/Birch family, which compares squares; and K as a function of its reduction lengths. g is needed because sqrt has infinite condition at 0 and the twin clips d² at 0: on the lane itself K reaches 6.4e7 ≈ 1/√eps, while on squares it is ≤ 2.4. K must vary with n because measured K grows like √n (dot: 4.1 at n = 32, 22.9 at n = 2048; distances: 4.4 at n = 512), so a constant fixed at n ≤ 32, including the record's PCA K = 3, will be breached by wide inputs. I recommend declaring the derived worst case (n+4 for distances, n+2 or n+3 for the RBF and SkewedChi2 samplers, max(n+4, m+2) for Nystroem) and recording the measured maximum (≤ 2.5 everywhere here) as evidence. The derived bound cannot be exceeded by any summation order, fused or not, and it still sits 10⁶ or more times below a real defect such as float32 rounding or a dropped term. PolynomialCountSketch also needs S to be row-wide, (1/D)·Σₘ Πₐ|FFT(aₐ)ₘ|, because its twin's FFT error is normwise; its K is measured (1.32) but not derived. PLANS should stop calling it a matrix projection. For the distance entry, spell the twin's expansion: ‖x‖² through row_sumsq (bit-exact) and ‖c‖² as numpy constants, so only the dot differs. A ulp bound is not even defined across CI machines: the twin alone differs by 4.5e18 ulps between OpenBLAS's SkylakeX and Sandybridge kernels, yet stays within K2 ≤ 1.27 in this form.


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| C1 (SOURCED): the KMeans/MiniBatchKMeans/BisectingKMeans/Birch twin is q = ((-2·(x@C.T)) + \|\|x\|\|²) + \|\|c\|\|², clipped by np.maximum, then sqrt. The dot is gemv with the FMA kernel. \|\|x\|\|² is bit-exact through row_sumsq, and a numpy re-implementation equals the twin, so the left-to-right dot is the only operation the entry cannot follow. | **CONFIRMED** | Installed sklearn 1.9.0: pairwise.py:389 (XX = row_norms), :399 (YY), :409-411 (-2*dot, += XX, += YY), :415 (xp.maximum), :426 (sqrt), :285 (the 'catastrophic cancellation' docstring). _kmeans.py:1133/1155-1157 and _birch.py:667/687 call euclidean_distances, and _bisect_k_means.py defines no transform of its own. extmath.py:89 uses einsum and :229 uses a @ b. numpy v2.5.1 matmul.c.src takes special_case/vector_matrix to @TYPE@_gemv at about lines 462/477/543; the report cited main, so line numbers drift by about 2. OpenBLAS develop dgemv_t_4.c:31-32 includes dgemv_t_microk_haswell-4.c for SKYLAKEX (vfmadd231pd at :49-52). My own run (v1_dist.py: different data, seeds and n up to 64, row_sumsq exec'd from the repo source itself) gave reimpl_mismatch_rows=0 and xx_mismatch_rows=0 over 4,104 rows. | 'Only the dot differs' holds on x86-64 baseline numpy. The repo's own row_sumsq docstring (_helpers.py:116-171) says an aarch64 build fuses the multiply-add, and the probe then returns False. The derived (n+4) bound does not need XX to be bit-exact: XX's error ≤ γn·\|\|x\|\|² sits inside γn·S2. So the ruling should not depend on row_sumsq. |
| C2 (MEASURED): comparing squares, K2 max is 2.40 over 136,725 lanes. On the lane itself K1 reaches 6.4e7 (about 1/sqrt(eps)) and ulps reach 4.6e18. The twin is within K2 ≤ 1.85 of the exact d². | **PARTLY** | The qualitative result reproduces independently (v1_dist.py: lognormal, uniform+offset, student-t and integer data; n = 3/16/64; MiniBatch, Bisecting and Birch with k=1101; 156,264 lanes). Pooled K2 max = 2.11 (p99.9 1.14), and max K2/(n+4) = 0.25. There are many 0-vs-nonzero clip flips. My figures do not match theirs. K1 max = 9.14e7, so 6.4e7 is not a ceiling: the ceiling is about sqrt(K2/eps) ≈ sqrt(K2)·6.7e7. Twin-vs-exact K2 max = 2.03, above the claimed 1.85 (n=64, lognormal and student-t). | Comparing squares with S2 holds with a K of about 2 at n ≤ 64, and the lane form fails at about 1/sqrt(eps)·sqrt(K2). The specific maxima (2.40, 6.4e7, 1.85) are sample maxima, not bounds. |
| C3 (DERIVED): \|d_n² − d_t²\| ≤ (n+4)·eps·S2. This comes from γn (any order, fused or not), two additions at 2u·S2 per side, max being 1-Lipschitz, and sqrt-then-square at eps·q per side. So the lane bound (n+4)·eps·S2/(d_n+d_t) is unbounded as d → 0. | **CONFIRMED** | I re-derived it line by line. Dot: 2·\|ΔD\| ≤ 2·2γn·Σ\|xc\| = 2γn·(2Σ\|xc\|) ≤ 2γn·S2 ≈ n·eps·S2. Additions: each result is ≤ S2(1+γn), so ≤ u·S2 each, 2u per side, 2·eps·S2 over both sides. Clip is a contraction. Square of fl(sqrt): (1+u)²−1 ≈ eps, giving eps·q per side and 2·eps·S2 in total. Sum: (n+4)·eps·S2 + O(eps²). The bound still holds if XX is not bit-exact. Measured K2/(n+4) ≤ 0.25 in my run. DuckDB 1.5.5 matches the unfused left-to-right model bit for bit (v0_duckdb.py: 4,000 rows of 24-term dots, sqrt(greatest(...)), ln, exp, cos: 0 mismatches, while an FMA chain would match only 2,701 of 4,000). I could not open the Higham PDF (egress blocked), so the equation number '3.5' is unverified. The γn·\|x\|ᵀ\|y\| any-order bound itself is standard. The Higham & Mary 2019 citation (SISC 41(5):A2815, doi 10.1137/18M1226312) was confirmed by search. | The ruling must say how check evaluates g. A naive float d_n*d_n − d_t*d_t adds up to about eps·S2 of its own error. Use (d_n−d_t)(d_n+d_t), exact arithmetic, or K+1. |
| C4 (MEASURED): with centres near the origin, S = 2Σ\|xc\| gives K = 2,450 (n=4), and S2 gives 1.05. | **CONFIRMED** | Their 2,450 comes from one differing lane out of 16,000; at n=16 and n=64 their exp7 had 0 differing lanes. My sweep (v2_ablation.py) over centre scale 0.1 and 0.01 at n = 4/16/64 gives K with 2Σ\|xc\| of 125-164 and 877-1,520 on 2-24 differing lanes per config. That growth goes like \|\|x\|\|/\|\|c\|\| and is unbounded. K2 with S2 stays ≤ 1.92. | The specific 2,450 is anecdotal. The point stands: S must contain \|\|x\|\|² + \|\|c\|\|², because a 1-ulp change of q ≈ \|\|x\|\|² cannot be bounded by the dot's own scale. |
| C5 (MEASURED): K is not constant in n. For a plain dot with S = Σ\|xc\|, K max is 2.2, 4.06, 6.0, 11.5, 22.9 at n = 8...2048, so the record's K=3 is exceeded at n=32. Distance K2 max is 0.57...4.36 (n = 2...512), growing roughly like √n. | **PARTLY** | v3_dot_n.py, fixed 200k lanes per n. Positive uniform data: 2.25, 3.73 (gemv_n 3.78), 7.5, 14.9, 23.3 (gemv_n 27.1), so K = 3 is breached at n=32. Mixed-sign N(0,1) data: 1.8, 2.7, 3.0, 2.4, 2.5 (gemv_n up to 2.9), with no growth to n=2048. The report's own exp5 RBF column (1.03 to 1.68, flat) shows the same. Distances (v10_dist_n.py): positive data 1.18/1.37/2.56/4.48 and centred data 1.96/1.81/1.83/3.71 at n = 8/32/128/512. Distances grow because near-centre rows and \|\|x\|\|² always give same-sign terms. | K grows (about n^0.4 over 8 to 2048) when the terms share a sign, which always holds for distances and for PCA on positive data. It does not grow for mixed-sign dots like RBF. That is enough to break a constant K, but 'measured K grows like √n' is not general. exp8 is a plain-dot proxy, not the record's PCA fixtures. |
| C6 (MEASURED): the twin is not bit-stable across OpenBLAS kernels. Sandybridge and Prescott differ from SkylakeX on 27-51% of KMeans lanes (up to 4.5e18 ulps, K2 ≤ 1.27); RBF differs by up to 2.4e6 ulps (K ≤ 2.35). Haswell and Zen give identical KMeans results but differ on RBF. | **CONFIRMED** | My own subprocess harness (v4_coretype.py: new state, n = 6/24/96, plus SkewedChi2). KMeans: Haswell and Zen 0 differing lanes; Sandybridge, Nehalem and Prescott 12-45% differ with 4.5e18 ulps (0.0 vs nonzero), K2 ≤ 1.50. RBF: up to 85% differ, up to 5.4e7 ulps, K ≤ 2.45. SkewedChi2: up to 78%, K ≤ 1.62. Source check: dgemv_t_4.c:31-32 (Haswell microkernel shared by SKYLAKEX) and dgemv_n_4.c:36-37 (dgemv_n_microk_skylakex-4.c). | The percentages and K maxima depend on the data (mine are 12-45%, K2 ≤ 1.50, K ≤ 2.45). On this build 'Zen' reports the Haswell core and 'Cooperlake' reports SkylakeX, so neither row is an independent kernel. OPENBLAS_CORETYPE also leaves numpy's own CPU dispatch (einsum for XX, exp/log/cos) unchanged, so a real CI CPU can differ more. |
| C7 (MEASURED): RBFSampler and SkewedChi2Sampler hold with S = c(Σ\|L_i W_ij\| + \|b_j\| + 1), K max 2.38 and 2.12 (2.51 in a smaller run). A magnitude-only scale reaches 302. numpy cos equals glibc; numpy log and exp (AVX512F kernels) are ≤ 1 ulp from glibc. | **PARTLY** | v5_samplers.py (t-distributed, offset 300, γ 0.05/2/scale; lognormal, sparse exponential with s=0.001, uniform to 1e3; n = 5/20/100): RBF K max 2.56 over 1.14M lanes; SkewedChi2 K max 2.77 over 855k lanes; magnitude scale up to 806. np.cos ≠ glibc on 0 of 1,995,000 values; np.log ≠ glibc on 138 of 75,000. The kernel attribution is wrong. In numpy v2.5.1 loops_exponent_log.dispatch.c.src (about lines 666-700 and 1316-1340), DOUBLE_exp/log on AVX512_SKX with NPY_CAN_LINK_SVML call __svml_exp8_ha and __svml_log8_ha. AVX512F_exp/log_DOUBLE (:720, :921) is the #else branch. The installed _multiarray_umath .so exports __svml_exp8_ha and __svml_log8_ha, and opt_func_info reports X86_V4. | The scale holds, but the measured maxima are higher on other data (2.56 and 2.77 > 2.51), so 'measured ≤ 2.5 everywhere' is false. numpy's exp and log on this host are SVML's _ha kernels, not numpy's own AVX512F ones. On an AVX2 machine they fall back to libm. Derived K: n+1 suffices for RBF (n+2 is conservative). |
| C8 (MEASURED): Nystroem(rbf) needs S_l = Σ_j \|N_lj\|·k_j·(1+γ·S2_j), measured K ≤ 0.83 over 140,000 lanes. The plain matvec scale Σ\|N k\| gives K = 216. | **CONFIRMED** | Source: kernel_approximation.py:1098-1108 (pairwise_kernels, then @ normalization_.T) and pairwise.py:1614-1621 (euclidean_distances squared, *= -gamma, exp). v7_nystroem.py (n = 6/12/40, m = 60/120/200, offsets 0/5/30/50, γ = default/0.2/3; 147,200 lanes): K ≤ 1.58 with the kernel-aware S; the plain matvec scale gives 21 to 1.6e4. | The measured maximum is sample-dependent (1.58, not 0.83), and the plain-scale failure is much worse than 216. My first-order count gives max(n+3, m+1); the report's max(n+4, m+2) is safely conservative. Only kernel='rbf' is covered. |
| C9 (MEASURED): PolynomialCountSketch is a count sketch followed by an FFT convolution. A per-lane scale fails (K = ∞ on structurally zero lanes), and the row-wide S = (1/D)Σ_m Π_d \|FFT(a_d)_m\| holds with K ≤ 1.32 (an O(p·log2 D) constant is expected). | **PARTLY** | Source confirmed: kernel_approximation.py:202-243 and scipy _basic_backend.py:77-86 (ducc0). Per-lane failure reproduced (v8_pcs_random.py): at n=4, D=100 the twin is nonzero on 11,962 of 12,600 exactly-zero lanes; at n=5, D=97 on 10,616 of 11,250. On random data the row-wide S gives K ≤ 1.40. It is not a bound, though. v6_pcs_adversarial.py builds an exactly representable integer x whose sketch a_1 is constant and whose a_2 sums to 0. Then the exact y is 0 on every lane and the exact row-wide S is 0, while the twin returns FFT noise on every lane: max 3.96e-15 at D=7 and 2.58e-13 at D=97. Even with S computed by a float FFT, K ≈ 2e15. The native direct convolution returns exact 0. With the normwise scale Π_d \|\|a_d\|\|_2, K is 0.10-0.36 on these inputs and ≤ 1.36 on the random data. | The proposed row-wide S has no finite worst-case K, so 'expect O(p·log2 D)' is refuted for that S. Use a normwise scale such as Π_d \|\|a_d\|\|_2, which follows the normwise FFT error bound and is ≥ the row-wide S by Cauchy-Schwarz/Parseval. The PLANS.md:40-43 wording ('project through a matrix') is indeed inaccurate. |

### What the report missed or got wrong

1. The PolynomialCountSketch scale is not safe. The row-wide S = (1/D)·Σ|ΠFFT| can be exactly 0 while the twin is not, for example on prime D (7, 97) with a constructed x (v6_pcs_adversarial.py), giving K = ∞. A normwise S = Π_d ||a_d||_2 is what the FFT's normwise error actually supports. It gives K ≤ 1.36 on random data and ≤ 0.36 on the adversarial case.

2. The measured maxima are not ceilings. Independent data exceed nearly every figure:
   - K1: 9.1e7 against 6.4e7.
   - Twin vs exact: 2.03 against 1.85.
   - RBF: 2.56 against 2.38.
   - SkewedChi2: 2.77 against 2.51, so 'measured ≤ 2.5 everywhere' is false.
   - Nystroem: 1.58 against 0.83.
   - Magnitude-only scale: 806 against 302.
   - Plain matvec scale for Nystroem: 1.6e4 against 216.
   This strengthens the report's own advice to declare derived K, not measured K.

3. 'K grows like √n' is true only for same-sign terms. Mixed-sign dots stay at about 2.5-3 up to n = 2048 (v3_dot_n.py, and the report's own exp5 RBF column). Distances always grow, because ||x||² and near-centre rows give same-sign terms: 3.7-4.5 at n = 512.

4. numpy exp and log on this host are SVML __svml_exp8_ha/__svml_log8_ha, not numpy's AVX512F_*_DOUBLE kernels. Those kernels are compiled only when SVML cannot be linked. On AVX2 machines numpy falls back to libm.

5. row_sumsq is bit-exact only on x86-64 baseline numpy. The repo's own docstring says aarch64 fuses. The report spells the entry as if bit-exact XX were portable. The derived (n+4) bound does not need it, and the report should say so.

6. OPENBLAS_CORETYPE emulates only the BLAS kernel, not numpy's own dispatch (einsum, SVML exp/log, cos). Its 'Zen' and 'Cooperlake' rows are aliases of Haswell and SkylakeX on this build.

7. The '10⁶ or more below a real defect' margin is overstated in two ways:
   - At n = 2048 the derived K·eps is 4.6e-13, so the margin to float32 rounding is about 1e5.
   - For output-side defects on distances the margin scales with d²/S2. A native that rounds its output to float32 passes on 5-9% of lanes (the near-centre ones) even at K = 2.5 (v9_defect_power.py). Fixture-level detection is still near-certain because far rows flag.

8. The ruling must say how check computes g and S. A float d_n² − d_t² or a float S2 adds its own rounding of about eps·S2, so check needs exact arithmetic or K+1.

9. Verified in the report's favour: DuckDB 1.5.5 evaluates the native spelling left to right, unfused, with glibc ln/exp/cos, bit for bit as the Python model does (v0_duckdb.py). That backs the assumption the report lists.

10. Could not verify: Higham's equation number (3.5) and Theorem 24.2. The PDF host was blocked.

Scripts: scripts/verify-distances/ (v0_duckdb.py, v1_dist.py, v2_ablation.py, v3_dot_n.py, v4_coretype.py, v5_samplers.py, v6_pcs_adversarial.py, v7_nystroem.py, v8_pcs_random.py, v9_defect_power.py, v10_dist_n.py).


### Does the recommendation follow? Yes

The core recommendation follows from the evidence: one ruling of the form |g(native) − g(twin)| ≤ K·eps·S, with a per-family S computed from the fitted state and the row, g = square for the distance family, and K derived as a function of the reduction lengths.

I re-derived (n+4) for distances line by line, and it still holds when ||x||² is not bit-exact. The RBF (n+1 suffices), SkewedChi2 and Nystroem constants are conservative. Measured constants are breached by independent data (SkewedChi2 2.77 > 2.5; K = 3 at n = 32 on positive data), which is exactly why derived K is the right choice.

Amendments needed:
- **PolynomialCountSketch:** replace the row-wide spectral S with a normwise one, e.g. Π_d ||a_d||_2 with a derived FFT constant. Otherwise mark it as having no unbreakable bound yet. As proposed, S can be 0 while the twin is not.
- **How check computes things:** the ruling must state how check computes g and S, exactly or with +1 on K, and add the absolute floor for underflow and overflow that the report already flags.
- **row_sumsq:** do not make the bound depend on its bit-exactness. That holds only on x86-64.
- **Margin claim:** soften '10⁶ below a defect'. The margin is about 1e5 at n = 2048. For distance outputs it depends on d²/S2, so fixtures must include far rows.
- **K(n) wording:** frame K(n) as needed for same-sign data, not as a general √n law.
