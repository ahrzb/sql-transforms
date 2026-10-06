# Matvec bound: error analysis, and how much the twin varies on its own

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


Scope: `loops/native/decisions/open/matvec-parity-bound.md`, which `power-parity-bound.md` and `additive-chi2-parity-bound.md` defer to.

Notation: u = 2^-53, eps = 2^-52 = 2u, γ_n = nu/(1−nu).
- S0 = Σ|x_i c_i|.
- S_rec = (S0 + |m·c|)/s is the record's term scale.
- S2 = (S0 + Σ|m_i c_i|)/s is the one proposed here.
- K(a,b) = |a−b| / (eps·S).

Scripts and outputs are in `R=scripts/matvec/`.

**Environment problem.** `import sql_transform` fails in this venv: pydantic's `_eval_type(prefer_fwd_module=...)` fails on CPython 3.14.0rc2. So `repo_fixtures.py` lifts `_step`/`_rows` verbatim from `catalog_test.py` with `ast`, and loads `_udf.py` as a standalone module.

---

## 0. What the twin actually computes (SOURCED + MEASURED)

**How PythonTransform calls the twin.** It is always one row at a time: `row = est.transform([vals])[0]` (`packages/sql-transform/sql_transform/_udf.py:314`). The batch paths also loop `__call__` per row (`_udf.py:196-203`, `237-244`), and so does `check._serve`. So the twin never takes the `gemm` path. A user who calls sklearn on a batch does.

**The calls inside.** sklearn computes `X @ components_.T`, then `-= mean_ @ components_.T`, then `/= scale` with scale clipped at eps (`.venv/.../sklearn/decomposition/_base.py:152-168`). For a single row, numpy's matmul dispatches on the shape (numpy v2.5.1 `numpy/_core/src/umath/matmul.c.src`):
- k = 1: `cblas_ddot` (`:464`, `:531-533`).
- k ≥ 2: `cblas_dgemv(..., CblasTrans, ...)` (`:477`, `:543-546`, `:147-159`). In OpenBLAS this is `dgemv_t` when `components_` is C-contiguous, and `dgemv_n` when it is F-contiguous.
- F-contiguous `components_` comes from the `covariance_eigh` solver: `Vt = eigenvecs.T` (`_pca.py:636`). In the fixtures this applied to 122 of 380 instances.

**The OpenBLAS 0.3.33 order** (sources downloaded from GitHub; the URL is `raw.githubusercontent.com/OpenMathLib/OpenBLAS/v0.3.33/kernel/x86_64/<file>`):
- **SkylakeX and Haswell, `dgemv_t`.** `KERNEL.SKYLAKEX:1` includes `KERNEL.HASWELL`, and `KERNEL.HASWELL:9-10` selects `dgemv_t_4.c` with `dgemv_t_microk_haswell-4.c` (`dgemv_t_4.c:31-33`).
  - Lanes are grouped by index (`dgemv_t_4.c:299-300`, `346/366/383/395`).
  - The first 4⌊k/4⌋ lanes use 4 interleaved FMA accumulators (`vfmadd231pd`, microk `:49`), reduced as (s0+s2)+(s1+s3) (`:87-97`).
  - The next 2 lanes use an unfused SSE2 kernel with 2 accumulators, and the last lane an unfused 2×2-accumulator kernel.
  - The rows past ⌊n/4⌋·4 go through a GCC-contracted tail, `y + fma(a0,x0,a1·x1)…` (`:405+`).
  - The chunk length is NBMAX = 2048 (`:35`).
- **k = 1, `ddot`.** On SkylakeX this is AVX-512 with 32 FMA accumulators (`ddot_microk_skylakex-2.c:51-92`). On Haswell it is 16 accumulators with an unfused tail (`ddot.c:39-44`).
- **Sandybridge, Nehalem and Prescott.** They use the unfused C fallback, `temp += ((a0x0+a1x1)+a2x2)+a3x3`.

**MEASURED: the twin's row order is emulated bit-exactly** (`$R/emul.py`).
- `verify_order2.py`, random m ∈ [1,80) ∪ [2030,2100), k ∈ [1,14):
  - SkylakeX: 10,640/10,640 lanes bit-exact.
  - Haswell: 10,640/10,640 bit-exact.
  - Sandybridge: 10,609/10,640; every miss has k = 1 (that `ddot` kernel is not emulated).
- On the catalog fixtures (`debug_emu.py`): 21,256/21,256 lanes match for C-contiguous `components_` or k = 1. On F-contiguous `components_` (`dgemv_n`, not emulated) 574 of 4,092 lanes miss.

The twin's order is therefore a function of five things: (CPU coretype, k, the lane index mod 4, `components_` memory layout, row vs batch). Reproducing it would need FMA. DuckDB 1.5.5 has no `fma`; checked with `duckdb_functions()`.

## 1. Error analysis: what K is provable, what K is measured, how K scales with n

**DERIVED.** The model is fl(a∘b) = (a∘b)(1+δ) and fma(a,b,c) = (ab+c)(1+δ), with |δ| ≤ u.
- **The entry (left to right, unfused).** Term i suffers n−i+2 roundings (term 1 suffers n), so |ŝ_N − s| ≤ γ_n S0. This is Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1, eq. (3.4).
- **Any order, any tree, fused or not.** Every term suffers at most n roundings, so |ŝ − s| ≤ γ_n S0 (Higham eq. (3.5)). A tree of depth d gives γ_{d+1}; pairwise summation gives γ_{⌈log2 n⌉+1}.
- **SkylakeX `dgemv_t` (the emulated order).** Each term suffers at most ⌈n/4⌉ FMA roundings, plus 2 horizontal adds, plus 1 tail add, so the bound is γ_{n/4+3}.
- **The lane.** r = fl(fl(d − mc)/s), with s bit-identical on both sides (sqrt is correctly rounded).
  - |r_N − r_T| ≤ [(γ_n + γ_h)·S0 + |Δmc|]/s + 2u(|r_N| + |r_T|).
  - |Δmc| ≤ 2γ_n Σ|m_i c_i| when the twin recomputes m·c on another BLAS, and 0 when the entry's constant is the twin's own.
  - Hence **|r_N − r_T| ≤ (n + 2)·eps·S2** for any kernel, any machine, row or batch. When m·c is shared, the bound is (n+2)·eps·S_rec.
  - If the SkylakeX kernel is assumed, it tightens to ((n + n/4 + 3)/2 + 2)·eps·S.
- **Higham & Mary 2019, probabilistic** (SIAM J. Sci. Comput. 41(5):A2815; not re-fetched, egress was blocked). If rounding errors are mean-independent, each side is ≤ λ√n·u·S0 with probability ≥ 1 − 2n·exp(−λ²(1−u)²/2). That gives K ≈ λ√n + 2. It is a bound with high probability under a model, not a theorem about all data.

**MEASURED: K against n** (`kvn_build.py`, `kvn_twin.py`, `kvn_analyze.py`; output in `kvn_analyze.out`).
- Setup: 500 rows × 16 components = 8,000 lanes per (dataset, n).
- "exact" is the correctly rounded x·c: Dekker TwoProduct pieces summed with `math.fsum`, checked against `math.fma`.
- Datasets:
  - **gauss**: orthonormal C, x ~ N(0,1).
  - **offset**: x = μ + N(0,1), μ ~ U(±100).
  - **pca**: `PCA(whiten=True, 16)` fitted on low-rank data with feature scales from 1e-2 to 1e4 and large means. Half the held-out rows lie within 1e-6·scale of the mean.

Dot-level maxima in eps·S0 units:

| n | gauss LTR−twin | offset LTR−twin | pca LTR−twin (p99) | pca LTR−exact | pca twin−exact | pca gemm−gemv | pca twin Sandybridge−SkylakeX | provable, any kernel: n | provable, SkylakeX kernel |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1.48 | 0.96 | 0.99 (0.99) | 1.07 | 0.99 | 1.61 | 0.99 | 4 | 4.0 |
| 32 | 1.88 | 1.08 | 3.90 (1.70) | 3.12 | 1.67 | 3.90 | 2.22 | 32 | 21.5 |
| 128 | 1.71 | 1.52 | 4.72 (3.18) | 5.69 | 3.38 | 5.27 | 4.53 | 128 | 81.5 |
| 512 | 1.92 | 1.87 | 5.50 (2.80) | 4.75 | 2.31 | 4.18 | 3.14 | 512 | 321.5 |
| 2048 | 1.66 | 1.28 | 12.4 (6.91) | 15.8 | 7.46 | 9.10 | 12.0 | 2048 | 1281.5 |

Whole PCA lanes, entry vs the SkylakeX row twin:

| n | max K (S_rec) | max K (S2) | max ulps |
|---|---|---|---|
| 4 | 0.98 | 0.98 | 1.6e10 |
| 32 | 1.97 | 1.95 | 8.2e11 |
| 128 | 3.11 | 2.36 | 5.9e11 |
| 512 | 4.57 | 2.75 | 1.4e11 |
| 2048 | 7.63 | 6.37 | 1.7e12 |

Against the Sandybridge twin, max K (S2) is 0.39, 1.73, 4.11, 3.27 and 9.47 for the same n.

How to read this:
- With zero-mean terms (gauss, offset), K does not grow with n: it stays around 1.5–1.9 from n = 4 to n = 2048.
- On structured PCA-like data it grows sub-√n: max K/√n ≤ 0.7 everywhere.
- Measured K is 10× (n = 32) to 160× (n = 2048) below the provable n.
- **The record's K = 3 fails from n = 128 on, and also at n ≤ 32 once more seeds are run** (next table).

**Reproducing the record** (`build_fixtures.py`, `twin_run.py`, `analyze_fixtures.py`; outputs `analyze_fixtures*.out`).

| | lanes | max ulps | p99 K | max K |
|---|---|---|---|---|
| Record, seeds 0–39 | 3,855 | 1,306 | 0.99 | 1.76 |
| Measured, seeds 0–39 (current generator) | 4,618 | 1,306 | 1.06 | 2.29 |
| Record, seeds 0–199 | 23,050 | 9,877,709,850 | 1.10 | 2.99 |
| Measured, seeds 0–199 (current generator) | 25,208 | 9,877,709,850 | 1.12 | 2.99 |
| Measured, seeds 0–999 (current generator) | 134,285 | 9.2e18 (sign flips) | 1.17 | **4.12** |

- The max-ulps figures and the 200-seed max K reproduce exactly.
- The lane counts and the 40-seed max K do not. The record does not state how it filtered lanes.
- The record's numbers match HEAD's fixture generator, not b851298's, despite the commit it names. On b851298's generator, run via `git show`, the 200-seed figures are 2.68e11 ulps and K 3.10.
- Over 1,000 seeds: LTR−exact reaches 4.67 and SkylakeX `gemv`−exact reaches 2.84 (eps·S0 units).

## 2. Is the twin a fixed reference? No (MEASURED)

Setup: one fitted model, pickled once and loaded unchanged in every configuration (no refit). The same rows go through each configuration, with `OPENBLAS_CORETYPE` and `OPENBLAS_NUM_THREADS` set per subprocess and confirmed with threadpoolctl. The Prescott setting reports "Katmai".

| pair, catalog PCA(whiten) fixtures | lanes differing | max ulps | max K (S_rec) | max K (S2) |
|---|---|---|---|---|
| **seeds 0–199 (25,208 lanes)** | | | | |
| entry vs SkylakeX twin | 4,819 | 9,877,709,850 | 2.99 | 2.99 |
| SkylakeX twin, 4 threads vs 1 | 0 | 0 | 0 | 0 |
| Haswell twin vs SkylakeX twin (k=1 and F-layout lanes only) | 680 | 568 | 19.8 | 1.76 |
| Sandybridge = Nehalem = Prescott twin vs SkylakeX twin | 8,418 | 2,026 | **237.6** | 2.86 |
| twin `transform(X)` (gemm) vs `transform([row])`, both SkylakeX | 4,696 | **9,877,709,850** | 2.99 | 2.99 |
| entry vs Sandybridge twin | 7,979 | 9,877,709,850 | **237.6** | 3.10 |
| **seeds 0–999 (134,285 lanes)** | | | | |
| Sandybridge twin vs SkylakeX twin | 45,541 | 9.5e18 | **1,730.8** | 3.10 |
| SkylakeX batch vs row | 26,350 | 1.3e19 | 4.12 | 4.12 |

On synthetic PCA data the twin also disagrees with itself by up to 1.4e13 ulps:
- batch vs row on SkylakeX: 1.4e13 at n = 32;
- Sandybridge vs SkylakeX: 1.4e13 at n = 32 and 2.2e12 at n = 2048.

Thread count changed nothing for `gemv` or `gemm` in any test, including `thread_probe.py` with n up to 4096 and k up to 64. Haswell and SkylakeX `gemm` differ at n = 2048 (K 3.3).

What this means:
1. **A ulp bound against the twin is not well defined.** The same model and the same row give answers 9.9e9 ulps apart on one machine, depending only on whether sklearn got one row or a batch. Across CPU kernels the gap reaches 1e12–1e19 ulps (sign flips included). The entry-vs-twin distance is the same size as the twin's own spread.
2. **The record's S misses cross-machine variation.** S_rec uses |m·c|, which is valid only when the entry's m·c constant is bit-identical to the twin's. That holds inside one `check` process, not when the twin runs on another BLAS: K reaches 237.6, and 1,730.8 at 1,000 seeds. With S2 every pair, entry or twin, stays ≤ 4.12.
3. Of the knobs tested, only the coretype (SkylakeX or Haswell vs the unfused C-fallback kernels) and the call shape (row vs batch) change the twin's answer. Thread count does not.

## 3. Is the term-scale criterion a standard notion?

- **Yes: it is the componentwise forward-error bound for dot products.** That is |fl(xᵀy) − xᵀy| ≤ γ_n |x|ᵀ|y| (Higham §3.1). Dividing by |xᵀy| gives a relative error ≤ γ_n·cond, with cond = Σ|p_i|/|Σp_i| (Ogita, Rump & Oishi 2005, SIAM J. Sci. Comput. 26(6):1955; their Dot2 bound is u|xᵀy| + γ_n²|x|ᵀ|y|). So "K·eps·S" is "K·eps·cond·|result|": a ulp bound scaled by the condition number. For several rounding sites, S is the first-order sum Σ_j |∂r/∂δ_j|, as in running-error analysis (Higham §3.3). Higham and ORO are cited from memory: the egress proxy blocked SIAM, Manchester, arXiv, Intel and nhigham.com.
- **`np.allclose` / `isclose`.** The test is |a−b| ≤ atol + rtol·|b|, with defaults rtol = 1e-5, atol = 1e-8 (`numpy/_core/numeric.py:2226`, `:2269`). The term-scale bound is the same shape with rtol = 0 and a per-lane atol = K·eps·S derived from the problem. A ulp bound is a pure rtol; neither a fixed atol nor ulps handles cancellation correctly.
- **numpy's own ulp tests.** `assert_array_max_ulp` and `assert_array_almost_equal_nulp` (`numpy/testing/_private/utils.py:1785`, `:1848`) target elementwise functions, not reductions.
- **skl2onnx.** Converter tests use a fixed absolute tolerance: `decimal=5` by default (`tests/test_utils/utils_backend.py:80`), passed to `assert_array_almost_equal` (`:330`), whose test is |a−b| < 1.5e-5 (`numpy/testing/_private/utils.py:1227`), on float32. The onnxruntime checkers file returned 404 at the guessed path.
- **Intel MKL CNR and ReproBLAS** (Demmel & Nguyen, binned summation). These give reproducibility across thread counts or code paths, not agreement with an unrelated BLAS. CNR is per code branch (`MKL_CBWR`). OpenBLAS's equivalent lever is pinning `OPENBLAS_CORETYPE`, x86-only. Both are from memory; I could not fetch the docs.
- **Correctly rounded dot product in SQL.** Measured in `duck_dot2.py` / `duck_dot2.out`, n = 32, 200k rows, one lane, one thread:
  - Ogita–Rump–Oishi Dot2, written with Dekker split, TwoProduct and TwoSum (no FMA; c pre-split as constants), runs in DuckDB bit-exactly to its spec on 5,000/5,000 sampled rows. It was correctly rounded on 5,000/5,000 rows; left to right was correctly rounded on 670/5,000.
  - Cost: 320 ms against 29 ms, **11× slower**; SQL text 13.8k against 1.2k characters; about 17 flops per term against 2.
  - It does not remove K. Even an exact entry differs from the twin by the twin's own error (SkylakeX `gemv`−exact up to 2.84 at n ≤ 32 and 7.46 at n = 2048). It roughly halves K: the provable bound becomes n/2 + 2.
  - The left-to-right entry is already a fixed reference. It is a fixed sequence of correctly rounded IEEE operations, deterministic on any IEEE platform without FTZ or x87 (DERIVED; not tested off this machine). Only the twin varies.
- **Constants must go through a string literal.** A bare numeral is parsed as DECIMAL and rounds twice in DuckDB: 45/2,000 random doubles came out wrong, against 0/2,000 through a string. confit already casts through a string (`packages/confit/confit/sql.py:276-285`). My harness initially did not, which is how this showed up.

## 4. How `check` should compute S and K

- **The scale, per lane k**, from fitted state plus the row exactly as `transform` sees it:
  - `s_k = max(sqrt(explained_variance_[k]), eps)`, as sklearn clips.
  - `S_k = (Σ_i |x_i·C[k,i]| + Σ_i |mean_[i]·C[k,i]|) / s_k`.
  - The second sum is the cross-machine term. Use |m·c| only if the bound is meant to hold just in the process that translated the step.
- **Overflow.** Compute S overflow-safely, for example with a common power-of-two scaling. One fixture lane had S = inf with finite results (−8.470005491905276e306 vs …126e306, 120 ulps); there the inequality passes vacuously.
- **Subnormal floor.** Add an absolute floor of n·2^-1074/s for subnormal products (DERIVED from the underflow model; no fixture lane needed it).
- **The rule.**
  - Pass if the values are equal, both NaN, or the same infinity.
  - Fail if exactly one is non-finite.
  - Otherwise require |a − b| ≤ K·eps·S + floor.
- **K, per family, as a function of the dot length n** (n = number of features).
  - Provable: K = n + 2 for any kernel, any machine, row or batch. It is about 34 for the catalog's n ≤ 32.
  - Probabilistic option: λ√n + 2.
  - Measured envelope: K_max ≤ 0.7√n on every dataset here, and 4.12 at n ≤ 32 over 134k lanes.
- **Detection power is barely affected.** A structural bug (wrong index, sign, mean, scale, or a missing term) moves a lane by about |x_i c_i| ≈ S/n. That exceeds (n+2)·eps·S for any n below about 10^7. One-ulp constant errors pass under both K = 3 and K = n + 2. They are caught instead by the other half of `check`: the entry against DuckDB's own definition, bit-exact.

**Flagged as not confirmed:**
- ARM and Apple Accelerate kernels (not available here).
- Exact Higham–Mary, ORO and MKL CNR wording (sources blocked).
- Why OpenBLAS threading never changed `gemv` results at 4096×64.
- The record's lane filter.



## The researcher's recommendation

Take Option 1, the term-scale bound, with two corrections; drop the measured K = 3. First, check should compute the scale per lane as S = (Σ_i |x_i·c_ki| + Σ_i |m_i·c_ki|)/s_k, with s_k clipped at eps as sklearn does, from components_, mean_, explained_variance_ and the input row. The record's |m·c| term lets cross-machine twins reach K = 1,731, while every entry or twin pair measured stays within 4.12 eps of this S. Second, declare K as the provable K = n + 2, with n the number of features. That follows from Higham's γ_n bound, which holds for any summation order, FMA or not, on both sides, plus two roundings. It therefore holds for every OpenBLAS kernel, CPU, thread count, and for batch as well as row calls. A measured K is not a bound in exactly the sense the record faults ulps for: the record's K = 3 is already exceeded at 1,000 seeds (4.12) and at n ≥ 128 (up to 7.6 on a lane, 12.4 on the dot). The looseness costs little: measured K stays at or below 0.7·√n, and real translation bugs move a lane by about S/n, far above (n+2)·eps·S. Ulp-sized errors such as a misparsed constant are caught by the bit-exact entry-vs-DuckDB half of check either way. Record the measured maximum next to the declared K as documentation, and make check handle non-finite results and an overflowing S explicitly. Do not make the entry correctly rounded by default: it costs about 11× in DuckDB and only halves the distance to a twin that already varies by up to 1e13 ulps across CPUs and call shapes. The same rule settles the power and additive-chi2 records: in each, S is the lane's first-order sum of rounding sensitivities over the operations whose order or library differs.


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| The twin is not a fixed reference: batch vs row differs by up to 9,877,709,850 ulps at 200 seeds (the same maximum as entry vs twin), and at 1,000 seeds batch vs row reaches 1.3e19 ulps and Sandybridge vs SkylakeX reaches 9.5e18. So a ulp bound against the twin is not well defined. | **CONFIRMED** | I wrote my own twin runner (verify-matvec/v_twin.py) and ran it under 5 coretypes on the report's fixtures1000.pkl. First I regenerated 44 seeds from catalog_test.py with my own AST extraction; they match the pickle bit for bit (v_fixcheck.py). Results (v_analyze.out), 200 seeds over 25,208 finite-S lanes: entry vs SKX row differs on 4,819 lanes, max 9.878e9 ulps; SKX batch vs SKX row differs on 4,696 lanes, max 9.878e9. At 1,000 seeds: batch vs row max 1.302e19; SNB vs SKX max 9.466e18; entry vs SKX max 9.234e18. | Context: of the lanes over 1e9 ulps, 12 of 15 (batch vs row) and 12 of 14 (SNB vs SKX) have a ±1e300 EDGES input (v_worst2.py). There the two results differ by hundreds of orders of magnitude, e.g. -1.9e254 vs 1.74, with K≈0.08. That is plain arithmetic garbage on both sides, more than 'sign flips'. The 9.9e9 lanes near the mean are the meaningful case. |
| S_rec = (Σ\|x_i c_i\| + \|m·c\|)/s does not bound cross-machine differences (K 237.6 at 200 seeds, 1,730.8 at 1,000). With S2 = (Σ\|x_i c_i\| + Σ\|m_i c_i\|)/s, every pair stays ≤ 4.12. | **PARTLY** | The numbers reproduce exactly: SNB vs SKX K(S_rec) is 237.568 at 200 seeds and 1,730.781 at 1,000. With S2 every pair is ≤ 4.118. Nehalem and Prescott equal Sandybridge on 0 differing lanes. Haswell vs SKX K(S_rec) reaches 27.5 at 1,000 seeds (all F-layout lanes). | The blow-up only happens when the entry's m·c constant and the twin's m·c come from different BLAS kernels. check never sees that pairing: it translates and runs the twin in one process (_check.py: query(step) and _serve run in-process). In every coretype process, sklearn's spelling (mean_.reshape(1,-1)@C.T) and C@mean_ give bit-identical m·c on all 135,034 lanes. Within check, S_rec and S2 both give max K 4.12. S2 is still the better choice, but for robustness: a cross-machine deployment contract, and freedom in how the translator computes m·c. Current check does not fail with S_rec. |
| The record's K=3 is not a bound: it reproduces at 200 seeds (2.99) but reaches 4.12 at 1,000 seeds. On PCA-like data K grows with n (3.90, 4.72, 5.50, 12.4). On zero-mean data it stays at 1.5-1.9. Every measured maximum is ≤ 0.7·√n. | **PARTLY** | K=3 is exceeded: entry vs SKX max K = 4.119, at an n=29 lane (v_kn.py). On my own different synthetic data (v_syn_build.py: Toeplitz with a common offset, heterogeneous scales and means; v_syn_an.out), the LTR−SKX dot reaches 2.96/3.39 at n=32, 5.20/4.87 at 128, 7.03/7.82 at 512 and 11.03/15.06 at 2048. Lane K(S2) reaches 13.1 vs SKX and 16.05 vs SNB at n=2048. The zero-mean control stays at 1.3–2.3 for every n. The growth claim and the flat zero-mean claim both reproduce. | 'Every measured maximum ≤ 0.7·√n' is false on the report's own data. On the fixtures: K=4.12 at n=29 (0.77√n), K=1.86 at n=3 (1.08√n), K=1.49 at n=2 (1.05√n). In the report's kvn_analyze.out: gauss n=4 LTR−twin is 1.48 (0.74√n) and gemm−gemv at n=4 is 1.61 (0.81√n). A constant floor of about 2 from the subtract and divide steps means no c·√n envelope holds at small n. Something like 2 + c·√n would be needed. |
| Provable: \|native − twin\| ≤ (n+2)·eps·S2 for any order, fused or not, and (n+2)·eps·S_rec when m·c is shared. The SkylakeX dgemv_t order tightens it to ((n+n/4+3)/2+2)·eps·S. Higham–Mary gives about λ√n+2. | **PARTLY** | I re-derived it line by line. Each side's dot is within γ_n·S0 (any binary tree, FMA or not: at most n roundings per term). \|Δmc\| ≤ 2γ_n Σ\|m_i c_i\|. Subtraction plus division add ≈2u per side, so ≈4u·\|r\| ≤ 2·eps·S. Since 2γ_n ≈ n·eps, the total is (n+2)·eps·S2; with a shared mc it is (n+2)·eps·S_rec. This holds to first order. The threaded and chunked OpenBLAS orders are still trees, so the bound covers them. A web search snippet confirms Higham–Mary Thm 2.4's form: γ̃_n(λ)=exp(λ√n·u + nu²/(1−u))−1, with probability ≥ 1−2exp(−λ²(1−u)²/2). The Higham ASNA equation numbers could not be fetched (egress blocked). | (a) Strictly, the bound is 2γ_n·S2 + 2u(1+u)(\|r_N\|+\|r_T\|), i.e. (n+2)·eps·S2·(1+O(nu)). K=n+3 would be a clean rigorous constant. (b) The SkylakeX tightening is wrong for the two lanes handled by dgemv_kernel_4x2 (k mod 4 ≥ 2). That kernel uses one SSE2 accumulator of 2 doubles per lane (dgemv_t_4.c:69-130), so its depth is about n/2+2, not n/4+3. The tightening also does not apply to F-layout (dgemv_n), k=1 ddot, gemm, or threaded calls. (c) The Higham–Mary 2019 model is independent, mean-zero errors, per the snippet. 'Mean-independent' is the later relaxation (Connolly–Higham–Mary). |
| The twin row order is gemv/ddot, depending on coretype, k, lane index mod 4, components_ layout and row vs batch. The emulation is bit-exact (10,640/10,640). DuckDB 1.5.5 has no fma, so the entry cannot follow the order. | **PARTLY** | I wrote my own driver for the report's emulator on new shapes (v_emu.py): n up to 6,199 crossing NBMAX chunks, k up to 22, every n mod 4, cancelling data. It is bit-exact on 6,970/6,970 lanes for SkylakeX and the same for Haswell (single thread). duckdb_functions() has no fma or fused function. The OpenBLAS v0.3.33 sources match the cited structure: KERNEL.SKYLAKEX includes HASWELL, the 4x4 kernel uses ymm FMA with the (s0+s2)+(s1+s3) reduction, NBMAX is 2048. The sklearn and _udf.py line citations are correct. | The list of order determinants is incomplete. When m·n ≥ 115200·GEMM_MULTITHREAD_THRESHOLD = 460,800 (interface/gemv.c:127), the thread count changes the order. driver/level2/gemv_thread.c splits the lanes into widths that are not multiples of 4, and for dgemv_n it splits the summation dimension and reduces. The grouping is by position relative to 4⌊k/4⌋, not literally 'index mod 4'. |
| PythonTransform always calls est.transform([vals]) one row at a time. apply_batch and the arrow wrapper loop __call__, so check's twin never exercises gemm. | **CONFIRMED** | _udf.py:314 is `row = est.transform([vals])[0]`. apply_batch (:196-203) and _arrow_scalar_batch (:237-244) loop per row. confit/functions.py:15-17 states the contract as 'as if the definition ran once per call'. _check.py:_serve uses DuckDBInferFn with udfs=[step]. |  |
| OPENBLAS_NUM_THREADS 1 vs 4 changed no result, up to n=4096, k=64, for gemv and gemm. Haswell vs SkylakeX differ only where k=1 or components_ is F-contiguous (680/25,208 lanes, max 568 ulps). Their gemm also differs at n=2048. | **REFUTED** | Every thread test in the report was below OpenBLAS's gemv threading threshold. 4096×64 = 262,144 is under 460,800 (interface/gemv.c:127 in v0.3.33), so OpenBLAS ran single-threaded and the null result is an artefact. Above the threshold, my v_thr.py (SkylakeX, 1 vs 4 and 1 vs 3 threads) gives these row-path (gemv) differences:<br>- C layout: 30/144 lanes at (32768,18); 36/240 at (32768,30); 38/240 at (20000,30).<br>- F layout: 40/40 at (100000,5) and 56/56 at (100000,7).<br>Relative differences reach 1.5e-14. gemm was unchanged in all my tests. The Haswell figure reproduces: 680 lanes, 568 ulps at 200 seeds, all from F-layout lanes; no k=1 lane differs, since k=1 occurs only with n=1 here. | Thread count does change the twin once m·n ≥ 460,800. It is irrelevant for catalog fixtures (m·n ≤ 1,024) but not for wide data. Haswell vs SkylakeX gemm differs at n ≤ 32 too: 885 batch lanes at 200 seeds, 522 of them C-layout. The difference is not limited to n=2048. None of this breaks the provable bound, since the threaded orders are still trees. It does undercut every 'measured' characterisation of the twin. |
| Dot2 in DuckDB SQL is bit-exact to spec and correctly rounded on 5,000/5,000 rows, at 11.0× the LTR cost. The twin is up to 2.84 eps·S0 from exact (7.46 at n=2048). Dot2 only halves K. | **PARTLY** | I wrote my own Dot2 as nested subqueries (v_dot2.py). At 1 thread it is 11.3× LTR (325.8 vs 28.8 ms per 200k rows); at 4 threads 13.8×. It matches the spec on 5,000/5,000 rows and is correctly rounded on 5,000/5,000. Twin−exact on the fixtures is 2.844 and LTR−exact is 4.671 (v_twin_dot.py), matching the report. | Correct rounding is only shown on well-conditioned dots (cond ≈ 3 in my data). Dot2 is not correctly rounded in general: its ORO bound is u\|xᵀy\| + γ_n²\|x\|ᵀ\|y\|. 'Only halves K' is true for the provable bound but understates the measured effect at large n on structured data. There the LTR entry dominates: LTR−exact is 11–15 against twin−exact 2.9–5.7 at n=2048, so an accurate entry cuts measured K by about 2.7–3.8×. The report also never considers a pairwise entry order, which costs the same flops as LTR (see missed_or_wrong). |
| A bare DuckDB numeric literal is parsed as DECIMAL and rounds twice (45/2,000 wrong vs 0/2,000 through a string). confit already casts through a string. | **CONFIRMED** | In my own test (v_duck_lit.py), 38/1,492 non-exponent reprs came out wrong as bare numerals and 0/3,000 through '...'::DOUBLE. typeof(0.1234567890123456) is DECIMAL(17,16), and exponent-form numerals are DOUBLE, so they are not affected. confit/sql.py:276-285 already casts through repr and documents why in a comment, so this is a rediscovery, not a new risk. |  |

### What the report missed or got wrong

1. The recommendation's '≤ 0.7·√n' envelope is false on the report's own data. On the fixtures, K=4.12 at n=29 (0.77√n), 1.86 at n=3 and 1.49 at n=2 (about 1.05–1.08√n). The report's gauss n=4 row gives 0.74√n and its gemm−gemv n=4 row 0.81√n. Its own two sentences ('≤ 0.7√n' and '4.12 at n ≤ 32') contradict each other.

2. The thread-count conclusion is an artefact of testing below OpenBLAS 0.3.33's gemv threading threshold, m·n < 460,800 (interface/gemv.c:127). Above it, 1 vs 4 or 3 threads changes 13–100% of row-path lanes (verify-matvec/v_thr.py). That supports declaring a provable K rather than a measured one, but it falsifies the report's list of what determines the twin's order. The report's open question 'why threading never changed gemv at 4096×64' has this answer.

3. The S_rec failure (K=1,731) does not occur in check as built. The translator and the twin run in the same process, and m·c is bit-identical across both numpy spellings under every coretype. S2 is the right call for a cross-machine contract or a translator free to compute m·c its own way. The record's owner should decide on that basis, not on the claim that S_rec is currently broken.

4. Wrong claim: 'ulp-sized errors such as a misparsed constant are caught by the bit-exact entry-vs-DuckDB half of check'. That half (_check.py:104-113) compares confit's serving of the native SqlFunction against DuckDB running the same definition. A wrong constant produced by the translator appears identically on both sides, so neither half catches it under either K. Both proposals tolerate ulp-level constant errors anyway.

5. Missed option: sum the entry as a balanced pairwise tree instead of left to right. It uses the same flops and SQL size and has no FMA.
   - The entry's provable error drops from γ_n to γ_{⌈log2 n⌉+1}, so K falls to about n/2+log2(n)/2+2, the same as Dot2's at no runtime cost.
   - Measured K vs SKX drops from 5.52 to 1.44 (toep) and from 13.11 to 3.52 (hetero) at n=2048 (v_pairwise.py).
   - Expression depth becomes log2 n. DuckDB 1.5.5's default max_expression_depth=1000 already rejects a nested LTR expression with 999 terms (500 parses; v_depth.py). The report's n=512/2048 LTR scenarios may not even run as plain nested SQL.

6. The SkylakeX 'tightened' bound ignores the 4x2 kernel's 2-accumulator lanes, which have depth about n/2. It also does not apply to F-layout (dgemv_n; 122 of 380 fixture instances, covariance_eigh), gemm, or threaded calls.

7. The proposed rule 'fail if exactly one side is non-finite' would spuriously fail when one order overflows and the other does not, e.g. a quotient near DBL_MAX. The current ulp check passes inf vs DBL_MAX at 1 ulp. No fixture lane hit this (0 cases), but the rule should pass when S is non-finite or near overflow.

8. Haswell vs SkylakeX gemm differs already at n ≤ 32: 522 C-layout fixture lanes at 200 seeds. It is not only an n=2048 effect.

9. The headline 1e18–1e19 ulp gaps come mostly from rows with ±1e300 inputs, where every side is arithmetic garbage (K≈0).

10. 'The same rule settles the power and additive-chi2 records' has no supporting analysis. Those families involve libm functions, not only summation order, and their S would have to be derived per family.


### Does the recommendation follow? Yes

The core recommendation follows from the evidence:
- Use a per-lane term-scale bound with S2 and a provable K that depends on n.
- Drop the measured K=3.
- Do not pay about 11× for Dot2 by default.

The derivation holds to first order, and it covers everything I found, including the threaded orders the report missed, because those are still summation trees. The measured K of 3 is exceeded (4.12 at n=29) and grows with n on structured data, as my own synthetic data reproduces.

Several supporting arguments are wrong or overstated and should be fixed before an owner rules:
- The '≤0.7√n' envelope is false.
- 'Thread count does not matter' is false above m·n=460,800.
- The other half of check does not catch translator constant errors.
- The K=1,731 motivation for S2 is a cross-machine scenario that check never exercises. S2 should be justified as a deployment-portability and robustness choice.

Further changes:
- Declare K=n+3, or (n+2)(1+2nu), to be strictly rigorous rather than first-order.
- Specify that S is computed overflow-safely, and that a non-finite or overflowing S, or one-sided overflow, passes rather than fails.
- Consider a pairwise entry order before ruling. It costs nothing, roughly halves the provable K, cuts measured K up to about 4× at large n, and keeps SQL expression depth at log2 n, under DuckDB's default limit of 1000.
- Do not extend the rule to the power and chi2 records without per-family derivations of S.

Artifacts: scripts/verify-matvec/ (v_twin.py, v_analyze.out, v_kn.py, v_syn_an.out, v_thr.py, v_emu.py, v_dot2.py, v_duck_lit.py, v_pairwise.py, v_depth.py, src/ with OpenBLAS v0.3.33 sources).
