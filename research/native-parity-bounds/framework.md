# How others state numeric parity, and the general bound

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


Scripts and outputs are in `R=scripts/framework/`. Run them with `.venv/bin/python <script>` from `$R`.

**Environment caveat.** `sql_transform` does not import in this venv: Python 3.14.0rc2 with pydantic 2.13.4 fails with `_eval_type() got an unexpected keyword argument 'prefer_fwd_module'`. So I copied the catalog's fixture generator verbatim into `$R/fixtures.py` (it is `catalog_test.py:596-835` at 113fba7).

Notation: u = 2^-53, eps = 2u, γ_n = nu/(1-nu). "K" is |a-b| / (eps·S).

## TL;DR

1. **The twin is not a ulp-accurate reference, so a ulp bound against it can't be met.** Its own error, against the correctly rounded lane, is:
   - PCA: up to **1.28e14 ulps** (p99 19, p99.9 148, 25,208 lanes);
   - chi2: up to **7.1e6 ulps**;
   - Yeo-Johnson: up to **137 ulps**.

   A small ulp bound against the twin asks the entry to reproduce the twin's rounding noise on badly conditioned lanes.
2. **The twin isn't reproducible either, and it varies as much as the entry differs from it.**
   - Across OpenBLAS kernels on the same fixtures: up to 2,026 ulps.
   - Batched transform against row-by-row, same machine: **9,877,709,850 ulps**, the same worst lane as entry vs twin.
   - Measured in units of the term scale, every one of these pairs stays at K ≤ 3.1.
3. **The record's matvec numbers reproduce, but a measured constant K is not a bound.**
   - Entry vs a twin on the Sandybridge kernel gives K = **3.10**, above the proposed PCA bound of 3.
   - A structured row at n = 32 gives **3.75**.
   - Max K grows roughly 0.2–0.37·√n with the width n: about 17 at n = 8192.
   - The bound that holds whatever the data is **K = n + 2**.
4. **chi2 is bit-exact against a twin pinned to numpy's baseline kernels** (7,199,988 lanes, 0 differ). Its whole gap comes from numpy's AVX-512 `log` and `cosh`. The `cosh(πjs)` "fitted constant" itself moves by 1 ulp with the CPU.
5. **Recommendation: generalize the ruling from "ulps of the result" to "ulps of the lane's error scale S".** Each family declares S and a K *derived* from per-operation error bounds (K may depend on n), with a measurement that confirms it with headroom. Today's ulp bounds are the special case S = |y|.

## 1. Theory

### Why a ulp bound of the result fails (DERIVED)

Take a lane y = Σ t_i (for PCA the terms are x_i·c_ki, then −M).
- Any two evaluation orders, with or without FMA, satisfy |T − N| ≤ 2γ_n Σ|t_i|.
- In ulps of y that is about (n/2)·κ, where κ = Σ|t_i| / |Σ t_i| is the componentwise relative condition number of the sum.
  - Higham, *Accuracy and Stability*, 2nd ed., §1.6 and §4.2.
  - Ogita, Rump & Oishi (SISC 2005) use cond = 2κ: [paper](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf) (abstract via search; the PDF was egress-blocked).

So "max ulps over N seeds" measures max κ over the data. It grows without bound as lanes approach 0. The record's 9.9e9-ulp lane sits at 0.33 eps of S, so κ ≈ 3e10. A ulp bound on such lanes bounds the data, not the implementation.

The general rule (Higham §1.5–1.6): forward error ≲ cond × backward error. A ulp bound on the result is meaningful only where cond = O(1).

### The general replacement (DERIVED, first order)

Both sides evaluate the same expression DAG. They differ only at a set D of nodes v_j: reductions summed in another order, and kernels from another library. At node j, side X has relative error |δ_j^X| ≤ r_j^X:
- an IEEE operation: u;
- a k-ulp kernel: 2k·u;
- an n-term reduction: γ_{n-1} on its term magnitudes.

Nodes that are bit-identical on both sides cancel to first order. Linearizing:

    |T − N| ≤ Σ_{j∈D} |∂y/∂v_j|·|v_j|·(r_j^T + r_j^N) + O(u²) ≤ K·eps·S(x) + τ

where S(x) = Σ_{j∈D} |∂y/∂v_j||v_j| is a linearized running-error scale (Higham §3.3). It is cheap to evaluate from the row and the fitted state. τ is an underflow floor.

**When this is a real bound, independent of the data:**
- **(i) S is computable from (x, fitted state) in float64.** Over-estimate it by a (1+γ_m) slack.
- **(ii) Each r_j comes from a stated per-operation bound.**
  - IEEE operations: proven.
  - glibc: documented, not proven. glibc "does not aim for correctly rounded results"; its goal is "within a few ulp", and its testsuite flags only errors above 9 ulp ([glibc manual/math.texi](https://github.com/bminor/glibc/blob/master/manual/math.texi)).
  - numpy's SVML kernels: the PR states "maximum ULP error across all the implementations is 4 ULP" ([numpy#19478](https://github.com/numpy/numpy/pull/19478)).
- **(iii) The O(u²) terms are negligible.** This fails at overflow edges, where one side is ±inf. Those need a separate non-finite rule.
- **(iv) No branch divergence, or f is continuous across the branch.** Examples: scipy's `w < 709.78` switch, and Kahan's `u == 1` guard.
- **(v) Underflow adds an absolute floor**, τ ≈ (n+1)·2^-1074/scale. Under gradual underflow, + and − are exact when the result is subnormal, and × errs by at most 2^-1075 absolute.

**Consequence: nodes shared bit-for-bit contribute nothing to S.** This is why Box-Cox (shared glibc `log`; only `expm1` is spelled) keeps a 4-ulp bound, while Yeo-Johnson (its `log1p` differs) does not.

### What K is

**Matvec (DERIVED).** Any evaluation tree of n products, FMA or not, sends each term through at most n roundings. So |fl − exact| ≤ γ_n·Σ|x_i c_i| per side.
- This is Higham (3.5). Jeannerod & Rump (SIMAX 34(2), 2013) sharpen it to n·u for "any evaluation order" ([SIAM](https://epubs.siam.org/doi/abs/10.1137/120894488); abstract via search, the PDF was blocked).
- Add the −M and ÷scale roundings and both sides: **|T−N| ≤ (2γ_n + 2γ_2(1+γ_n))·S ≈ (n+2)·eps·S**, with S = (Σ|x_i c_ki| + |M_k|)/scale_k.
- If M is computed on a different BLAS than the twin's (translation on machine A, twin served on machine B), replace |M| with Σ|m_i c_ki| ("S_full"); K then becomes 2n+2.

**The typical K is about √n.** Under the model of independent, mean-zero rounding errors, γ_n becomes ~λ√n·u with high probability (Higham & Mary, SISC 41(5), 2019, [doi](https://epubs.siam.org/doi/10.1137/18M1226312)). Kahan's 1996 "Improbability of Probabilistic Error Analyses" ([pdf](https://people.eecs.berkeley.edu/~wkahan/improber.pdf), not fetched) argues that rounding errors are correlated in structured data. The structured row in §2.2 is such a case: the measured K reaches (n−2)/8.

## 2. Measurements (x86-64 AVX-512, numpy 2.5.1, OpenBLAS 0.3.33, glibc 2.39, DuckDB 1.5.5)

### 2.1 PCA(whiten=True), catalog fixtures, seeds 0–199 (`e1_gen.py`, `e1_twin.py`, `e1_analyze.py` → `e1_out_200.txt`)

Kernels were selected with `OPENBLAS_CORETYPE`. Only five kernels are distinct: Cooperlake and SapphireRapids fall back to SkylakeX, Zen to Haswell, Atom and Barcelona to Nehalem, Core2 and Prescott to Katmai. Of those five, Sandybridge, Nehalem and Katmai give bit-identical answers. Thread count makes no difference: 4 threads and 1 thread are identical.

| pair (25,193 finite lanes) | lanes differ | max ulps | p99 ulps | p99 K | max K |
|---|---|---|---|---|---|
| entry (left to right) vs twin[SkylakeX] | 4,819 | **9,877,709,850** | 10 | 1.12 | **2.99** |
| entry vs twin[Haswell] | 4,686 | 9,877,709,850 | 9 | 1.11 | 2.99 |
| entry vs twin[Sandybridge = Nehalem = generic] | 3,754 | 9,877,709,850 | 7 | 1.09 | **3.10** |
| twin[SkylakeX] vs twin[Haswell] (S_full) | 680 | 568 | 1 | 0.67 | 1.76 |
| twin[SkylakeX] vs twin[Sandybridge] (S_full) | 8,418 | 2,026 | 18 | 1.18 | 2.86 |
| twin batched vs row-by-row, same machine (`e1_batch.py`) | 4,696 of 25,208 | **9,877,709,850** | – | – | 2.99 |
| twin vs exact (Fractions, S_full) | – | – | – | 0.98 | 1.94 (Sandybridge 2.16) |
| entry vs exact (S_full) | – | – | – | 1.10 | 3.04 |
| twin vs correctly rounded, in ulps (`e1_ulps_exact.py`) | 11,848 not exact | **1.28e14** | 19 | – | – |

**Other results:**
- The projected-mean constant M differs across BLAS kernels on 10,586 of 25,348 lanes.
- There are 0 sign flips in every pair.
- Width matters (all at 200 seeds): K ≤ 1.76 for n ≤ 4, ≤ 1.64 for n in 5–16, and ≤ 2.99 for n in 17–32.

**Checking the matvec record's numbers:**
- Max ulps reproduce exactly: 9,877,709,850 at 200 seeds and 1,306 at 40 seeds.
- Max K 2.99 and p99 1.10 reproduce at 200 seeds (mine: 2.99 and 1.12).
- My lane counts are higher: 25,193 against the record's 23,050, and 4,611 against 3,855 at 40 seeds. At 40 seeds my max K is 2.29 against the record's 1.76. I could not explain the difference in lane selection.

### 2.2 Is K a constant? (`e4_kn.py`, synthetic, the twin called row by row on SkylakeX)

Rows sit near a positive mean, so x·c and M cancel. These lanes use components with all-positive entries ("positive c").

| n | 4 | 16 | 32 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|---|---|
| max K | 0.74 | 1.33 | 1.54 | 2.80 | 5.64 | 10.3 | 17.0 |
| p99 K | 0.45 | 0.72 | 1.02 | 1.80 | 3.45 | 7.41 | 13.0 |

With mixed-sign components, K stays ≤ 1.5 at every n.

**Structured row.** Take x = [1, 2^-53 × (n−2), −1] with c = 1. The left-to-right sum gives 0.0 (each half-ulp addend is lost to round-half-even). The twin gives exactly half the true sum, on both SkylakeX and Sandybridge. So K = (n−2)/8: 3.75 at n = 32, 127.75 at n = 1024, with a 100% relative difference.

### 2.3 AdditiveChi2Sampler (`e2_draws.py`, `e2_twin.py` under `NPY_DISABLE_CPU_FEATURES`, `e2_analyze.py` → `e2_out.txt`)

There are 599,999 values of x: uniform on (0, 1e3), log-uniform on (1e-300, 1e300), and 1 + N(0, 1e-3). That gives 12 lanes per x over the default (s, j).

**Which numpy kernels differ from glibc** (`numpy.lib.introspect.opt_func_info` reports the kernel in use):
- numpy `log` on X86_V4 (AVX-512) differs from glibc on 3,223 values (max 1 ulp).
- On X86_V3 and baseline, `log` is bit-equal to glibc. So are `log1p` and `expm1`.
- `np.cos` and `np.sin` equal glibc on 1,000,000 draws under both V4 and baseline.
- `np.cosh(π·j·s)` on V4 is 1 ulp from glibc for 3 of the 6 default (s, j) (`e2_cosh.py`).

| pair | max ulps (worst lane type) | max K, S = f·(1+\|θ\|) |
|---|---|---|
| twin[V4] vs entry (glibc) | 7,482 (sin, s=0.5, j=1) | **1.031** (sin, s=0.4, j=3) |
| twin[V4] vs twin[V2] | 7,482 | 1.031 |
| twin[V2 baseline] vs entry with glibc cosh (`e2_pinned.py`) | **0 of 7,199,988 lanes differ** | 0 |
| twin vs exact, and entry vs exact (Decimal, 60 digits, 30k subsample) | the twin itself is up to **7,130,376 ulps** from correct | ≤ 1.38 on both sides |

The chi2 record said "K = 1 holds with room". It does not hold: K reaches 1.031 here.

### 2.4 Yeo-Johnson, standardize=False (305 λ × 3,000 x = 915,000 lanes)

The entry here spells Goldberg's log1p and Kahan's expm1 on glibc. S = (1+|w|)|t|, with w = λ·log1p(x).

| pair | lanes differ | max ulps | max K |
|---|---|---|---|
| twin[V4] vs entry | 233,077 | 128 | **3.00** |
| twin[V4] vs twin[V2] | 79,049 | 63 | 2.47 |
| twin[V2] vs entry | 261,922 | 128 | 3.00 |
| vs exact (30k subsample) | twin 137 ulps from correct | – | twin 1.40, entry 2.17 |

Pinning numpy's kernels does not make Yeo-Johnson exact: the spelled log1p and expm1 are not glibc's.

### 2.5 DuckDB (`e3_duckdb.py`, `e3_sum.py`)

- **Row-wise sums match Python.** The entry's `x0*c0 + x1*c1 + …` matches Python's left-to-right result on all 7,868 PCA lanes tested (no FMA, no reassociation).
- **`SUM(DOUBLE)` over a 20M-row native table is not reproducible with more than one thread:**
  - with threads = 2 or 4, 8 runs gave **8 distinct results** (relative error against the exact sum ≤ 6.5e-13);
  - with threads = 1 it is deterministic.
- **`fsum` gave 2 distinct results in 4 runs** at threads = 2 and 4. That contradicts the claim in [duckdb discussion #12693](https://github.com/duckdb/duckdb/discussions/12693) that `kahan_sum` is deterministic.

### 2.6 What an allclose test would need (`e1_allclose.py`)

- **The needed atol depends on the data's units.** On the PCA fixtures with rtol = 1e-12, passing every lane needs atol = 3.5e-5. That figure is set by one cancelling lane with large terms, so it changes with the data's scale.
- **Low rtol makes the test vacuous on most lanes.** At rtol = 1e-15, the 1e300 edge rows force atol = 2.6e283, which passes 23,045 of 25,208 lanes whatever they hold.

## 3. How others state parity

| system | criterion | defaults / scope | source |
|---|---|---|---|
| numpy.testing | `assert_allclose`: \|a−d\| ≤ atol + rtol\|d\|. `assert_array_almost_equal_nulp`: \|x−y\| ≤ nulp·spacing(max) | rtol 1e-7, atol 0; nulp 1 | `numpy/testing/_private/utils.py:1671-1686, 1785-1820` (installed) |
| numpy's own accuracy tests | `assert_array_max_ulp` against a correctly rounded reference | float64 log, log1p, expm1, sin, cos, exp, tan, log10: **1 ulp**, on a validation set of a few hundred points; run only on Linux with AVX, "avoid testing inconsistent platform library implementations" | `numpy/_core/tests/test_umath_accuracy.py:22-27,83`; `data/umath-validation-set-*.csv` |
| numpy SVML | AVX-512 kernels, Linux only, ≤ 4 ULP | 18 functions (1.22) | [1.22.0 notes](https://github.com/numpy/numpy/blob/main/doc/source/release/1.22.0-notes.rst), [PR #19478](https://github.com/numpy/numpy/pull/19478) |
| scikit-learn | `transform` on subsets: `assert_allclose(atol=1e-7)`; on reordered samples: `atol=1e-9`. Its testing helper says atol "should be adjusted manually … in case there are null values" | its own transform is **not** batch-invariant to the bit | `sklearn/utils/estimator_checks.py:2092-2125, 2160-2168`; `utils/_testing.py:175-190` |
| PyTorch `assert_close` | \|a−e\| ≤ atol + rtol\|e\| | (rtol, atol): f64 (1e-7, 1e-7), f32 (1.3e-6, 1e-5), bf16 (1.6e-2, 1e-5), f16 (1e-3, 1e-5) | [torch/testing/_comparison.py:65-75](https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py) |
| PyTorch determinism | "not guaranteed across PyTorch releases, individual commits, or different platforms … CPU and GPU" | – | [randomness notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html) (via search; fetch blocked) |
| ONNX backend tests | `np.testing.assert_allclose` | rtol 1e-3, atol 1e-7 | `onnx/backend/test/case/node/__init__.py:256`, `runner/__init__.py:194` |
| skl2onnx | a tree is "not a continuous function. Therefore, even a small dx may introduce a huge discrepency"; advice: keep the same types up to the tree | – | [plot_ebegin_float_double.py](https://github.com/onnx/sklearn-onnx/blob/main/docs/tutorial/plot_ebegin_float_double.py) |
| Intel oneMKL CNR | bitwise only within the same code branch (`MKL_CBWR`) and a constant thread count. STRICT mode drops the thread condition for gemm, symm, hemm and trsm only | – | [Intel CNR docs](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/reproducibility-conditions.html) (via search; fetch blocked) |
| OpenBLAS | no guarantee. The FAQ blames "specialized cpu instructions that combine multiplications and additions" and splitting work across threads. `OPENBLAS_CORETYPE` picks the kernel | – | [docs/faq.md](https://github.com/OpenMathLib/OpenBLAS/blob/develop/docs/faq.md) |
| ReproBLAS | order-independent binned summation, about 4× slower than MKL's dot at n = 4096 | – | [Ahrens, Demmel & Nguyen, EECS-2016-121](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.html) (via search) |
| IEEE 754-2019 §11 | reproducible results only for programs "translated into … reproducible operations on reproducible formats". Languages should let users forbid value-changing optimizations (reassociation, contraction). Elementary functions are reproducible only when correctly rounded (§9.2) | – | secondary sources only, e.g. WG21 P3375; **the standard's text was not fetched** |
| CORE-MATH | correctly rounded libm. glibc ≥ 2.43 takes 7 binary64 functions from it (acosh, asinh, atanh, erf, erfc, lgamma, tgamma), **not** log, exp, cos | – | [core-math](https://core-math.gitlabpages.inria.fr/) (via search) |

**What these practices share:**
- Test suites use allclose: a tolerance, tuned per suite, not a promise.
- Vendors promise bitwise results only on a pinned code path, thread count and platform.
- Nobody promises parity across kernels in ulps of the result.

## 4. The five alternatives

| option | holds whatever the data? | checkable by `native.check` | promise to a consumer | how it fails |
|---|---|---|---|---|
| (a) K·eps·S, **K derived** (e.g. n+2) | **yes**, given conditions (i)–(v) | yes: one S per lane from est and row, NaN-safe | "within rounding of the twin, scaled by the lane's condition"; S = \|y\| on well-conditioned lanes (≈ K ulps) | 100% relative difference on cancelling lanes (it is the twin's own noise); a loose K at large n still catches every non-rounding bug |
| (a′) K·eps·S, K measured (3, 1) | **no**: 3.10 (Sandybridge), 3.75 (structured row), 1.031 (chi2) | yes | same, but can be breached by the next seed, the next CPU or a wider n | the record's own objection to option 2 applies |
| (b) allclose atol+rtol per family | no: atol depends on the data's units (3.5e-5 here, 2.6e283 with edge rows) | yes | none that scales | vacuous on small lanes, fails on large-term lanes |
| (c) ulps with an absolute floor | no (it is (b)) | yes | same as (b) | same as (b) |
| (d) pinned twin (baseline numpy, fixed OpenBLAS kernel and threads) | yes for chi2 (**bit-exact**), no for matvec (no BLAS kernel is left-to-right) or Yeo-Johnson | needs a subprocess (environment variables must be set before import) | parity with a *reference* twin, not with the deployed AVX-512 twin (up to 7,482 ulps from it for chi2, 2,026 for PCA) | changes what "twin" means; good only as a CI regression guard |
| (e) correctly rounded reference, each side within k·eps·S | yes (it is (a) split in two) | Fractions for matvec (~0.3 ms per lane); Decimal for log/cos (~50 µs+); mpmath is not installed | "as accurate as the twin" (measured K vs exact: PCA twin 2.16, entry 3.04; chi2 1.38; Yeo-Johnson 1.40 / 2.17) | cost; still needs S (the twin is 1.3e14 ulps from exact) |

## 5. Proposed amended ruling (closes matvec, power and chi2 together)

> A native entry is bit-exact wherever it performs the twin's operations in the twin's order. Otherwise, per lane, `|entry − twin| ≤ K·eps·S(x) + τ`. Each family declares:
>
> - **S, the lane's error scale:** a formula over the row's inputs and the fitted state. It is the condition-weighted magnitude of the operations whose rounding may differ between twin and entry. For a lane that is one rounding of one kernel, S = |lane|, which is today's ulp bound.
> - **K, derived** from the per-operation error bounds of both sides (IEEE operations; documented ulp bounds of libm and numpy kernels). K may depend on the lane's width.
> - **A measurement on the catalog's fixtures** whose maximum is at most K/2.
>
> Where either side, or S, is not finite, the lane compares as with bound 0 (NaN = NaN; ±inf by sign; NULL = NULL). τ is the declared underflow floor. A family that cannot state a finite S and K for a configuration does not serve it.

**What each record must then declare:**

| family | S | K (derivation) | measured |
|---|---|---|---|
| matvec (PCA and kin) | (Σ\|x_i c_ki\| + \|M_k\|)/scale_k. Use Σ\|m_i c_ki\| in place of \|M_k\| if M may come from another BLAS | **n + 2** (γ_n per side + 2 roundings); 2n+2 with S_full | ≤ 3.10 at n ≤ 32; ~0.3√n typical |
| chi2 | f·(1+\|θ\|), with f = √(2xs/C) and θ = j·s·ln x | per side max(2k_L+2, 2k_c+3)·u. k = 1 everywhere: **K = 5**; numpy's documented SVML k_L = 4: K = 8 | 1.031 |
| Yeo-Johnson | (1+\|w\|)·\|t\| (κ_expm1(w) ≤ 1+\|w\|) | twin (r_log1p + r_expm1 + 2u)(1+\|w\|) ≈ 6u; entry: Goldberg log1p ≤ ~5–6u and expm1 ≤ 3 ulp → ≈ 14u. Together **K ≈ 10** (first order; the expm1 term is measured, not proven) | 3.00 |
| any family + standardize | (S_t + \|μ\|)/σ | K_t + 2 | Box-Cox 362 ulps / Yeo-Johnson 1,650 ulps on fixtures (records); K not measured here |

## 6. Risks to state in the ruling

- **Cancelling lanes.** A lane that cancels to about 0 can differ by 100% relatively, or be 0.0 against 8.9e-16. Downstream comparisons, splits, bin edges and sign tests can flip.
  - For a split at τ, P(flip) ≈ p_lane(τ)·K·eps·S per row: negligible, but not zero.
  - The twin already has this property across CPUs (8,418 of 25,193 PCA lanes differ between kernels) and between batched and single-row calls.
  - Measured sign flips: 0 in every comparison here.
- **Compositions.** If a bounded lane feeds a discontinuous step (a tree, a binner), the composition is not covered. Suggested rule: the entry may take the other branch only where the twin's value is within K·eps·S of the threshold.
- **"Constants" that are not constant.** M = mean_·components_ᵀ and cosh(πjs) are computed per machine by the twin. When the translation and the twin run on different CPUs, S must include their computation scale.
- **DuckDB on non-glibc hosts.** DuckDB's `ln`, `exp` and `cos` are glibc's only on Linux/glibc, so "entry = DuckDB" is per-platform. This follows from IEEE 754 §11 and glibc's documented goals; I did not measure it.

## Not confirmed

- The text of IEEE 754-2019 §11, Higham's equation numbers, the Jeannerod–Rump / Higham–Mary / Kahan PDFs, and the Intel, PyTorch and DuckDB doc pages were all egress-blocked. Those claims rest on search-index abstracts or excerpts.
- The glibc per-function ulp table was not fetched; its path 404'd.
- I could not explain why my fixture lane counts are higher than the record's.



## The researcher's recommendation

Amend the owner ruling from "a small per-family ulp bound" to "a per-family bound in ulps of the lane's error scale": per lane, |entry − twin| ≤ K·eps·S(x) + τ. S is a declared, cheap formula over the row and the fitted state: the condition-weighted magnitude of the operations whose rounding can differ between the two sides. Today's ulp bounds are the special case S = |y|, so the exact and ulp-bound families are unchanged. K must be *derived* from per-operation error bounds and may grow with width: n+2 for matvec, about 5–8 for chi2, about 10 for Yeo-Johnson. The fixture measurement confirms it with at least 2× headroom; it does not fix it. A measured constant is already falsified: 3.10 on a Sandybridge-class BLAS kernel and 3.75 on a structured n = 32 row, against the proposed PCA bound of 3, and 1.031 against chi2's proposed 1. Reject allclose and absolute floors: their atol depends on the data's units (3.5e-5 on these fixtures, 2.6e283 with the edge rows). Use the pinned reference only as an extra CI guard where it buys bit-exactness: chi2 is bit-exact against a baseline-numpy twin over 7.2M lanes. Use the correctly rounded reference as an occasional audit, not the gate. The decisive fact for the owner: the twin itself is up to 1.3e14 ulps from the correct PCA lane, and it differs from itself across BLAS kernels and between batched and single-row calls by the same K ≈ 3 the entry shows. The S-bound therefore promises exactly what the twin can promise about itself. The ruling should add two clauses: lanes where either side or S is not finite must match exactly, and a discontinuous consumer may branch differently only where the twin's lane is within K·eps·S of its threshold.


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| The twin is far from the correctly rounded lane (PCA up to 1.28e14 ulps, p99 19, p99.9 148, 25,208 lanes; chi2 7,130,376 ulps; Yeo-Johnson 137 ulps), so a small ulp bound against it can't be met. | **PARTLY** | I regenerated the PCA(whiten) fixtures myself (verify-framework/v1_twin.py, v1_analyze.py, v1_report.py; fixtures.py is a verbatim copy of catalog_test.py:596-835 apart from the PythonTransform shim) and computed an exact Fraction reference. Result: 25,208 lanes, 11,848 not exact, max 1.28e14 ulps, p99 19, p99.9 148, the same as the report. I did not recompute the chi2 and Yeo-Johnson figures with Decimal, but e2_out.txt shows them and the method looks sound. The problem is the inference. On the worst PCA lane the twin and the entry are bit-identical (-0.00021315488712691233 on both; exact -0.0002166). The 1.28e14 ulps is shared error, so it says nothing about whether an ulp bound against the twin can be met. Part of that error is the rounding of the constant M, which both sides share: measured against the exact lane given the twin's M, p99 falls from 19 to 6, the max to 8.95e13, and not-exact lanes to 8,680. For chi2 and Yeo-Johnson, entry and twin have the identical error against exact (7,130,376 and 136.8 ulps on the same lanes). That comes from the problem's conditioning, not from the twin. | The twin is not ulp-accurate. But the evidence that a small ulp bound can't be met is entry vs twin (9.9e9 ulps), not twin vs exact. The 1.3e14 lane is one where entry equals twin exactly. |
| The twin isn't reproducible either. Across OpenBLAS kernels: up to 2,026 ulps, 8,418 of 25,193 lanes, max K 2.86 in units of the term scale. Batched vs row-by-row: 9,877,709,850 ulps, K 2.99, same worst lane. sklearn's checks accept this with assert_allclose(atol=1e-7). | **PARTLY** | The numbers reproduce. SkylakeX vs Sandybridge: 8,418 lanes differ, 2,026 ulps, K 2.861. Batched vs row-by-row: 4,696 lanes differ, 9,877,709,850 ulps, K 2.992, and the batched result equals the entry's value on that lane. Three problems. (a) K 2.86 uses S_full (Σ\|m_i c_i\|). With the record's own term scale (\|M\|), twin[SkX] vs twin[SB] reaches max K 237.6 (p99 5.78), because M itself moves with the kernel. (b) The served twin never batches: _udf.py:316 calls est.transform([vals])[0] one row at a time, so batch variance does not happen in serving. (c) sklearn's subset check (estimator_checks.py:2125) uses atol 1e-7 plus sklearn's default float64 rtol of 1e-7. The worst lane differs by 3.5e-5 absolute (2e-6 relative, against a tolerance of 1.85e-6), so that check would reject it. It passes only on its benign 20x3 uniform data. | The twin varies across kernels at K ≤ 2.9 only when S_full is used; with the record's S it is K ≈ 238. sklearn's tolerances would not accept the 9.9e9-ulp lane. Batch variance is not a property of the served twin. |
| The matvec record's numbers reproduce (9,877,709,850 ulps, K 2.99), but K = 3 is not a bound: Sandybridge reaches 3.10; the structured row gives K = (n-2)/8 = 3.75 at n = 32; max K grows about 0.2-0.37·sqrt(n) (2.8 at n = 128, 17.0 at n = 8192). | **CONFIRMED** | Independent rerun: max 9,877,709,850 ulps, max K 2.992, p99 K 1.124 on 25,193 lanes; entry vs the Sandybridge twin gives K 3.096. The structured row reproduces with 2 components: the twin returns 0.500 of exact, so K = 3.75 at n = 32 on both SkylakeX and Sandybridge. With 1 component the twin returns about 0.97 of exact, giving K ≈ 7.25 at n = 32 and about 248 at n = 1024, so (n-2)/8 understates the structured case. On real PCA(whiten) fits of wide, correlated, positive data (v2_kn.py), max K is 2.31 at n = 128, 3.32 at n = 256 and 5.41 at n = 1024, about 0.2·sqrt(n). Synthetic positive components (v7_wide.py) give 8.7 at n = 2048 and 15.7 at n = 8192. I did not rerun the 1,306 ulps at 40 seeds. | The structured worst case is closer to (n-2)/4 than (n-2)/8 (1-component call). |
| Any order, with or without FMA, errs by at most γ_n·Σ\|x_i c_i\| per side, so \|twin - entry\| ≤ (2γ_n + 2γ_2(1+γ_n))·S ≈ (n+2)·eps·S. With M from another BLAS, use Σ\|m_i c_i\| and K becomes 2n+2. | **PARTLY** | Line-by-line rederivation: per side ŷ = (d̂ - M)(1+δ1)(1+δ2)/s with \|d̂ - d\| ≤ γ_n·A, so the error is ≤ (γ_n + γ_2 + γ_nγ_2)·S. Two sides give ≈ 2(n+2)u = (n+2)·eps. That part is correct, given that M is the same double on both sides; sklearn's _transform (subtract M, then divide by sqrt(explained_variance_) clipped at eps) matches. The S_full case is double-counted. \|Δd̂\| ≤ 2γ_n·A and \|ΔM\| ≤ 2γ_n·B, so the total is 2γ_n(A+B) = n·eps·S_full·scale, and K stays n+2, not 2n+2. Also, 'measured 3.04 (entry vs exact) is well inside K = 34' compares a one-sided measurement with the two-sided bound; the one-sided bound is (n+2)/2 = 17. Jeannerod-Rump's n·u for any order is as cited; the γ_n bound with FMA follows from counting roundings. It assumes no underflow or overflow. | K = n+2 holds with S_full too (2n+2 is a factor-2 over-estimate). Compare entry vs exact against (n+2)/2. |
| chi2's whole gap comes from numpy's AVX-512 kernels: np.log differs from glibc on 3,223/599,999 (max 1 ulp); np.cosh is 1 ulp off on 3 of the 6 default (s,j); a baseline/V3 twin is bit-exact against the entry on glibc (0 of 7,199,988); on AVX-512 max K is 1.031, breaking 'K = 1 holds with room'. | **PARTLY** | Independent check (v3_*.py): 750k new draws, the twin being sklearn's AdditiveChi2Sampler.transform itself, the entry spelled in DuckDB SQL (not Python math), 7 configurations up to j = 7. np.log[V4] vs DuckDB ln: 2,588 of 750,000 differ, max 1 ulp. np.log[V2] vs DuckDB ln: 0 differ, and DuckDB ln equals math.log. np.cosh[V4] differs from glibc for (0.8,1), (0.4,1), (0.4,2) and 7 other pairs; np.cosh[V2] differs for none. The V2 twin vs the DuckDB entry with glibc cosh: 0 lanes differ in every configuration, which confirms bit-exactness. V4 max K: 1.043 at (s=0.4, j=3), 1.61 at (0.3, 5), 1.39 at (0.7, 5). But sklearn loops j over range(1, sample_steps) (kernel_approximation.py:795). So the actual defaults are (0.5,1), (0.4,1) and (0.4,2), where max K is 0.904 here; the report's own e2_out.txt max on them is 0.745. The 'six default (s,j)' is an off-by-one inherited from the chi2 record. The 1.031 breach needs sample_steps=4 with an explicit sample_interval. | K = 1 is broken for the family (up to 1.61 on legal non-default configurations) but held on the true defaults in both measurements. cosh is off on 2 of the 3 real default pairs. |
| Comparable systems state parity as test tolerances (numpy rtol 1e-7/atol 0; PyTorch f64 (1e-7,1e-7), f32 (1.3e-6,1e-5); ONNX rtol 1e-3/atol 1e-7; sklearn atol 1e-7); bitwise only on pinned paths (MKL CNR); OpenBLAS no promise; glibc not correctly rounded, 'a few ulp', test flags >9 ulp; numpy SVML ≤4 ULP, AVX-512 Linux only. | **CONFIRMED** | numpy utils.py:1671 has rtol=1e-7, atol=0. PyTorch _DTYPE_PRECISIONS is as quoted (around lines 55-68 on main, not 65-75). ONNX node/__init__.py on main: rtol 1e-3, atol 1e-7 (lines 267-268). sklearn estimator_checks.py:2125 uses atol=1e-7. glibc math.texi verbatim: 'does not aim for correctly rounded results', 'within a few ulp', 'only flags results larger than 9ulp'. numpy PR #19478: 'maximum ULP error across all the implementations is 4 ULP', Linux only. The Intel CNR page was egress-blocked; search snippets are consistent. The OpenBLAS FAQ's FMA sentence is about LAPACK test noise, so 'no promise' is an inference from silence. | Nuance that matters for chi2's K = 8: the numpy 1.22 SVML list (exp2, log2, log10, expm1, log1p, cbrt, trig, hyperbolic) does not include log or exp. np.log's AVX-512 f64 kernel is not SVML, so the 4-ULP figure does not document it. numpy's own f64 validation tests hold log/log1p/expm1/sin/cos/exp/cosh to 1 ulp, on 147-714 points. |
| An allclose bound depends on the data's units: rtol 1e-12 needs atol 3.5e-5; at rtol 1e-15 the 1e300 rows force atol 2.6e283, vacuous on 23,045 of 25,208 lanes. | **CONFIRMED** | v6_allclose.py on my own lanes: rtol 1e-12 needs atol 3.51e-5; rtol 1e-15 needs 2.55e283, with 23,045 of 25,208 lanes having \|twin\| below it. Excluding lanes with S > 1e100, rtol 1e-15 needs atol 320. Context the report omits: numpy's or sklearn's default assert_allclose (rtol 1e-7, atol 0 or 1e-7) fails on exactly 1 of 25,208 lanes, the 9.9e9-ulp lane. The S-bound is itself an allclose with rtol 0 and a per-lane atol of K·eps·S(x). The real difference is a data-derived atol, not 'no tolerance'. |  |
| DuckDB evaluates x0*c0 + x1*c1 + ... exactly like Python left-to-right (7,868 lanes, 0 differ); SUM(DOUBLE) over a 20M-row native table gave 8 distinct results in 8 runs at threads 2 and 4; fsum gave 2 distinct in 4 runs, contradicting the forum claim; the oracle makes no promise for float aggregates. | **CONFIRMED** | v8_duck_l2r.py: 12,708 full lanes, including the -M and ÷scale steps, differ on 0 (no FMA, no reassociation). v5_sum.py on a 16M-row native table: threads=1 is deterministic; at threads 4 and 8, SUM gives 8 distinct results in 8 runs, fsum 2 distinct in 8, kahan_sum 2 distinct in 4 (at 4 threads). Discussion #12693: the 'kahan_sum is deterministic' line is the issue author's own test, not a maintainer statement. I did not check the DuckDB docs behind 'makes no promise'. This bears little on the decision, since the entries are row-wise expressions. | The determinism claim came from the poster, not from DuckDB maintainers. |
| Yeo-Johnson with S = (1+\|λ·log1p(x)\|)·\|t\|: entry vs twin max K 3.00 over 915,000 lanes (128 ulps); twin AVX-512 vs baseline K 2.47 (63 ulps); pinning numpy does not make it exact because DuckDB has no log1p/expm1; derived first-order K ≈ 10. | **PARTLY** | sklearn 1.9 PowerTransformer.transform calls scipy.stats.yeojohnson (_data.py:3470), and scipy uses expm1(λ·log1p(x))/λ, so the twin used in the report is right. Independent rerun (v4_*.py): 836,000 new lanes including \|x\| ~ 1e-8, entry spelled in DuckDB SQL. Entry vs twin[V4] gives max K 3.335 (256 ulps); vs twin[V2] 3.482; V4 vs V2 2.584 (110 ulps). So the measured 3.00 is specific to the report's draws and is exceeded, consistent with the report's own thesis that measured constants are not bounds. DuckDB 1.5.5 has no log1p or expm1 (checked in duckdb_functions()), and the V2 twin still differs on 254,567 lanes. The K ≈ 10 is first-order and rests on an expm1 error that was measured, not proven, and on per-function libm ulps that glibc does not document. | Measured max K is at least 3.48. K ≈ 10 is an estimate, not a derived bound. |

### What the report missed or got wrong

1. S vs S_full is the biggest gap. The entry bakes in M = mean_·components_ᵀ at translation time, while the twin recomputes M on every call on whatever host serves it (sklearn _transform). M differs across OpenBLAS kernels on 10,586 of 25,348 constants. With the record's term scale (|M|), I measured max K = 237.6 (p99 5.78) for entry(M from SkylakeX) vs a Sandybridge twin, and 19.8 vs a Haswell twin. The report's headline that the twin varies at K ≤ 2.9 depends on quietly switching to S_full. S_full has to be the default S for matvec, not an optional variant. With S_full, K = n+2 still holds; the report's 2n+2 double-counts the two error terms.

2. The 'decisive fact' is mis-stated. The 1.3e14-ulp twin lane is bit-identical to the entry, and much of the twin's distance from exact is the rounding of M, which both sides share. Twin-vs-exact figures do not bear on parity.

3. The chi2 'default (s,j)' list is off by one, an error inherited from the record. sklearn loops j over range(1, sample_steps), so the defaults are (0.5,1), (0.4,1) and (0.4,2), where K ≤ 0.90 in both measurements. The breaches are on legal non-default configurations, and are larger than reported: up to 1.61 at (s=0.3, j=5). cosh differs on 2 of the 3 real default pairs.

4. The served twin is row-by-row (_udf.py:316), so batched-vs-row variance never occurs in serving. sklearn's own subset-check tolerance (atol 1e-7, rtol 1e-7) would reject that 3.5e-5 discrepancy, so 'sklearn accepts this' is wrong.

5. For the transcendental families, 'K derived, holds whatever the data' is overstated:
- Nothing in glibc guarantees per-kernel ulp bounds: the manual says 'a few ulp' and the tests flag only errors above 9 ulp. With k = 9, chi2's formula gives K ≈ 21, not 5.
- np.log's AVX-512 float64 kernel is not one of the SVML functions in the 1.22 notes, so the 4-ULP figure behind 'K = 8' is borrowed from other kernels.
- For Yeo-Johnson, K ≈ 10 rests on a measured expm1 error.

Only matvec gets a true derived bound.

6. The proposed rule that lanes with a non-finite side or S must match exactly can be broken by rounding alone near overflow. With x = [1e308, 1e308, 0, ..., -1e308], the OpenBLAS twin returns 1e308 for n ≥ 8, while left-to-right returns inf. The report flags overflow edges in its condition (iii) but then writes an exact-match rule that legitimate entries will fail.

7. The structured-row K is understated. A 1-component call gives K ≈ 7.25 at n = 32, about (n-2)/4, not the 2-component (n-2)/8.

8. Yeo-Johnson's measured max K is draw-dependent: 3.48 on my draws vs 3.00.

9. Minor: the PyTorch and ONNX line numbers have drifted on main, and 'measured 3.04 inside K = 34' should be compared against the one-sided 17.


### Does the recommendation follow? Yes

The core recommendation follows from the evidence. The evidence is: no small ulp bound exists, measured constants keep getting exceeded (3.10, 3.48, 1.61, about 0.2·sqrt(n) growth on real wide PCA), and allclose's atol depends on the data's units. Given that, a per-lane bound |entry - twin| ≤ K·eps·S(x) + τ, with S declared per family and K derived where possible, is the right shape. For matvec it is rigorous (K = n+2), and my reruns confirm it holds with large headroom.

It needs these amendments before an owner ruling:
- Make S_full (Σ|x_i c_i| + Σ|m_i c_i|) the matvec S, because M is baked in at translation and recomputed by the twin on the serving host. With |M| I measured K = 238. Keep K = n+2 (not 2n+2).
- Drop the 1.3e14 'decisive fact', which is a lane where entry and twin agree. Replace it with the cross-kernel variance, measured with S_full.
- State plainly that for chi2 and Yeo-Johnson, K is derived from assumed kernel ulp bounds that glibc and numpy do not guarantee. Either declare the assumed per-kernel ulps, so they are checkable by a probe like function.kernel_distance, or call K provisional.
- Fix the chi2 default (s,j) set.
- Replace 'non-finite must match exactly' with a rule that tolerates overflow-edge disagreement: allow one side ±inf only where the other's magnitude is within K·eps·S of DBL_MAX.

The pinned-baseline chi2 guard is well supported: bit-exact against a DuckDB-spelled entry over all the configurations I tried. Keeping the exact reference as an audit, not the gate, is reasonable.
