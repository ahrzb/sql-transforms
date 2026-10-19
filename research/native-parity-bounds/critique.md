> Completeness critique over all the research notes and their verifications, 2026-10-06. It settles the contradictions between notes and lists what none of them answered.

# Completeness critique: open decisions of the native loop (repo at 113fba7)

The critic scripts are in `scripts/critic/` (c1–c8). They are listed at the end.

## 1. Contradictions between reports

| # | Contradiction | Settled? | Resolution |
|---|---|---|---|
| 1 | **Which float64 `log` numpy runs.** The framework verifier says it is not SVML. The distances report says it is numpy's own AVX512F (Tang) kernel. The chi2 and power reports, and three verifiers, say it is SVML `__svml_log8_ha`. | **Yes (c6)** | It is SVML. `opt_func_info` reports log as X86_V4. `np.log` differs from glibc on 27,529 of 1M draws with \|x−1\|<0.06; the Tang kernel hands that interval to glibc, so it would show 0 there. **Consequence:** framework's chi2 K=8 uses the "4 ULP" figure for numpy 1.22's low-accuracy kernels, which does not apply. The HA tier applies: ≤1 ulp claimed, ~0.6 measured. **Caveat:** a GCC/clang build on AVX-512 without SVML (macOS x86, `-Ddisable-svml`) runs Tang, and nobody measured Tang's distance from glibc. |
| 2 | **chi2 K=1 at sklearn's real defaults.** The framework verifier says it held there (0.904). The chi2 report (1.241) and its verifier (1.222) say `sample_steps=3` breaks it. | **Yes (c7)** | It is breached. I took 459 values of x where `np.log` ≠ glibc (random mantissa, \|ln x\| up to ~690). At `sample_steps=3` (s=0.4, j=1,2) the max K is **1.143**, with 19 of 1,836 lanes above 1. At `sample_steps=2` (s=0.5) it stays at 0.886. This matches the theory: A_sup = 2/σ(0.4) = 1.25. The framework's draws did not include log mismatches at large \|ln x\|. Both the record and the framework report also list the default (s, j) pairs off by one. |
| 3 | **Thread count.** The matvec and framework reports say it never changes the twin; the matvec verifier says it does. | **Yes (source)** | OpenBLAS 0.3.33 `interface/gemv.c:127` keeps gemv single-threaded while m·n < 115200·4 = 460,800, and every "no effect" test was below that. Above it the twin changes with thread count. This does not matter for catalog fixtures (m·n ≤ 1,024), but it does for wide data. The derived bounds still hold, because threaded orders are still summation trees. |
| 4 | **Matvec K under the full scale.** The framework report says 2n+2; the matvec report and framework verifier say n+2. | **Yes (algebra)** | It is n+2: the two dots differ by ≤ 2γₙΣ\|x·c\| and the two copies of M by ≤ 2γₙΣ\|m·c\|, about n·eps·S_full together, plus ~2·eps·S for the subtraction and division. The framework report counts this twice. Declare **n+3**, which also absorbs the O(nu) terms and the rounding in check's own S. |
| 5 | **Default S for matvec.** The matvec report uses S2 = S_full. Its verifier says the record's S never fails inside check. The framework report keeps \|M\| as the default; its verifier says S_full must be the default. | **Needs an owner ruling (§2.2)** | All four are factually right. Inside one process M is bit-identical, so every pair stays ≤ 4.12 under either S. The twin, though, recomputes M on every call (`-= mean_ @ components_.T`), and M differs between BLAS kernels on 10,586 of 25,348 constants. Across hosts the record's S gives K = 237.6 at 200 seeds and 1,730.8 at 1,000. S_full ≥ S_rec and costs no detection power, so use S_full unless the contract is same-host only. |
| 6 | **chi2 K.** The chi2 report says 3, from max(d·A, κ). The framework report says 5 (k=1) or 8. | **Partly** | The chi2 derivation is sharper: it uses the one-double distance between the two logs directly. The framework's 8 rests on row 1's wrong kernel. But K=3 is tight. Legal configurations reach 2.57 (s=0.2501, j=3) and 2.80 (j=17). κ becomes 3 if glibc's cos and sin are 1 ulp (libm-test-ulps) rather than 0.55. A cosh constant computed under a different dispatch adds about 1. The framework's own rule that the measurement be ≤ K/2 would force K ≥ 5.6 on those configurations. **Choice:** a per-(s, j) K from the formula (the registry already supports per-estimator bounds), or a flat K ≈ 5 with the cosh constant's source pinned. |
| 7 | **Rule for non-finite lanes.** Matvec: fail if exactly one side is non-finite. Power: pass if \|w − ln DBL_MAX\| ≤ K·eps·(1+\|w\|). Framework: non-finite must match exactly. | **No; every version was refuted by its verifier** | Overflow on one side only can be plain rounding: OpenBLAS returns 1e308 where left-to-right gives inf for n ≥ 8, and `standardize` with `scale_` < 1 moves the overflow point away from w ≈ 709.78. A naively computed S also overflows to inf, and then every lane passes; this happened on one real fixture lane. **One rule is needed, stated on the output:** one side ±inf passes only if the other side is finite, has the same sign and is within K·eps·S of DBL_MAX. Compute S scaled (with frexp) so it cannot overflow. NaN matches only NaN. Separately, the entry's Kahan expm1 returns NaN at overflow and needs a fix. |
| 8 | **"Measured K grows like c·√n."** Matvec says ≤ 0.7√n, distances ~0.2√n, framework 0.2–0.37√n. | **Yes** | None of these is an envelope. At small n the constant term dominates (K 1.86 at n=3). Mixed-sign data do not grow at all, while same-sign data grow about n^0.4. A constructed n=32 row reaches K ≈ 7.25, about (n−2)/4. The derived n+2 is the only real bound, and it is within about 4× of what can actually be reached. |

Measured Yeo-Johnson maxima also disagree (power 2.84, framework 3.00, verifiers 3.48 and 3.91). They depend on the draws, and all sit under the derived 9–11.

## 2. Decision-relevant questions no report answered

1. **Can a bounded step be part of a composition? This decides most of option 1's payoff.**
   - Today `compose.py` (lines 14–18, 86–91, 232–236) refuses any step whose bound is not 0 anywhere in a Pipeline, ColumnTransformer or FeatureUnion, even as the last step.
   - So option 1 on its own still leaves these NotNative: `make_pipeline(StandardScaler(), PCA())`, StandardScaler→KMeans, and any bounded family inside a ColumnTransformer.
   - The owner needs to rule on three cases:
     - (a) **Bounded step last in a Pipeline, or a part of a union.** No new theory is needed: check computes S from the step's own input, which is the output of the bit-exact steps before it, computed in Python.
     - (b) **Bounded step feeding continuous steps.** S has to be carried through them (to first order, Σ\|∂y/∂v\|·S_v). Nobody derived this.
     - (c) **Bounded step feeding discontinuous steps** (KBins, Binarizer, spline knots, Isotonic, thresholds). No bound exists, so keep refusing.
2. **Is the contract same-host or cross-host?**
   - Same-host means translation, twin and DuckDB on one machine, as `check` runs them. Cross-host means SQL translated on CI and compared against a twin on the serving host.
   - This one choice decides:
     - S_rec vs S_full (row 5);
     - whether values the twin recomputes on every call can be treated as shared constants: PCA's M, KMeans's ‖c‖² via einsum, chi2's cosh(πjs) via SVML;
     - whether chi2's "declare K = 0 where `kernel_distance` reads 0" means anything, since that probe runs on the translating host.
   - There is a precedent: `function.py` already gates serving on `kernel_distance` measured at translation time, so today's contract is implicitly "on the translating platform".
3. **What happens when an assumed kernel accuracy does not hold?**
   - The derived K for power (9–11) and chi2 (3–5) assume glibc's ~0.52-ulp ln/exp and SVML-HA ≤ 1 ulp. Neither is guaranteed: glibc's manual promises "a few ulp" and its tests flag only errors above 9 ulp.
   - Undecided: should K be probe-gated, refusing to serve when the probe exceeds the assumed distance as `function.py` does? And are non-glibc DuckDB hosts in scope? With 1-ulp ln/exp the Yeo-Johnson constants become 10.5 and 12.5, above the recommended 10 and 12.
4. **check's API change was never sized.**
   - Today the bound is an integer: `Entry.ulps`, `per_estimator → int`, and `_same(a, b, ulps)`.
   - Option 1 needs a per-lane tolerance computed from:
     - the fitted estimator;
     - the row the step actually sees;
     - for distances, both outputs and a comparison map g (squares);
     - plus K(n).
   - That is per-family code reading sklearn internals, the same kind of "per-estimator copy" the tolerated record names as its obstacle.
   - check also silently skips rows where the twin raises. The sparse finding (periodic degree-0 spline compares 0 rows) and the tolerated finding (4.9% of gate rows skipped) both need tests to assert `compared == n`, or "raises iff the twin raises".
5. **Should already-served bounds be restated in the new form?**
   - Box-Cox is served at ulps=4 (`power.py:69`), and that figure is a measurement.
   - Critic c4/c5: over 400M targeted lanes, 692 sit at exactly 4 ulps and none at 5. The bound holds, but with no headroom; the first-order worst case is ~9 ulps.
   - FunctionTransformer's `_BOUNDS`, and the `kernel_distance` probe that gates them, use the biased `exp(uniform)` construction (`function.py:214`).
   - No report asked whether these move to "derived K, measured headroom" at the same time.
6. **Discontinuities downstream in the user's SQL.** Only the framework report raises this. A cancelling lane can come out 0.0 on one side and 8.9e-16 on the other, or flip sign. A consumer's `CASE WHEN lane > t` can then branch differently. check cannot verify this. Is it part of the contract, or only documented?
7. **Provenance of the matvec record's numbers.**
   - Its lane counts (3,855 and 23,050) were reproduced by nobody; everyone gets 4,618 and 25,208.
   - Its numbers match HEAD's generator, not b851298 as the record states.
   - c8 tried four filters; none reproduces both counts:

     | filter | seeds 0–39 | seeds 0–199 |
     |---|---|---|
     | drop rows with \|x\| ≥ 1e300 | 4,178 | 22,738 |
     | \|y\| < 2.6e283 | 4,258 | 23,047 |
     | finite lanes | 4,618 | 25,208 |
     | the record | 3,855 | 23,050 |

   - The record should state its lane filter and the commit it was measured on.
8. **Environment.** In this venv `import sql_transform` fails (CPython 3.14.0rc2 with pydantic 2.13.4), so the repo's pytest does not even collect. Every report worked around this with shims or verbatim copies. No claim about a bounded family went through the repo's own `check` end to end.

## 3. REFUTED and PARTLY verdicts, and what changes

**Matvec**
- **REFUTED: thread count makes no difference.** It does above m·n = 460,800. This argues for a derived K; the recommendation stands.
- **PARTLY: S_rec blows up.** It never fails inside check. S_full has to be justified by the cross-host contract (Q2), not by a current failure.
- **PARTLY: K ≤ 0.7√n.** False. Drop the envelope and use n+3.
- **PARTLY: the SkylakeX tightening.** It is wrong for the 4x2 kernel (depth about n/2) and for F-layout, gemm and threaded calls. Never declare a kernel-specific K.
- **PARTLY: Dot2.** The report missed a pairwise-tree entry order:
  - same flop count as left-to-right;
  - provable γ_{log₂n+1};
  - cuts measured K about 4× at n=2048;
  - keeps SQL nesting depth at log₂n. Nested left-to-right SQL hits DuckDB's default `max_expression_depth` of 1000: in the verifier's test 999 terms were rejected and 500 parsed.
  - The owner should consider it the default spelling for wide n.
- **Wrong: "the other half of check catches constant errors."** `_check.py:104-113` compares the native entry served by confit against DuckDB running the same definition, so the baked constants are identical on both sides. Nothing catches a constant error smaller than K·eps·S, which is harmless.

**Distances**
- The row-wide spectral S for PolynomialCountSketch is refuted: S can be 0 while the twin is not (prime D). Use the normwise ∏‖a_d‖₂, or do not serve it.
- Every measured maximum was exceeded on independent data (SkewedChi2 2.77 against "≤ 2.5"). Declare derived K.
- "K grows like √n" holds only when the terms share a sign.
- numpy's exp and log are SVML here.
- `row_sumsq` is bit-exact only on x86-64, so the bound must not depend on it.
- The margin to a real defect is about 1e5 at n=2048. A float32-rounded output passes on 5–9% of near-centre lanes, so the fixtures need rows far from the centres.

**Power**
- Measured K and the kernel-disagreement percentages depend on the draws; the real tail is about 3.9.
- The derived K assumes glibc-grade ln/exp. Declare 11 and 13, or scope the claim to glibc.
- The overflow rule has to be stated on the output, because `standardize` moves the overflow point.
- In DuckDB `NaN >= 0` is TRUE, so the entry needs an explicit NaN arm and an `isinf` arm.
- Box-Cox keeps its own K of 5, not the family-wide 12.

**Chi2**
- **REFUTED: "a non-SVML build means libm, so declare K = 0".** On AVX-512 those builds run Tang. Only the probe can decide, and nobody has measured the distance d for Tang.
- K=3 is tight, and cosh(πjs) is a second kernel difference. Gate on the sin/cos probes too, and pin where the cosh constant comes from.
- The repo's own probe uses the same biased `exp(uniform)` construction the report criticises in the record.

**Sparse**
- In Spline's degree-0 periodic configuration the sparse twin raises on every row, which needs a second carve-out.
- A lifted guard without the probe fix ends in NotNative (a safe refusal), not a mistranslation.
- KNeighborsTransformer and RadiusNeighborsTransformer can be served under option 2 through an author-written densifying wrapper.
- Drift through a sparse Pipeline intermediate is ≤ 1 ulp for the scalers. For Normalizer l1 it reaches 4 ulps, caused by summation order.
- The report missed:
  - a fourth densify site, `model/_foreign.py:128`;
  - the MissingIndicator(`sparse=True`) guard;
  - the width check in the test helpers.

**Tolerated**
- The counts shrink: 565 rows come back fully finite (not 625), and NULL comes back fully finite in 48 configurations (not 51).
- **REFUTED: option 2 costs at least as much as the guard.** Reject option 2 on policy, not on cost.
- The periodic-spline breach is wider (n_knots ≥ degree + 3). Adding ±inf to EDGES alone exposes nothing; the gate also needs a `handle_missing="zeros"` periodic fixture.
- The guard's exactness was never prototyped. The degree-0 constant spline raises a numpy broadcast error, which is not validation, so a probed guard would miss it.
- A cheaper spelling exists: `Σ x_j*0.0 <> 0` at 0–31 ns per feature.

**Framework**
- The "decisive" 1.3e14-ulp lane is one where the entry equals the twin; drop it.
- sklearn's own tolerances would reject the 9.9e9-ulp lane, and the served twin never batches.
- K for S_full is n+2, not 2n+2.
- Measured Yeo-Johnson K is 3.48.

## 4. The fact each open record's owner most needs

- **matvec-parity-bound.md:** The proposed K=3 is already exceeded: 4.12 at n=29 over 1,000 seeds, and about 7.25 on a constructed n=32 row. By contrast, \|entry − twin\| ≤ (n+2)·eps·S_full, with S_full = (Σ\|xᵢcₖᵢ\| + Σ\|mᵢcₖᵢ\|)/sₖ, is a theorem for every summation order, FMA, kernel, thread count and batch shape. Every pair measured stays ≤ 4.12 under it. Also, the record lists KMeans and kin among the families it decides, but option 1 as written does not cover them: they need squares compared with S = ‖x‖²+‖c‖²+2Σ\|xc\|, because on the lane itself K reaches about 1e8.
- **power-parity-bound.md:** Option 3's exit condition can never deliver bit-exactness. The twin is SVML on Linux AVX-512 and libm everywhere else, and on the catalog fixtures it differs from itself by 886 ulps (K ≈ 1.9), with natural 1e19-ulp zero and sign-flip lanes. The derived K (9.05 for Yeo-Johnson, 11.05 with `standardize`, glibc-grade ln/exp) covers the adversarial measured tail of about 3.9.
- **additive-chi2-parity-bound.md:** K=1 fails at sklearn's own `sample_steps=3` default (theory 1.25; measured 1.143–1.241). The derived ceiling (below 3) is nearly reached (2.8) on legal configurations, so declare K per (s, j) or about 5, and take cosh(πjs) from the twin's own `np.cosh`. Disabling numpy's X86_V4 kernels makes the twin bit-exact (0 of 23.4M lanes).
- **sparse-outputs.md:** The record's premise "declared width and lanes unchanged" is false. The fit probe turns a sparse output into a 0-d object array and declares `struct<f0>`, so densifying only at serve time just swaps one error for another. Densify at the fit probe and at the serve call, plus `model/_foreign.py:128`. After that, OneHotEncoder and KBinsDiscretizer equal their dense configurations bit for bit, by construction.
- **tolerated-differences.md:** The tolerance turns failing queries into plausible answers that the gate cannot see. One NULL makes a 1,000-row KBins query succeed with that row in the top bin, where the twin fails the whole query. NULL comes back fully finite in 48 of 160 configurations. check skips every twin-raise row (4.9% of the gate), and EDGES has no ±inf. A lane-0 `error()` guard costs under 2% of the twin, but nobody has shown it never raises where the twin answers.

## Critic experiments

- **c1–c5** (`c1_boxcox_served.py` … `c5_boxcox_targeted.py`): the twin is `scipy.special.boxcox`, which equals glibc expm1(λ·log x)/λ on 100% of 1M draws. Over 400M random and alignment-targeted lanes the served Box-Cox bound reached exactly 4 ulps on 692 lanes and was never exceeded.
- **c6** (`c6_log_kernel.py`): the SVML fingerprint for numpy's `log`.
- **c7** (`c7_chi2_default.py`): chi2 at the true defaults; `sample_steps=3` gives max K 1.143.
- **c8** (`c8_lanecount.py`): the filters tried on the matvec record's lane counts; none reproduces them.
