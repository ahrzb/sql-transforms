# Inference parity: do the differences change downstream predictions?

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).

> **Verification pending**: this report has not been adversarially re-run yet.


Scripts and outputs are in `$R = scripts/inference`. Per-case JSON is in `$R/results/`. The wide table is `$R/compact.md`.

## Answer

1. **N differs from T by about as much as T differs from itself.** That holds for the fraction of entries that differ, for the max |Δ|, and for K (in eps·S units). For two of the four families, N is bit-identical to a legitimate twin:
   - **PCA:** N equals the twin run with `OPENBLAS_CORETYPE=Sandybridge` (row and batch). 0 of 12.2M entries differ, in 11 of 12 cases. In the 12th (pca_all/synth_redund), only the 3 near-null lanes differ, because the `mean_@components_.T` constant was taken under the other kernel.
   - **AdditiveChi2Sampler:** N equals the twin on numpy without AVX-512. 0 of 20.7M entries differ, in 4 of 4 cases.
2. **New rows on lanes that carry signal: no prediction changes, for N or for any T'.** Across LR, DT, RF, HGB and KMeans argmin there was 0 label change and 0 leaf change (DTreg included). That covers 29 cases and 1.63M rows for N, and 397 comparisons (22.2M rows) for T'. Scores move only in the last bits: LR predict_proba by ≤7.9e-15 and linear regression by ≤1.7e-13. T' moves them by the same amount.
3. **Predictions do change in two configurations. The twin changes them there too, at similar rates.**
   - **HGB on rows that reproduce training values ("atoms").**
     - N: 6,569 rows took a different HGB branch, and 19 labels flipped (over 1.63M rows).
     - Worst T' (sse_box/batch): 4,956 rows and 12 labels. avx2_box/batch: 4,940 rows and 13 labels.
     - The twin's own training/serving skew (fit_transform vs row-by-row serving) on 177k training rows: 3,349 rows and 7 labels.
   - **PCA(whiten=True) keeping every component, on rank-deficient data.** A whitened null component is pure rounding noise of size O(1). Label flips out of 70k rows:

     | | LR | DT | RF | HGB |
     |---|---|---|---|---|
     | N | 114 | 284 | 253 | 645 |
     | batch twin vs row twin | 109 | 296 | 266 | 594 |
     | Haswell twin | 75 | 198 | 174 | 455 |
     | twin's own fit_transform skew (20k training rows) | 32 | 90 | 3 | 169 |

   The pathologies section explains both.
4. **Bit-identical features and scores are not achievable, and the twin does not achieve them either.** The twin's output for a row changes with:
   - the OpenBLAS kernel (CPU);
   - numpy's SIMD (CPU);
   - batch versus single row, and even the batch size;
   - fit_transform versus transform.

   "Everything the same" therefore holds at the label level: N stays inside the twin's own run-to-run and machine-to-machine envelope, and it fails only where the twin already fails against itself.

## Setup

**T (the twin).** The repo's own `PythonTransform`, called once per row (`est.transform([vals])[0]`, then `float()`), in the default env: OpenBLAS 0.3.33 SkylakeX, numpy 2.5.1 with AVX512_SKX. The `sql_transform` package does not import on this Python 3.14rc2 + pydantic, so `_udf.py` is loaded standalone.

**T' (the twin under other conditions).** One subprocess per configuration, with the env set before import:

| configuration | what it changes | how it differs from T |
|---|---|---|
| batched `transform(X)` | gemm instead of a single-row call | differs on PCA and KMeans |
| `OPENBLAS_NUM_THREADS=1` | BLAS threads | identical to the default, row and batch (0 differences) |
| `OPENBLAS_CORETYPE=Haswell` | BLAS kernel | the row path differs for PCA; identical for KMeans |
| `OPENBLAS_CORETYPE=Sandybridge` | no-FMA BLAS kernel | — |
| `NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL,AVX512_SPR` | numpy's log/log1p/expm1/exp become glibc's (0 differences vs `math` on 2×10⁵ draws each) | — |
| "avx2_box" | Haswell + no-AVX-512 numpy | — |
| "sse_box" | Nehalem + no-AVX2 numpy | — |
| default rerun | nothing | bit-identical, deterministic |

**N (the native-like entry).** The entry's operation order in IEEE double precision, with no FMA:
- **PCA:** `(Σᵢ xᵢcₖᵢ left to right − mc_k)/scale_k`, with mc taken from numpy.
- **KMeans:** the twin's formula `sqrt(max((−2·x·c + Σx²) + |c|², 0))` summed left to right, with |c|² from numpy einsum. The alternative `Σ(xᵢ−cᵢ)²` is reported as N_direct.
- **Yeo-Johnson (`standardize=True`):** Goldberg's `z·(ln u/(u−1))` for log1p and Kahan's `(u−1)·(w/ln u)` for expm1 (as in `power.py`), then `(t−mean_)/scale_`.
- **Chi2:** glibc log, cos and sin.

**N checked against DuckDB 1.5.5.** DuckDB evaluated the SQL spelling on 1,500-row samples of all 30 cases. 0 mismatches in about 1.6M values.

**Data.** Test rows (n_test, about 5×10⁴–7×10⁴ per case) are resampled rows with 10% jitter, rounded to each column's recorded precision and clipped to its range, plus the training rows themselves.
- Datasets: breast_cancer, wine, digits (integer pixels), diabetes, a 3-class synthetic set, a synthetic set with 4 exactly redundant columns (`synth_redund`), and a positive heavy-tailed synthetic set (`synth_pos`).
- california_housing could not be downloaded (proxy returned 403).
- PCA and KMeans are fitted on standardized inputs; Yeo-Johnson and chi2 on raw inputs.
- `kmeans64` = KMeans(64): it has singleton clusters on breast (16) and digits (5).

**Downstream models and outputs.** The models were fitted the way Pipeline.fit does it, on `fit_transform`. All of them are evaluated in a single default-env process on same-shape arrays, so any difference comes from the features. Outputs measured:
- LogisticRegression (predict, predict_proba, decision_function)
- LinearRegression
- DecisionTreeClassifier and DecisionTreeRegressor (`apply` on float32 input, as sklearn routes)
- RandomForest with 100 trees (`apply`, predict, predict_proba)
- HistGradientBoosting (`_raw_predict`: a raw score that differs means a branch changed)
- argmin of the KMeans distances

## Per-case results

Each cell reads N / T'. T' is the maximum over the 14 T' comparators (7 configurations × row/batch). "fit" is the twin's own skew: fit_transform output vs row-by-row serving of the same training rows. K is |Δ|/(eps·S), with S the per-lane term scale from the records. For KMeans, K is measured on the squared distance D.

| case | lanes | entries ≠ | max \|Δ\| | K max | LR max ΔP | Lin max Δ | labels LR/DT/RF/argmin | DT+RF leaf changes | HGB branch rows N/T'/fit | HGB labels N/T'/fit |
|---|---|---|---|---|---|---|---|---|---|---|
| pca_all/breast | 30 | .79/.79 | 3.4e-14/3.4e-14 | 3.9/4.0 | 2.3e-15/2.3e-15 | 2.2e-15/2.2e-15 | 0/0 | 0/0 | 186/192/192 | 0/0/0 |
| pca_all/wine | 13 | .58/.58 | 2.7e-15/2.7e-15 | 2.7/2.7 | 4.4e-16/5.6e-16 | 1.1e-15/1.1e-15 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| pca_all/digits | 64 | .80/.80 | 2.5e-14/2.5e-14 | 5.4/5.4 | 5.9e-15/6.3e-15 | 1.3e-14/1.4e-14 | 0/0 | 0/0 | 999/1043/817 | 0/0/0 |
| pca_all/diabetes | 10 | .60/.60 | 6.9e-15/6.9e-15 | 2.9/2.9 | 5.6e-16/5.6e-16 | 1.1e-13/1.1e-13 | 0/0 | 0/0 | 145/145/115 | 0/0/0 |
| pca_all/synth | 20 | .69/.69 | 2.7e-15/2.7e-15 | 3/3 | 6.7e-16/6.7e-16 | 6.7e-16/8.9e-16 | 0/0 | 0/0 | 250/250/250 | 1/2/2 |
| **pca_all/synth_redund** | 20 | .72/.73 | **6.0/6.2** | 3/3 | 0.023/0.025 | 0.0096/0.010 | **LR 114/114, DT 284/296, RF 253/266** | 45,786/54,114 | 42,563/42,563/11,984 | **645/645/169** |
| pca_95/breast | 10 | .68/.68 | 6.2e-15/6.2e-15 | 3.9/4.0 | 1.2e-15/1.3e-15 | 1.1e-15/1.1e-15 | 0/0 | 0/0 | 147/147/166 | 0/1/1 |
| pca_95/wine | 10 | .57/.57 | 2.7e-15/2.7e-15 | 2.7/2.7 | 5.6e-16/5.6e-16 | 8.9e-16/8.9e-16 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| pca_95/digits | 40 | .79/.79 | 2.5e-14/2.5e-14 | 5.4/5.4 | 7.9e-15/7.9e-15 | 1.2e-14/1.2e-14 | 0/0 | 0/0 | 986/986/742 | 0/0/0 |
| pca_95/diabetes | 8 | .57/.57 | 1.8e-15/1.8e-15 | 2.9/2.9 | 4.4e-16/5.6e-16 | 1.1e-13/1.1e-13 | 0/0 | 0/0 | 125/132/132 | 0/0/0 |
| pca_95/synth | 19 | .66/.66 | 2.7e-15/2.7e-15 | 3/3 | 5.6e-16/6.7e-16 | 8.9e-16/8.9e-16 | 0/0 | 0/0 | 254/254/225 | 2/2/2 |
| pca_95/synth_redund | 15 | .64/.64 | 2.7e-15/2.7e-15 | 3/3 | 4.4e-16/4.4e-16 | 4.4e-16/4.4e-16 | 0/0 | 0/0 | 89/89/87 | 1/1/0 |
| kmeans8/breast | 8 | .36/.22 | 2.3e-14/1.9e-14 | 3.9/3.4 | 3.3e-15/2.8e-15 | 2.0e-15/1.3e-15 | 0/0 | 0/0 | 70/55/55 | 1/0/0 |
| kmeans8/wine | 8 | .29/.15 | 3.8e-15/3.3e-15 | 3.4/3.4 | 1.6e-15/1.3e-15 | 1.3e-15/1.3e-15 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| kmeans8/digits | 8 | .41/.12 | 2.1e-14/1.8e-14 | 6.5/3.2 | 1.3e-14/1.2e-14 | 3.6e-14/2.8e-14 | 0/0 | 0/0 | 313/161/117 | 0/0/0 |
| kmeans8/diabetes | 8 | .27/.14 | 4.0e-15/3.1e-15 | 3.1/3.1 | 7.8e-16/7.8e-16 | 1.7e-13/1.1e-13 | 0/0 | 0/0 | 52/40/40 | 1/1/1 |
| kmeans8/synth | 8 | .31/.07 | 3.6e-15/1.8e-15 | 3.6/3.3 | 2.4e-15/2.4e-15 | 1.8e-15/1.8e-15 | 0/0 | 0/0 | 118/24/24 | 3/0/0 |
| kmeans64/breast | 64 | .29/.19 | **3.4e-7/3.4e-7** | 4.1/3.3 | 4.4e-9/6.2e-9 | 1.3e-8/1.8e-8 | 0/0 | 0/0 | 88/69/66 | 0/0/0 |
| kmeans64/digits | 64 | .33/.11 | **6.7e-7/9.5e-7** | 6.4/3.1 | 7.4e-10/7.4e-10 | 4.1e-8/1.1e-7 | 0/0 | 0/0 | 511/329/241 | 0/0/0 |
| kmeans64/synth | 64 | .29/.08 | 3.6e-15/2.2e-15 | 4.3/3.2 | 1.8e-14/1.4e-14 | 2.8e-14/2.1e-14 | 0/0 | 0/0 | 199/84/80 | 1/1/1 |
| yj_std/breast | 30 | .27/.11 | 1.7e-14/1.7e-14 | 1.4/0.95 | 4.2e-15/3.8e-15 | 2.7e-14/2.7e-14 | 0/0 | 0/0 | 766/483/0 | 0/0/0 |
| yj_std/wine | 13 | .27/.11 | 1.3e-14/1.0e-14 | 1.3/1.0 | 4.4e-15/4.1e-15 | 3.1e-15/2.9e-15 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| yj_std/digits | 64 | .07/.08 | 7.1e-16/2.4e-15 | 0.86/1.1 | 1.3e-15/1.8e-15 | 3.6e-15/3.6e-15 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| yj_std/diabetes | 10 | .38/.04 | 1.3e-15/1.3e-15 | 3.5/2.6 | 4.4e-16/2.2e-16 | 5.7e-14/5.7e-14 | 0/0 | 0/0 | 18/5/0 | 1/0/0 |
| yj_std/synth | 20 | .40/.16 | 4.4e-15/2.2e-15 | 2.6/2.4 | 7.8e-16/5.6e-16 | 1.3e-15/6.7e-16 | 0/0 | 0/0 | 107/55/0 | 0/2/0 |
| yj_std/synth_pos | 20 | .33/.14 | 2.4e-15/1.8e-15 | 1.1/0.9 | 6.7e-16/6.7e-16 | 6.7e-16/4.4e-16 | 0/0 | 0/0 | 1146/500/0 | 8/4/0 |
| chi2/breast | 90 | 2e-4/2e-4 | 1.1e-14/1.1e-14 | 0.56/0.56 | 1.7e-15/1.7e-15 | 1.4e-13/1.4e-13 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| chi2/wine | 39 | 5e-4/5e-4 | 4.4e-16/4.4e-16 | 0.56/0.56 | 7.2e-16/7.2e-16 | 0/0 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| chi2/digits | 192 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |
| chi2/synth_pos | 60 | 8e-4/8e-4 | 8.9e-16/8.9e-16 | 0.41/0.41 | 4.4e-16/4.4e-16 | 4.4e-16/4.4e-16 | 0/0 | 0/0 | 0/0/0 | 0/0/0 |

**Max ulps (N vs T, and the largest T' vs T):**

| family | N | T' |
|---|---|---|
| PCA | 5.1e4–1.4e7 | up to 1.7e7 |
| synth_redund, kmeans64 | 9.2e18 / 4.5e18 (sign change, or 0 against nonzero) | same |
| kmeans8 | 5–51 | 3–42 |
| Yeo-Johnson | 3.7e3–3.4e5 | 136–2.7e5 |
| chi2 | 4–35 | 4–35 |

**Within the twin itself:**
- Yeo-Johnson: twin on an AVX2 box vs AVX-512 twin differs on 4–16% of entries. N vs T differs on 7–40%. N vs the AVX2-box twin differs on 13–40%.
- For KMeans, N differs on more entries than T' (27–41% vs 7–22%), but by the same size and K.
- N_direct (Σ(xᵢ−cᵢ)²) differs on 37–53% of entries and has the same downstream outcome: 0 LR/DT/RF/argmin label changes, 2,235 HGB branch rows, 3 HGB labels.

**Single row vs batch (default env).**
- pca_all/breast: computing one row at a time vs the full batch differs on 75% of entries. Batches of 2–2048 rows agree with the full batch.
- pca_all/synth: one row at a time differs on 68%, and batches of 2–2048 rows still differ from the full batch on 14%.
- kmeans8/synth: one row at a time differs on 6%.
- The same row placed at 7 different positions in a batch gives 2 distinct outputs.

## Analytic flip model vs observation

A row changes branch at split s (lane f, threshold t) iff t lies between x_T and x_C. sklearn DT/RF compare `float32(x) ≤ t`, so for them t is the float32 rounding boundary, but the same argument applies.

**E[changes] = Σ_s [ c_s(t)·E|Δ_f| + A_s·π_s ]**
- c_s(t): rows per unit x that reach s, near t. Estimated as 16/(2h), where h is the distance to the 16th nearest row.
- A_s: rows whose value is exactly the threshold value, up to rounding (atoms).
- π_s ≈ ½·P(x_C ≠ x_T) on that lane.

**1. Continuous term.**
- Inputs: |Δ| ≈ K·eps·S ≈ 1e-15, with K ≲ 5 and S ~ 1–10 on standardized lanes. ρ ~ 0.1–1 per unit.
- Per split per row this gives about 1e-16. Summed over a path (DT depth 10–25; RF 100 trees; HGB 100–1000 trees) it stays below about 3e-13 per row per model.
- For LR, ρ_margin(0)·E|Δmargin|, with h = 2e-4…0.1, gives ≤ 5e-15 per row.
- Plugged in per case, the expected number of changes over each test set (n_test, about 5–7×10⁴ rows) is:

| | N | T' |
|---|---|---|
| DT | 4e-13 – 6e-10 | similar |
| RF (10 trees × 10) | 5e-11 – 2e-8 | 6e-11 – 2e-8 |
| HGB, continuous part | 8e-13 – 4e-8 | — |
| LR | 7e-14 – 3e-10 | — |
| argmin | 2e-12 – 3e-11 | — |

- **Observed: 0 in every case.** The T' estimates are within 1–4× of N's. For example, pca_all/synth RF: 4.2e-9 for N and for Sandybridge, 2.7e-9 for Haswell.
- At 10⁹ rows a day that is about 3e-4 RF leaf changes a day. A label change additionally needs the forest's vote to be within one tree.

**2. Atom term (HGB).** sklearn computes HGB bin thresholds with `np.percentile(..., method="averaged_inverted_cdf")` once a lane has more than 255 distinct training values. That method returns training values themselves. In `noise_splits.py`, essentially every HGB split has gap 0 (for example 2,221 of 2,221 on breast).
- A served row that reproduces the threshold-defining training row (PCA, KMeans), or just its feature value (the per-feature transforms Yeo-Johnson and chi2), sits one rounding away from t.
- Predicted straddles = Σ over atom pairs of ½·P(lane differs). The table gives predicted / observed (rows with a branch change):

| case | atom pairs | N | batch twin | AVX2-box twin | fit_transform |
|---|---|---|---|---|---|
| pca_all/breast | 1,251 | 458/426 (186) | 439/452 (192) | 313/235 (118) | 441/452 (192) |
| pca_all/digits | 8,786 | 3,480/3,316 (999) | 3,469/3,382 (1,043) | 2,279/1,408 (569) | 2,766/2,648 (817) |
| pca_all/synth | 1,500 | 511/493 (250) | 510/530 (250) | 353/293 (150) | 510/530 (250) |
| kmeans8/digits | 15,514 | 3,170/3,094 (313) | 914/1,295 (161) | 0/0 | 897/1,034 (117) |
| yj_std/breast | 9,384 | 1,349/1,590 (771) | 0/0 | 558/1,117 (487) | 0 |
| yj_std/synth_pos | 13,739 | 2,203/2,145 (1,146) | 0/0 | 937/779 (500) | 0 |
| chi2/breast | 8,595 | 0.1/0 | 0/0 | 0.1/0 | 0 |
| wine (every family) | 0 | — | — | — | — |

- Wine has no atom pairs: its 178 training rows give at most 255 distinct values per lane, so thresholds are midpoints. The same holds for Yeo-Johnson on digits.
- In every case except the noise-lane one, the rows with an HGB branch change are exactly the rows straddling an atom (186=186, 999=999, 250=250, 313=313). Outside digits, every HGB branch change was on a copy of a training row. Digits has 530 test rows that are exact duplicates of training rows; 224 of those changed branch.
- sklearn DT and RF thresholds are midpoints of neighbouring float32 values, so they never sit on an atom. They had 0 changes outside noise lanes.

**3. Stress test: rows placed on the decision surface.** Rows were bisected to adjacent doubles across the LR boundary, the DT root and HGB root splits, and the KMeans Voronoi boundary: 2,000 rows each. Each cell counts rows whose prediction differs from T:

| case/target | N | batch twin | Haswell twin | Sandybridge twin | no-AVX-512 numpy | AVX2 box |
|---|---|---|---|---|---|---|
| pca_95/breast LR | 392 | 427 | 213 | 392 | 0 | 213 |
| pca_95/breast DT root | 259 | 241 | 170 | 259 | 0 | 170 |
| pca_95/breast HGB root | 221 | 217 | 166 | 221 | 0 | 166 |
| pca_all/synth_redund LR | 599 | 675 | 441 | 599 | 0 | 441 |
| kmeans8/breast LR | 435 (direct 653) | 374 | 0 | 214 | 0 | 0 |
| kmeans8/breast argmin | 614 (direct 512) | 614 | 0 | 311 | 0 | 0 |
| yj_std/breast LR | 623 | 0 | 0 | 0 | 540 | 540 |
| yj_std/breast DT/HGB root | 0 | 0 | 0 | 0 | 0 | 0 |
| chi2/breast (all) | 0 | 0 | 0 | 0 | 0 | 0 |

- With all the probability mass on the boundary, N and the twin's legitimate variants change 10–34% of predictions. They are the same order, and Sandybridge equals N for PCA.
- The single-lane Yeo-Johnson boundaries came out 0. Only 2–4 distinct input doubles sit at such a boundary, and N equals T on those doubles. This is the atom mechanism again.

## Pathological configurations

1. **Whitened null components.** This occurs with PCA(whiten=True) keeping all components on data with exact linear dependencies (redundant columns, a sum of parts, complete one-hot encodings).
   - `explained_variance_` = 0 is clipped to scale = eps. The lane is then (x·c − m·c)/eps: rounding noise of size O(1), with std 2.36. Lanes with ev ~1e-16 are noise at the 1e-8 level.
   - Every model used those lanes: HGB on all of them, DT with importance 0.022, and an LR coefficient.
   - N vs T: max |Δ| = 6.0. Batch vs row: 6.2. Haswell: 4.0.
   - K stays ≤ 3, so the proposed K·eps·S bound holds, yet the lane carries no signal.
   - Labels flip for the twin against itself just as for N (see point 3 above). pca_95 drops these lanes and is clean.
   - Digits does not trigger it: its null directions are constant-zero pixels, which contribute exactly 0.
2. **HGB atoms.** These are thresholds equal to training values, met by rows that reproduce training rows or values. They are common whenever a population is re-scored, records are duplicated, or the per-feature transforms see data on a recorded-precision grid. This is not specific to the native entry: any recomputation hits them, including the twin's own serving vs fit_transform (192 of 569 breast training rows) and a different CPU.
3. **KMeans rows on a center (singleton clusters).**
   - The true distance is 0. The twin gives `sqrt(max(noise, 0))`, which is 0 or 2.4e-7–9.5e-7. N gives 0 or 1.7e-7, on different rows. fit_transform gives a third pattern: 9.5e-7 where T gives 0. N_direct gives exactly 0.
   - Because of this, |Δ| reaches 3.4e-7 for N and 9.5e-7 for T', and 4.5e18 ulps (0 against nonzero).
   - No DT/RF split fell inside [0, 1e-6] except 20 RF splits on kmeans64/digits, and 0 leaves changed. A threshold inside that noise floor needs two or more training rows sharing a center with different labels (duplicates).
4. **Discrete inputs with per-feature transforms.**
   - chi2 on integer pixels (digits) is bit-exact for N and for every T': numpy's SIMD log equals glibc's on 1–16.
   - Yeo-Johnson on digits has at most 17 values per lane, so HGB uses midpoints and there are no atoms.
   - Exactly-zero lanes (chi2 at x = 0, Yeo-Johnson of 0 standardized) are computed identically everywhere, so they are safe atoms.
5. **Batch composition.** The twin is not a function of the row unless it is served one row at a time (single row vs batch numbers above). NUM_THREADS does not matter: 0 differences.

## Conclusions for the owner's goal

- **What the data supports as a criterion.** "Native vs twin lies within twin vs twin" holds here, family by family, at both the feature level and the prediction level. It is the bar the twin itself meets across CPUs and batch sizes.
- **A stronger statement is available for PCA and chi2.** N is bit-equal to a legitimate twin: OpenBLAS Sandybridge kernel for PCA, numpy without AVX-512 for chi2. `native.check` could gate bit-exactly against that reference twin instead of using an eps·S bound.
- **KMeans and Yeo-Johnson are not bit-equal to any twin.**
  - KMeans: the twin's |x|² comes from numpy einsum's reduction order.
  - Yeo-Johnson: the entry spells log1p and expm1 (Goldberg, Kahan) because DuckDB has neither. With glibc's log1p and expm1 it would equal the non-AVX-512 twin.

  Their N–T differences are still the same order as T'–T, with K ≤ 6.5 and 3.5 against T' ≤ 3.4 and 2.6.
- **Noise lanes.** Refuse, or at least flag, PCA/whiten components whose scale is clipped or whose variance is ≲ eps·max. Their output is not a function of the row for the twin either. The same applies to their kin in the matvec families.
- **HGB atoms.** They change HGB branches, and occasionally labels, for N and for every non-identical twin at comparable rates. They cannot be removed by any entry that is not the training computation itself.

## Caveats

- Downstream models run in one env. Their own BLAS (LR decision_function) is held fixed.
- The RF analytic term uses 10 of the 100 trees, scaled ×10.
- HGB label counts are small; 19 vs 13 is not a significant difference.
- sql_transform could not be imported, so the repo's PythonTransform was loaded from `_udf.py`.
- california_housing was unavailable offline.

