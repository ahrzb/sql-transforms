# Tolerated differences: what the entries answer where the twin raises

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


**Scope.** Repo at `113fba7`, read-only. sklearn 1.9.0, numpy 2.5.1, DuckDB 1.5.5, confit release build, x86-64 AVX-512.
`$R` = `scripts/tolerated`. Run any script as `cd /home/user/sql-transforms && uv run --frozen python $R/<script>`.
**Environment caveat.** This venv runs Python 3.14.0rc2, where pydantic 2.13.4 breaks `import sql_transform` (the `typing._eval_type(prefer_fwd_module=…)` TypeError). The repo's own pytest collection fails here for the same reason. Every script first imports `$R/shim.py`, which drops that kwarg. No repo file was changed.

## 0. Answers to the five questions

1. **Does "the entry may answer" ever give a silently plausible finite value?** Yes, often. I probed all 160 catalog configurations (every `catalog_test.FIXTURES` factory). The twin raised on 1,396 of 5,104 probe rows, and the entry raised on none of them. On 625 of those 1,396 rows the entry answered entirely in finite numbers:
   - 485 rows (59 configs, 17 classes): a lane that reads the bad input is finite.
   - 140 rows: no lane reads it (selectors, MissingIndicator).
   - A pinned Box-Cox with λ<0 (an 18th class) adds more cases.
2. **NULL, the common SQL case.** For 51 of 160 configs, a NULL feature that the twin rejects comes back finite. In a 1,000-row call with a single NULL, the twin query fails outright. The entry returns all 1,000 rows and puts the NULL row in KBins' **top bin**.
3. **The finite answers are mostly sklearn's own arithmetic, but not always.** 573 of 625 equal `sklearn.config_context(assume_finite=True)`, i.e. the twin with its finiteness check turned off. The other 48 are artefacts of DuckDB treating NaN as larger than every number:
   - Binarizer: NaN gives 1.0 (numpy gives 0.0).
   - Isotonic with `clip`: NaN gives `fp[-1]` (numpy gives NaN).
   - Periodic spline: ±inf gives a basis value (scipy gives NaN).
4. **A generic guard is not exact; a per-leaf probed guard is cheap.**
   - Read-off-the-tags guards both miss raises and raise where the twin answers (forbidden).
   - An exact guard costs **16–37 ns/feature/row** for ±inf only and **34–93 ns/feature/row** for ±inf+NaN. The twin costs **~98–134 µs/row**.
5. **Recommendation:** require the guard. Section 7 says how.

## 1. Method (MEASURED, `$R/probe.py`, `analyze*.py`, `reasons.py`)

**Fitting.** Each configuration is fitted on 40 rows: 3 features drawn from uniform(1, 10), or 1 feature for Isotonic. Encoders get categories {1, 2, 3}. Imputers and indicators get their marker in feature 0 only.

**Probe rows.** One feature is set to a probe value and the others sit at their fit median. The probes are +inf, −inf, NaN, NULL, ±1e308, 0, −1, ±1e6, 0.5. Every feature position is probed, which gives 33 rows per configuration.

**Serving.** Each row is served one at a time through confit, once with `PythonTransform` (the twin) and once with `to_native(step)`. The SQL is `_registry.query`, the same one `check` serves.

**Lanes that read a feature.** A lane counts as reading feature p if the twin's output in that lane varies over 16 finite values of p.

**Classes, for rows where the twin raises:**
- **a:** every lane that reads the bad input is non-finite or NULL.
- **b:** every such lane is finite.
- **ab:** mixed.
- **b0:** no lane reads the input, and the whole row is finite.
- **c:** the entry raises too.

**Where the twin answers:** 3,708 rows, of which **0 DIFF and 0 REVERSE**. On this probe set, "never the reverse" holds.

**Totals over the 1,396 twin-raise rows:** a 760, b 425, ab 60, b0 140, a0 11, **c 0**.

## 2. What the twin rejects, and why (SOURCED, MEASURED)

The finiteness check is `check_array`'s `ensure_all_finite`, set at each estimator's call site:
- **allow-nan (rejects ±inf only):** Standard, MinMax, MaxAbs, Robust (`preprocessing/_data.py:570, 1119, 1391, 1774`), Quantile (`:3000, 3091`), Power (`:3627`), encoders (`_encoders.py:1054, 1602`), TargetEncoder (`_target_encoder.py:447`).
- **True (rejects ±inf and NaN):** Normalizer, Binarizer, PolynomialFeatures, KBins, Isotonic, FeatureAgglomeration, and FunctionTransformer with `validate=True`.

Exceptions that tags do not predict:
- **Selectors** use `ensure_all_finite=not allow_nan` (`feature_selection/_base.py:114`). VarianceThreshold sets `allow_nan=True` (`_variance_threshold.py:139`), so it does no finiteness check at all and **±inf passes**. Measured: the twin answers inf.
- **Imputers.** SimpleImputer and MissingIndicator reject NaN when `missing_values` is not NaN (`impute/_base.py:351-353, 949-951`). MissingIndicator's tag says `allow_nan=True` regardless (`:1145`).
- **Object arrays.** For object dtype, `_assert_all_finite` checks only NaN, and only when NaN is not allowed (`utils/validation.py:119-122`). Encoders build an object array when the row holds a string (`_encoders.py:54-56`). Measured (`$R/extra.py` A): on a (string, number) row, OneHot/Ordinal/Target **answer** ±inf as an unknown category. On an all-numeric row they **raise**. The entry already matches the twin in both cases.
- **SplineTransformer** uses `ensure_all_finite=(handle_missing != "zeros")` (`_polynomial.py:1001`). With `"zeros"`, ±inf passes.
- **Domain checks**, beyond finiteness:
  - Box-Cox `nanmin(X) <= 0` (`_data.py:3633`).
  - Unknown category under `handle_unknown="error"`.
  - Isotonic `out_of_bounds="raise"`.
  - Spline `extrapolation="error"` (values beyond the knots).
  - MissingIndicator `error_on_new` (NaN in a feature that had none at fit).
- **Pipelines** validate at every step, including intermediate values. Measured on Pipeline[9] (SimpleImputer → StandardScaler → MinMaxScaler(clip)): a finite input of 1e308 overflows to inf after scaling, and the twin raises in MinMaxScaler.

## 3. Estimator × input → twin → entry → class (MEASURED, `probe.json`, `$R/examples.py`)

| estimator (configs) | twin raises on | entry answers there | class |
|---|---|---|---|
| StandardScaler (4), RobustScaler (5) | ±inf | ±inf | a |
| MinMaxScaler clip=False (2) / clip=True (2) | ±inf | ±inf / clipped to `feature_range` ends | a / **b** |
| MaxAbsScaler clip=False / clip=True | ±inf | ±inf / ±1 | a / **b** |
| Normalizer l1, l2, max (3) | ±inf, NaN | inf: that lane NaN, **the other lanes 0.0**; NaN: all lanes NaN | ab / a |
| Binarizer (3) | ±inf, NaN | +inf→1, −inf→0, **NaN/NULL→1** (numpy: 0) | **b** |
| SimpleImputer, NaN marker (6) | ±inf | ±inf passed through | a |
| SimpleImputer, numeric marker (2) | ±inf, NaN | value lane ±inf/NaN; indicator lane 0.0 | ab |
| MissingIndicator (3) | ±inf; NaN when error_on_new or numeric marker | 0.0 lanes | **b / b0** |
| VarianceThreshold (1) | nothing | equals the twin (inf) | = |
| 10 selectors (KBest, Percentile, Fpr, Fdr, Fwe, Generic, FromModel, RFE, RFECV, SFS) | ±inf or NaN in **any** column | a kept column passes through; **a dropped column leaves the row finite** | a / **b0** |
| PolynomialFeatures (5) | ±inf, NaN | inf/NaN products (bias lane 1.0) | a |
| One-Hot / Ordinal, handle_unknown="error" (3) | ±inf, unknown category, NaN not seen at fit | NULL lanes for that feature | a |
| One-Hot / Ordinal, other (5); TargetEncoder (2) | ±inf (all-numeric row) | unknown encoding: −1, all-zeros, infrequent, or target mean | **b** |
| KBinsDiscretizer (10) | ±inf, NaN | **+inf/NaN/NULL → top bin**; −inf → bin 0 | **b** |
| QuantileTransformer (5) | ±inf | +inf→1, −inf→0 | **b** |
| SplineTransformer (15, handle_missing="error") | ±inf, NaN; past the knots under "error" | NaN/NULL → 0.0 lanes; ±inf → boundary basis values (constant, periodic), ±inf lanes (linear), NaN (continue, error) | **b** / ab / a |
| IsotonicRegression (8) | ±inf, NaN; outside [X_min, X_max] under "raise" | "nan"/"raise": NaN; **"clip": ±inf and NaN → fp[-1] / fp[0]** | a / **b** |
| FunctionTransformer validate=False (21) / validate=True (9) | nothing / ±inf, NaN | equals the twin / f(x): mostly ±inf or NaN; **reciprocal(±inf)=±0, exp(−inf)=0** | = / a, **b** |
| FeatureAgglomeration (7) | ±inf, NaN | cluster mean inf/NaN | a |
| PowerTransformer box-cox (2 fixtures + pinned) | ±inf, x≤0 | x≤0 → NaN; +inf → inf for λ≥0, **→ −1/λ for λ≤−1e−19**. Measured λ=−50, −1, −1e−3 → 0.02, 1, 1000 (`extra.py` B) | a / **b** |
| Pipeline (10), ColumnTransformer (7), FeatureUnion (6) | union of the parts' rules, intermediate values included | e.g. (MinMax→Binarizer) b; Pipeline[9] at 1e308 → clipped to 1 | a / **b** / ab |

**Counts:**
- **b/ab** (a lane that reads the bad input is finite): 59/160 configs across 17 classes, plus PowerTransformer with λ<0.
- **Any fully-finite twin-raise row** (adds b0): 27 of 33 classes in the probe, 28 with PowerTransformer.
- **Never finite on a twin-raise row:** only StandardScaler, RobustScaler, PolynomialFeatures and FeatureAgglomeration. VarianceThreshold never raises.

## 4. Do the finite answers at least match sklearn with validation off? (MEASURED)

Of the 625 finite twin-raise rows:
- **573 equal** `est.transform(row)` under `sklearn.config_context(assume_finite=True)`.
- **48 differ.** All 48 are DuckDB's NaN total order (NaN equals NaN and is greater than every number; repo `native/_helpers.py:30`; DuckDB docs, "SQL Quirks", https://duckdb.org/docs/lts/sql/dialect/sql_quirks — from a search snippet, since a direct fetch was blocked by the proxy) turning NaN into a branch:

| case | entry | sklearn without validation |
|---|---|---|
| Binarizer, NaN | 1.0 | 0.0 |
| Isotonic(clip), NaN | `fp[-1]` | NaN |
| Spline(periodic), ±inf | basis value | NaN |

The same applies to pipelines and column transformers that contain these.

**Consequence (DERIVED):** "non-finite in → non-finite out" is not a stable property in a DuckDB plan. Any downstream `x < t` CASE turns NaN back into a definite branch, as Binarizer, KBins and Isotonic(clip) already show.

## 5. Batch semantics and the gate's blind spot (MEASURED)

- **Batch** (`extra.py` C). KBins(ordinal, quantile, 5 bins) over 1,000 rows, with one NULL at row 517:
  - Twin: `ValueError: Input X contains NaN`, and the whole call fails.
  - Entry: 1,000 rows; row 517 = `{f0: 4.0}`, the top bin.
  
  The record's ground ("no served answer changes") is literally true. But the observable change is error → answer, at the granularity of a **whole query**.
- **Uncompared gate rows** (`$R/gate_skips.py 2`, seeds 0–1, all fixtures, 51 s). `check` skips every row where the twin raises (`_check.py:124-125`). That is 624 of 12,800 gate rows (**4.9%**):
  - OrdinalEncoder 22.1%
  - OneHotEncoder 19.0%
  - PowerTransformer 13.1%
  - IsotonicRegression 12.7%
  - SplineTransformer 10.3%
  - KBinsDiscretizer 6.5%
- **No ±inf in the gate.** The gate's `EDGES` (`catalog_test.py:599`) contains no ±inf. So apart from FunctionTransformer's own SPECIALS test, the record's lead example (an infinite feature) is never served by the gate.
- **Side finding: a real parity breach where the twin *answers*** (`$R/spline_zeros.py`, `$R/spline_zeros_check.py`).
  - Configuration: `SplineTransformer(degree=1, n_knots≥4, extrapolation="periodic", handle_missing="zeros")` at x=±inf.
  - Twin: NaN in every lane. Entry: lane f1 = 0.0.
  - `native.check` raises `ParityError … step nan, native 0.0`; it does not trigger with n_knots 2–3 or degree 3.
  - Cause (DERIVED): `fmod(±inf, p)` is NaN; DuckDB routes NaN to the last interval, where the spelling folds f1 to the constant 0.0.
  - This is outside the tolerance. It is reachable only because the gate never serves ±inf.

## 6. The alternative: trap like the twin

### 6.1 Generic guards are not exact (MEASURED, `analyze2.py`)

Each guard is evaluated against the 1,396 twin-raise rows, using the top-level estimator's tags.

| guard | catches | misses | over-raises (forbidden) |
|---|---|---|---|
| g1: any ±inf | 794 | 602 | 134 (FunctionTransformer validate=False 126, VarianceThreshold 6, ColumnTransformer 2) |
| g2: ±inf, or NaN when `allow_nan` is False | 1,240 | 156 | 364 |
| g3: g2, skipped when `no_validation` | 1,240 | 156 (domain checks, MissingIndicator numeric marker) | 112 (Pipeline 36, ColumnTransformer 40, FeatureUnion 30, VarianceThreshold 6) |

Compositions' tags do not describe their parts: Pipeline and ColumnTransformer copy only `pairwise` and `sparse`, so `allow_nan` reads False. A NaN headed for an imputer therefore over-raises. The answer to Q4's "is it sufficient" is **no**, on both counts:
- it misses domain checks and intermediate overflow;
- read from tags, it raises where the twin answers.

### 6.2 An exact construction (DERIVED from the rules in §2)

For a leaf estimator E, the twin raises when **(any j in C_E with x_j in F_E) or D_E(x)**:
- **C_E** is E's input columns.
- **F_E** is one of {}, {±inf}, {±inf, NaN}, per column. It is fixed by the call site and by the row's dtype, which the declared types fix.
- **D_E** is E's domain predicate.

For compositions:
- Pipeline: R(E₂∘E₁)(x) = R(E₁)(x) ∪ R(E₂)(E₁(x)).
- ColumnTransformer / FeatureUnion: the union over parts, each on its own columns.

So the guard can be built per leaf without copying sklearn's code:
- **(i) F_E by probing.** At translation time, call `est.transform` with ±inf and NaN in each column. This is the same technique `encode.py` already uses. It costs 3k transform calls, about 10 ms at 32 features.
- **(ii) Predicates on the leaf's own input expressions.** In a pipeline these are intermediate expressions, which catches the 1e308 → inf case.
- **(iii) Hoist to the final lane 0.** Collect the predicates through a side channel and OR them into the composed lane 0. A selector can drop a sub-step's lane 0, so leaving the guard inside the sub-step is not safe.
- **(iv) Domain checks.** Five already exist as NaN/NULL arms (power.py:66, isotonic.py:145, spline "error", encode.py's NULL ELSE) or are one comparison (MissingIndicator `error_on_new`). Switch them to `error()`.
- **Bounded entries.** For PowerTransformer and FunctionTransformer, an intermediate within ≤4 ulps of DBL_MAX could be inf on one side and finite on the other. I flag this but did not measure it.

Placement is verified (`extra.py` D): a lane-0 `CASE … THEN error()` fires when the query reads only field f3, in confit and in DuckDB alike. This is the same mechanism as the existing unknown-id trap, which I also reproduced.

### 6.3 Cost (MEASURED; `$R/guard_spell.py` interleaved, 7×5 calls of 10,000 rows, median; `$R/guard_cost.py`)

| entry (32 features) | none µs/row | ±inf `abs(x)=inf` | ±inf+NaN `abs(x)=inf OR x=NaN` | ±inf+NaN `x>MAX OR x<−MAX` | per-lane NaN-out (option 2) |
|---|---|---|---|---|---|
| StandardScaler | 1.43 | +0.52 (16 ns/feature) | +1.09 (34 ns) | +1.30 (41 ns) | +0.99 (31 ns) |
| KBins(ordinal, 5) | 3.42 | +1.17 (37 ns) | +2.78 (87 ns) | +2.97 (93 ns) | +2.70 (84 ns) |

- **Twin reference, 4 features, 64-row calls:** StandardScaler 97.8 µs/row, KBins 99.2, Quantile(100) 133.6.
- **Guard vs twin:** under 1.5% of the twin's cost at 32 features. The speedup stays above 50×.
- **Guard vs entry:** 36–90% on the cheapest entries.
- **Build time:** +0.02–0.15 s at 32 features.
- **Spellings agree:** all of them trap the same inputs in confit and DuckDB (verified on ±inf, NaN, NULL, ±DBL_MAX, 5e−324, −0.0).

### 6.4 Option 2 is dominated (DERIVED, confirmed by the last column above)

Let P_j be the predicate for column j, with cost c(P_j).
- **Guard:** cost Σ_j c(P_j), applied once on lane 0.
- **Non-finite-out:** each lane ℓ needs the predicates of every column S_ℓ it reads, so the cost is Σ_ℓ Σ_{j∈S_ℓ} c(P_j) ≥ Σ_j c(P_j). Equality holds only for elementwise entries. Normalizer, PolynomialFeatures and OneHot pay more unless shared work is reused.

So option 2 costs at least as much as the guard. It still turns errors into answers, and it does not survive a downstream NaN comparison (§4).

## 7. Unconfirmed / limits

- The probe uses one controlled fit per configuration, not the fixture generator's widths, string mixes or multiple instances. The object-dtype encoder path was tested separately.
- I did not probe the λ≈0 and λ>0 Box-Cox branches beyond inf.
- "Plausible" is judged per lane read. Whether a downstream model consumes those lanes is application-dependent.
- The timings are one machine, release confit, in-process. The noise is about ±10% (IQR in `guard_spell.out`).
- The DuckDB NaN-order documentation is cited from a search result, because the proxy blocked a direct fetch. The behaviour itself is measured here (`fin_range` traps NaN identically in confit and DuckDB).


## The researcher's recommendation

Require the guard: make the entry trap like the twin. Do not accept the tolerance as stated, and do not accept it with a "non-finite in → non-finite out" condition.

Accepting as stated is not grounded. In 59 of 160 configurations (18 of 33 classes) the entry turns a query the twin would have stopped into finite, plausible answers. This includes the most common SQL case: a NULL lands in KBins' top bin, Binarizer answers 1, and Isotonic(clip) answers its top value. The gate cannot see any of it, because it skips twin-raise rows and never serves ±inf.

The non-finite-out condition does not hold today for those same configurations. Enforcing it would need the same per-column predicates as the guard, at the same or higher cost. And DuckDB's NaN-above-everything order turns NaN back into a plausible branch downstream, so even a non-finite output is not safe.

The record's stated obstacle, a per-estimator copy of sklearn's validation, is much smaller than it sounds:
- Probe each fitted leaf for which of {+inf, −inf, NaN} it rejects per column. `encode.py` already uses this technique.
- Apply the predicates to the leaf's own input expressions, so pipelines' intermediate overflow is caught.
- OR them into the composed lane 0 as `CASE … THEN error()`, which fires on any field read in both confit and DuckDB.
- Switch the five existing NaN/NULL domain arms (Box-Cox, isotonic "raise", spline "error", encoder unknown, MissingIndicator `error_on_new`) to `error()`.

The cost is 16–93 ns per feature per row, against the twin's ~100 µs. With it, `check` can assert "raises iff the twin raises" on the 4.9% of rows it now skips. Whatever you rule, add ±inf to the gate's EDGES: that alone exposes a real parity breach where the twin answers (periodic degree-1 spline with `handle_missing="zeros"`).


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| Across the 160 configs (5,104 probe rows), the twin raised on 1,396 rows and the entry never did (c = 0). The entry answered fully-finite rows on 625 of them: 485 b/ab rows (59 configs, 17 classes) and 140 b0 rows. 0 REVERSE and 0 DIFF on the 3,708 rows the twin answers. | **PARTLY** | I rewrote the probe (verify-tolerated/vprobe.py: my own serving, one row at a time, and my own lane-dependency test). With the report's fit design it gives 5,104 rows, 1,396 twin raises, c = 0, and 3,708 '=' rows with 0 DIFF and 0 REVERSE. With a different design (4 features, gamma data, seed 12345, a fitted row as base) it gives 6,688 rows, 1,812 twin raises, c = 0, and still 0 DIFF and 0 REVERSE. However, recounting the report's own probe.json shows that 'fully finite' is wrong for 60 of the 625 rows. In probe.py the 'ab' class requires at least one non-finite lane that reads the bad input. The (class, all lanes finite) counts are b 425 True, b0 140 True, ab 60 False. | 565 twin-raise rows come back fully finite (425 b + 140 b0). Another 60 'ab' rows mix finite and non-finite lanes. The 485 / 59 / 17 split depends on how lane dependency is detected: my variant gives 55 configs and 16 classes. The 1,396, c = 0 and 0 REVERSE/DIFF figures hold. |
| In 51 of 160 configs a NULL comes back finite where the twin raises. The named examples hold: KBins top bin, Binarizer 1.0, Spline('error') 0.0 lanes, Isotonic(clip) fp[-1], MissingIndicator(-1) 0.0, and selectors drop the column. With 1,000 rows and one NULL, the twin fails and the entry returns 1,000 rows with the NULL row in the top bin. | **PARTLY** | I refitted every named example myself (vexamples.py) and all of them reproduce. KBins(quantile, ordinal, 5) answers NULL -> 4.0 and -inf -> 0. Binarizer(0.5) answers NULL -> 1.0 where numpy gives 0.0. Spline(handle_missing='error') answers NULL -> all 0.0. Isotonic(clip) answers NaN -> 10.0917, its top value, where sklearn with assume_finite gives NaN. MissingIndicator(-1) answers NaN -> 0.0. SelectKBest answers NaN or inf in its dropped column -> a finite row. The batch case also reproduces: over 1,000 rows with a NULL at row 517, the twin fails with 'Input X contains NaN' and the entry returns 1,000 rows with row 517 = {f0: 4.0}, the top bin. The count is the weak part. Of the 51 configs, 3 (FeatureUnion[5], SimpleImputer[6], SimpleImputer[7]) are 'ab' rows that still have a NaN lane. | A NULL comes back fully finite in 48 configs (my re-probe also finds 48). The other 3 are partly NaN. |
| 573 of the 625 finite answers equal sklearn with assume_finite=True. The other 48 are artefacts of DuckDB's NaN ordering (Binarizer, Isotonic clip, periodic spline at ±inf), so non-finite-in -> non-finite-out does not survive downstream DuckDB comparisons. | **PARTLY** | 573 + 48 = 621, not 625. The other 4 rows are ones where sklearn still raises with assume_finite=True (MissingIndicator's error_on_new, a domain check). Restricted to the 565 truly fully-finite rows, the split is 525 equal, 36 differ and 4 still raise. All of the differing rows are NaN-related: Binarizer and its pipelines, Isotonic(clip) on NaN, and periodic spline at ±inf (fmod(inf) = NaN, which is then routed). DuckDB 1.5.5 measured directly: NaN > inf is true, NaN = NaN is true, and NaN <> 0 is true. DuckDB documents this ('NaN compares equal to NaN and greater than any other floating point number') on its Numeric Types and PostgreSQL Compatibility pages. The cited sql_quirks page is blocked by the proxy, so I could not confirm that exact page. | The downstream-comparison argument applies equally to every NaN the twin itself returns: StandardScaler and imputers on NaN, and spline with handle_missing='zeros'. It is a property of DuckDB consumers in general, not evidence against the non-finite-out option in particular. |
| Generic guards are not exact. g1 misses 602 rows and over-raises 134. The tag-based g3 misses 156 and over-raises 112 (VarianceThreshold, plus Pipeline, ColumnTransformer and FeatureUnion whose own tags say allow_nan=False). | **CONFIRMED** | I recomputed the guards on my own re-probe (vguards.py) and got the same numbers: g1 602 / 134, g2 156 / 364, g3 156 / 112, with the same per-class split. Caveat: g3 is a weak baseline, because it reads the composition's own tags, which catalog_test._runs already documents as wrong. A guard built from leaf tags via _runs still misses 156 but over-raises only 22 (VarianceThreshold 6, Pipeline 6, ColumnTransformer 10). The conclusion stands either way: no tag-based guard is exact. |  |
| The sklearn rules: selectors use ensure_all_finite=not allow_nan and VarianceThreshold sets allow_nan=True; object-dtype input is checked only for NaN, so encoders answer ±inf on a row that holds a string; SplineTransformer skips validation under 'zeros'; Box-Cox rejects nanmin(X) <= 0; imputers reject NaN when missing_values is not NaN. | **CONFIRMED** | Every cited line in the venv's sklearn says what the report claims: feature_selection/_base.py:114, _variance_threshold.py:139, utils/validation.py:119-122, _encoders.py:54-56, _polynomial.py:1001, _data.py:3633, impute/_base.py:351-353 and 949-951, and the MissingIndicator tag at :1145. I also ran the object-dtype path myself (venc.py): OneHotEncoder(ignore) and OrdinalEncoder(use_encoded_value) on a (string, ±inf/NaN) row are answered by the twin, and the entry matches. Nuance: on object input the check is skipped entirely when NaN is allowed, which is the encoders' case. |  |
| An exact lane-0 guard costs +16 to +37 ns/feature/row for ±inf only and +34 to +93 for ±inf+NaN. The twin costs about 98-134 µs/row. The trap fires on any single field read, in confit and in DuckDB. | **PARTLY** | Timed with my own harness (vcost.py: 32 features, 10k-row calls, 9×3 interleaved, median). StandardScaler: OR-of-abs=inf +17.3 ns, ±inf+NaN +35.6 ns, which reproduces the report. KBins: +51 and +108 ns, above the report's range. A cheaper spelling the report missed, one test `(x0*0.0 + … + xk*0.0) <> 0`, traps ±inf, NaN and NULL for +0 to +31 ns per feature. A two-level ±inf-only version costs +0.3 to +29 ns. Each spelling traps the same probe values in confit and in DuckDB when the query reads only f5 and the bad value is in x3. A row with a NULL id still returns NULL under a guard, as with the twin. The twin at 32 features with 1,000-row calls costs 154-222 µs/row. So the guard is cheap next to the twin, but adds 37-100% to the cheapest entries. | The cost depends on how the guard is spelled. With the report's spelling it is about 17-108 ns/feature; with a single-test spelling it is about 0-31 ns/feature. Either way it is well under 2% of the twin. |
| Option 2 (non-finite in -> non-finite out) costs at least as much as the guard: Σ over lanes and the columns each reads ≥ Σ over columns. Measured +0.99 vs +1.09 µs/row for StandardScaler(32) and +2.70 vs +2.78 for KBins(32). | **REFUTED** | The inequality assumes option 2 needs a predicate in every lane, and that the guard's sum runs over the same columns. Neither holds. First, StandardScaler, RobustScaler, PolynomialFeatures, FeatureAgglomeration and unclipped MinMax/MaxAbs are all class 'a' in both probes: their arithmetic already turns non-finite input into non-finite output, so option 2 costs nothing there while the guard costs +0.5-1.1 µs/row at 32 features. The +0.99 µs measured for StandardScaler times a predicate it does not need. Second, a selector's dropped columns are read by no lane, so option 2 needs no predicate for them, but the guard must test them because the twin validates every input column. The guard sums over all columns; option 2 sums only over columns some lane reads. | Option 2 costs about the same as the guard only for entries built on comparisons (KBins, Binarizer, Quantile, Spline, Isotonic(clip), clip scalers, encoders). For propagating entries and selectors it is cheaper. The remaining argument against it is policy: it still turns errors into answers. |
| The gate leaves 4.9% of its rows uncompared (624 of 12,800, seeds 0-1), with OrdinalEncoder 22.1%, OneHotEncoder 19.0% and so on. EDGES has no ±inf, so infinite inputs are gated only in FunctionTransformer's SPECIALS test. | **CONFIRMED** | I counted the rows independently, calling est.transform on each gate row without confit or check() (vgate.py). Result: 624 of 12,800 (4.9%), with every per-class share identical. 0 of 12,800 gate rows hold ±inf. By cause: NaN/NULL 347, unknown category 129, isotonic bounds 69, spline knots 51, Box-Cox non-positive 21, MissingIndicator new-missing 5, intermediate inf 2. EDGES is at catalog_test.py:599. The only other ±inf in tests is _helpers_test.py, which checks helpers against numpy, not against the twin. |  |
| Parity breach where the twin answers: SplineTransformer(degree=1, n_knots≥4, periodic, handle_missing='zeros') at ±inf. The twin answers NaN, the entry answers 0.0 in f1, and check raises ParityError. It does not trigger with n_knots 2-3 or with degree 3. | **PARTLY** | The breach is real (vspline.py), but wider than the report says. It triggers at degree 1 with n_knots 4, 5, 6 and 9; at degree 2 with n_knots 5, 6 and 9 (lane f2); and at degree 3 with n_knots 6 and 9 (lane f3). It does not trigger at degree 2 with n_knots 3-4 or degree 3 with n_knots 4-5. The pattern is n_knots ≥ degree + 3, with the bad lane f{degree}. The report tested degree 3 only at small n_knots, so 'not with degree 3' is wrong. I also ran the gate's own check with ±inf added to EDGES over all 160 fixtures × 3 seeds (1,081 rows holding ±inf): no breach. It appears only after adding a handle_missing='zeros' periodic fixture (vgate_inf.py), which then breaches at degree 1 with 4 knots and degree 3 with 6 knots for every seed. | The breach covers any periodic spline under 'zeros' with n_knots ≥ degree + 3. Adding ±inf to EDGES alone exposes nothing: the gate also needs a handle_missing='zeros' fixture. |

### What the report missed or got wrong

1. 'Fully finite' is overstated throughout. 60 of the 625 rows (class 'ab') have a non-finite lane by construction. The real figures are 565 fully-finite rows and 48 configs (not 51) where NULL comes back fully finite. 573 + 48 also leaves out 4 rows where sklearn with assume_finite=True still raises.

2. The gate recommendation is wrong as stated. The report says adding ±inf to EDGES 'alone exposes a real parity breach'. I ran the gate's check with ±inf in EDGES over all 160 fixtures × 3 seeds (1,081 rows holding ±inf) and found no breach. The breach appears only when a handle_missing='zeros' fixture is added, which the report itself notes is missing. The breach is also wider than reported: any periodic spline under 'zeros' with n_knots ≥ degree + 3, including degree 3, which the report says does not trigger.

3. There is a twin-raise source the 'exact guard' construction does not cover. SplineTransformer(degree=0, extrapolation='constant', n_knots≥3) above the knots raises a numpy broadcast ValueError, which is an sklearn bug, not validation. The entry answers 0.0, as spline.py's own docstring records. Probing for ±inf/NaN cannot find this, and it is not one of the report's 'five domain arms'. So the claim that an exact guard is a small job is DERIVED, not demonstrated: no prototype was run across the catalog showing 0 misses and 0 over-raises. Over-raises are forbidden ('never the reverse'), and the risk from ulp-bounded intermediates near DBL_MAX is noted in the report but not measured.

4. The Option-2 dominance derivation is wrong. Propagating entries (StandardScaler, RobustScaler, PolynomialFeatures, FeatureAgglomeration, unclipped scalers) satisfy non-finite-out at zero cost. Selectors need no predicate on the columns they drop. The guard is cheaper only for entries built on comparisons.

5. The NaN-ordering argument against non-finite-out applies equally to the NaN values the twin itself legitimately returns. It describes how DuckDB consumers behave, not a defect specific to Option 2.

6. Misidentified example. Pipeline[9] is (SimpleImputer→StandardScaler)→PolynomialFeatures→Binarizer: x² overflows and Binarizer raises, and the entry answers Binarizer's 0/1. It is not 'SimpleImputer→StandardScaler→MinMaxScaler(clip)… clipped to 1'. Pipeline[8], which is that chain, shows no twin raise at ±1e308.

7. The recommendation's '18 of 33 classes' pairs the 59 configs with PowerTransformer λ<0, which is not among those 59 configs.

8. The guard cost depends on how it is spelled. A single `Σ x_j*0.0 <> 0` test traps ±inf, NaN and NULL for 0-31 ns/feature, against the report's 34-93 (108 for KBins in my run). The report's figures are an upper bound for one spelling.

9. A middle option is not considered: keep the tolerance but pin the entry to sklearn's own assume_finite=True arithmetic. That would fix the 48 NaN-order artefacts and give the tolerance a contract sklearn itself defines, though it would not stop answers like KBins putting NULL in the top bin.

10. Box-Cox at +inf answers -1/λ, which is 1e19 at λ=-1e-19 and 3.3e11 at λ=-3e-12. Those values are finite but not 'plausible'. This is a small point and does not change the conclusion.


### Does the recommendation follow? Yes

The core ruling, not to accept the tolerance as stated, follows from evidence I reproduced independently:
- The entry never raises where the twin does.
- In about 75 configs it returns fully-finite answers there, many of them plausible. For example, a NULL lands in KBins' top bin and the whole 1,000-row query succeeds where the twin's query fails.
- The gate cannot see this. It skips 4.9% of its rows and serves no ±inf.
- A lane-0 error() guard fires on any single field read, in confit and in DuckDB, and does not fire on NULL-id rows. It costs under 2% of the twin (154-222 µs/row at 32 features), so it is affordable.

Three supporting parts are weaker than presented:
1. Exactness. 'Require the guard' rests on an exactness argument that is derived, not demonstrated. At least one twin-raise source is not validation: the degree-0 constant spline's broadcast error. Any over-raise breaks 'never the reverse', so the ruling should require a prototype to pass a raises-iff-twin-raises check across the catalog before it replaces the tolerance. A guard that traps exactly where sklearn validation traps, and still tolerates misses elsewhere, is the defensible version.
2. Rejecting Option 2 on cost. This relies on a refuted inequality: for propagating entries and selectors Option 2 is cheaper or free. Option 2 should be rejected on policy (it still turns errors into answers), not on cost.
3. The EDGES step. 'Add ±inf to EDGES; that alone exposes the breach' is wrong. The gate also needs a handle_missing='zeros' spline fixture, and the breach covers more configurations than reported (n_knots ≥ degree + 3, not just degree 1).

The guard costs in the report are an upper bound. A single-test spelling is cheaper.

Scripts:
- scripts/verify-tolerated/vprobe.py, with vsum.py to summarise its output
- vexamples.py
- vspline.py
- vgate.py
- vgate_inf.py
- vcost.py
- vguards.py
- vboxcox.py
- vmisc.py
- venc.py

DuckDB NaN ordering sources: https://duckdb.org/docs/current/sql/data_types/numeric and https://duckdb.org/docs/lts/sql/dialect/postgresql_compatibility (both seen through search; direct fetches to duckdb.org are blocked).
