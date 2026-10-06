# Sparse outputs: what sklearn returns and how the step should take it

> Research note for the native loop's open decisions, 2026-10-06. Written by a research agent, then re-run by an independent adversarial verifier (its verdicts are at the end, and override the report where they disagree). Paths under `scripts/` are relative to this folder; see [README.md](README.md).


Scope: `loops/native/decisions/open/sparse-outputs.md`. Environment: sklearn 1.9.0, scipy 1.18.0, numpy 2.5.1, duckdb 1.5.5, CPython **3.14.0rc2**.

**Environment flag.** `sql_transform` does not import in this venv. pydantic 2.13.4 calls `typing._eval_type(prefer_fwd_module=…)`, which rc2 does not have, so `uv run pytest` fails at collection. I could not run the repo's own tests. My scripts load a shim first (`$S/_shim.py`). It strips that keyword and changes nothing in the repo.

`$S = scripts/sparse`. Every script and its `.out` file is there.

---

## 1. What fails today, and where (MEASURED: `q1_containers.py`, `q1b_end_to_end.py`, `q1c_width1.py`)

| | default `sparse_interface="spmatrix"` | `sklearn.config_context(sparse_interface="sparray")` (new in 1.9, `sklearn/_config.py:199-207`) |
|---|---|---|
| `est.transform([row])` | `csr_matrix` (1,k), float64 (MissingIndicator: `csc_matrix`, bool) | `csr_array` (1,k) |
| `[0]` | `csr_matrix` (1,k) | **1-D `coo_array` (k,)** |
| `list(row)` then `float()` | `[csr_matrix]`, then `TypeError … not 'csr_matrix'` (the record's error, reproduced exactly) | k `np.float64` values, and `float()` succeeds |
| `np.asarray(out)` | **0-d object array** | 0-d object array |
| `.toarray()` | ndarray (1,k), dtype kept | ndarray (1,k) |
| `.todense()` | `np.matrix`. `[0]` stays 1×k, so it is unsafe | ndarray |

sklearn builds a `csr_array` internally and converts it on the way out (`_encoders.py:1085-1092`, `utils/_sparse.py:11-34 _align_api_if_sparse`).

**The record's "declared width and lanes are unchanged" is not what happens today.** The fit probe is `probe = np.asarray(est.transform(feats[idx][:1]))` with `width = probe.shape[1] if probe.ndim > 1 else 1` (`_projection.py:352-353`). A sparse output becomes a 0-d object array, so the probe sees width 1. `get_feature_names_out` returns k names, which does not match, so the names fall back to `f0`. The step is declared `struct<f0: double>`. Through `SQLProjection`:

- **Whole-struct read** with `OneHotEncoder()` or `KBinsDiscretizer(encode='onehot')`:
  - spmatrix: fit succeeds, then DuckDB `transform` and confit `infer` raise the TypeError on the first row.
  - sparray: the same calls raise `UDFError: produced 3 values, declared 1`.
- **Field read** `t(...).c_red`: fit raises `no output field 'c_red'; it fits to ['f0']`. The error is misleading.
- **k = 1** (`OneHotEncoder(drop='if_binary')` on one binary column): the declaration is right (`struct<c_b>`).
  - spmatrix raises at serve.
  - sparray **serves correct values by accident**, because a 1-D `coo_array` iterates to scalars.

So densifying only in `_udf` would just swap the TypeError for the width-mismatch error. **Both the fit site and the serve site have to densify.**

## 2. Inventory (MEASURED: `q2_inventory.py`, all 70 `all_estimators('transformer')` minus out-of-scope, dense float64 input)

| transformer (coverage.md status) | sparse for dense input when | dense switch | measured shape |
|---|---|---|---|
| OneHotEncoder (native) | **default** `sparse_output=True` | `sparse_output=False` | (1,186) |
| KBinsDiscretizer (native) | **default** `encode='onehot'` | `'onehot-dense'` / `'ordinal'` | (1,19) |
| RandomTreesEmbedding (not yet) | **default** `sparse_output=True` | `sparse_output=False` | (1,1795) |
| KNeighborsTransformer (not yet) | **always** (graph, width = n_fit) | **none** | (1,60) |
| RadiusNeighborsTransformer (not yet) | **always** | **none** | (1,60) |
| SplineTransformer (native) | `sparse_output=True` | default | (1,28) |
| MissingIndicator (native) | `sparse=True` (`'auto'` is dense for dense input) | `sparse=False`/`'auto'` | csc, bool |
| FunctionTransformer (native) | when `func` returns sparse | author's func | |
| ColumnTransformer (composition) | at fit, per instance, when overall density < `sparse_threshold` (0.3) (`_column_transformer.py:994-1003`) | `sparse_threshold=0` | OHE-only part: sparse; Scaler(3 cols)+OHE: dense |
| FeatureUnion (composition) | **any** part sparse (`pipeline.py:2128-2129`) | every part dense | (1,190) |
| Pipeline (composition) | last step sparse, or a sparse-preserving step after a sparse one (OHE→MaxAbsScaler, OHE→StandardScaler(with_mean=False)) | inner params | (1,186) |
| PolynomialFeatures, SimpleImputer(add_indicator), SparseRandomProjection | sparse only for sparse **input**, which the step never passes (it calls `transform([list])`) | n/a | dense |

LabelBinarizer and MultiLabelBinarizer (`sparse_output`) are out of scope per coverage.md.

The record names 2 sparse-by-default transformers; the in-scope count is **5**: OHE, KBins, RTE, KNT, RNT.

- Option 1 serves all rows of the table.
- Option 2 can only name a remedy where a dense switch exists. KNT, RNT and a sparse-returning FunctionTransformer could never be served. FeatureUnion and Pipeline need edits to inner parts.

## 3. Value equality

**What densifying does to each value (SOURCED + DERIVED).** `toarray()` ends in scipy's `csr_todense`, which runs `Bx_row[Aj[jj]] += Ax[jj]` into a zeroed buffer (scipy v1.18.0, `scipy/sparse/sparsetools/csr.h:289`). So each lane is D(v) = (+0.0) + v. Under round-to-nearest:
- D(v) = v exactly for every v ≠ ±0, including NaN, ±inf and subnormals.
- D(±0) = +0.
- Duplicate stored entries are summed.

Densifying can therefore change a value only where a −0.0 is stored. MEASURED in `q3e_toarray_negzero.py`: a stored −0.0 comes out as +0.0 for csr, csc and 1-D coo, through both `toarray` and `todense`.

**OneHotEncoder and KBins are identical by construction (SOURCED).** `OneHotEncoder.transform` builds the CSR with `data = np.ones(...)`, then returns `_align_api_if_sparse(out)` if sparse, else `out.toarray()` (`_encoders.py:1083-1094`). The dense configuration *is* `.toarray()` of the same matrix. KBins one-hot uses an inner `OneHotEncoder(sparse_output=self.encode == "onehot")` (`_discretization.py:395-399`), so the same holds.

MEASURED in `q3_values.py`, per row and compared on raw bits:

| family | configs | rows | lanes | differing rows |
|---|---|---|---|---|
| OHE (handle_unknown ignore / infrequent_if_exist / warn; drop first / if_binary; min_frequency, max_categories; dtype float32 / int64; NaN and None categories; ±0.0 categories; unknowns) | 20 | 4,062 | 41,615 | **0** (no stored zeros) |
| KBins (quantile / uniform / kmeans × 2, 5, 9 bins; ±0, ±5e-324, ±1e300) | 9 | 1,386 | 21,098 | **0** |
| RandomTreesEmbedding / MissingIndicator | 1 / 2 | 80 / 120 | 17,200 / 360 | **0** |
| SplineTransformer (degree 0–5 × n_knots 2, 3, 6 × 5 extrapolations × bias × knots; probes on knots, nextafter, ±0, NaN, ±1e12) | 268 | 30,775 | 270,548 | **93**, all in degree 0 + `constant` |
| ColumnTransformer `sparse_threshold=1` vs `0` | 3 | 189 | 756 | **10**, all −0.0 vs +0.0 |
| FeatureUnion (OHE sparse vs OHE dense, plus a dense part) | 2 | 126 | 25,326 | **25**, all −0.0 vs +0.0 |

Where the table shows differences:

- **SplineTransformer, degree 0, `extrapolation='constant'`, x > xmax.** Here `-degree:` is `0:`.
  - The sparse branch assigns the whole basis row, so the sparse twin answers f_max (`_polynomial.py:1180`).
  - The dense branch's slice is empty (`_polynomial.py:1187`). With one spline the dense twin answers 0.0 (93 rows). With more than one it raises a shape mismatch (390 rows over 8 configs), while the sparse twin answers.
  - So the sparse Spline twin is **not** "the same values in another container" in that corner. Everywhere else it was bit-identical.
- **ColumnTransformer and FeatureUnion.** When the stacked output is sparse, dense parts go through `sparse.hstack`, which does not store −0.0. A dense part's −0.0 therefore comes out as +0.0. MEASURED in `q3d_negzero.py`; the cases were a FunctionTransformer `np.negative` part and a StandardScaler part fed −0.0.

**A sparse intermediate inside a Pipeline, with the same fitted state (MEASURED: `q3b_pipeline_paths.py`, 200 rows per pair):**
- Producers OHE, KBins and RTE into MaxAbs, Standard(with_mean=False), Robust(with_centering=False), Normalizer l1/l2/max, PolynomialFeatures(2), Binarizer, VarianceThreshold and SimpleImputer: **0 differing lanes**.
- Producer Spline (real-valued) into MaxAbs, Standard, Robust: up to 1 ulp. Into Normalizer l2 / l1: up to 1 / 3 ulps.

DERIVED: the sparse branch of each scaler computes x·fl(1/s); the dense branch computes fl(x/s) (`_data.py:1129` vs `:1134`, `:1395` vs `:1399`, `:1779` vs `:1784`). The first has relative error ≤ 2u + u², the second ≤ u, so they can differ by a few ulps. For x ∈ {0, 1} they agree exactly: 1·fl(1/s) = fl(1/s). Sums of 0/1 values are exact integers in any order, which covers the Normalizers.

**The dense configuration of a pipeline is a different model.** `Pipeline(OHE(), StandardScaler(with_mean=False))` and its dense version fit different `scale_` values (sparse `mean_variance_axis` vs dense variance). Their outputs differed by 5–15 ulps on every row (203/203). So option 2's suggested fix silently changes the author's fitted model.

**Explicit zeros carry meaning only in neighbour graphs.** `KNeighborsTransformer(mode='distance')` and `RadiusNeighborsTransformer` store 0.0 for a neighbour at distance 0 (MEASURED: stored `[0.0, 0.67, …]`). Densifying merges that with "not a neighbour". The DOUBLE lane type cannot express the difference either way.

## 4. How peer converters handle it (SOURCED, fetched 2026-10-06)

| system | sparse-default OHE | KBins `onehot` |
|---|---|---|
| skl2onnx | Always emits dense `FloatTensorType([N, k])` and ignores `sparse_output` (`skl2onnx/shape_calculators/one_hot_encoder.py:18-20`; reshape at `operator_converters/one_hot_encoder.py:262`) | **Refuses at conversion**: `RuntimeError("onehot encoding not supported. ONNX does not support sparse tensors…")` (`operator_converters/k_bins_discretiser.py:24-30`) |
| hummingbird | Dense: `torch.cat(...).float()` (`operator_converters/_one_hot_encoder_implementations.py:69,108`) | Treats it like `onehot-dense`: `if self.encode in ["onehot-dense","onehot"]` (`_discretizer_implementations.py:54`) |
| sklearn `set_output(transform="pandas")` | Refuses at transform (`_encoders.py:1033-1040`; generic check `utils/_set_output.py:299-304`) | Same generic refusal |
| sklearn ColumnTransformer, dense stack | Densifies parts itself: `Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]` (`_column_transformer.py:1125`) | |

URLs: `https://github.com/onnx/sklearn-onnx/blob/main/skl2onnx/...` and `https://github.com/microsoft/hummingbird/blob/main/hummingbird/ml/operator_converters/...`. I did not check MLflow.

## 5. Cost (MEASURED: `q5_cost.py`, `q5b_width_costs.py`; median of 5, OPENBLAS_NUM_THREADS=1, one OHE feature with k categories)

| k | `transform` sparse (µs) | `transform` dense config (µs) | `.toarray()` (µs) | densify share of step | bytes/row |
|---|---|---|---|---|---|
| 10 | 346 | 296 | **1.06** | 0.3 % | 80 |
| 100 | 258 | 253 | 1.11 | 0.4 % | 800 |
| 1,000 | 282 | 264 | 1.43 | 0.4 % | 8 KB |
| 10,000 | 377 | 365 | 2.83 | 0.2 % | 80 KB |
| 100,000 | 1,972 | 2,009 | 16.5 | 0.2 % | 800 KB transient |

KBins one-hot (4 features × 5 bins): sparse 783 µs, dense 720 µs, `toarray` 1.09 µs. The sparse transform is already within about 10 % of the dense one.

Densifying is never the width problem.

- **Per row:** `PythonTransform.return_types` is recomputed on every `__call__` (`_udf.py:316`) at 0.6–0.8 µs per lane, which is **70.6 ms per row at k = 10⁵**. The dense configuration pays this too.
- **At construction:** `__post_init__`'s case-collision check is O(k²): 3.19 s at k = 10⁴, roughly 5 min extrapolated at 10⁵.

Those are separate issues and apply to every transformer at that width.

## 6. Design

**Densify at three sites, with one helper.** Use `.toarray()`, not `np.asarray` (gives 0-d object) and not `.todense()` (gives `np.matrix` for spmatrix):
```python
def _dense(out):
    m = sys.modules.get("scipy.sparse")          # no scipy import; a sparse result implies it is loaded
    return out.toarray() if m is not None and m.issparse(out) else out
```
1. `_udf.py:314`: `row = _dense(est.transform([vals]))[0]`. This covers both sparse interfaces and every format, plus `Named` / `OrderSensitive`, which forward `transform`.
2. `_projection.py:352`: `probe = np.asarray(_dense(est.transform(feats[idx][:1])))`. Without this the declared struct stays `f0`.
3. `native/encode.py:110` (`_probe`): `np.asarray(sparse, dtype=float64)` raises `ValueError`, which `_probe` reads as "the twin raises here". MEASURED in `q6_probe_trap.py`: with `sparse_output=True`, every probe returns `None`, against `[1,0,0]`, `[0,1,0]`, `[0,0,0]` dense. Dropping the `sparse_output` guard without this fix would produce an all-raise translation.

**Check of the fix (MEASURED: `q6_fix_demo.py`, sites 1 and 2 monkeypatched outside the repo).** I ran OHE(handle_unknown='ignore'), KBins(onehot) and Spline(sparse_output=True) against their dense configs, under both interfaces. Probes covered unknown, NULL, −0.0 and out-of-range values. In every case the declared structs were identical, and DuckDB `transform` and confit `infer` were bit-equal.

**What the catalog needs after option 1:**
- **OHE and KBins as leaves:** remove the sparse guard and reuse the existing translation. This is exact by construction.
- **Spline:** accept `sparse_output=True` except degree 0 + `constant`, which keeps `NotNative` or gets its own above-xmax lanes.
- **Pipeline:** a sparse step that is not last should refuse, alongside the existing `_float64_out` check. The 0/1 producers measured exact, but Spline did not, and this is a per-consumer argument.
- **FeatureUnion with any sparse part:** refuse, or wrap the other parts' lanes in `+ 0.0`. In DuckDB 1.5.5, `-0.0 + 0.0 = +0.0` and `x + 0.0` is not folded away (`q3d_negzero.py`). I did not check confit's handling.
- **ColumnTransformer:** keep refusing when `sparse_output_` is set.

**Tests to add:**
- `PythonTransform` and `SQLProjection` over `OneHotEncoder()` and `KBinsDiscretizer(encode='onehot')`:
  - declared struct equals the dense config's, with a field read such as `.c_red` working at fit;
  - output bit-equal to the dense config on known, unknown, NULL and −0.0 rows;
  - repeated under `config_context(sparse_interface='sparray')`.
- The width-1 case (`drop='if_binary'`).
- Flip the `onehot` case of `catalog_test.py:1056` (`test_kbins_refuses`) to `check(...) == n`.
- Composition refusals: a sparse step that is not last in a Pipeline, and a FeatureUnion with −0.0.

**Option 2 and the hybrid.** Option 2 does fail earlier. But once fix (2) is in, the step's fit already reads the true width and names, so most of that benefit is gone. Option 2 also:
- has no remedy for KNeighborsTransformer, RadiusNeighborsTransformer or a sparse FunctionTransformer;
- forces edits inside FeatureUnion and Pipeline parts;
- changes downstream fitted state (5–15 ulps).

A width warning tied to sparsity would point at the wrong cause: the dense configuration has the same k lanes and the same per-lane cost. If a width guard is wanted, it should be a separate decision that covers every step.

**Unverified:** MLflow behaviour; confit's handling of `x + 0.0`. The transform timings are specific to this machine: 250–2,000 µs here, against the ~118 µs in goal.md for a different transformer.


## The researcher's recommendation

Take option 1, but apply the densify at three sites through one `_dense(out)` helper (`.toarray()` behind `scipy.sparse.issparse`, looked up via `sys.modules`), not only in `_udf`. The sites are the serve call (`_udf.py:314`), the fit probe (`_projection.py:352`) and the catalog's OneHotEncoder probe (`native/encode.py:110`). Today the fit probe declares sparse steps as width-1 `f0`, so a serve-only fix just changes the error. For OneHotEncoder and KBinsDiscretizer the densified values equal the dense configuration bit-for-bit by construction, since sklearn's dense path is `.toarray()` of the same CSR (measured 0 differences over ~5.4k rows). Their catalog entries can therefore drop the guard and reuse the existing translation. Do not apply 'only the container differs' to everything else:
- Spline sparse needs a carve-out for degree 0 + `constant` (f_max vs 0/raise).
- Pipeline compositions should refuse a sparse step that is not last (Spline intermediates shift scalers by up to 3 ulps).
- FeatureUnion and ColumnTransformer with sparse output should keep refusing, or canonicalise dense parts' lanes with `+ 0.0`, because scipy turns −0.0 into +0.0.

Option 2 has no remedy for KNeighborsTransformer, RadiusNeighborsTransformer or a sparse FunctionTransformer, and its "set sparse_output=False" advice changes downstream fitted state by 5–15 ulps. The hybrid's width warning would point at the wrong cause: densifying costs ≤0.4% per row, and the real width costs (`return_types` per call, the O(k²) name check) hit dense configurations equally, so treat them as a separate, general decision.


## Adversarial verification

| claim | verdict | evidence | correction |
|---|---|---|---|
| Today the fit probe np.asarray(sparse) is 0-d object, so a sparse step is declared struct<f0>; whole-struct read fails at first row (TypeError under spmatrix, 'produced k values, declared 1' under sparray); field read fails at fit with 'fits to ['f0']'; densifying only in _udf would not fix it, _projection.py:352 must densify too. | **CONFIRMED** | My own script v1_e2e.py used different data (4 string categories, 5k rows) and KBins(strategy='uniform'). np.asarray(csr) has shape () and dtype object. Under spmatrix, OHE() and KBins(onehot) declare struct<f0: double>, and both DuckDB transform and confit infer raise 'float() argument ... not csr_matrix'. Under sparray they raise 'produced 4 values, declared 1' (OHE) and 'produced 3 values, declared 1' (KBins). The field read .k_b fails at fit with MarginalizeError '... it fits to ['f0']'. I also patched only PythonTransform.__call__ with .toarray() (v12_serveonly.py). The step is still declared struct<f0> and serve fails with 'produced 3 values, declared 1'. The line numbers (_udf.py:314-315, _projection.py:352-353) are correct. |  |
| sklearn 1.9 adds sparse_interface (default 'spmatrix'); under 'sparray' est.transform([row])[0] is a 1-D coo_array that float() accepts, so a width-1 sparse output (OHE(drop='if_binary')) serves correctly by accident while spmatrix raises. | **CONFIRMED** | _config.py:199-207 documents sparse_interface, versionadded 1.9. utils/_sparse.py:_align_api_if_sparse converts at return time. Under sparray, [0] is coo_array (3,) and iterates to np.float64 values. Under spmatrix, [0] is csr_matrix (1,3). OHE(drop='if_binary') declares struct<b_q> and serves under sparray, matching the expected values on all 300k rows in DuckDB (v1b_threads.py), but raises the TypeError under spmatrix. Extra finding: the container is chosen from the config in force at serve time. A model fitted under sparray and then served outside the context raises the TypeError. So the 'accident' depends on the caller's environment, not on the model. |  |
| For OHE and KBins, densifying is bit-identical to the dense configuration by construction (dense path = out.toarray() of the same CSR; KBins one-hot goes through an inner OHE); 0 differing rows over 20 OHE and 9 KBins configs. | **CONFIRMED** | The source matches the claim. _encoders.py:1085-1094 builds csr_array(data=np.ones) and returns out.toarray() when not sparse_output. _discretization.py:395-399 builds OneHotEncoder(sparse_output=self.encode == 'onehot'). My own sweep (v3_ohe_kbins.py) ran under both interfaces:<br>- OHE: 648 configs (handle_unknown × drop × min_frequency × max_categories × dtype float64/float32/int64; numeric with ±0/NaN and string with None; unknown probes), 129,600 rows. 0 rows differ on raw bits or dtype, and no zeros are stored.<br>- KBins: 48 configs (uniform/quantile/kmeans × 2/4/7/11 bins × dtype; ±0, ±5e-324, ±1e300, ±1e308), 11,856 rows. 0 rows differ. |  |
| Densify is D(v) = +0.0 + v per lane (scipy csr_todense into a zeroed buffer): exact except stored -0.0 -> +0.0; sparse CT/FU stacking drops a dense part's -0.0, so sparse and dense configs differ on ±0 lanes; DuckDB x + 0.0 reproduces this and is not folded. | **CONFIRMED** | I fetched scipy v1.18.0 csr.h. Line 289 is `Bx_row[Aj[jj]] += Ax[jj];`, inside csr_todense. A stored -0.0 comes out as +0.0 for csr, csc, coo, bsr, dia and dok (matrix and array) and for 1-D coo. Exception: lil_matrix/lil_array.toarray() keeps -0.0, so D(v) depends on the format. This does not matter in practice because sklearn never returns lil. csr_array(dense [-0.0, 1.0]).nnz is 1. ColumnTransformer with FunctionTransformer(np.negative) plus OHE: sparse_threshold=0.3 (sparse) vs 0 (dense) differ on 19/60 rows, all +0.0 vs -0.0. FeatureUnion: 20 differing lanes, all ±0. In DuckDB, signbit(x + 0.0) is false for x = -0.0 from a table column. I also checked confit, which the report listed as unverified: SQLProjection('SELECT x + 0.0 ...').infer_batch gives +0.0 for x = -0.0, the same as DuckDB. |  |
| SplineTransformer(sparse_output=True) differs from dense at degree=0, extrapolation='constant', x > xmax (sparse answers f_max; dense answers 0.0 with one spline or raises with more); elsewhere sparse and dense were bit-identical across the other 258 configs. | **PARTLY** | _polynomial.py:1180 (sparse `[mask, -degree:]`) vs 1187 (dense empty slice) is correct. My sweep (v5_spline.py) reproduces both effects: with n_knots=2, sparse returns 1.0 and dense 0.0 (8 rows); with more splines, dense raises and sparse answers (32 rows). The 'elsewhere bit-identical' part is incomplete. The researcher's q3_values.py skips this case explicitly (`if extrap == 'periodic' and degree == 0: continue`), and the report never says so. In that configuration the sparse twin raises ValueError('inconsistent shapes', scipy/sparse/_base.py:836) on every row (432/432 probe rows, 12 configs), while the dense twin serves. | Sparse Spline differs from dense in two corners. At degree 0 + 'constant' it gives different values. At degree 0 + 'periodic' the sparse twin raises on every row while the dense twin answers. Both need carve-outs. |
| In-scope sparse-by-default for dense input: OHE, KBins, RTE, KNT, RNT (5, not 2); opt-in: Spline, MissingIndicator(sparse=True), FunctionTransformer, ColumnTransformer, FeatureUnion, Pipelines; KNT/RNT have no dense switch, so option 2 can never serve them. | **PARTLY** | I swept all 84 all_estimators('transformer') with default constructors on dense input (v6_inventory.py). Sparse by default: KBinsDiscretizer, KNeighborsTransformer, OneHotEncoder, RadiusNeighborsTransformer, RandomTreesEmbedding, plus TfidfTransformer, which coverage.md marks out of scope. KNT and RNT have no output-format parameter (their params are algorithm, leaf_size, metric, metric_params, mode, n_jobs, n_neighbors/radius, p). The inventory is right. 'Option 2 can never serve them' is too strong: an author can append FunctionTransformer(lambda X: X.toarray(), accept_sparse=True). That is clumsy, but it works. | The count of 5 is correct. Under option 2, KNT and RNT need an author-written densifying wrapper, not 'never'. |
| Densify costs 1.06 us/row at k=10 and 16.5 us at k=1e5 (~0.2-0.4% of the step); return_types is recomputed per call at ~0.6-0.8 us/lane (70.6 ms/row at 1e5) and __post_init__'s collision check is O(k^2) (3.19 s at 1e4); these hit dense configs equally. | **CONFIRMED** | v7_cost.py (OPENBLAS_NUM_THREADS=1, string categories): toarray takes 1.03 us at k=10, 1.29 us at k=1e3 and 24 us at k=1e5, which is 0.02-0.23% of transform. return_types costs 1.2-2.0 us/lane on this (loaded) machine. __post_init__ takes 0.075 s at k=1e3 and 6.1 s at k=1e4; the ×82 growth for 10× k is quadratic. The absolute numbers depend on the machine, but the conclusion holds. The report also understates the per-row width cost: on the DuckDB path, _scalar recomputes _lanes three times per row (via __call__ line 316, then `self.return_types` at line 184 and `self.return_names`). |  |
| The catalog's OHE probe (encode.py:110) raises ValueError on a sparse twin, so every probe returns None and lifting the guard without densifying there would produce an all-raise translation; Pipelines with a sparse intermediate need a guard: 0/1 producers exact, Spline into scalers/normalizers up to 3 ulps (x*fl(1/s) vs fl(x/s), _data.py:1129/1134, 1395/1399, 1779/1784). | **PARTLY** | np.asarray(csr, dtype=float64) does raise ValueError('setting an array element with a sequence'), and _probe returns None. The consequence is wrong, though. I stripped the guard from _encode and ran to_native(strict=True) on a sparse OHE (v9_probe.py). It raises NotNative('OneHotEncoder: blocks of 3 lanes in all') through encode.py:97-98. That is a safe refusal, not an all-raise translation.<br><br>Pipelines (v8_pipe.py, same fitted consumer, sparse vs dense input): OHE, KBins and RTE into 7 consumers show 0 diffs. Spline gives:<br>- StandardScaler(with_mean=False), MaxAbsScaler, RobustScaler(with_centering=False): ≤1 ulp<br>- Normalizer l2: 2 ulps<br>- Normalizer l1: 4 ulps, not 3<br>- Normalizer max: 0<br><br>The mechanism is also misattributed. The cited _data.py lines are StandardScaler, MaxAbsScaler and RobustScaler only. Normalizer's sparse path (sparsefuncs_fast: a sequential sum over nonzeros, then `X_data[j] /= sum_`) differs from the dense path (np.sum or row_norms via einsum) in summation order, not by a reciprocal multiply. For the scalers, fl(x·fl(1/s)) and fl(x/s) round two reals less than 1 ulp apart, so they differ by at most 1 ulp, not 'a few'. | Without the probe fix, the catalog refuses (NotNative); it does not mistranslate. The Spline-intermediate drift is ≤1 ulp for scalers and data-dependent for Normalizer (4 ulps seen), caused by summation order. |
| A two-site densify (fit probe plus __call__) makes OHE(), KBins(onehot) and Spline(sparse_output=True) declare the same struct as their dense configs and serve bit-equal on DuckDB transform and confit infer, under both interfaces. | **CONFIRMED** | In v10_fix.py I patched the source of both sites myself (exec of modified SQLProjection._fit_step and PythonTransform.__call__, outside the repo). Probes covered unknown, NULL, -0.0, -5e-324, ±1e300 and out-of-range values. Under spmatrix and sparray, OHE(handle_unknown='ignore'), KBins(4 bins, onehot) and Spline(degree=3, constant) have equal declared structs, and DuckDB transform and confit infer_batch are bit-equal to their dense configs. The field read .c_red works at fit and serve. Spline degree 0 + constant still differs, as the report says. Without the patch, the same script reproduces the failures. |  |
| skl2onnx emits dense [N,k] for OHE regardless of sparse_output but refuses KBins(onehot); hummingbird densifies both; sklearn ColumnTransformer densifies sparse parts with .toarray() when the stack is dense. | **CONFIRMED** | Fetched from GitHub main: skl2onnx shape_calculators/one_hot_encoder.py:20 sets FloatTensorType([instances, categories_len]), and neither the shape calculator nor the converter mentions sparse_output. operator_converters/k_bins_discretiser.py:24-30 raises RuntimeError('onehot encoding not supported. ONNX does not support sparse tensors...'). In hummingbird, _discretizer_implementations.py:54 is `if self.encode in ["onehot-dense", "onehot"]`, and _one_hot_encoder_implementations.py:69/108 returns torch.cat(...).float(). sklearn _column_transformer.py:1125 is `Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]`. Density decision: :994-1003. FeatureUnion: pipeline.py:2128. |  |

### What the report missed or got wrong

1. **A fourth densify site is missing.** `sql_transform/model/_foreign.py:128` (`_EstimatorTransform`, used by `Transform.from_estimator`) runs `np.asarray(instance.transform(...))` and then `out[:, i]`. I ran `Transform.from_estimator(KBinsDiscretizer(n_bins=3, strategy='uniform'), ...)`: transform raises `IndexError: too many indices for array: array is 0-dimensional`, while `onehot-dense` serves. The shared `_dense` helper should cover this site, or the decision should say this API is out of scope.

2. **The Spline carve-out is incomplete.** The researcher's `q3_values.py` skips periodic + degree 0 without saying so. In that configuration the sparse twin raises `ValueError` on every row (432/432) while the dense twin answers. If the catalog lifts the Spline sparse guard with only the degree 0 + constant carve-out, it would serve SQL values for a model whose Python step can never serve a row. `native/_check.check()` skips any row where the step raises, for any exception, so such a check compares 0 rows. It passes only if the test asserts `compared == n`.

3. **The probe-trap consequence is overstated.** Without the `encode.py:110` fix, lifting the guard gives `NotNative('blocks of N lanes in all')`, a safe refusal (verified). The fix is still needed to make sparse OHE native, but it is not a correctness hazard.

4. **MissingIndicator(sparse=True) is left out of the catalog follow-ups.** `impute.py:69-70` guards it too, and densified output equals the dense config (0/120 rows differ in my run; the report's own run also showed 0).

5. **Pipeline numbers are fixture-specific, and the Normalizer mechanism is misattributed.**
   - Spline into Normalizer l1 reaches 4 ulps, not 3.
   - The cause for Normalizer is summation order, not `x*fl(1/s)`.
   - For the three scalers the drift is at most 1 ulp, not 'a few'.
   - The refit drift between the sparse and dense Pipeline(OHE, StandardScaler(with_mean=False)) grows with n: 3, 16 and 65 ulps max output difference at n = 37, 203 and 1000. So '5–15 ulps' describes one fixture. The direction of the argument holds.

6. **The blanket Pipeline refusal goes further than the evidence.** 0/1 producers (OHE, KBins, RTE) were bit-exact into all seven consumers I tested, so a per-producer rule would keep common pipelines like OHE→MaxAbsScaler.

7. **Smaller points:**
   - `lil.toarray()` keeps -0.0, so D(v) = +0 + v depends on the format.
   - confit's `x + 0.0` does canonicalise -0.0 (I verified it), which closes the report's open item.
   - The repo's test helpers `catalog_test.py:767/943/1015` and `compose_test.py:31` read width as `np.asarray(est.transform(X[:1])).shape[1]`. They will raise `IndexError` on sparse steps, so the proposed test flips need `_dense` there too.
   - sklearn chooses the sparse container from the config in force at serve time (thread-local). The same fitted model serves or fails depending on the caller's config, which is one more reason to densify in the step.


### Does the recommendation follow? Yes

The core recommendation follows from evidence I reproduced independently: option 1, densifying with `.toarray()` behind `issparse` at both the `SQLProjection` fit probe and the `PythonTransform` serve call. A serve-only fix just swaps errors, as verified. With both sites patched, OHE, KBins and Spline(degree 3) match their dense configs bit for bit on DuckDB and confit under both sparse interfaces. Densifying costs very little.

The OHE/KBins catalog follow-up (drop the guard, reuse the dense translation) is justified by construction and measurement. `discretize.py`'s else-branch already handles both onehot forms.

Corrections before adopting it:
1. **Sites:** use the helper at `model/_foreign.py:128` as well, or rule that path out of scope. The `encode.py:110` site is needed only to make OHE native; without it the catalog refuses safely, so it is not a correctness hazard.
2. **Spline:** carve out degree 0 + periodic as well as degree 0 + constant. With periodic, the sparse twin raises on every row.
3. **MissingIndicator:** add MissingIndicator(sparse=True) to the guards to drop.
4. **Pipelines:** the blanket refusal of non-last sparse steps is defensible as a conservative default. The evidence supports allowing 0/1 producers (OHE, KBins), and refusing only real-valued sparse intermediates such as Spline.
5. **FeatureUnion/ColumnTransformer:** refusing, or wrapping lanes in `+ 0.0`, is sound. `+ 0.0` now checks out on both DuckDB and confit.

The arguments against option 2 and the hybrid hold in substance:
- Densifying is not the width cost. The real width costs, `_lanes` recomputed three times per row and the O(k²) `__post_init__`, apply equally to dense configs.
- Pipeline refits really do change fitted state.

'No remedy for KNT/RNT' should read 'only an author-written densifying wrapper'.
