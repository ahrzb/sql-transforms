# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands.

## Next

Easiest first; each is one family, one PR.

1. **Selectors:** `VarianceThreshold`, `SelectKBest`, `SelectPercentile`,
   `SelectFpr`/`Fdr`/`Fwe`, `GenericUnivariateSelect`, `SelectFromModel`,
   `RFE`, `RFECV`, `SequentialFeatureSelector`: `transform` keeps a fitted
   column subset.
2. **Linear projections:** `PCA` (`whiten`), `IncrementalPCA`,
   `TruncatedSVD`, `FactorAnalysis`, `FastICA`, `GaussianRandomProjection`,
   `SparseRandomProjection`, `PLSSVD`/`PLSRegression`/`CCA`/`PLSCanonical`
   (x scores), `LinearDiscriminantAnalysis`: a matvec, so a measured bound
   (`_helpers.dot`).
3. **`PolynomialFeatures`** (`degree`, `interaction_only`, `include_bias`).
4. **Encoders over strings:** `OrdinalEncoder`, `OneHotEncoder`
   (`handle_unknown`, `drop`, infrequent categories), `TargetEncoder`
   (transform of new rows only). Needs string features in the fixtures.
5. **`KBinsDiscretizer`** (`encode="ordinal"`; `onehot-dense` after 4).
6. **Non-linear maps:** `PowerTransformer` (Yeo-Johnson, Box-Cox,
   `standardize`), `QuantileTransformer` (interpolation over quantiles),
   `SplineTransformer`, `FunctionTransformer` for numpy ufuncs with a SQL
   twin, `AdditiveChi2Sampler`, `SkewedChi2Sampler`, `RBFSampler`,
   `PolynomialCountSketch`.
7. **Distances to fitted centres:** `KMeans`, `MiniBatchKMeans`,
   `BisectingKMeans`, `Birch` (`transform` = distances).
8. **Compositions:** a step whose instances are `Pipeline`s of catalog
   entries (compose the translations), then `ColumnTransformer` and
   `FeatureUnion`.

## Needs from confit

- **Per-read expansion** (confit PLANS' first item for this catalog). A
  field read expands the function's whole body, so build time grows with
  reads × body. Through `to_native` (which builds the all-lanes query
  once), with one / three fitted groups, on master 6306c8f:
  `StandardScaler` 0.09 / 0.27 s at 32 features, 0.35 / 1.0 s at 64,
  1.6 s / past the 4M-token cap at 128; `MinMaxScaler(clip=True)`
  0.27 / 0.86 s at 32, 1.2 / 4.2 s at 64; `Normalizer` (every lane repeats
  the row norm) l1 1.6 s and l2 3.3 s at 16, and at 32 refused past the
  compiled-size limit, but only after 10 s (l1) and 26 s (l2) of building;
  max 1.8 s at 8, where it is capped meanwhile. Served since #336–#338: a
  constant CASE result counts as trap-free (a 32-lane step serves a
  64-row call in 331 µs, against 297 µs inline and 5,081 µs before), a
  named refusal past Cranelift's size limit, `greatest`/`least` without
  the exponential fold, and binary-search dispatch over many instances.

## Left Python

Configurations a translator declines (`NotNative`), each with its ground:

- `Normalizer(norm="l2")` where sklearn's `row_norms` does not accumulate
  as numpy's x86-64 baseline einsum kernel, which `_helpers.row_sumsq`
  follows; `_helpers.row_sumsq_is_numpys` probes it. On aarch64 numpy's
  `npyv_muladd_f64` is a fused `vfmaq_f64`, which SQL cannot spell
  (inferred from numpy's source; no aarch64 run yet).
- `MinMaxScaler(clip=True)` with a NaN `feature_range` bound.
- `SimpleImputer` fitted on object data (strings), whose output is not a
  float the step can return; a `missing_values` that is neither NaN nor a
  number (`None`, `pd.NA`, a string); `MissingIndicator(sparse=True)`.
- `Normalizer(norm="max")` over more than 8 features, until per-read
  expansion (above).
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`, `IsotonicRegression`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
