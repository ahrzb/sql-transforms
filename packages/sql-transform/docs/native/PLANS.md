# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands.

## Next

Easiest first; each is one family, one PR.

1. **Scalers, the rest:** `MinMaxScaler` (`clip`), `MaxAbsScaler`,
   `RobustScaler` (`with_centering`, `with_scaling`, `unit_variance`),
   `Normalizer` (norm `l1`/`l2`/`max`: a row reduction, check its order),
   `Binarizer`.
2. **`SimpleImputer`** for numbers: `mean`, `median`, `most_frequent`,
   `constant`; `add_indicator` adds lanes. `MissingIndicator`.
3. **Selectors:** `VarianceThreshold`, `SelectKBest`, `SelectPercentile`,
   `SelectFpr`/`Fdr`/`Fwe`, `GenericUnivariateSelect`, `SelectFromModel`,
   `RFE`, `RFECV`, `SequentialFeatureSelector`: `transform` keeps a fitted
   column subset.
4. **Linear projections:** `PCA` (`whiten`), `IncrementalPCA`,
   `TruncatedSVD`, `FactorAnalysis`, `FastICA`, `GaussianRandomProjection`,
   `SparseRandomProjection`, `PLSSVD`/`PLSRegression`/`CCA`/`PLSCanonical`
   (x scores), `LinearDiscriminantAnalysis`: a matvec, so a measured bound
   (`_helpers.dot`).
5. **`PolynomialFeatures`** (`degree`, `interaction_only`, `include_bias`).
6. **Encoders over strings:** `OrdinalEncoder`, `OneHotEncoder`
   (`handle_unknown`, `drop`, infrequent categories), `TargetEncoder`
   (transform of new rows only). Needs string features in the fixtures.
7. **`KBinsDiscretizer`** (`encode="ordinal"`; `onehot-dense` after 6).
8. **Non-linear maps:** `PowerTransformer` (Yeo-Johnson, Box-Cox,
   `standardize`), `QuantileTransformer` (interpolation over quantiles),
   `SplineTransformer`, `FunctionTransformer` for numpy ufuncs with a SQL
   twin, `AdditiveChi2Sampler`, `SkewedChi2Sampler`, `RBFSampler`,
   `PolynomialCountSketch`.
9. **Distances to fitted centres:** `KMeans`, `MiniBatchKMeans`,
   `BisectingKMeans`, `Birch` (`transform` = distances).
10. **Compositions:** a step whose instances are `Pipeline`s of catalog
    entries (compose the translations), then `ColumnTransformer` and
    `FeatureUnion`.

## Needs from confit

- **List-valued SQL functions**: served now. `SqlFunction` takes a
  `pa.list_(t, k)` return (a body of k expressions; `null_when` too), read
  whole or by constant index (`packages/confit/tests/test_list_literals.py`).
  Adopt it: `_lanes` stops refusing an unnamed width-k step.
- **`error()`**: served now, as a CASE result with a constant message
  (`packages/confit/tests/test_error_function.py`). Adopt it: each lane's
  dispatch gains `WHEN id IS NULL THEN NULL ELSE error('...')`, and goal.md's
  unknown-id difference goes.
- **A NULL struct**: served now. `SqlFunction(..., null_when=...)` makes
  the whole struct NULL under a condition, and field reads over it serve
  (`packages/confit/tests/test_struct_field_reads.py`). Adopt it: a struct
  step passes `null_when=lambda iid, *_: iid.isnull()`, and goal.md's
  NULL-struct difference goes.
- **Dispatch on many instances.** Each lane selects its instance with a CASE
  ladder, linear in the number of fitted groups. A step with hundreds of
  groups needs a constant lookup (an indexed list, or a params static).

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`, `IsotonicRegression`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
