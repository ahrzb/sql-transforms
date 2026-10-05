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

- **List-valued SQL functions.** A step with unnamed width-k output
  declares a fixed-size list return; `SqlFunction` refuses list returns, so
  such steps stay Python.
- **`error()`**, so an unknown instance id raises as the twin does (goal.md,
  "Tolerated differences").
- **A field read over a CASE-valued struct** (`(CASE ... END).p` refuses
  today), so a NULL id can answer a NULL struct.
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
