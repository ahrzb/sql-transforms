# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands.

## Next

Easiest first; each is one family, one PR.

1. **`SimpleImputer`** for numbers: `mean`, `median`, `most_frequent`,
   `constant`; `add_indicator` adds lanes. `MissingIndicator`.
2. **Selectors:** `VarianceThreshold`, `SelectKBest`, `SelectPercentile`,
   `SelectFpr`/`Fdr`/`Fwe`, `GenericUnivariateSelect`, `SelectFromModel`,
   `RFE`, `RFECV`, `SequentialFeatureSelector`: `transform` keeps a fitted
   column subset.
3. **Linear projections:** `PCA` (`whiten`), `IncrementalPCA`,
   `TruncatedSVD`, `FactorAnalysis`, `FastICA`, `GaussianRandomProjection`,
   `SparseRandomProjection`, `PLSSVD`/`PLSRegression`/`CCA`/`PLSCanonical`
   (x scores), `LinearDiscriminantAnalysis`: a matvec, so a measured bound
   (`_helpers.dot`).
4. **`PolynomialFeatures`** (`degree`, `interaction_only`, `include_bias`).
5. **Encoders over strings:** `OrdinalEncoder`, `OneHotEncoder`
   (`handle_unknown`, `drop`, infrequent categories), `TargetEncoder`
   (transform of new rows only). Needs string features in the fixtures.
6. **`KBinsDiscretizer`** (`encode="ordinal"`; `onehot-dense` after 5).
7. **Non-linear maps:** `PowerTransformer` (Yeo-Johnson, Box-Cox,
   `standardize`), `QuantileTransformer` (interpolation over quantiles),
   `SplineTransformer`, `FunctionTransformer` for numpy ufuncs with a SQL
   twin, `AdditiveChi2Sampler`, `SkewedChi2Sampler`, `RBFSampler`,
   `PolynomialCountSketch`.
8. **Distances to fitted centres:** `KMeans`, `MiniBatchKMeans`,
   `BisectingKMeans`, `Birch` (`transform` = distances).
9. **Compositions:** a step whose instances are `Pipeline`s of catalog
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
- **A constant CASE result that counts as trap-free.** A field read keeps
  each sibling `can_trap` (#328) cannot clear, and a typed constant
  standing as a CASE result (`CASE WHEN x IS NULL THEN CAST('0.0' AS
  DOUBLE) ELSE x END`, as `confit.sql` renders every float and as
  `coalesce(x, NaN)` binds) is not cleared, though the same constant inside
  arithmetic or a comparison is. Measured 2026-10-05, 32 lanes of
  `CASE WHEN id = 0 THEN f(x) END` per 64-row call: 216 µs when `f` is
  arithmetic over the parameter, 241 µs for `CASE WHEN x IS NULL THEN x
  ELSE x + 0.0 END`, 5,220 µs for `CASE WHEN x IS NULL THEN 0.0 ELSE x
  END`. Every entry has such arms (the NULL-as-NaN feature read, clip
  bounds, the norm guard, Binarizer's 1/0, an imputed statistic), so a
  read of a native step evaluates all its lanes: `StandardScaler` over 32
  features serves a 64-row call in 5,081 µs, against 256 µs for the same
  lanes written inline (the twin: about 500 µs a row). Every wide entry
  waits on this.
- **Per-read expansion** (confit PLANS' first item for this catalog). Build
  time grows with field reads × body: `to_native` (which builds the
  all-lanes query once) takes, with one / three fitted groups,
  `StandardScaler` 0.3 / 1.0 s at 32 features, 1.4 / 5.0 s at 64 and
  8 s / past the 4M-token cap at 128; `MinMaxScaler(clip=True)` 6.3 / 25 s
  at 64; `Normalizer` l1 1.8 s and l2 4.2 s at 16, l2 at 32 past
  Cranelift's function-size limit, and max (quadratic per lane) 2 s at 8,
  15 s at 12. `Normalizer(norm="max")` is served to 8 features meanwhile.
- **`greatest`/`least` in linear size.** They are composed as a CASE fold
  that clones the accumulator four times per argument
  (`src/specializer/frontend/functions.rs`): 6 arguments take 5.6 s to
  build and 8 fail Cranelift's function-size limit after 62 s.
  `_helpers.row_max` spells a row maximum as a tournament of two-way CASEs
  instead (quadratic).
- **Dispatch on many instances**: served now. A CASE whose leading arms are
  `id = <integer>` (8 or more) lowers to a binary search, and trap-free
  struct siblings no longer cost a lane per read: 1000 instances serve at
  about 0.11 us per row instead of 6.9 us. Build time still grows with
  field reads x instances (confit PLANS, "Per-read expansion").

## Left Python

Configurations a translator declines (`NotNative`), each with its ground:

- `Normalizer(norm="l2")` where sklearn's `row_norms` does not accumulate
  as numpy's x86-64 baseline einsum kernel, which `_helpers.row_sumsq`
  follows; `_helpers.row_sumsq_is_numpys` probes it. On aarch64 numpy's
  `npyv_muladd_f64` is a fused `vfmaq_f64`, which SQL cannot spell
  (inferred from numpy's source; no aarch64 run yet).
- `MinMaxScaler(clip=True)` with a NaN `feature_range` bound.
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
