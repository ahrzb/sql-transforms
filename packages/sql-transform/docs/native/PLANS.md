# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands.

## Next

Easiest first; each is one family, one PR.

1. **`KBinsDiscretizer`** (`encode="ordinal"`, `"onehot-dense"`).
2. **Non-linear maps:** `QuantileTransformer` (interpolation over quantiles),
   `SplineTransformer`, `FunctionTransformer` for numpy ufuncs with a SQL
   twin, `AdditiveChi2Sampler`.
3. **Compositions:** a step whose instances are `Pipeline`s of catalog
   entries (compose the translations), then `ColumnTransformer` and
   `FeatureUnion`.

## Waiting on the owner

- **Linear projections:** `PCA` (`whiten`), `IncrementalPCA`,
  `TruncatedSVD`, `FactorAnalysis`, `FastICA`, `GaussianRandomProjection`,
  `SparseRandomProjection`, `PLSSVD`/`PLSRegression`/`CCA`/`PLSCanonical`
  (x scores), `LinearDiscriminantAnalysis`: a BLAS matvec whose order the
  entry cannot follow, and whose error is not small in ulps of the result
  (decisions/open/matvec-parity-bound.md).
- **`PowerTransformer`'s Yeo-Johnson, and either method with
  `standardize=True`:** no small bound in ulps of the result, as for the
  matvec families (decisions/open/power-parity-bound.md, measured).
  Box-Cox with `standardize=False` is native, within 4 ulps.
- **Distances to fitted centres:** `KMeans`, `MiniBatchKMeans`,
  `BisectingKMeans`, `Birch` (`transform` = distances, through BLAS), and
  the samplers that project through a matrix: `RBFSampler`,
  `SkewedChi2Sampler`, `PolynomialCountSketch`. The same ruling.

## Needs from confit

- **A subexpression shared within one call.** A `Normalizer` lane is
  `x_j / g(norm(x))`, and every lane repeats the row norm verbatim, so the
  body is O(n²) in the features. Measured on master 477ca2f through
  `to_native`: l1 1.6 s and l2 3.5 s at 16 features; at 32 confit refuses
  past its compiled-size limit after 9 s (l1) and 20 s (l2), and at 48
  after 30 s (l1) and 55 s (l2; an internal Cranelift verifier error
  until #346 named it, master fa13d32). Evaluating identical pure
  subexpressions of a call once (or a local binding in a SQL function
  body) makes it O(n). The max norm is capped at 8 features meanwhile.
  Sent to the confit loop 2026-10-05.
- **Build time linear in the lanes read.** A step whose query reads L
  struct fields builds in time growing about as L^2.5, though its body is
  linear in L: `PolynomialFeatures(degree=2)` at 231 lanes builds in 1.6 s,
  496 in 6.5 s, 861 in 23 s, 1,326 in 68 s (master 477ca2f), and past
  2,000 lanes confit refuses at its expansion cap. The catalog test draws
  steps of at most 300 lanes meanwhile (`MAX_LANES`). Sent to the confit
  loop 2026-10-05, with a repro.
- **A call confit knows cannot trap.** `can_trap` counts every call as
  one that may trap (`ln`, `exp`, even unary minus), so a struct field
  read keeps the other lanes' calls and serving grows as the square of the
  width. Box-Cox (`native/power.py`), one instance, per row against the
  Python step: 0.3 vs 118 us at 1 feature, 39 vs 183 at 8, 90 vs 192 at
  12, 179 vs 209 at 16, 393 vs 267 at 24; build 0.15 s at 8, 2.1 s at 24,
  13 s at 32 with three instances, and at 64 (three instances) confit
  refuses past Cranelift's size limit after 26 s (master b926e88, 2026-10-05). Classifying total
  calls (`exp`, `pow`, `fneg`) as trap-free, and `ln` under a CASE arm
  whose condition excludes `x <= 0`, would make it linear. `PowerTransformer`
  is capped at 12 features meanwhile.

Served since this catalog began (#336–#339, #341, #346): a constant CASE
result counts as trap-free (a 32-lane step serves a 64-row call in 331 µs,
against 297 µs inline and 5,081 µs before); a named refusal past
Cranelift's size limit; `greatest`/`least` without the exponential fold;
binary-search dispatch over many instances; a field read expands its call
once (`StandardScaler` at 128 features and three groups builds in 1.2 s,
where it passed the token cap); a cast that cannot fail is trap-free (wide
`PolynomialFeatures` over BIGINT features built in 20-50 s, now 1-3 s);
DuckDB's parse depth; and a named refusal past Cranelift's 24-bit index
width, where a 48-feature `Normalizer` met a verifier error.

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
- A selector that keeps no feature, or whose `get_support()` raises (as
  its `transform` would).
- `OneHotEncoder(sparse_output=True)`, the default: a sparse output, which
  the Python step does not serve either (it reads a row with `float()`,
  and a sparse row is not a float). A note for the step, not the catalog.
- An encoder over a boolean feature: the fixtures make no boolean features
  yet, for any entry.
- `Normalizer(norm="max")` over more than 8 features, and any norm whose
  body confit does not build, until a subexpression is shared within a call
  (above).
- `PowerTransformer(method="yeo-johnson")` and `standardize=True`
  (waiting on the owner, above); Box-Cox over more than 12 features, until
  confit knows a call that cannot trap (above). Where the twin rejects
  x <= 0, the entry answers NaN (goal.md, "Tolerated differences").
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`, `IsotonicRegression`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
