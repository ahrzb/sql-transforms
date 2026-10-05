# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands.

## Next

Easiest first; each is one family, one PR.

1. **Non-linear maps:** `PowerTransformer` (Yeo-Johnson, Box-Cox,
   `standardize`), `QuantileTransformer` (interpolation over quantiles),
   `SplineTransformer`, `FunctionTransformer` for numpy ufuncs with a SQL
   twin, `AdditiveChi2Sampler`. In progress, wave 1 (subagents.md):
   `PowerTransformer` on `claude/native-power`
   (`session_01KRvmpULPVDQtRxRx6XrBRG`), `QuantileTransformer` on
   `claude/native-quantile` (`session_01Y2x9kvyGhHrtr9VTsDFv8R`).
2. **Compositions:** a step whose instances are `Pipeline`s of catalog
   entries (compose the translations), then `ColumnTransformer` and
   `FeatureUnion`.
3. **Wider fixtures:** `MAX_LANES` (300) was set while builds grew about as
   lanes^2.5; since #350 they grow about as lanes^1.4 (2,556 lanes: 3.2 s),
   so the fixtures can draw wider steps.

## Waiting on the owner

- **Linear projections:** `PCA` (`whiten`), `IncrementalPCA`,
  `TruncatedSVD`, `FactorAnalysis`, `FastICA`, `GaussianRandomProjection`,
  `SparseRandomProjection`, `PLSSVD`/`PLSRegression`/`CCA`/`PLSCanonical`
  (x scores), `LinearDiscriminantAnalysis`: a BLAS matvec whose order the
  entry cannot follow, and whose error is not small in ulps of the result
  (decisions/open/matvec-parity-bound.md).
- **Distances to fitted centres:** `KMeans`, `MiniBatchKMeans`,
  `BisectingKMeans`, `Birch` (`transform` = distances, through BLAS), and
  the samplers that project through a matrix: `RBFSampler`,
  `SkewedChi2Sampler`, `PolynomialCountSketch`. The same ruling.

## Needs from confit

- **A subexpression shared within one call** (the confit loop's ticket
  T1, in progress). A `Normalizer` lane is `x_j / g(norm(x))`, and every
  lane repeats the row norm verbatim, so the body is O(n²) in the
  features. Measured on master 5513891 through `to_native`: l1 1.4 s and
  l2 3.3 s at 16 features; at 32 confit refuses past its compiled-size
  limit after 6.9 s (l1) and 16.7 s (l2), at 48 after 23.7 s and 46.7 s.
  A native `Normalizer` call is 2x the twin's speed where other scalers
  are about 30x. Evaluating identical pure subexpressions of a call once
  (or a local binding in a SQL function body) makes it O(n). The max norm
  is capped at 8 features meanwhile. Sent to the confit loop 2026-10-05.
- **An early size refusal** (the confit loop's ticket T2, in progress):
  the refusals above arrive after Cranelift has spent its time (up to
  47 s), which `to_native` pays before falling back to Python.

Served since this catalog began (#336–#339, #341, #346, #348, #350,
#353): a constant CASE
result counts as trap-free (a 32-lane step serves a 64-row call in 331 µs,
against 297 µs inline and 5,081 µs before); a named refusal past
Cranelift's size limit; `greatest`/`least` without the exponential fold;
binary-search dispatch over many instances; a field read expands its call
once (`StandardScaler` at 128 features and three groups builds in 1.2 s,
where it passed the token cap); a cast that cannot fail is trap-free (wide
`PolynomialFeatures` over BIGINT features built in 20-50 s, now 1-3 s);
DuckDB's parse depth; a named refusal past Cranelift's 24-bit index
width, where a 48-feature `Normalizer` met a verifier error; struct reads
that build about linearly in the lanes read (`PolynomialFeatures` at 2,556
lanes: refused, then 287 s after #348, now 3.2 s after #350, master
5513891); and a dropped function's JIT memory freed (each build leaked two
memory mappings, so a process stalled at `vm.max_map_count` after about
32,000 builds; 6,000 builds now hold 477 mappings).

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
- `KBinsDiscretizer(encode="onehot")`, the default: a sparse output, as
  for `OneHotEncoder` above. `KBinsDiscretizer(dtype=np.float32)`: the
  twin rounds x to float32 before it bins it, which the entry does not
  spell (a cast to FLOAT would have to round as numpy does, unproven).
  Bin edges that are not sorted numbers (searchsorted's answer is then
  its search order's), which no strategy fits on finite data.
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`, `IsotonicRegression`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
