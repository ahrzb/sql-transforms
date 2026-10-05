# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one family, one PR.

1. **Non-linear maps:** `SplineTransformer`, `FunctionTransformer` for
   numpy ufuncs with a SQL twin (native T2, in progress), and
   `AdditiveChi2Sampler`.
2. **Compositions:** `ColumnTransformer` and `FeatureUnion`, composing
   entries as `compose.py` composes a `Pipeline`'s.
3. **Show served compositions in coverage.md:** sklearn's transformer
   list has no `Pipeline` (it is not a `TransformerMixin`), so the
   scoreboard does not show the one composition the catalog serves. A
   "served" note on composition rows, with `Pipeline` added from
   `catalog()`, would.
4. **`QuantileTransformer` as one search tree per feature:** its two
   `np.interp` searches bisect the same breakpoints (`-x` mirrors them,
   with the other endpoint of each interval closed), so one tree whose
   leaves compute both lines, with `x` at a breakpoint dispatched to the
   neighbour piece the mirrored search picks, keeps the twin's arithmetic
   and builds linearly (Needs from confit, "Two CASE trees"): the default
   1,000 quantiles would build in about 0.6 s per feature, against 1.4 s,
   and the caps could rise.

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
- **Two CASE trees in one expression that build in linear time.** One
  balanced CASE tree of q linear pieces over a DOUBLE builds linearly
  (0.09, 0.17, 0.40 s at q = 500, 1,000, 2,000); `0.5 * (tree(x) -
  tree(-x))` over `x = coalesce(p, NaN)`, the shape of
  `QuantileTransformer`'s entry (`np.interp` both ways), builds in 0.46,
  1.37, 4.48 s, and the entry at 4,000 quantiles in 16.2 s (release
  build, master 49acad5). One tree whose leaves hold both lines builds in
  0.28, 0.59, 1.25, 2.87 s up to 4,000. A confit-only reproduction is in
  the message sent to the confit loop (2026-10-05). The entry is capped at
  4,000 quantiles over the features and 4,000,000 in their squares
  meanwhile (about 7 s at most); Next, item 4, is the entry-side
  alternative.

Served since this catalog began (#336–#339, #341, #346, #348, #350,
#353, #362): a constant CASE
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
32,000 builds; 6,000 builds now hold 477 mappings); and total calls (`exp`,
a negation) and `ln` under a CASE guard that excludes x <= 0 as trap-free,
so a field read leaves the other lanes unevaluated (Box-Cox at 64
features, one instance: 3,448 us per row and a 15.6 s build before, 52 us
and 0.18 s with a guard arm the entry adds; a negation builds and serves
as `-1.0 * x` does, and the entries spell `-x` again).

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
- `QuantileTransformer(output_distribution="normal")`: scipy's
  `norm.ppf` has no SQL twin. Past 4,000 quantiles over an estimator's
  features or 4,000,000 in their squares, where builds pass about 7 s
  (Needs from confit, "A CASE tree that builds in linear time").
  Quantiles unsorted or partly NaN (never seen in 3,000 fits; a
  feature missing everywhere is served), quantiles further apart than a
  double spans, or a platform whose `np.interp` fuses its multiply-add
  (`quantile.interp_is_numpys` probes it).
- `PowerTransformer(method="yeo-johnson")` and `standardize=True`
  (waiting on the owner, above). Where the twin rejects
  x <= 0, the entry answers NaN (goal.md, "Tolerated differences").
- A `Pipeline` with a step that is not a catalog entry, or one
  registered with a bound (a later step does not keep it bounded:
  `x - mean_` near `mean_`); with `transform_input` (which only transforms
  fit metadata, so the refusal is conservative); a step before the last
  whose output is not float64 (`MissingIndicator`'s booleans, an encoder's
  or discretizer's `dtype`: exact 0/1 or small integers either way, not
  yet shown to read the same downstream); passthrough steps only, over a
  string feature (the step's `float()` raises). A `set_output` container
  between steps is not examined yet.
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`, `IsotonicRegression`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
