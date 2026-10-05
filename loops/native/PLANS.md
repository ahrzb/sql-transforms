# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one family, one PR.

1. **Non-linear maps:** `SplineTransformer`, `AdditiveChi2Sampler`.
2. **Compositions:** `ColumnTransformer` and `FeatureUnion`, composing
   entries as `compose.py` composes a `Pipeline`'s.
3. **Show served compositions in coverage.md:** sklearn's transformer
   list has no `Pipeline` (it is not a `TransformerMixin`), so the
   scoreboard does not show the one composition the catalog serves. A
   "served" note on composition rows, with `Pipeline` added from
   `catalog()`, would.
4. **Re-measure the caps set before #350:** the fixtures' `MAX_LANES`
   (300) and `quantile.py`'s `MAX_QUANTILES` (2,000) were set while builds
   grew about as lanes^2.5; since #350 they grow about as lanes^1.4
   (2,556 lanes: 3.2 s, master 5513891).
5. **A bound per configuration.** An entry's ulp bound is its class's
   (`translates(cls, ulps=)`), so `FunctionTransformer`, bit-exact for the
   identity and the exact functions, refuses `np.exp`, `np.log`,
   `np.log2`, `np.tan` (1 ulp from DuckDB's on x86-64 with AVX-512),
   `np.log10` (2) and `np.cbrt` (3), measured over 1,600,000 draws
   (`function.py`, 2026-10-05). A translator that declares its own bound
   per estimator would serve them within those, once each is measured over
   200 seeds of fixtures.

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
- **A call confit knows cannot trap.** `can_trap` counts every call as
  one that may trap (`ln`, `exp`, even unary minus), so a struct field
  read keeps the other lanes' calls and serving grows as the square of the
  width. Box-Cox (`native/power.py`), one instance, per row against the
  Python step: 0.3 vs 118 us at 1 feature, 39 vs 183 at 8, 90 vs 192 at
  12, 179 vs 209 at 16, 393 vs 267 at 24; build 0.15 s at 8, 2.1 s at 24,
  13 s at 32 with three instances, and at 64 (three instances) confit
  refuses past Cranelift's size limit after 26 s (master b926e88,
  2026-10-05). Classifying total calls (`exp`, `pow`, `fneg`) as
  trap-free, and `ln` under a CASE arm whose condition excludes `x <= 0`,
  would make it linear. `PowerTransformer` is capped at 12 features
  meanwhile. Sent to the confit loop 2026-10-05.
- **`sin` and `cos` under a guard.** DuckDB's `sin` and `cos` raise on an
  infinity, so `can_trap` counts them as trapping even under
  `CASE WHEN abs(x) = inf THEN NaN ELSE sin(x) END` (or
  `x = inf OR x = -inf`), and a struct field read evaluates every lane's
  call: `FunctionTransformer(np.sin)`, three instances, 1,024-row batches,
  release build, against the Python step, 13.7 vs 20.8 us per row at 8
  features, 26.8 vs 25.8 at 10, 36.6 vs 29.1 at 12, 178 vs 49 at 24
  (supervisor's measurement on 851cf08, master with #362, 2026-10-05). A
  guard rule for them, as #362 gave `ln` and `sqrt`, would make it linear;
  the entry caps `sin` and `cos` at 8 features meanwhile. Every other
  `FunctionTransformer` spelling serves 128 features.
- **A negation as cheap as a product.** A DOUBLE `-x` builds and serves
  far slower than `-1.0 * x`, which is the same double: a 32-feature
  `QuantileTransformer` (3 quantiles) built in 2.8 s and served 64 rows in
  32 ms with `-x`, against 0.35 s and 1.1 ms with the product (master
  b926e88). The entry spells the product meanwhile. Sent to the confit
  loop 2026-10-05.

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
- `QuantileTransformer(output_distribution="normal")`: scipy's
  `norm.ppf` has no SQL twin. Past 2,000 quantiles per estimator (summed
  over its features), a cap set before #350 (Next, item 3). Quantiles unsorted or partly NaN (never seen in 3,000 fits; a
  feature missing everywhere is served), quantiles further apart than a
  double spans, or a platform whose `np.interp` fuses its multiply-add
  (`quantile.interp_is_numpys` probes it).
- `PowerTransformer(method="yeo-johnson")` and `standardize=True`
  (waiting on the owner, above); Box-Cox over more than 12 features, until
  confit knows a call that cannot trap (above). Where the twin rejects
  x <= 0, the entry answers NaN (goal.md, "Tolerated differences").
- `FunctionTransformer` with a `func` other than the identity and numpy's
  `abs`, `fabs`, `negative`, `positive`, `conjugate`, `square`, `sqrt`,
  `reciprocal`, `floor`, `ceil`, `trunc`, `rint`, `sign`, `sin`, `cos`
  (lambdas, partials, user functions, other ufuncs); with `kw_args`; over
  a string feature, or a boolean one except for the identity (numpy keeps
  a boolean row boolean). The transcendentals 1-3 ulps from DuckDB's wait
  on a bound per configuration (Next, item 5); `log1p` and `expm1` have no
  DuckDB function; `sin` and `cos` only where `kernel_is_confits` finds
  numpy's kernel bit-equal to confit's, and over at most 8 features
  (Needs from confit, `sin` and `cos` under a guard).
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
