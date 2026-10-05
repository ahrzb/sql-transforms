# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one family, one PR.

1. **Non-linear maps:** `SplineTransformer`, `AdditiveChi2Sampler`.
2. **Show served compositions in coverage.md:** sklearn's transformer
   list has no `Pipeline` (it is not a `TransformerMixin`), so the
   scoreboard does not show the one composition the catalog serves. A
   "served" note on composition rows, with `Pipeline` added from
   `catalog()`, would.
3. **`QuantileTransformer` as one search tree per feature:** its two
   `np.interp` searches bisect the same breakpoints (`-x` mirrors them,
   with the other endpoint of each interval closed), so one tree whose
   leaves compute both lines, with `x` at a breakpoint dispatched to the
   neighbour piece the mirrored search picks, keeps the twin's arithmetic
   and builds linearly (Needs from confit, "Two CASE trees"): the default
   1,000 quantiles would build in about 0.6 s per feature, against 1.4 s,
   and the caps could rise.
4. **A bound per configuration.** An entry's ulp bound is its class's
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

- **Shared subexpressions kept lazy inside untaken CASE arms** (a
  regression from #363, the highest of these). Lowering evaluates every
  shared subexpression before the first item, so in a step whose instance
  arms are CASE trees that share subtrees, every row computes every shared
  subtree. `QuantileTransformer`'s widest fixture (27 features, three
  instances, small-integer features whose breakpoints repeat) serves 64
  rows in 21,480 us after #363 against 1,153 before, where the twin takes
  about 27,000 (release build, master 89e99fc against c3b42ea); with one
  instance it is unchanged (862 against 967), with two 2x slower (2,444
  against 1,194). Sent to the confit loop with the reproduction,
  2026-10-05.
- **A `greatest` that builds linearly.** One `greatest` over n DOUBLEs
  builds in 0.20 s at n = 32, 1.0 s at 64 and 6.1 s at 128, and past
  Cranelift's size limit at 256, where a sum of the same n builds in 3 to
  11 ms; nested two-way `greatest` calls repeat their arguments (a
  balanced tree of them over a 32-feature `Normalizer` row: 87 s); and the
  expansion cap counts a body before shared subexpressions are found, so
  the CASE tournament the max norm used refuses at 48 features (release
  build, master f0fa925). `Normalizer(norm="max")` spells one `greatest`
  and is capped at 48 features (6.4 s) meanwhile.
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
  meanwhile (about 7 s at most); Next, item 3, is the entry-side
  alternative.
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

Served since this catalog began (#336–#339, #341, #346, #348, #350,
#353, #358, #362, #363): a constant CASE
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
as `-1.0 * x` does, and the entries spell `-x` again); a repeated pure
subexpression computed once per row, and a refusal past Cranelift's
registers right after lowering (a `Normalizer` repeats its norm in every
lane: l1 and l2 at 32 features were refused after 6.9 s and 16.7 s,
master 5513891, and now build in 0.10 s and 0.22 s and serve a row in 1.6
and 5.1 us against the twin's 250 to 300; at 128 features they build in
1.8 s and 5.1 s; the max norm's cap goes from 8 features to 48).

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
- `Normalizer(norm="max")` over more than 48 features, until confit's
  `greatest` builds linearly (Needs from confit).
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
- `FunctionTransformer` with a `func` other than the identity and numpy's
  `abs`, `fabs`, `negative`, `positive`, `conjugate`, `square`, `sqrt`,
  `reciprocal`, `floor`, `ceil`, `trunc`, `rint`, `sign`, `sin`, `cos`
  (lambdas, partials, user functions, other ufuncs); with `kw_args`; over
  a string feature, or a boolean one except for the identity (numpy keeps
  a boolean row boolean). The transcendentals 1-3 ulps from DuckDB's wait
  on a bound per configuration (Next, item 4); `log1p` and `expm1` have no
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
- A `ColumnTransformer` or `FeatureUnion` with a part that is not a
  catalog entry, or one registered with a bound; a sparse output
  (`sparse_output_`); a `set_output` container (the step's
  `transform(...)[0]` misreads a DataFrame row); a string column passed
  through (the step's `float()`); column names, which need a DataFrame,
  and a scalar column, which hands the part a 1-D array. In a
  `ColumnTransformer`, a part whose first step is a
  `FunctionTransformer(func, validate=False)`: it is handed an object
  array, on which `np.sqrt` raises and the exact functions answer as
  Python floats do (conservative for those). A weight that is not a
  double, on a float32 output (multiplied in float32), or an integer
  weight on an output that may not be float64 or over a boolean feature
  (an integer product has no -0.0); a weighted passthrough in a
  `FeatureUnion` (the twin multiplies the step's list, which raises). A
  `ColumnTransformer` with a passthrough part before the last step of a
  `Pipeline` (its object output, as for the `Pipeline` rule above).
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`, `IsotonicRegression`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
