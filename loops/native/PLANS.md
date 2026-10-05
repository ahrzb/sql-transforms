# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one family, one PR.

1. **Non-linear maps:** `AdditiveChi2Sampler`.
2. **`QuantileTransformer` as one search tree per feature:** its two
   `np.interp` searches bisect the same breakpoints (`-x` mirrors them,
   with the other endpoint of each interval closed), so one tree whose
   leaves compute both lines, with `x` at a breakpoint dispatched to the
   neighbour piece the mirrored search picks, keeps the twin's arithmetic
   and builds linearly (Needs from confit, "Two CASE trees"): the default
   1,000 quantiles would build in about 0.6 s per feature, against 1.4 s,
   and the caps could rise.
3. **A bound per configuration.** An entry's ulp bound is its class's
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
  meanwhile (about 7 s at most); Next, item 2, is the entry-side
  alternative.

- **A value bound once in a SQL function body, and a build linear in the
  parameters.** A function body is substituted as text, so an expression
  read twice is spelled twice, and a recurrence whose every step reads
  the previous one twice doubles per step. `SplineTransformer`'s de Boor
  recurrence does (scipy's order, which the entry must keep): one lane of
  one feature at degree 3, 5 knots, is about 7 KB of SQL, at degree 5
  about 33 KB, and 32 features of degree 5 expand past the 4,000,000-token
  cap; `periodic` repeats its mapped `x` (a remainder) at every read.
  Apart from size, the build grows with the parameters times the body:
  a confit-only function of 320 struct lanes, each a 9-arm CASE of
  polynomial arithmetic over one of its DOUBLE parameters, builds in 2.7 s
  over 4 parameters and 7.3 s over 32 (0.25, 0.63, 1.9, 7.3 s at 4, 8,
  16, 32 parameters of 10 lanes each; release build, master 8a67154,
  2026-10-05); the reproduction is in #384's description. 32 features
  of degree 3, 8 knots, build in about 22 s (`error`) and 44 s
  (`continue`); the spline entry refuses past an estimated 7 s build
  meanwhile (spline.py, `_build_estimate`). A binding (a `let`, or a
  nested function whose arguments are evaluated once) would make the
  recurrence linear in the degree.

Served since this catalog began (#336–#339, #341, #346, #348, #350,
#353, #358, #362, #363, #374, #375, #377): a constant CASE
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
1.8 s and 5.1 s); `greatest` lowered as a running extreme (the max norm
at 48 features built in 6.4 s and stopped there; it now builds in 0.21 s,
and in 1.6 s at 128 features, uncapped); `round(DOUBLE)` and `sin`/`cos`
under a guard on `abs(x)` as trap-free (`FunctionTransformer(np.sin)` at
32 features served a row in 100 us against the twin's 56; now 2.4 us,
and 19.5 us at 128, uncapped, with the guard spelled through `abs`); and
a subexpression shared only where it costs no row anything (a
three-instance `QuantileTransformer` served 64 rows in 21,480 us after
#363; now 1,033; release build, master 8a67154).

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
- `KBinsDiscretizer(encode="onehot")`, the default: a sparse output, as
  for `OneHotEncoder` above. `KBinsDiscretizer(dtype=np.float32)`: the
  twin rounds x to float32 before it bins it, which the entry does not
  spell (a cast to FLOAT would have to round as numpy does, unproven).
  Bin edges that are not sorted numbers (searchsorted's answer is then
  its search order's), which no strategy fits on finite data.
- `QuantileTransformer(output_distribution="normal")`: scipy's
  `norm.ppf` has no SQL twin. Past 4,000 quantiles over an estimator's
  features or 4,000,000 in their squares, where builds pass about 7 s
  (Needs from confit, "Two CASE trees in one expression").
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
  on a bound per configuration (Next, item 3); `log1p` and `expm1` have no
  DuckDB function; `sin` and `cos` only where `kernel_is_confits` finds
  numpy's kernel bit-equal to confit's.
- A `Pipeline` with a step that is not a catalog entry, or one
  registered with a bound (a later step does not keep it bounded:
  `x - mean_` near `mean_`); with `transform_input` (which only transforms
  fit metadata, so the refusal is conservative); a step before the last
  whose output is not float64 (`MissingIndicator`'s booleans, an encoder's
  or discretizer's `dtype`: exact 0/1 or small integers either way, not
  yet shown to read the same downstream); passthrough steps only, over a
  string feature (the step's `float()` raises). A `set_output` container
  between steps is not examined yet.
- `SplineTransformer(sparse_output=True)`: a sparse output
  (decisions/open/sparse-outputs.md). `extrapolation="linear"` at
  `degree=0, n_knots=2` over two or more features: the twin's running
  `degree` (spline.py) continues two lanes of one from the second feature
  on, and writes a row above the knots into the previous feature's lane.
  Knots that are not sorted, partly NaN, or span past a double, and a
  spline whose `c` is not sklearn's shape (no fit makes these). A step
  past an estimated 7 s build, per estimator (32 features of degree 3,
  8 knots, and wider; Needs from confit, "A value bound once"), and any
  step where scipy's `BSpline` does not round as the unfused recurrence
  (`spline.bspline_is_scipys`, an FMA build). Where the twin raises the
  entry answers: NaN past the knots under `extrapolation="error"`, 0.0
  for NaN under `handle_missing="error"`, and 0.0 above the knots under
  `extrapolation="constant"` at `degree=0`.
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
- `IsotonicRegression` with float32 thresholds (the twin casts its input
  to float32), past 8,000 thresholds (about 6 s to build; one CASE tree
  builds in about 0.7 ms a threshold, 22 s at 20,000), with thresholds
  further apart than a double spans, or on a platform whose `np.interp`
  fuses its multiply-add (`quantile.interp_is_numpys`). Where the twin
  raises (`out_of_bounds="raise"` outside the range, NaN, infinity), the
  entry answers NaN, or the constant of a one-threshold fit (goal.md,
  "Tolerated differences").
- `FeatureAgglomeration` with a `pooling_func` other than `np.mean`.
  `np.max` and `np.min` included: on a tie of signed zeros numpy's SIMD
  reduction answers the zero its lane order reaches, neither the first
  nor the last tied operand, so `greatest`/`least` (the first) part from
  it. Spelling numpy's reduction lanes (CPU-dependent, with a probe as
  `row_sumsq_is_numpys` has) would serve them.
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `RandomTreesEmbedding`,
  `BernoulliRBM`, `NMF`/`MiniBatchNMF`, the dictionary learners,
  `SparsePCA`, `LatentDirichletAllocation`,
  `NeighborhoodComponentsAnalysis`. Each needs a design first.
