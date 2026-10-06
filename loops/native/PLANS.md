# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one family, one PR.

## Waiting on the owner

- **`AdditiveChi2Sampler`:** its lanes are `factor * cos(j * (s *
  log(x)))` and the same with `sin`, and numpy's `log`, up to 1 ulp from
  DuckDB's `ln`, reaches `cos` and `sin` unchanged in absolute terms, so
  near a zero the result parts by any number of ulps (8,192 over 400,000
  draws), while within 0.91 eps of the lane's term scale
  (decisions/open/additive-chi2-parity-bound.md).
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
  tree(-x))` over `x = coalesce(p, NaN)` builds in 0.46, 1.37, 4.48 s
  (release build, master 49acad5; a confit-only reproduction is in the
  message sent to the confit loop, 2026-10-05). The catalog no longer
  needs it: `QuantileTransformer` answers both searches from one tree
  whose leaves compute both lines (T12), 0.5, 1.3, 2.7-3.2 s at 1,000,
  2,000, 4,000 quantiles over one feature against 1.4, 4.7, 23 s for the
  two trees (release build, one container, 2026-10-05); capped at 8,000
  quantiles over an estimator's features (6-8 s at that sum). Low priority
  for confit: a future entry that needs two trees in one expression would
  raise it again.
- **A value bound once in a SQL function body.** A function body is
  substituted as text, so an expression read twice is spelled twice, and
  a recurrence whose every step reads the previous one twice doubles per
  step. `SplineTransformer`'s de Boor recurrence does (scipy's order,
  which the entry must keep): one lane of one feature at degree 3, 5
  knots, is about 7 KB of SQL, at degree 5 about 33 KB, and 32 features
  of degree 5 expand past the 4,000,000-token cap; `periodic` repeats its
  mapped `x` (a remainder) at every read. #387 (a shared value computed
  just before its first reader) took the build from growing with the
  parameters times that text to about linear in it: 32 features of degree
  3, 8 knots, build in 1.1-2.4 s (21-46 s before), and the entry's 7 s
  cap now refuses only degree 5 at 7 knots from 32 features (`continue`,
  `periodic`) or 64 (the others), and degree 4 at 64 (`continue`, 5
  knots) (spline.py, `_build_estimate`). The size still doubles per
  degree, so the need stands: a binding (a `let`, or a nested function
  whose arguments are evaluated once) would make the recurrence linear in
  the degree, and serve the steps past the token cap.
- **A value bound once, for tree embeddings.** `RandomTreesEmbedding`'s
  fastest spelling is one nested CASE per tree answering its leaf id, each
  lane `leaf = k`: confit computes the CASE once per row and serves 0.11 us
  a lane (30 trees of depth 5, 693 lanes, 75 us a row). But the body is
  text, so the CASE is spelled in every lane of its tree, quadratic in the
  tree's leaves: the default `RandomTreesEmbedding(n_estimators=100,
  max_depth=5, sparse_output=False)` fitted on `normal(size=(2000, 8))`
  expands past the 4,000,000-token cap (the CASE spelling, a reproduction,
  is in the T16 PR's description). The entry spells each lane as its
  leaf's path instead (trees.py), linear in the text, at 0.22 us a lane;
  a binding would serve the CASE spelling at every width the entry takes
  (release build, 2026-10-06).

Served since this catalog began (#336–#339, #341, #346, #348, #350,
#353, #358, #362, #363, #374, #375, #377, #387, #390): a constant CASE
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
#363; now 1,033; release build, master 8a67154); a shared value
computed just before its first reader (a `SplineTransformer` of 32
features, degree 3, 8 knots, built in 21-46 s, now 1.1-2.4 s; the build
no longer grows with the parameters times the body); and `cbrt` as glibc's,
as DuckDB's is (confit parted from DuckDB on 99,438 of 200,000 draws, by
up to 3 ulps; now on none, so `FunctionTransformer(np.cbrt)` is served
within its 3 ulps of numpy).

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
- `KBinsDiscretizer(encode="onehot")`, the default: a sparse output, as
  for `OneHotEncoder` above. `KBinsDiscretizer(dtype=np.float32)`: the
  twin rounds x to float32 before it bins it, which the entry does not
  spell (a cast to FLOAT would have to round as numpy does, unproven).
  Bin edges that are not sorted numbers (searchsorted's answer is then
  its search order's), which no strategy fits on finite data.
- `QuantileTransformer(output_distribution="normal")`: scipy's
  `norm.ppf` has no SQL twin. Past 8,000 quantiles over an estimator's
  features, where builds reach 6-8 s (`quantile.MAX_QUANTILES`).
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
  (bit-exact) and `exp`, `log`, `log2`, `tan` (within 1 ulp), `log10`
  (within 2) and `cbrt` (within 3) (lambdas, partials, user functions,
  other ufuncs); with `kw_args`; over a string feature. Over boolean
  features only (a row none NULL is a boolean array), a function whose
  answers on False and True are not its doubles' on 0.0 and 1.0
  (`function._on_booleans`): `negative`, `positive` and `sign` raise,
  `reciprocal` answers int8 (1/False is 0), `sin`, `cos`, `tan` and `exp`
  answer float16. `log1p`
  and `expm1` have no DuckDB function. `sin` and `cos`,
  and the bounded functions, only where `kernel_distance` finds numpy's
  kernel within the function's bound of confit's (numpy picks its kernel
  by CPU). A bounded function does not compose: a `Pipeline`,
  `ColumnTransformer` or `FeatureUnion` refuses it, naming its bound.
- A `Pipeline` with a step that is not a catalog entry, or one
  registered with a bound (a later step does not keep it bounded:
  `x - mean_` near `mean_`); with `transform_input` (which only transforms
  fit metadata, so the refusal is conservative); a step before the last
  whose output is not float64 (`MissingIndicator`'s booleans, an encoder's
  or discretizer's `dtype`: exact 0/1 or small integers either way, not
  yet shown to read the same downstream); passthrough steps only, over a
  string feature (the step's `float()` raises); over boolean features
  only, a step before the last that hands on a dtype other than bool or
  float64 (`FunctionTransformer(np.sqrt)`'s float16; a selector's,
  `Binarizer`'s or the identity's booleans are served, read as 0/1 by
  the next step). A `set_output` container between steps is not examined
  yet.
- `SplineTransformer(sparse_output=True)`: a sparse output
  (decisions/open/sparse-outputs.md). `extrapolation="linear"` at
  `degree=0, n_knots=2` over two or more features: the twin's running
  `degree` (spline.py) continues two lanes of one from the second feature
  on, and writes a row above the knots into the previous feature's lane.
  Knots that are not sorted, partly NaN, or span past a double, and a
  spline whose `c` is not sklearn's shape (no fit makes these). A step
  past an estimated 7 s build, per estimator: degree 5 at 7 knots from
  32 features (`continue`, `periodic`) or 64 (the others), degree 4 at 5
  knots and 64 features of `continue`, and everything past confit's
  token cap (Needs from confit, "A value bound once"); and any
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
  weight on an output that may not be float64 (an integer product has no
  -0.0): in a `ColumnTransformer` over any boolean feature (handed as a
  Python bool, which a selector keeps), in a `FeatureUnion` over boolean
  features only, where the part hands a boolean row back not float64; an
  encoder part over a boolean feature in a `ColumnTransformer` (handed a
  Python bool in an object array, which sklearn matches its own way:
  categories [nan] answer [True, False] as objects and raise on it as
  booleans; the entry probes the step's rows); a weighted passthrough in a
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
- `RandomTreesEmbedding(sparse_output=True)`, the default: a sparse
  output (decisions/open/sparse-outputs.md). Past 25,000 path steps (a
  leaf's depth, summed over the leaves) per estimator, an estimated 7 s
  build (`trees.MAX_PATH_STEPS`): 250 trees of depth 5 (7.8 s), and
  one unbounded tree over 2,000 rows has 33,688 steps. Where the twin
  raises (±inf, a value past float32's range) the entry answers a leaf
  (goal.md, "Tolerated differences").
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `BernoulliRBM`,
  `NMF`/`MiniBatchNMF`, the dictionary learners, `SparsePCA`,
  `LatentDirichletAllocation`, `NeighborhoodComponentsAnalysis`. Each needs a design first.
