# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one PR.

1. **The build estimates on confit #461** (reading 6, finding 76).
   Since #456, a struct read field by field builds in about the square of
   its fields, past the estimates of `trees.py`, `spline.py` and
   `quantile.py`: 100 trees of depth 6 (estimate 3.4 s) build in 18 s,
   250 of depth 5 (6.5 s) in 42-44 s, 10 quantiles over 800 features in
   60-63 s (`to_native`; release build, master 98033e8, uv's CPython).
   The confit loop traced the cost to its check for a subquery at each
   SELECT item (`subquery::has_subquery`), which walked the whole call
   again at every field read. #461 (merged 2026-10-15, master 620112e)
   keeps the answer for each call: on the confit loop's container 4,000
   fields build in 1.00-1.12 s (23.6-27.8 s before) and the catalog's
   250 trees of depth 5 (5,982 fields) in 5.8-6.2 s (158 s); above 4,000
   fields the build still grows faster than linearly (8,000 fields:
   3.2 s). Measure the builds of these three families again with reading
   6's scripts (`/mnt/project-files/transforms-loop/reading-6/`:
   `widths_r6.py`, `struct_list.py`, `steps_at.py`); if a step still
   builds past its estimate, fit that estimate again. Then remove the
   need "A struct read field by field that builds about as fast as before
   #456" below.

## Ruled 2026-10-06, to build

The owner approved every recommendation in `decisions/closed/` (records
and research in `decisions/research/2026-10-06/`), amended: bit-exact is the
default, and a bound above 0 serves only on the caller's request (it can
flip HistGradientBoosting labels on repeated training values). T19 built
that request, `to_native(step, allow_bound=True)`. In order:

1. **Families on the parity bound.** Since T24 a family declares its error
   scale, `translates(cls, scale=ErrorScale(k, s, tau, g))`, and
   `native.check` holds each row to it (`_registry.ErrorScale` says what
   each operand is). T37 served the first family, `PCA`, `IncrementalPCA`,
   `TruncatedSVD` and `GaussianRandomProjection` (project.py: S_full,
   K = n + 3, summed pairwise), and `report-format.md` lists error scales.
   - The rest of the linear projections, each with its own S: `FastICA`
     subtracts the mean before the product (S over |x_i - m_i|·|c_ki|);
     the PLS family and `LinearDiscriminantAnalysis` center and scale, then
     project; `FactorAnalysis` chains two products, so the first one's
     error passes through the second; `SparseRandomProjection` sums in
     scipy's CSR order, which SQL may follow bit-exactly behind a kernel
     probe.
   - `PowerTransformer` Yeo-Johnson and `standardize=True`: K 10 / 12 / 7;
     restate served Box-Cox as K = 5. Needs NaN and `isinf` arms.
   - `AdditiveChi2Sampler`: K = 4 (3 with the twin's own cosh), 0 where the
     probes read 0; ship `sample_steps=1` (bit-exact) first. Since T29
     the kernel probe draws random significands (`_random_doubles`).
   - Distances to fitted centres (`KMeans` and kin): compare squares,
     K = n + 5. `RBFSampler` n + 3, `SkewedChi2Sampler` n + 4, `Nystroem`
     max(n + 5, m + 3). `PolynomialCountSketch` is an FFT, not a matvec:
     not served until a normwise S is derived.
2. **Still open for the owner:** a bounded step inside a composition
   (decisions/open/bounded-steps-in-compositions.md). `compose.py` keeps
   refusing one meanwhile.

## Needs from confit

- **A struct read field by field that builds about as fast as before
  #456.** Since confit #456 (0def508), a query that reads a struct output
  field by field, one SELECT item a field, as `query(step)` does, builds
  in a time that grows about as the square of the fields read. 4,000
  fields over 8 parameters build in 0.92-1.01 s at 4bb88f5 and in
  5.84-6.13 s on master; read whole, 0.60-0.67 s on both. In the catalog,
  250 trees of depth 5 build in 42-44 s as a struct (5.9-6.5 s at reading
  5) and 2.0-2.2 s as a list. T35's catalog code on confit 0def508 builds
  the widest bench forest in 0.60-0.64 s, T36's on 4bb88f5 in 0.25-0.33 s
  (release build, uv's CPython, 2026-10-14). A confit-only reproduction
  is in `/mnt/project-files/transforms-loop/confit-field-reads-456.py`,
  sent to the confit loop on 2026-10-14. Delivered by confit #461
  (merged 2026-10-15, master 620112e). Measure ("Next", item 1), then
  remove this item.
- **A struct output read field by field, built once for the call.**
  Where a query reads each field of a struct output one by one
  (`to_native`'s trial build does, and so does a serving query that names
  fields), confit builds each field read on its own. confit #430 served
  the input guard's part (2026-10-06): a field read no longer builds the
  guard's tests again, and the guard runs once a row. Since then (release
  build, master b4b8d03, 2026-10-14) the forest and spline estimates are
  fitted again (T28), every encoder measured builds, in 0.03 to 5.2 s,
  `StandardScaler` at 256 features builds in 0.46 s (1.4 s before) and
  `PolynomialFeatures` of degree 2 over 64 features in 2.0 s (4.6 s).
  The rest is open, low priority for confit: a struct read field by
  field still builds 1.2 to 3.7 times as slowly as a list
  (`RandomTreesEmbedding` of 100 trees of depth 5: 1.6 s against 0.7 s),
  and this sets the forest, spline and quantile caps (`QuantileTransformer`
  over 800 features of one leaf: 5.7 s against 1.1 s, master 94d7bb1). A
  confit-only struct of 2,000 fields over 64 parameters builds in 1.1 to
  1.4 s read field by field and 0.09 s read whole (loops/confit/PLANS.md).
  When it lands, measure the builds again and refit the estimates.
- **A named refusal, not a Cranelift verifier error.** `OneHotEncoder`
  over 16 string features of 125 categories (2,000 fields) stopped after
  31 s with `cranelift define: verifier error`, an internal error, where
  confit names the limit it passes for the other wide encoders (release
  build, T27's code on master e1f15b9). Since #430 it builds in 1.2 s,
  and no step measured meets the error (2026-10-14). The need stays open
  in confit's plans, for a function that passes the limit only after
  Cranelift's optimizer. The catalog leaves such a step Python either
  way.
- **Two CASE trees in one expression that build in linear time.** One
  balanced CASE tree of q linear pieces over a DOUBLE builds linearly
  (0.09, 0.17, 0.40 s at q = 500, 1,000, 2,000); `0.5 * (tree(x) -
  tree(-x))` over `x = coalesce(p, NaN)` builds in 0.46, 1.37, 4.48 s
  (release build, master 49acad5; a confit-only reproduction is in the
  message sent to the confit loop, 2026-10-05). The catalog no longer
  needs it: `QuantileTransformer` answers both searches from one tree
  whose leaves compute both lines (T12), 0.5, 1.3, 2.7-3.2 s at 1,000,
  2,000, 4,000 quantiles over one feature against 1.4, 4.7, 23 s for the
  two trees (release build, one container, 2026-10-05); capped since T34
  at an estimated 7 s build (`quantile._build_estimate`). Low priority
  for confit: a future entry that needs two trees in one expression would
  raise it again.
- **A long OR of tests that builds as fast as one `greatest`.** The input
  guard of the widest bench `Pipeline` joined 392 tests by OR in each of
  its 9 instances, and the step built in 2.4 to 2.7 s (2.2 to 2.3 s as a
  balanced OR). With one `greatest` or `least` test for each kind of
  rejected value (T35) it builds in 0.9 to 1.2 s, and in 0.6 to 0.7 s
  without the guard (release build, master 9976d96). The catalog no
  longer needs it. Low priority for confit.
- **A cheaper pass over a function body in Python.** confit's
  `SqlFunction` writes a body's values read twice as bound values
  (`sql.shared_texts`, #412) in Python, and it rebuilds every node of the
  body to do so. For `QuantileTransformer` at 8,000 quantiles over one
  feature it takes 1.7 to 1.9 s beside a build of 6.9 to 7.3 s, in a
  `to_native` of 8.8 to 9.1 s (release build, master 94d7bb1, T34). The
  build estimates do not count it. Low priority for confit. A confit-only
  reproduction, `/mnt/project-files/transforms-loop/confit-function-body-pass.py`
  (a tree of 8,000 leaves: 0.68 s in `SqlFunction` beside 4.34 s for the
  build and one call, master 98033e8), was sent to the confit loop on
  2026-10-14; it recorded it at low priority.
Served since this catalog began (#336–#339, #341, #346, #348, #350,
#353, #358, #362, #363, #374, #375, #377, #387, #390): a constant CASE
result counts as trap-free (a step of 32 output fields serves a 64-row call
in 331 µs, against 297 µs inline and 5,081 µs before); a named refusal past
Cranelift's size limit; `greatest`/`least` without the exponential fold;
binary-search dispatch over many instances; a field read expands its call
once (`StandardScaler` at 128 features and three groups builds in 1.2 s,
where it passed the token cap); a cast that cannot fail is trap-free (wide
`PolynomialFeatures` over BIGINT features built in 20-50 s, now 1-3 s);
DuckDB's parse depth; a named refusal past Cranelift's 24-bit index
width, where a 48-feature `Normalizer` met a verifier error; struct reads
that build about linearly in the output fields read (`PolynomialFeatures`
at 2,556 output fields: refused, then 287 s after #348, now 3.2 s after #350, master
5513891); and a dropped function's JIT memory freed (each build leaked two
memory mappings, so a process stalled at `vm.max_map_count` after about
32,000 builds; 6,000 builds now hold 477 mappings); and total calls (`exp`,
a negation) and `ln` under a CASE guard that excludes x <= 0 as trap-free,
so a field read leaves the other output fields unevaluated (Box-Cox at 64
features, one instance: 3,448 us per row and a 15.6 s build before, 52 us
and 0.18 s with a guard arm the entry adds; a negation builds and serves
as `-1.0 * x` does, and the entries spell `-x` again); a repeated pure
subexpression computed once per row, and a refusal past Cranelift's
registers right after lowering (a `Normalizer` repeats its norm in every
output field: l1 and l2 at 32 features were refused after 6.9 s and 16.7 s,
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
within its 3 ulps of numpy); and a value bound once in a SQL function body
(#412): a `confit.sql` node that the body reads twice or more, the same
object, binds once and is computed once per row where it cannot trap. The
spline's recurrence no longer doubles per degree in the build: degree 5, 7
knots, 32 features with `continue` passed the token cap, and now builds in
1.9-2.1 s, and 21 features in 1.1 s against 5.2 s. A tree's CASE built once
and read by every output field of its tree: `trees.py` spells forests so
since T22, and the default `RandomTreesEmbedding` serves a row in 253 us
against 528 us for the paths (a struct return, release build, master
dc3ed3b). A value that a body builds once and reads many times, read at
about the cost of a parameter (#459, which builds a small CASE that
cannot trap, and a field read's NULL guard, without branches): the
catalog reads each numeric feature as `coalesce(x, NaN)`, so 256 such
features read by 16 sums of products served 64 rows in 17.2-18.5 ms, and
now in 1.5 ms. `PCA` over 256 features of 64 components serves 64 rows in
3.7-4.3 ms, against 77-83 ms and the twin's 30-34 ms, and builds in
1.1-1.5 s, against 5.2-5.9 s (T39; release build, master bea1372,
2026-10-15).

## Left Python

Configurations a translator declines (`NotNative`), each with its ground:

- Without `to_native(step, allow_bound=True)`, every configuration within
  a bound above 0 (T19): Box-Cox (4 ulps), and `FunctionTransformer`'s
  `exp`, `log`, `log2`, `log10`, `tan` and `cbrt` where the kernel probe
  does not read 0 (the owner's amendment in
  `decisions/closed/matvec-parity-bound.md`).
- `Normalizer(norm="l2")` where sklearn's `row_norms` does not accumulate
  as numpy's x86-64 baseline einsum kernel, which `_helpers.row_sumsq`
  follows; `_helpers.row_sumsq_is_numpys` probes it. On aarch64 numpy's
  `npyv_muladd_f64` is a fused `vfmaq_f64`, which SQL cannot spell
  (inferred from numpy's source; no aarch64 run yet).
- `MinMaxScaler(clip=True)` with a NaN `feature_range` bound.
- `SimpleImputer` fitted on object data (strings), whose output is not a
  float the step can return; a `missing_values` that is neither NaN nor a
  number (`None`, `pd.NA`, a string).
- A selector that keeps no feature, or whose `get_support()` raises (as
  its `transform` would).
- An encoder whose twin raises on every value of a feature: a string
  feature fitted only on NaN, which sklearn does not match to the None
  that the step hands. Every call raises in Python too.
- `KBinsDiscretizer` bin edges that are not sorted numbers (searchsorted's
  answer is then its search order's), which no strategy fits on finite
  data.
- `QuantileTransformer(output_distribution="normal")`: scipy's
  `norm.ppf` has no SQL twin. A fit whose build is estimated past 7 s
  (`quantile._build_estimate`, from the leaves and the features): 10
  quantiles over 800 features build in 11.4 s. 8,000 quantiles over one
  feature build in 6.9 to 7.4 s and are served (master 94d7bb1, T34).
  Quantiles unsorted or partly NaN (never seen in 3,000 fits; a
  feature missing everywhere is served), quantiles further apart than a
  double spans, or a platform whose `np.interp` fuses its multiply-add
  (`quantile.interp_is_numpys` probes it).
- `PowerTransformer(method="yeo-johnson")` and `standardize=True`
  (on the parity bound, above).
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
  yet. A composition before the last step of a `Pipeline` that is a
  `ColumnTransformer`'s part: the twin hands it an object array, and the
  input guard does not probe what it hands on. A sparse step hands the
  next one a sparse matrix, which it reads by its own sparse branch
  (T36, `compose._SPARSE_READERS`): a step that is not known to answer
  the 0/1 matrix as its dense array (`QuantileTransformer`, which maps
  only the stored values; a `FunctionTransformer` with a `func`;
  `Binarizer` with a negative threshold; anything else outside the
  readers), and a sparse step before the last that hands on values other
  than 0 and 1 (a sparse `SplineTransformer`, a scaler, `Normalizer`, an
  imputer that fills a stored 1 with another value): measured to part by
  an ulp or more.
- A step whose twin raises on the input guard's probe row: each feature
  at 1.0, False or an empty string, or the row its family gives
  (`Entry.base`, a fitted category or knot). Nothing is left to probe
  from.
- `SplineTransformer(sparse_output=True)` where its twin parts from the
  dense one (T36, spline.py "Sparse"): at degree 0 under `constant`
  (above the knots it answers the basis at the last knot) and `periodic`
  (it raises on every row); under `handle_missing="zeros"` over two or
  more features with `continue`, `error` or `periodic` (sklearn zeroes a
  NaN's output fields in the first feature only); over NaN knots (scipy's
  design matrix raises). `extrapolation="linear"` at
  `degree=0, n_knots=2` over two or more features: the twin's running
  `degree` (spline.py) continues two output fields of one from the second
  feature on, and writes a row above the knots into the previous feature's
  output field.
  Knots that are not sorted, partly NaN, or span past a double, and a
  spline whose `c` is not sklearn's shape (no fit makes these). A step
  past an estimated 7 s build, per estimator (`spline._build_estimate`):
  of the 359 configurations measured, none up to 64 features, and 7 of
  96 and 128 features that build in 8.0 to 12.5 s (since T28; 43 before
  confit #430); and any
  step where scipy's `BSpline` does not round as the unfused recurrence
  (`spline.bspline_is_scipys`, an FMA build). `extrapolation="error"`
  with a feature whose knots are NaN: the twin raises on every number
  there.
- A `ColumnTransformer` or `FeatureUnion` with a part that is not a
  catalog entry, or one registered with a bound; a `set_output` container (the step's
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
  to float32), past an estimated 7 s build (`isotonic._build_estimate`,
  at about 11,700 thresholds: 12,314 build in 7.4 s on master 94d7bb1),
  with thresholds further apart than a double spans, or on a platform
  whose `np.interp` fuses its multiply-add (`quantile.interp_is_numpys`).
- `FeatureAgglomeration` with a `pooling_func` other than `np.mean`.
  `np.max` and `np.min` included: on a tie of signed zeros numpy's SIMD
  reduction answers the zero its lane order reaches, neither the first
  nor the last tied operand, so `greatest`/`least` (the first) part from
  it. Spelling numpy's reduction lanes (CPU-dependent, with a probe as
  `row_sumsq_is_numpys` has) would serve them.
- `RandomTreesEmbedding`: a forest whose build is estimated past 7 s (`trees._build_estimate`): over 8 features, past
  about 5,990 output fields, such as 300 trees of depth 5 (6,961 fields,
  9.8 s) and 3 unbounded trees over 2,000 rows (8.4 s). Since T28, 160
  trees of depth 5, 100 trees of depth 6 and one unbounded tree over
  4,000 rows are served.
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `Nystroem`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `BernoulliRBM`,
  `NMF`/`MiniBatchNMF`, the dictionary learners, `SparsePCA`,
  `LatentDirichletAllocation`, `NeighborhoodComponentsAnalysis`. Each needs a design first.
