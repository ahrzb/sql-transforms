# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

Easiest first; each is one family, one PR.

1. **One CASE per tree** (#412). Build each tree's nested CASE once, and
   read that object in every output field of the tree. On the default
   forest (2,286 output fields) it serves 118 us a row against 202 us for
   the paths, and builds in 1.28 s against 1.84 s (the confit loop's
   measurement, release build of 4865d9f). Then fit the trees cap again on
   the new spelling: reading 4 measured 0.25 ms a path step at 1,173 steps
   and 0.44 ms at 21,353 (finding 56), where `trees.py` assumes 0.28 ms.
2. **The kernel probe draws random significands**
   (`function.kernel_distance`). A quarter of its draws are `exp(uniform)`,
   on which two accurate `log` kernels always agree (#404, `chi2.md` §4).
   The registered bounds hold on 2,000,000 random-significand draws in each
   of three ranges (numpy 2.5.1 against DuckDB 1.5.5, 2026-10-06): `log`
   1, `log2` 1, `log10` 2, `exp` 1, `tan` 1, `cbrt` 3. Since T19 a probe
   that reads 0 makes a bounded function bit-exact on that platform: it
   serves by default, is checked at 0 and composes, as `sin` and `cos`.
3. **"Lane" in the catalog's code means an output field.** GLOSSARY.md
   defines a lane as the machine type that holds a value in a built
   function. Rename `MAX_LANES`, `_registry._lanes` and the docstrings'
   "lanes" as a ticket of its own: every module uses the word.

## Ruled 2026-10-06, to build

The owner approved every recommendation in `decisions/closed/` (records
and research in `decisions/research/2026-10-06/`), amended: bit-exact is the
default, and a bound above 0 serves only on the caller's request (it can
flip HistGradientBoosting labels on repeated training values). T19 built
that request, `to_native(step, allow_bound=True)`. In order:

1. **The parity bound in `native.check`:** per output field
   `|g(entry) - g(twin)| <= K*eps*S + tau`, with S and K declared per
   family, S computed overflow-safely, and the one-sided-infinity rule on
   the output (matvec-parity-bound.md, Recommendation 1 and 6).
2. **The input guard** (tolerated-differences.md): probe each leaf for the
   values its twin rejects, trap in the first output field, check "raises
   iff the twin raises" on every row. The row generator draws ±inf since
   T20, and the periodic spline breach at ±inf is fixed (#406).
3. **Densify sparse outputs** (sparse-outputs.md): one helper at
   `_udf.py:314`, `_projection.py:352`, `model/_foreign.py:128` and
   `native/encode.py:110`; then drop the sparse guards of OneHotEncoder,
   KBinsDiscretizer, MissingIndicator and SplineTransformer (not degree 0).
4. **Families on the parity bound:**
   - Linear projections (`PCA` and kin): S_full, K = n + 3, summed pairwise;
     refuse whitened components whose scale was clipped.
   - `PowerTransformer` Yeo-Johnson and `standardize=True`: K 10 / 12 / 7;
     restate served Box-Cox as K = 5. Needs NaN and `isinf` arms.
   - `AdditiveChi2Sampler`: K = 4 (3 with the twin's own cosh), 0 where the
     probes read 0; ship `sample_steps=1` (bit-exact) first. The probe's
     draws are in "Next" (the kernel probe).
   - Distances to fitted centres (`KMeans` and kin): compare squares,
     K = n + 5. `RBFSampler` n + 3, `SkewedChi2Sampler` n + 4, `Nystroem`
     max(n + 5, m + 3). `PolynomialCountSketch` is an FFT, not a matvec:
     not served until a normwise S is derived.
5. **Still open for the owner:** a bounded step inside a composition
   (decisions/open/bounded-steps-in-compositions.md). `compose.py` keeps
   refusing one meanwhile.

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
within its 3 ulps of numpy); and a value bound once in a SQL function body
(#412): a `confit.sql` node that the body reads twice or more, the same
object, binds once and is computed once per row where it cannot trap. The
spline's recurrence no longer doubles per degree in the build: degree 5, 7
knots, 32 features with `continue` passed the token cap, and now builds in
1.9-2.1 s, and 21 features in 1.1 s against 5.2 s. A tree's CASE built once
and read by every output field of its tree serves the default
`RandomTreesEmbedding` in 118 us a row, against 202 us for the shipped
paths (the confit loop's measurements, release build of 4865d9f).

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
  number (`None`, `pd.NA`, a string); `MissingIndicator(sparse=True)`.
- A selector that keeps no feature, or whose `get_support()` raises (as
  its `transform` would).
- `OneHotEncoder(sparse_output=True)`, the default: a sparse output, which
  the Python step does not serve either (it reads a row with `float()`,
  and a sparse row is not a float). A note for the step, not the catalog.
- `KBinsDiscretizer(encode="onehot")`, the default: a sparse output, as
  for `OneHotEncoder` above. Bin edges that are not sorted numbers
  (searchsorted's answer is then its search order's), which no strategy
  fits on finite data.
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
  (decisions/closed/sparse-outputs.md). `extrapolation="linear"` at
  `degree=0, n_knots=2` over two or more features: the twin's running
  `degree` (spline.py) continues two lanes of one from the second feature
  on, and writes a row above the knots into the previous feature's lane.
  Knots that are not sorted, partly NaN, or span past a double, and a
  spline whose `c` is not sklearn's shape (no fit makes these). A step
  past an estimated 7 s build, per estimator (`spline._build_estimate`):
  of the configurations measured, none up to 64 features, and ten of 96
  and 128 features that build in 7.1 to 15 s; and any
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
  output (decisions/closed/sparse-outputs.md). Past 25,000 path steps (a
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
