# Native catalog plans

The working list for [the loop](README.md): open work only, highest value
first. Remove an item when it lands. What is in flight, and who has it, is
on the board: [tickets.md](tickets.md).

## Next

1. The rest of "Ruled 2026-10-06, to build", item 2.

## Ruled 2026-10-06, to build

The owner approved every recommendation in `decisions/closed/` (records
and research in `decisions/research/2026-10-06/`), amended: bit-exact is the
default, and a bound above 0 serves only on the caller's request (it can
flip HistGradientBoosting labels on repeated training values). T19 built
that request, `to_native(step, allow_bound=True)`. In order:

1. **Families on the parity bound.** Since T24 a family declares its error
   scale, `translates(cls, scale=ErrorScale(k, s, tau, g, on))`, and
   `native.check` holds each row to it (`_registry.ErrorScale` says what
   each operand is); since T50, `scaled=` names the estimators it applies
   to, the others bit-exact (kernel.py). T37 served the first family,
   `PCA`, `IncrementalPCA`, `TruncatedSVD` and `GaussianRandomProjection`
   (project.py: S_full, K = n + 3, summed pairwise), and
   `report-format.md` lists error scales.
   - The rest of the linear projections, each with its own S. T41 served
     the centered ones (`FastICA`, the PLS family and
     `LinearDiscriminantAnalysis`: project.py, `_project_centered`).
     T57 served `SparseRandomProjection`, bit-exact in scipy's CSR order
     behind a kernel probe (project.py, `_sparse_project`).
     T58 served `FactorAnalysis` (project.py, `_factor_analysis`),
     K = n + k + 1: its twin computes the inverse `cov_z` with LAPACK on
     each call, by Cholesky where the matrix is exactly symmetric and by
     LU elsewhere, and its bits change with the host's kernels and thread
     count, so S reads D, a first-order bound on the error of either
     path's inverse on any host (`_factor_prior`). It stays Python where
     theta, the bound's second order, is past 2**-10 (measured only where
     A's condition number is past 1e14), or where two pivot candidates
     are within a ratio of 1 + 2**-8.
   - T47 served the distances to fitted centres (`KMeans`,
     `MiniBatchKMeans`, `BisectingKMeans` and `Birch`: distance.py), which
     compare squares (`ErrorScale.g`), K = n + 5. T51 served
     `RBFSampler`, K = n + 3 (kernel.py, its own build estimates), and
     T52 `SkewedChi2Sampler`, K = n + 4 (the RBF entry over ln(x + s),
     numpy's `log` within 1 ulp). T54 served `Nystroem` of the rbf kernel,
     K = max(n + 5, m + 3) (kernel.py, `_nystroem`): its S reads each
     kernel at the least squared distance within the rounding of the
     distance, which the research's S, with the kernel itself, does not
     bound where the rounding moves the kernel's argument by 1 or more.
     `PolynomialCountSketch` is an FFT, not a matvec: not served until a
     normwise S is derived.
2. **A bounded step inside a composition** (ruled 2026-10-15,
   [decisions/closed/bounded-steps-in-compositions.md](decisions/closed/bounded-steps-in-compositions.md)).
   T42 served option 1: with `allow_bound=True`, a step whose bound is
   above 0 serves as the last step of a `Pipeline`, or as an unweighted
   part of a `ColumnTransformer` or `FeatureUnion`, and `native.check`
   holds each output field to the bound of the step or part that gives it,
   at the input its twin reads (`compose._pipeline_bounds`,
   `_parts_bounds`). T59 served option 2 for the affine steps and a part's
   weight: each step after a bounded one gives its output fields a carried
   bound (`_registry.Entry.carry`, `_carry.py`), from the bound of each
   field it reads at the twin's value and its own rounding. The affine
   scalers carry one, as do the linear projections (project.py),
   `FeatureAgglomeration`'s mean, the selectors and `FunctionTransformer`
   of None, `np.negative`, `np.abs` and the like. Any other step after a
   bounded one refuses by name. Left: carries for continuous steps that
   are not affine (`Normalizer`, `PowerTransformer`, the kernel maps, the
   functions within a bound of their own); a map `g` (a distance's square)
   through a composition; after a discontinuous step, `native.check`
   reports the flip rate in place of a bound. Open with the owner: the
   rows where a bounded step's sum overflows on one side only, where a
   later step's validation raises on that side alone
   ([decisions/open/traps-where-a-sum-overflows.md](decisions/open/traps-where-a-sum-overflows.md)).

## Needs from confit

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
  field still builds 1.1 to 9 times as slowly as a list, more as the
  fields grow, and this sets the forest, spline and quantile caps:
  `RandomTreesEmbedding` of 100 trees of depth 5 (2,286 fields) in 0.77 s
  against 0.39 s, of 800 trees (18,360 fields) in 21.5 s against 4.4 s,
  and `QuantileTransformer` over 800 features of one leaf in 4.6 s against
  0.5 s (T40, release build, master 318575b with confit #461). Since T53
  it sets `AdditiveChi2Sampler`'s cap too: with one sample step over 1,024
  features, a struct builds in 8.3 to 9.6 s and a list in 0.85 to 0.92 s,
  and the cap refuses both, as the entry cannot tell them apart (release
  build, master 6b7e7f1). Since T55 it sets `PowerTransformer`'s cap too:
  a Box-Cox of 960 features builds in 4.8 s as a list and 14.4 s as a
  struct (release build, master 1a14ce6), and standardized, of 768
  features, in 4.1 to 4.5 s and 9.0 to 9.3 s (T56, master 1592bfd). Since
  T57 it sets `SparseRandomProjection`'s cap too, most where a struct's
  fields repeat a sum: confit computes such a sum once a row (`share.rs`),
  and a struct read field by field then builds in about 17 µs a field and
  repeated sum. 1,024 fields that repeat 512 sums of two entries build in
  7.2 s as a struct and 0.14 s as a list, and 1,024 fields of their own
  sums in 0.42 s and 0.16 s (release build, master daf0e4c with T57's
  project.py). A confit-only struct of 2,000 fields over 64 parameters builds in 1.1 to
  1.4 s read field by field and 0.09 s read whole (loops/confit/PLANS.md).
  The distances to fitted centres grow faster, about as k³: a struct of
  1,000 centres over 2 features builds in 8.6 s, of 2,000 in 73 s
  (1.1 s as a list), and confit refuses one of 4,000 after 475 s, as
  past the code generator's size limit (T47, release build, master
  c06a15a), so
  their cap takes about 920 centres over 2 features
  (`distance._build_estimate`). When it lands, measure the builds again
  and refit the estimates.
- **Deep expressions.** confit's parser refuses a bound value whose SQL
  nests about 512 additions deep ("a let: sql parser error":
  `Normalizer(norm="l2")` over 1,024 features, whose sum of squares
  follows numpy's order), and `SqlFunction` writes a bound value's SQL by
  recursion, so a chain 1,024 deep raises `RecursionError` (finding 86),
  which `_registry._translate` refuses by name since T48: the step stays
  Python. A family within an error scale sums as a balanced tree, log2(n)
  deep; of the entries measured, only `Normalizer(norm="l2")`, bit-exact
  in numpy's order, meets it. The input guard joins its tests in a chain
  (`_registry.any_of`): Box-Cox's, one test a feature (a standardized
  step joins its edge tests into it, T56), meets the limit at 1,024
  features (T55), past its cap. `SparseRandomProjection`, bit-exact in
  scipy's order, meets it too: each stored entry nests the sum two levels
  deeper, so confit's depth limit of 1,000 refuses a component of 494
  entries as a struct, and the entry refuses one of more than 480 by name
  (T57, `project._MAX_STORED`): at a density of 1/3, about 1,400
  features or more. Low priority for confit.
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
- **A small value read many times, computed once a row.** confit shares a
  subexpression of six nodes or more (`share.rs`, `MIN_SHARED_SIZE`), so
  it computes a feature's `coalesce(x, NaN)`, five nodes, again at each
  read. Where the input guard reads it too, each read is slow (reading 7,
  finding 85): with confit alone, 64 sums of 256 products over such values
  served a row in 32 to 33 µs, 17 to 18 µs without the guard, and 19 to
  22 µs over `coalesce(x, NaN) - 0.0`, seven nodes, which confit shares
  (release build, master 9f74bc0). The catalog no longer needs it: since
  T44 a translator reads each number as `x - 0.0` (`_registry._apart`),
  and reads it twice in one arm where each of its reads sits in its own
  CASE arm (`isotonic.py`). But a build that reads many shared values
  grows with them: `PCA` at 1,024 features of 48 components builds in
  6.2 s reading `x - 0.0`, and in 3.9 to 5.0 s reading `coalesce(x, NaN)`,
  so a projection reads `x - 0.0` only within its cap's reach
  (`project._reads_apart`). A list return of two fields or more over
  shared values builds about as their number squared, where a struct
  read field by field and a single field do not: `PCA` of 2 components
  over 4,096 features, read `x - 0.0`, builds in 11.3 s as a list and
  5.9 s as a struct (T48, finding 87; release build, master af89dae), so
  `project._centered_build_estimate` has a term in n² for a list. Low
  priority for confit. A confit-only reproduction is
  `/mnt/project-files/transforms-loop/t44/reads3.py`: send it to the
  Confit Loop when it runs again.
- **A logarithm computed once a row.** confit's pass that computes a
  repeated value once a row (`share.rs`) shares only a value that cannot
  raise, judged node by node without the facts of its CASE, so it never
  shares `ln`, bare or under a CASE that rules its raise out.
  `SkewedChi2Sampler` (T52, kernel.py) reads ln(x_i + s) in each of the m
  products of x_i, so confit computes n·m logarithms a row, not n. At 32
  features of 64 components a call of 64 rows serves in 5.9 ms as a list
  and 11.8 ms as a struct, and in 0.78 and 0.74 ms where `exp`, which
  confit shares, stands in for `ln`; the builds take 0.48 and 1.70 s,
  against 0.20 and 0.30 s (release build, master 220a5e8). A struct read
  field by field also carries every field's logarithms (the first need
  above), so it builds about as n·m²: 2,048 components over one feature
  ran out of 16 GB of memory. So T52 stays Python past 2,048 products,
  where a call can take longer than the twin's. When it lands, measure
  again, lift that cap (`kernel._SKEWED_PRODUCTS`) and refit
  `kernel._skewed_build_estimate`. Low priority for confit: no other entry
  reads a logarithm in many products. Send it to the Confit Loop when it
  runs again.
- **A definition that DuckDB runs with each bound value written once.**
  confit's `SqlFunction.register` spells each bound value out at each of
  its reads (`sql_body`), so a value that many output fields read is
  written once in each. Each of `Nystroem`'s m kernels is read by all m
  output fields (T54, kernel.py), so its definition grows as m²·n: 25 MB
  of SQL at 8 features of 100 components, and at 64 of 100 DuckDB ran out
  of 16 GB of memory as `native.check` registered it. confit itself builds
  that step in 1.2 s. So `check` holds `Nystroem` to its definition only
  at few components, and the tests keep to them. Low priority for confit.
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
2026-10-15). A struct read field by field that builds about as fast as
before #456 (#461, which answers confit's check for a subquery once per
call, where #456 walked the whole call again at each field read): 4,000
fields over 8 parameters build in 0.54-0.63 s read field by field, against
5.84-6.13 s on #456 and 0.92-1.01 s before it; 250 trees of depth 5 in
2.4-2.6 s, against 42-44 s at reading 6 and 5.9-6.5 s at reading 5; 10
quantiles over 800 features in 7.9 s, against 11.4 s before #456 (the
trial build; T39 and T40, release build, masters bea1372 and 318575b,
2026-10-15).

## Left Python

Configurations a translator declines (`NotNative`), each with its ground:

- Without `to_native(step, allow_bound=True)`, every configuration within
  a bound above 0 (T19): `PowerTransformer` (its error scales), and
  `FunctionTransformer`'s `exp`, `log`, `log2`, `log10`, `tan` and `cbrt`
  where the kernel probe does not read 0 (the owner's amendment in
  `decisions/closed/matvec-parity-bound.md`).
- `Normalizer(norm="l2")` where sklearn's `row_norms` does not accumulate
  as numpy's x86-64 baseline einsum kernel, which `_helpers.row_sumsq`
  follows; `_helpers.row_sumsq_is_numpys` probes it. On aarch64 numpy's
  `npyv_muladd_f64` is a fused `vfmaq_f64`, which SQL cannot spell
  (inferred from numpy's source; no aarch64 run yet). At 1,024 features
  or more, where numpy's order nests its sum of squares too deep for
  confit ("Deep expressions" above): confit's parser refuses it at 1,024
  features, and at 2,048 `SqlFunction` recurses past Python's limit,
  which `to_native` refuses by name since T48 (finding 86).
- The linear projections (project.py), each only with `allow_bound=True`:
  a `PCA` or `IncrementalPCA` fitted in float32 (it projects its mean in
  float32); a whitened `PCA` component of rounding noise; a step past an
  estimated 7 s build (`project._build_estimate`, and
  `_centered_build_estimate` where the twin subtracts a mean, since each
  product then reads that difference); a fitted mean, scale or
  matrix not all finite. `LinearDiscriminantAnalysis(solver="lsqr")`, whose
  twin raises. A `FastICA` that whitens rounding noise (a singular value
  of 0, at most sqrt(eps) of the largest, or raised to the "eigh" solver's
  floor), or whose fit broke down: unwhitened, rows of the unmixing matrix
  that are not unit vectors (on data that is not white); under
  "unit-variance", a row of `components_` that is not finite (a source of
  no spread).
- `SparseRandomProjection` over sparse components (project.py,
  `_sparse_project`): a `components_` other than the CSR matrix the fit
  makes, or not all finite; a component of more than 480 stored entries,
  whose sum in scipy's order confit does not parse; a step past an
  estimated 7 s build (`project._sparse_build_estimate`); a platform whose
  scipy does not sum a CSR product unfused in stored order
  (`project.csr_sum_is_scipys`). Of density 1, its dense components serve
  only with `allow_bound=True`, as the linear projections above.
- `FactorAnalysis` (project.py, `_factor_analysis`), only with
  `allow_bound=True`: a fit whose theta is past 2**-10, where
  `I + Wpsi @ components_.T` is too ill-conditioned for a bound that holds
  on any host; a fit where two pivot candidates of the entry's LU of that
  matrix are within a ratio of 1 + 2**-8; a matrix with no finite inverse;
  a fitted mean, component or noise variance not all finite, or a noise
  variance not positive; a step past an estimated 7 s build
  (`project._factor_build_estimate`).
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
  quantiles over 800 features build in 9.0 s, 2,000 over 8 features in
  8.3 s, and 16,000 over one feature in 7.4 s. 12,000 quantiles over one
  feature build in 4.8 s and are served (T46 on master 12aca98).
  Quantiles unsorted or partly NaN (never seen in 3,000 fits; a
  feature missing everywhere is served), quantiles further apart than a
  double spans, or a platform whose `np.interp` fuses its multiply-add
  (`quantile.interp_is_numpys` probes it).
- `PowerTransformer` with `allow_bound=True`: a lambda not finite; a step
  past an estimated 7 s build (`power._build_estimate`, which takes about
  347 features of Yeo-Johnson and 702 of Box-Cox, and standardized about
  313 and 682); a kernel probe that does not find confit's `ln` and `exp`
  the C library's, or, for Yeo-Johnson, numpy's `log1p` and `expm1` within
  1 ulp of it (`power.kernel_distances`); a twin that overflows on no
  single edge (`power._edges`, not seen in the measurements); standardized,
  a scaler other than the one the fit makes, a `mean_` or `scale_` not
  finite, or a `mean_` of 2**970 or more, where `t - mean_` can overflow on
  one side only (`power._scaler`).
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
  by CPU). A bounded function composes only as the last step or an
  unweighted part, as any bounded step (item 2 above).
- A `Pipeline` with a step that is not a catalog entry, or a step whose
  bound is above 0 before the last (no bound is derived through a later
  step: `x - mean_` near `mean_`); a `ColumnTransformer` or `FeatureUnion`
  with a weighted part whose bound is above 0, or one whose error scale
  compares through a map; a `Pipeline` with `transform_input` (which only
  transforms fit metadata, so the refusal is conservative); a step before
  the last whose output is not float64 (`MissingIndicator`'s booleans, an
  encoder's or discretizer's `dtype`: exact 0/1 or small integers either
  way, not yet shown to read the same downstream); passthrough steps only, over a
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
  past an estimated 7 s build, per estimator (`spline._build_estimate`),
  at about 170 to 230 features of degree 3 and 8 knots, or 110 to 160 of
  degree 5: of the 175 configurations measured (T46 on master 12aca98),
  none up to 64 features, 2 of 128 features (degree 5, 8 knots: 7.2 and
  8.5 s), 4 of the 6 of 192 features (8.0 to 14.4 s) and all 9 of 256 and
  320 features (confit refuses the struct return of the three of 256
  features at degree 5 and of 320 under `continue`, past 4,000,000
  tokens); and any
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
  at about 26,800 thresholds: 28,524 build in 7.4 s, T46 on master
  12aca98),
  with thresholds further apart than a double spans, or on a platform
  whose `np.interp` fuses its multiply-add (`quantile.interp_is_numpys`).
- `FeatureAgglomeration` with a `pooling_func` other than `np.mean`.
  `np.max` and `np.min` included: on a tie of signed zeros numpy's SIMD
  reduction answers the zero its lane order reaches, neither the first
  nor the last tied operand, so `greatest`/`least` (the first) part from
  it. Spelling numpy's reduction lanes (CPU-dependent, with a probe as
  `row_sumsq_is_numpys` has) would serve them.
- `RandomTreesEmbedding`: a forest whose build is estimated past 7 s
  (`trees._build_estimate`): over 8 features, past about 10,000 output
  fields, such as 300 trees of depth 6 (11,402 fields, 8.1 s) and 2
  unbounded trees over 5,000 rows (10,000 fields, 8.7 s), and over 128
  features past about 5,200, such as 250 trees of depth 5 (5,751 fields,
  8.4 s; T46 on master 12aca98). 400 trees of depth 5 over 8 features
  (9,210 fields, 6.3 s) are served.
- `AdditiveChi2Sampler`: a step whose build is estimated past 7 s
  (`kernel._additive_chi2_build_estimate`, T53): past about 900 features
  of one sample step, 148 of three or 12 of 100. 3 steps over 256
  features build in 19.5 s as a list and 26.5 s as a struct (master
  6b7e7f1).
- `Nystroem` of a kernel other than rbf (each needs its own error scale),
  only with `allow_bound=True`; a gamma that the twin does not multiply by
  as a double (a long double, an array, a factor -γ above 0 or not
  finite); a component or a normalization not finite; a platform whose
  numpy `exp` is more than 1 ulp from confit's on the kernel probe; a step
  whose build is estimated past 7 s (`kernel._nystroem_build_estimate`,
  T54): past about 276 components over 1 to 8 features, 200 over 256, 77
  over 1,024 or 25 over 2,048, and past 3,100 features of one component.
  300 components over 8 features build in 7.5 s as a list and 8.3 s as a
  struct (master 3e22aa6).
- Any step confit does not build (past its expansion cap or Cranelift's
  function size): `to_native` builds it first.

## Later

- The CPython of the Nix dev shell (#428) has no profile-guided
  optimization. On it, a 64-row call of a twin took a median of 1.23 times
  as long (reading 5, section 3). Measure the timings of a reading on the
  same CPython each time, or pin a CPython with this optimization in
  `flake.nix`. The ruling is
  [the worker environment](decisions/closed/worker-environment.md).
- In the dev shell, `maturin develop --release` stops, because Nix marks
  its CPython as externally managed. Use `--skip-install`, and write this
  in the build steps of the worker brief.
- The input guard's probe calls the twin's `transform` once per feature
  and special value (`_registry._probed`): 3n calls of a row of n
  features. `QuantileTransformer` transforms each feature in a Python
  loop and lets NaN through, so each NaN probe transforms every feature:
  at 800 features of 10 quantiles, the probe takes 8.0 to 9.1 s of
  `to_native`'s 19.7 to 20.4 s (reading 7, finding 84; the trial build
  takes 9.4 to 10.9 s, and the cap estimates only that). Probe a special
  value over all features in one call of n rows first, and probe the
  features one at a time only where that call raises.
- One time for every cap (reading 8, finding 92). T46 fitted the forest,
  isotonic, quantile and spline estimates on the trial build alone; T47,
  T48 and T51 to T53 fitted theirs on the whole warm `to_native`, which
  also runs the probe. Since T53, each estimate's docstring says which
  time it estimates. So a cap of 7 s bounds the trial build in T46's
  families and the whole `to_native` in the others: `PCA` over 4,600
  features of one component takes 5.6 to 5.9 s in `to_native`, of which
  the trial build is 1.5 to 1.6 s. Fit T46's four estimates again on the
  warm `to_native`, so that each cap counts the probe, after the cheaper
  probe above.

- Since T44's reads, a forest over many features builds in up to 1.22
  times the time (100 trees of depth 5 over 128 features: 2.7 s, against
  2.2 s in one process) and serves a row in 0.80 to 0.93 times it. A
  forest whose estimate passes the cap could read each number as it is,
  as a projection does (`project._reads_apart`): the cap reaches about
  5,200 output fields at 128 features since T46, and reached 6,800 before
  T44's reads (master 318575b).

- An ordinal output field over a number feature's classes is a chain of
  one arm a class, after a test of both ends (`encode._field`, T46). A
  bisection over the sorted classes, as `KBinsDiscretizer`'s ordinal bins
  do (T45), would test about log2(k) of k classes: at 64 features of 64
  classes, a row of a 1,024-row call takes 37 µs with the chain (release
  build, master 12aca98 with T46).

- At one feature and one output field, the twins of `FastICA`, the PLS
  classes and `FactorAnalysis` answer a zero's sign by the rows of the
  call: in a call of one row, `np.dot` of 1×1 arrays answers the product
  itself, -0.0 included, and in a call of more rows it adds the product
  to 0.0. Since T43 the entries answer the one-row sign, which `check`
  compares, and the bound allows either sign. In T58's parity run, the
  twin of `FactorAnalysis` answered 0.0 where its entry answered -0.0 in
  321,024 fields of calls of 5,000 rows, in 19 fits of one feature. So which sign matches the twin
  depends on the number of rows in each call of the twin.

- Transformers that read their fit samples: `KNNImputer`, `Isomap`,
  `LocallyLinearEmbedding`, `KernelPCA`, `KNeighborsTransformer`,
  `RadiusNeighborsTransformer`, `IterativeImputer`, `BernoulliRBM`,
  `NMF`/`MiniBatchNMF`, the dictionary learners, `SparsePCA`,
  `LatentDirichletAllocation`, `NeighborhoodComponentsAnalysis`. Each needs a design first.
