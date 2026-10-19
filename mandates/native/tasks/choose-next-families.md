# Choose the next families

**Done when:** the tracker has one task for each "not yet" class in [coverage.md](../../../packages/sql-transform/spec/native/coverage.md) that can be served, ordered by value, and each class that cannot be served has its reason in this task's findings or a question.
**State:** not started. **Next:** read the installed `transform` of each "not yet" class.

## Now

17 classes are "not yet". They were filed as needing a design, with no triage against the installed twins.

## Wanted

For each class: the twin's `transform` operations with file and line in the installed sklearn, the nearest existing translator, effort (S, M or L), and whether a closed proposal covers its bound. Reviewers flagged `NeighborhoodComponentsAnalysis` (`_nca.py`, `np.dot(X, components_.T)`) and `BernoulliRBM` (matvec, intercept, `expit`) as possibly cheap; verify both.

Also weigh these open configurations of served classes:

- Normalizer l2 on hosts whose einsum order differs; deep exact reductions.
- Projections fitted in float32, noisy whitened components and degenerate ICA;
  FactorAnalysis near ill-conditioned inverses or pivot ties.
- SimpleImputer over string output or nonnumeric missing markers; selectors
  retaining no features; encoders with no answer for a feature.
- QuantileTransformer normal output (`norm.ppf` needs a SQL spelling), and
  interpolation on hosts that use fused multiply-add.
- FunctionTransformer: `log1p`, `expm1`, user callables, keyword arguments,
  boolean/float16 and object-array paths not yet served.
- Compositions: non-float64 intermediate outputs, set_output containers,
  named/scalar DataFrame columns, object-array nested compositions, sparse
  readers with values other than 0/1, weighted non-double outputs and parts
  with a comparison map. Derive or refuse each branch without changing the twin.
- Sparse spline differences, including degree-0 and missing-value branches;
  platform-dependent fused recurrence; malformed knots and overflow edges.
- FeatureAgglomeration max/min pooling: derive numpy's tied-zero lane order and
  probe it before serving; greatest/least alone do not follow it.
- Nystroem kernels other than rbf each need their own error scale;
  PolynomialCountSketch needs a derived normwise scale for its FFT.
- Depth, build budget and kernel-probe refusals remain intentional until a
  measured capability or a closed ruling changes their premise.

And these measurements:

- Compare twin timings under the same CPython build. The Nix CPython has no
  profile-guided optimization; do not infer a native speed change from that.
- Measure whether forests near the cap should avoid cost-only shared reads,
  as projections already do; keep the spelling centralized in `_registry.py`.
- Replace ordinal numeric category chains with bisection if retained measurements
  show value; use `KBinsDiscretizer`'s sorted-bin search as the nearby example.

## Involved

- `packages/sql-transform/spec/native/coverage.md`
- the installed `sklearn` package

## Out of scope

Implementing any class.
