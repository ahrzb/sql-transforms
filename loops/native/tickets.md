# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. Wave 1 (KBinsDiscretizer #351,
QuantileTransformer #357, PowerTransformer #356) ran before the board
existed and is merged. A worker's prompt is [`../worker-brief.md`](../worker-brief.md),
then [`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T1 | `Pipeline` of catalog entries | `claude/native-pipeline` | — | `catalog_test.py`, `__init__.py`, PLANS | `session_018LHu15JPWd1LYRNLLTHeZJ` | #368 | merged |
| T2 | `FunctionTransformer` over numpy functions with an exact SQL twin | `claude/native-function` | — | `catalog_test.py`, `__init__.py`, PLANS | `session_01KLp5t39aq6tePxHvz7kJwi` | #367 | in review |
| T3 | move the loop to `loops/native/` | `claude/native-loops` | #361 (merged) | every native doc; T1 and T2 merge master after it | inline | #366 | merged |
| T4 | adopt confit #362; re-measure the caps (Box-Cox width, `MAX_QUANTILES`, `MAX_LANES`) | `claude/native-caps` | — | `catalog_test.py` (`MAX_LANES`), PLANS | inline | #369 | merged |
| T5 | adopt confit #363 and #358: `Normalizer` re-measured, the max norm as one `greatest` | `claude/native-shared` | — | `_helpers.py` (`row_max`), PLANS | inline | | in review |

Next up, once a slot frees and the account's usage warning clears:
`SplineTransformer`, then `AdditiveChi2Sampler`, then `ColumnTransformer`
and `FeatureUnion` (after T1).

## T1: `Pipeline` of catalog entries

**Goal.** A step whose instances are fitted `sklearn.pipeline.Pipeline`s
of catalog entries serves natively: the composition of the entries'
translations, bit-exact. Pipelines are what users deploy, so this turns
the catalog's 26 entries into the pipelines built from them.

**The twin.** sklearn 1.9 `Pipeline.transform` (`sklearn/pipeline.py`):
each step's `transform` on the previous step's output, in order,
`"passthrough"` and `None` steps skipped (`_iter(with_final=True,
filter_passthrough=True)`). Read it in the installed sklearn, including
what `transform_input` does.

**Pointers.**
- `sql_transform/native/_registry.py`: a translator is
  `(est, x, types) -> list[S.Expr]`; `catalog()` maps an estimator class
  to its `Entry` (`translate`, `ulps`). Compose through `catalog()`: the
  first step gets the step's feature expressions and declared types, every
  later step the previous step's lanes with `types` all `pa.float64()`
  (the twin hands it a float64 array).
- A new module `compose.py` registers `Pipeline` (exact class only).

**Rules.**
- Serve a pipeline only when every step is a catalog entry registered at
  0 ulps; otherwise raise `NotNative` naming the step. A non-zero bound
  composed with a later step is not bounded by it (a 4-ulp Box-Cox lane
  followed by `StandardScaler`'s `x - mean_` can be any number of ulps
  apart). Register `Pipeline` at 0.
- A step that is not in the catalog, a `transform_input`, or anything
  else that changes what `transform` computes: `NotNative`, naming it.
- Nested pipelines compose naturally; serve them.
- Expression size multiplies with depth (no shared subexpressions in
  confit yet: its ticket T1, PR #363). Let `to_native`'s trial build
  refuse what confit cannot build; measure build and serve time for the
  widest pipelines you fixture and record them in the PR.

**Fixtures** (your own `FIXTURES[Pipeline]` block): two- and three-step
pipelines across families, e.g. `SimpleImputer` then `StandardScaler`;
`StandardScaler` then `PolynomialFeatures`; `OneHotEncoder(dense)` then
`MaxAbsScaler` over string features; `KBinsDiscretizer` then
`OneHotEncoder(dense)`; `MinMaxScaler` then `Binarizer`;
`StandardScaler` then `SelectKBest`; one with a `"passthrough"` step; one
nested. Check how `Pipeline.__sklearn_tags__` reports input tags (the
generator's string features and holes follow them) and say what you found.

**Acceptance.** Bit-exact on the fixtures (8 seeds in the gate; run
`NATIVE_SEEDS=200` once over your configurations and report it); every
refusal named and tested; gate green; coverage regenerated. `coverage.md`
lists `Pipeline` under "composition": leave `coverage.py`'s categories as
they are, and say in the PR how you would show a served composition.
Changes to `_registry.py`, `_helpers.py` or `_check.py`: avoid them; if
one is unavoidable, keep it minimal and name it first in the PR.

## T2: `FunctionTransformer` over numpy functions with an exact SQL twin

**Goal.** `sklearn.preprocessing.FunctionTransformer` serves natively
where its function has an exact SQL spelling: the identity and numpy's
elementwise functions whose result is the same double in DuckDB.

**The twin.** sklearn 1.9 `FunctionTransformer.transform`
(`sklearn/preprocessing/_function_transformer.py`): validate if
`validate=True`, then `func(X, **(kw_args or {}))`, `func=None` being the
identity. Read it in the installed sklearn.

**Pointers.** A new module `function.py`; translators are
`(est, x, types) -> list[S.Expr]` (see `scalers.py`); `_helpers.f64`,
`_helpers.isnan`.

**Rules.**
- Serve: `func=None`; numpy functions exact on both sides, such as
  `np.abs`, `np.negative` (spell it `-1.0 * x`: confit serves a product
  faster than a negation, PLANS), `np.positive`, `np.square`, `np.sqrt`
  (IEEE sqrt is correctly rounded on both sides; DuckDB's `sqrt` of a
  negative raises where numpy answers NaN, so guard it), `np.reciprocal`,
  `np.floor`, `np.ceil`, `np.trunc`, `np.rint` (half to even: check
  which DuckDB function rounds that way), `np.sign`. Check every one
  against numpy on signed zeros, NaN, infinities and subnormals.
- Transcendental functions (`np.log`, `np.exp`, `np.log1p`, `np.expm1`,
  `np.sin`, ...): numpy's float64 kernels on x86-64 with AVX-512 are its
  own SIMD code, not glibc's (measured in #356), and DuckDB has no
  `log1p`/`expm1`. Serve one only with a probe proving this platform's
  numpy answers as the SQL does, or with a bound measured over 200 seeds
  and cited; a bound that is not small goes to
  `decisions/open/power-parity-bound.md`'s question, not into code.
- `kw_args` that change the result, `validate=False` inputs that are not
  numeric rows, a `func` that is not one of the served functions (lambdas,
  partials, user functions): `NotNative`, naming it.

**Fixtures** (your own `FIXTURES[FunctionTransformer]` block): the
identity with `validate` True and False, and each served function, over
the generator's regimes (edges, NULLs, small values).

**Acceptance.** Bit-exact on the fixtures (8 seeds in the gate; run
`NATIVE_SEEDS=200` once over your configurations and report it); every
refusal named and tested; gate green; coverage regenerated; the PR lists
each function served or refused with the reason.
