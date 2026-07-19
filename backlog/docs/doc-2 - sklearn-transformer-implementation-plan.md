---
id: doc-2
title: sklearn transformer implementation plan
type: other
created_date: '2026-07-18 14:01'
---
# sklearn transformer implementation plan

The prioritized list of sklearn transformers to implement, and the parity target for
the sklearn integration work (strategy — fallback-first, then native per transformer —
canonical: decision-4; scope + parity harness below). Unlike [the DataFusion function catalogue](<doc-1 - DataFusion-function-catalogue.md>) (an auto-generated,
*exhaustive* engine surface), this is a **curated** list: we deliberately scope to
**tabular preprocessing** and will never implement most of sklearn. It's a plan, not
a catalogue of everything.

Each transformer ships in two forms (see the BACKLOG item's fallback / native-swap split):
first a **Python fallback** (wraps the real sklearn object; trivially correct), then
a **native** engine implementation (a SQLTransform/expression, no sklearn call at
serve time). Both are validated by the per-transformer **differential parity harness**
(see below): our output must be bit-identical to sklearn's across a param + input
matrix.

The **"native machinery"** column is the load-bearing insight: most high-priority
transformers map onto engine features **already shipped** (window aggregates,
`PARTITION BY`, static `LookupJoin`, struct/list + `UNNEST`), which is exactly why
the spine was built first. "Learns" ties to the execution-model *state shape*
(scalar / list / code-map / per-group table).

Priority tiers reflect **what a served tabular request actually touches**, not raw
sklearn popularity — target users do mixed numeric/categorical tabular models
(prediction, recommendation with high-cardinality IDs).

## Tier 0 — co-first (the native-swap phase builds these first)

| Transformer | Learns (state shape) | Native machinery | Parity notes |
|---|---|---|---|
| `StandardScaler` | mean, scale (scalar) | window agg `AVG`/`STDDEV OVER ()` — **shipped** | `with_mean`/`with_std` toggles; population vs sample std (sklearn uses population `ddof=0`). |
| `SimpleImputer` | fill value: mean/median/most_frequent/constant (scalar) | mean→window agg (shipped); median/mode→percentile/mode agg (**partial** — needs the agg) | `strategy`, `fill_value`, `add_indicator` (emits an extra missing-indicator column — fan-out). |
| `OrdinalEncoder` | category→code map (code-map) | static `LookupJoin` + unknown→LEFT-lookup miss — **shipped** (incl. the nullability fix) | `handle_unknown`/`unknown_value`, `encoded_missing_value`; category order determinism. |
| `OneHotEncoder` | category list (list) | `array_agg`/distinct + fan-out via `unnest(struct)` — **shipped** (rich types) | `handle_unknown`, `drop` (first/if_binary), `min_frequency`/`max_categories` (infrequent grouping), sparse vs dense. **The fan-out + unknown-category stress case.** |

## Tier 1 — close behind (near-free once Tier 0 exists)

| Transformer | Learns (state shape) | Native machinery | Parity notes |
|---|---|---|---|
| `MinMaxScaler` | data min, max (scalar) | window `MIN`/`MAX OVER ()` — **shipped** | `feature_range`; clip behavior on unseen out-of-range at transform. |
| `MaxAbsScaler` | max abs (scalar) | window agg — **shipped** | — |
| `RobustScaler` | median, IQR quantiles (scalar) | percentile agg — **partial** (needs quantile agg; DataFusion has `approx_percentile_cont`, but exact-parity may need exact quantiles) | `quantile_range`, `with_centering`/`with_scaling`; **exact vs approx quantile is a parity risk.** |
| `TargetEncoder` | per-category target mean (per-group table) | `PARTITION BY` group agg — **shipped** | sklearn's cross-fit/smoothing at *fit* is subtle — parity needs matching its shrinkage + CV scheme exactly. |

## Tier 2 — later (real work, lower serving frequency)

`KBinsDiscretizer` (bin edges — quantile/uniform/kmeans strategies; fan-out if
`encode=onehot`), `PowerTransformer` (Box-Cox/Yeo-Johnson lambdas), `QuantileTransformer`
(learned quantile fn — large state), `Normalizer` (row-wise L1/L2 — no fit state,
per-row), `Binarizer` (threshold — no fit state), `KNNImputer` (needs neighbor lookup —
expensive at serve, likely fallback-only). Prioritize by real demand.

## Structural glue (not leaves, but required — fallback phase)

`Pipeline` (sequencing), `ColumnTransformer` (column routing + horizontal concat — the
assembly-parity surface), `FunctionTransformer` (stateless wrap of an arbitrary fn —
fallback-only unless the fn is expressible).

## Out of scope (not tabular preprocessing — do not implement)

- **Target-side, not feature-side:** `LabelEncoder`, `LabelBinarizer`,
  `MultiLabelBinarizer` (operate on `y`, not the feature matrix).
- **Text:** `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`.
- **Decomposition / projection:** `PCA`, `TruncatedSVD`, `GaussianRandomProjection`.
- **Feature selection:** `SelectKBest`, `VarianceThreshold`, `SelectFromModel`.
- **Kernel / nonlinear expansion:** `PolynomialFeatures` (borderline — cheap but
  rarely on the serving hot path; revisit only if asked), `SplineTransformer`,
  `Nystroem`, `RBFSampler`.

## Differential parity harness (per transformer)

Every transformer — fallback and native — runs through a **parametrized parity
matrix**: our `transform` output must be bit-identical to the real sklearn object's,
across (a) a spread of the parity-sensitive params in the tables above and (b) input
edge cases: nulls, unseen categories, single row vs batch, integer/float/string
dtypes, empty/degenerate columns. This is the oracle behind the native-swap phase's per-transformer
native swaps (native diffed against the same matrix the fallback passes), and it's
distinct from — and feeds — the end-to-end **assembly**-parity harness (the ColumnTransformer-glue
slice, whole `ColumnTransformer` vector). Leaf correctness here; assembly correctness there.

## Integration strategy & scope

Strategy (native-goal + opaque-fallback, fallback-first) canonical record: **decision-4**.
In scope as the primary serving goal (supersedes the old README `sklearn.*` surface).
Correctness/coverage first — the simplest bit-identical implementation wins even if it
isn't yet the zero-copy path; the optimized serving path (DRAFT-3) is validated against
the parity harness this work delivers.

- **Two integration directions, compose-first:** (a) *compose* — our transformers are
  sklearn estimators the user drops into their own `Pipeline`/`ColumnTransformer` (primary;
  incremental, low-friction adoption, one transformer at a time, coexisting with sklearn);
  (b) *consume* — accelerate a whole already-fitted sklearn pipeline handed to us
  (secondary). Estimator-interface compliance is what makes (a) work (DRAFT-1) and is the
  gating requirement.
- **Transformer coverage**, ranked by "what a served request touches" (not raw popularity):
  `SimpleImputer` + `StandardScaler` (numeric) and `OrdinalEncoder` + `OneHotEncoder`
  (categorical, co-first — target audience is mixed numeric/categorical, incl.
  recommendation with high-cardinality IDs). Other scalers near-free once one scaler
  exists; `TargetEncoder` close behind. (Full tier table above.)
- **The real unlock is the structural glue, not the leaves:** `Pipeline` (sequencing) and
  `ColumnTransformer` (column routing + output concatenation). Build these alongside the
  first leaves — bare transformers can't run a realistic pipeline.
- **Unknown-category handling is a designed-in requirement, not a flag:** cold-start unseen
  IDs are the common case in serving/recommendation. Match sklearn's `handle_unknown` /
  `drop` / infrequent-category semantics exactly.
- **Acceptance test = end-to-end assembly parity:** the full feature vector (width + column
  order + values) must be bit-identical to `ColumnTransformer.transform()`, because the
  downstream model consumes it positionally and a mislabeled column is a *silent* wrong
  prediction. Per-transformer correctness in isolation is not sufficient.
- **First-class Python fallback per transformer** (run the real sklearn object) so partial
  coverage ships and the native surface grows incrementally. Fallback is not free at
  serving time — it drags the DataFrame back onto the request path (see DRAFT-3 serving +
  DRAFT-4 benchmark).
- **Open sub-question:** whether/how the SQL authoring surface (`sklearn.standardize(col)`-
  style, goal 1) maps onto this. Both integration directions work with fitted
  sklearn-estimator objects; the SQL authoring front-end is a separate question.
