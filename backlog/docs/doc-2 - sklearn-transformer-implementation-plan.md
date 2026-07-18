---
id: doc-2
title: sklearn transformer implementation plan
type: other
created_date: '2026-07-18 14:01'
---
# sklearn transformer implementation plan

The prioritized list of sklearn transformers to implement, and the parity target for
the [sklearn integration](../../docs/BACKLOG.md#sklearn-transformer-integration--functionality--parity)
work. Unlike [the DataFusion function catalogue](<doc-1 - DataFusion-function-catalogue.md>) (an auto-generated,
*exhaustive* engine surface), this is a **curated** list: we deliberately scope to
**tabular preprocessing** and will never implement most of sklearn. It's a plan, not
a catalogue of everything.

Each transformer ships in two forms (see the BACKLOG item's Phase A / Phase B split):
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

## Tier 0 — co-first (M1 Phase B builds these first)

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

## Structural glue (not leaves, but required — Phase A)

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
dtypes, empty/degenerate columns. This is the oracle behind Phase B's per-transformer
native swaps (native diffed against the same matrix the fallback passes), and it's
distinct from — and feeds — the end-to-end **assembly**-parity harness (Phase A2,
whole `ColumnTransformer` vector). Leaf correctness here; assembly correctness there.
Tracked in [BACKLOG.md](../../docs/BACKLOG.md#per-transformer-differential-parity-harness).
