# Vision

sql-transform is becoming a **compose-in, sklearn-compatible feature-transform
library with a fast serving path**: typed transformers you drop into an existing
sklearn pipeline that run online at low latency, with an optional SQL surface for
authoring them declaratively. Two capabilities define it:

1. **Authoring** — express a transform as a typed, composable unit (optionally as a
   SQL expression) instead of hand-rolled Python/sklearn glue.
2. **Fast inference** — once fitted, apply it to single rows or batches at low
   latency, without going back through a SQL engine.

Everything below serves one of those two, in service of a single near-term goal.

## The bet

Near-term, the target is concrete: **a well-adopted alternative to raw sklearn
preprocessing transformers**, for people serving those transforms online at low
latency. The feature-store direction is a *later* expansion, not the current goal —
though, as the third hook shows, its groundwork falls out naturally. Three things,
in order, are meant to earn adoption.

### 1. It composes in and just works — the wedge

Not "swap your imports." The unit of adoption is dropping *one* of our transformers
into an existing sklearn pipeline and having it compose seamlessly with sklearn's
own transformers, the surrounding pandas/numpy/joblib tooling, and the model. That
requires implementing sklearn's estimator interface (`fit`/`transform`/
`get_feature_names_out`, `get_params`/`set_params`, cloneable) so ours are
first-class citizens *inside* a stock `Pipeline`/`ColumnTransformer`. Adoption is
incremental — one transformer at a time, coexisting with sklearn — far lower
friction than wholesale replacement. It is also the answer to ONNX's weak spot: no
export/convert step, no separate runtime, no new mental model — you stay in Python,
in your existing pipeline, with your own objects.

### 2. It's dramatically faster at n = 1 — the reason to switch

Good DX gets someone to look; speed is what makes leaving raw sklearn rational.
Both are required — great DX at equal speed just means "keep using sklearn."

The speedup has a sharper source than "make it quick": the cost it targets is the
**intermediate representation**, not the arithmetic. In a typical sklearn serving
path a single request becomes a `dict` and then a pandas `DataFrame` so a
`ColumnTransformer` can run over it — and the transform itself (subtract a mean,
divide by a scale, look up a category) is nanoseconds. Building the DataFrame to
hang that arithmetic off of is hundreds of microseconds: block manager, index,
dtype inference, a Series per column. At `n = 1` that fixed cost never amortizes,
so it *is* the latency. The same cost is invisible in training/batch land, where it
spreads across 100k rows and vanishes — which is exactly why online serving is the
case that suffers. Memory follows the same shape: a DataFrame per request is a
burst of allocation and garbage on every call, and under concurrent load that churn
surfaces as tail-latency jitter.

So the bet for inference is `input → typed values → feature buffer → model.predict`,
**never materializing a dict-of-columns or a DataFrame on the request path**. That
is also why transforms are reimplemented rather than called faster — sklearn's API
is DataFrame/array-in, so owning the transform logic is the only way to skip the
intermediate. The per-transformer math is trivial; the value is entirely in the
representation choice. The correctness bar is the opposite and unforgiving: the
assembled feature vector must be bit-identical — same width, same column order — to
what sklearn's `ColumnTransformer` would have produced, because a downstream model
consumes it positionally and a mislabeled column returns a *confidently wrong*
prediction, not an error. End-to-end assembly parity, not per-transformer
correctness in isolation, is the real acceptance test. One consequence: falling back
to the real sklearn object for a transform is *not free* — it drags the DataFrame
back onto the request path, so hot-path features want native implementations even
when individually uncommon.

### 3. It exposes a legible, enforced feature contract — the moat

The feature vector a model consumes is normally an opaque positional array, and
that opacity is the root of the everyday pains of production ML: training/serving
skew (the serving side reconstructs a contract nobody wrote down, and it drifts),
un-debuggable predictions (decoding an anonymous vector to explain a score), and
un-handoffable pipelines (a new owner has no spec of what the thing consumes or
emits). Pipelines rot the same way and for the same reason — stringly-typed
dict-passing, schemas that live only in someone's head, a column rename that breaks
something three steps downstream with no warning until a garbled number shows up in
production.

The bet is that **strong typing is the antidote**. Every transformer carries
explicit, checkable schemas — Pydantic models for its input row (`__THIS__`), its
learned state (`__STATE__`), and its output — validated when the transformer is
built (columns checked against the model) and again at call time. Because we own
every transform, we can emit a typed, validated output schema that traces each
feature back to the raw input that produced it. Two things make this more than
sklearn's brittle `get_feature_names_out()`: *enforcement* — a schema mismatch
errors at the boundary instead of yielding a silently wrong prediction — and
*provenance* — raw column → feature, which is what makes a prediction debuggable and
a pipeline handoff-able. The result turns a pile of glue code into a set of
transformers whose types document and enforce how they fit together, so a growing
pipeline stays legible instead of collapsing under its own weight.

This contract is **decoupled from how transforms are authored**: SQL is one optional
authoring surface for people who want declarative, inspectable transforms; the typed
contract is delivered either way, including for a pipeline assembled purely through
the compose-in API. And it is the bridge forward — a typed, validated,
provenance-carrying feature contract *is* a proto feature-store contract, which is
why that expansion is a natural next step rather than a pivot.

## How it works today

Two phases, two engines:

- **`fit(table)`** — runs the SQL through DataFusion (full SQL: aggregates, window
  functions, GROUP BY). Extracts the *state* a transform needs (e.g. `MEAN(age)`)
  into a typed Pydantic model, then rewrites the SQL to reference that precomputed
  state (`__STATE__`) plus the raw input row (`__THIS__`) instead of recomputing
  aggregates.
- **`transform(table)`** — batch execution through DataFusion: the rewritten SQL
  (`__THIS__ CROSS JOIN __STATE__`) runs against the batch as `__THIS__` and the
  frozen fit-time state as a one-row `__STATE__` table. Vectorized, columnar.
- **`infer(row)` / `infer_batch(rows)`** — low-latency execution through the small
  native interpreter (`InferFn`, via pyo3), row-at-a-time, against the same frozen
  state. No SQL engine at call time — just expression eval, scans, and joins
  against typed Pydantic rows. Accepts dicts or Pydantic models; returns typed
  output models.

Both paths run the *same* rewritten SQL against the *same* frozen state, so they
return identical values on the normal numeric path. The split just picks the
engine: DataFusion for large batches, the native interpreter for online inference.
This is what makes goal 2 possible: fit pays the cost of a real query engine once;
inference pays only for a lean interpreter walking a plan.

See [SQL_SUPPORT.md](SQL_SUPPORT.md) for the detailed feature-by-feature tracker
(execution engine vs. authoring front-end).

## What we have

**Authoring / fit (`sql_transform/`, DataFusion-backed)**
- `SQLTransform(sql).fit(table)` — DataFusion executes the SQL, extracts window/agg
  state into a synthesized Pydantic `StateModel`.
- Auto-synthesized Pydantic model for `__THIS__` (input row schema) when the user
  doesn't supply one; user can pass their own `this_model`.
- SQL rewrite pass: window-aggregate SQL → plain `__STATE__`/`__THIS__` column-ref
  SQL, ready for the native interpreter.

**Inference (`InferFn`, Rust/pyo3)**
- Typed row tables in, typed output model out — Pydantic on both sides, validated
  at build time (columns checked against the model) and at call time.
- `infer(tables_dict)` and `infer(**kwargs)` (plus merge of both), single-row and
  batch.
- SQL support: SELECT projections, WHERE, INNER/CROSS JOIN, static-table
  `LookupJoin` (row table joined to a preloaded `pa.Table` by key — no per-row
  Python callback).
- Expressions: arithmetic, comparisons, `CAST`, `UPPER/LOWER/TRIM/SUBSTR/CONCAT`,
  `ABS`, `ROUND`, `COALESCE`, `NULLIF`; NULL propagation rules matched to SQL
  semantics; clean `ValueError`s instead of panics (div/mod by zero, bad casts).
- Output type: user-supplied `output_model`, or statically inferred/synthesized
  from the query.

**Not yet supported in the native interpreter**
- `CASE WHEN`
- `GROUP BY` / aggregates at inference time (by design — aggregation only happens
  during `fit`; open question whether inference ever needs it, e.g. for online
  aggregation use cases)
- `LEFT`/`RIGHT`/`OUTER` JOIN (only INNER/CROSS/LookupJoin today)
- `ORDER BY`, `LIMIT`
- Subqueries, CTEs
- sklearn transform functions (`sklearn.standardize(...)` etc. — README documents
  these but they predate the Rust rewrite and are not wired into the current
  interpreter/rewrite pipeline; status needs verification)

## Roadmap

The sequenced path toward these goals — milestones and progress — lives in
[ROADMAP.md](ROADMAP.md); the underlying scoped tasks and open questions live in
[BACKLOG.md](BACKLOG.md). This doc stays focused on what the project is and where
it's headed, not the step-by-step.
