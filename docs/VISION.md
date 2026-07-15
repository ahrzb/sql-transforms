# Vision

sql-transform lets you define ML feature transforms as SQL, and run them two ways:

1. **Easy authoring** — write a transform as a SQL expression instead of hand-rolled
   Python/sklearn boilerplate.
2. **Fast inference** — once fitted, apply that same SQL to single rows or batches
   at low latency, without going back through a SQL engine.

Everything else in this doc serves one of those two goals.

## Typed transforms — why it matters

Feature and training pipelines rot into unmaintainable messes: stringly-typed
dict-passing, schemas that live only in someone's head, a column rename that
breaks something three steps downstream with no warning until a garbled number
shows up in production. sql-transform's bet is that **strong typing** is the
antidote.

Every transformer carries explicit, checkable schemas — Pydantic models for its
input row (`__THIS__`), its learned state (`__STATE__`), and its output —
validated when the transformer is built (columns checked against the model) and
again at call time. That makes a transformer a typed, composable unit: you can
see what it consumes and produces without running it, refactor it without fear,
and catch a mismatch at the boundary instead of downstream. The aim is to turn a
pile of glue code into a set of transformers whose types document and enforce how
they fit together — so a growing pipeline stays legible instead of collapsing
under its own weight.

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
  Rust interpreter (`InferFn`, via pyo3), row-at-a-time, against the same frozen
  state. No SQL engine at call time — just expression eval, scans, and joins
  against typed Pydantic rows. Accepts dicts or Pydantic models; returns typed
  output models.

Both paths run the *same* rewritten SQL against the *same* frozen state, so they
return identical values on the normal numeric path. The split just picks the
engine: DataFusion for large batches, the Rust interpreter for online inference.
This is what makes goal 2 possible: fit pays the cost of a real query engine
once; inference pays only for a lean interpreter walking a plan.

See [SQL_SUPPORT.md](SQL_SUPPORT.md) for the detailed feature-by-feature tracker
(execution engine vs. authoring front-end).

## What we have

**Authoring / fit (`sql_transform/`, DataFusion-backed)**
- `SQLTransform(sql).fit(table)` — DataFusion executes the SQL, extracts window/agg
  state into a synthesized Pydantic `StateModel`.
- Auto-synthesized Pydantic model for `__THIS__` (input row schema) when the user
  doesn't supply one; user can pass their own `this_model`.
- SQL rewrite pass: window-aggregate SQL → plain `__STATE__`/`__THIS__` column-ref
  SQL, ready for the Rust interpreter.

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

**Not yet supported in the Rust interpreter**
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

Deferred tasks and open questions live in [BACKLOG.md](BACKLOG.md) — when work is
pushed out of current scope, it lands there. This doc stays focused on what the
project is and how it works today.
