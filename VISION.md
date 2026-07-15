# Vision

sql-transform lets you define ML feature transforms as SQL, and run them two ways:

1. **Easy authoring** — write a transform as a SQL expression instead of hand-rolled
   Python/sklearn boilerplate.
2. **Fast inference** — once fitted, apply that same SQL to single rows or batches
   at low latency, without going back through a SQL engine.

Everything else in this doc serves one of those two goals.

## How it works today

Two phases, two engines:

- **`fit(table)`** — runs the SQL through DataFusion (full SQL: aggregates, window
  functions, GROUP BY). Extracts the *state* a transform needs (e.g. `MEAN(age)`)
  into a typed Pydantic model, then rewrites the SQL to reference that precomputed
  state (`__STATE__`) plus the raw input row (`__THIS__`) instead of recomputing
  aggregates.
- **`transform(table)` / `_infer(row)`** — the rewritten SQL runs through a small
  Rust interpreter (`InferFn`, via pyo3), row-at-a-time, against the fitted state.
  No DataFusion, no aggregation engine, at inference time — just expression eval,
  scans, and joins against typed Pydantic rows.

This split is what makes goal 2 possible: fit pays the cost of a real query engine
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

## Open questions / roadmap candidates

- **Decided, in progress:** replace `_rewrite.py`'s DataFusion-plan-walk with a
  sqlglot-based rewrite of the original SQL text. Why: DataFusion's Python
  bindings don't fully expose plan introspection (`Expr::WindowFunction` has no
  `to_variant()` support in the installed version — needed an undocumented
  workaround), and that gap would recur for every new construct the rewrite
  tries to understand. This project already has precedent for the same lesson:
  the Rust `InferFn` interpreter parses SQL with `sqlparser` directly rather
  than via DataFusion's logical planner, for an analogous reason (see
  [[project_phase2_interpreter_pyo3_gotchas]]). DataFusion's role is unchanged
  otherwise — it remains the sole execution engine; sqlglot only changes how
  `fit()` figures out what to rewrite. v1 scope is deliberately narrow: simple
  projection + simple equality joins, not full SQL. Out-of-scope constructs
  raise a clear error at rewrite time rather than failing confusingly in
  `InferFn` later. Track progress in [SQL_SUPPORT.md](SQL_SUPPORT.md)'s Layer 2
  table.
- Bring sklearn-style transforms (scaling, encoding, binning) onto the new
  fit/state/InferFn pipeline, or decide they're out of scope for v0.
  See [[project_goal_and_planning]].
- Decide if/how `CASE WHEN` and outer joins matter for real feature-engineering
  SQL before investing — prioritize by what authoring goal 1 actually needs.
- Codegen / compiled inference path (older README roadmap item) — superseded by
  the Rust `InferFn` interpreter; probably drop from roadmap unless interpreter
  overhead becomes a measured bottleneck.
- No `VISION.md`/`TODO.md` existed before this; this file is the first cut —
  revise as the plan for each open question above gets decided.
