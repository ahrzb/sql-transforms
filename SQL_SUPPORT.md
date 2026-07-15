# SQL Support Tracker

Two layers, two different SQL surfaces. Keep both current as capability lands —
grep the "Source" column's file when in doubt, this doc drifts.

## Layer 1 — Execution engine (`InferFn`, Rust interpreter, `src/*.rs`)

Runs at inference time (`transform()`/`_infer()`), row-at-a-time, no DataFusion.

| Feature | Status | Source |
|---|---|---|
| SELECT projection, aliases | ✅ | `plan.rs` |
| WHERE | ✅ | `plan.rs:99` |
| INNER JOIN / CROSS JOIN | ✅ | `plan.rs` `RelNode::Join`/`CrossJoin` |
| Static-table lookup join (row ⋈ preloaded `pa.Table`) | ✅ | `plan.rs` `RelNode::LookupJoin`, `lookup.rs` |
| Arithmetic `+ - * /` `%` | ✅ | `expr_build.rs` |
| Comparisons `= <> < > <= >=` | ✅ | `expr_build.rs` |
| `AND` / `OR` / `NOT` | ✅ | `expr_build.rs` |
| `CAST` (INT/FLOAT/STR/BOOL) | ✅ | `expr.rs` `eval_cast` |
| `UPPER LOWER TRIM SUBSTR/SUBSTRING CONCAT` | ✅ | `expr.rs` `eval_builtin` |
| `ABS ROUND` | ✅ | `expr.rs` |
| `COALESCE NULLIF` | ✅ | `expr.rs` |
| NULL propagation (SQL semantics) | ✅ | `expr.rs`, `expr_build.rs` |
| Clean errors (div/mod by zero, bad cast, missing attr) | ✅ | `plan.rs`/`expr.rs` `InterpError` |
| `CASE WHEN` | ❌ | not implemented |
| `LIKE` | ❌ | not implemented |
| `IN (...)` / `IN` subquery | ❌ | not implemented |
| `BETWEEN` | ❌ | not implemented |
| `IS NULL` / `IS NOT NULL` | ❌ | not implemented |
| `LEFT`/`RIGHT`/`FULL OUTER` JOIN | ❌ | only INNER/CROSS/LookupJoin |
| `GROUP BY` / aggregates | ❌ by design | aggregation only happens in `fit()`, not at inference |
| `ORDER BY` / `LIMIT` | ❌ | not implemented |
| Subqueries / CTEs | ❌ | not implemented |
| Window functions | ❌ | fit-phase only, not in InferFn |

## Layer 2 — Transformer authoring (`SQLTransform.fit()`, DataFusion + rewrite, `sql_transform/_state.py` + `_rewrite.py`)

This is the SQL you actually write as a user. `fit()` runs it through full
DataFusion, then `_rewrite.py` converts the top-level projection list into the
narrower Layer-1 SQL. **The rewrite step is the bottleneck** — it currently only
understands plain columns and binary-op arithmetic in the SELECT list, so most of
what DataFusion itself supports at `fit()` time can't survive into `transform()`.

**Parser swap in progress:** `_rewrite.py` currently walks DataFusion's Python
*logical plan* objects, not the SQL text — this hit a real wall building
window-aggregate detection (`Expr::WindowFunction` isn't wired into
`to_variant()` in the installed `datafusion` version; had to fall back to an
undocumented `node.method(raw_expr)` calling convention). That's a DataFusion
Python-*binding* completeness gap, not a query-planning one — expect it to recur
for any new construct (`WHERE`, `CASE`, functions) the rewrite tries to walk.
Decision: move the rewrite/analysis step onto **sqlglot** parsing the *original*
SQL text directly (stable, documented AST, built for exactly this job).
DataFusion's role doesn't shrink — it stays the sole *execution* engine (`fit()`
still runs real queries through it to compute aggregate values); sqlglot only
replaces how `_state.py`/`_rewrite.py` figure out *what* to rewrite. v1 target
scope for the sqlglot rewrite: simple projection + simple equality joins (see
rows below) — not "arbitrary SQL." Out-of-scope constructs must raise a clear
`ValueError` from the rewrite step itself, not silently pass through to a
confusing `InferFn` failure later.

| Feature | Status | Source |
|---|---|---|
| Window aggregate, no `PARTITION BY`/`ORDER BY` (e.g. `AVG(age) OVER ()`) | ✅ | `_state.py` |
| Plain column reference in SELECT | ✅ | `_rewrite.py` `_column_to_sql` |
| Binary-op arithmetic in SELECT (e.g. `age - AVG(age) OVER ()`) | ✅ | `_rewrite.py` `_expr_to_sql` |
| Required alias on every SELECT item | ✅ (enforced) | `_rewrite.py` |
| `PARTITION BY` window aggregates | ❌ explicitly rejected | `_state.py` raises `NotImplementedError` |
| `ORDER BY` window aggregates | ❌ explicitly rejected | `_state.py` raises `NotImplementedError` |
| Simple equality JOIN in authored SQL (row⋈row, row⋈static) | 🔜 v1 target of the sqlglot rewrite | not started |
| `WHERE` in the authored SQL | ❌ deferred past v1 | rewrite only walks the projection list |
| Function calls in SELECT (`UPPER(...)`, `CAST(...)`, etc.) — even though Layer 1 supports them | ❌ deferred past v1 | `_expr_to_sql` only handles `Column`/`BinaryExpr`/`Alias` |
| `CASE WHEN` in authored SQL | ❌ deferred past v1 | same gap as above, and Layer 1 doesn't support it either |
| Non-equality or outer JOIN in authored SQL | ❌ deferred past v1 | Layer 1 itself only supports inner-equality joins anyway |
| `GROUP BY` (non-window aggregation) | ❌ | `_state.py` only recognizes window-agg display syntax |
| `sklearn.*` transforms (`standardize`, `minmax_scale`, `onehot_encode`, etc.) | ❌ not implemented | README advertises these; no `sklearn` reference anywhere in `sql_transform/` or `src/` as of 2026-07-15 — treat README's sklearn section as aspirational, not current |

## Reading this table

Layer 1 (the interpreter) is currently *more* capable than Layer 2 (the authoring
front-end) exposes — e.g. `WHERE`, joins, and string functions all work in
`InferFn` today but can't be reached by writing `SQLTransform(sql)` because
`_rewrite.py` doesn't pass them through. Closing that gap (making the rewrite
pass handle more of what DataFusion accepts) is probably higher leverage than
adding new Layer-1 features, since goal 1 (easy authoring) is bottlenecked there.
The sqlglot rewrite (see above) is the first step of that, scoped deliberately
narrow (projection + equality joins) rather than chasing full parity in one pass.
See [[project_goal_and_planning]] in memory for the two project goals this maps to.
