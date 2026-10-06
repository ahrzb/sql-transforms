# SQL Support Tracker

Two layers, two different SQL surfaces. Keep both current as capability lands —
grep the "Source" column's file when in doubt, this doc drifts.

## Layer 1 — Execution engine (`InferFn`, native interpreter, `src/*.rs`)

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

## Layer 2 — Transform authoring (`sql-transform`)

The [authoring contract](../packages/sql-transform/docs/contract.md) defines
the current SQL surface. It covers authored `(F, T) -> R` transforms, fit,
composition, supported decorrelation and bounded window marginalization.

`SQLTransform` supports general relation computations. `SQLProjection` adds
the row-local rule. Confit checks serving admission separately.

## Reading this tracker

Layer 1 describes the execution engine. Use the authoring contract for
Layer 2, rather than inferring authoring support from an engine feature.
A batch projection can succeed while Confit refuses its serving SQL.
