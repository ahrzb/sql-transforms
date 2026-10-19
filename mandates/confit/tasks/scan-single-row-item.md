# Evaluate single-row scan items

**Done when:** A non-aggregate scan item returns NULL over no rows and traps per request over multiple retained rows.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Grouped scalar items use SINGLE-join semantics and refuse multi-row groups at build. DuckDB traps for multiple matches even under an untaken CASE. In seeds 40,100,000..44,099,999, 952 refusals named multiple rows: DuckDB trapped in 839 and served in 112 when no request matched.

## Wanted

Count the rows retained by each request before returning its item. Preserve item evaluation before LIMIT.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: `Binder::scan_value`.
- `packages/confit/src/specializer/plan.rs`: `AggPost::row`.
- `packages/confit/src/duckdb/mod.rs`: `more_rows`, `materialize_rows`.
- `packages/confit/tests/test_scalar_subquery.py`: SINGLE semantics.

## Out of scope

Batch-dependent request-table subqueries.
