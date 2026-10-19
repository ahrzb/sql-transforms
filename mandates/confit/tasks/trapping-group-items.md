# Evaluate trapping grouped items per request

**Done when:** An item that traps in one static group can serve unmatched requests and trap matching ones.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Group post-items currently run at build (`AggPost`, `post_values`) and refuse any group trap or no-row trap. DuckDB evaluates once per distinct outer value, even under CASE. In seeds 30,100,000..32,099,999, 19 no-row traps refused, including seven where DuckDB served.

## Wanted

Route eligible groups of at most 100 rows through the scan. Evaluate the post-item at the subquery operator, including over no rows.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: `agg_post`, `scan_serves`, `scan_value`.
- `packages/confit/src/specializer/plan.rs`: `Fold::post`, `Fold::traps`, `JoinSpec::after_where`.
- `packages/confit/src/specializer/lower.rs`: `emit_fold`.
- `packages/confit/tests/test_scalar_subquery.py`: post-item traps.

## Out of scope

Increasing scan bounds or silently moving a SELECT subquery before WHERE.
