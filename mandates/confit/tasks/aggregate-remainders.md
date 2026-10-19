# Serve remaining grouped aggregate forms

**Done when:** One measured exact aggregate remainder serves with explicit remaining bounds.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Exact sum/product at decimal128(38,0) belong here rather than the i128 lane. Non-column ORDER BY terms, DISTINCT constants and count(DISTINCT *) currently refuse; DuckDB rejects count(DISTINCT *) in binding. DOUBLE reductions outside whole output columns, nonfinite input and out-of-range bounds remain refused by agreement.

## Wanted

Measure exact product and sum overflow first. Keep the other aggregate syntax refusals executable; do not lift DuckDB binder errors.

## Involved

- `packages/confit/src/duckdb/mod.rs`: aggregate materialization.
- `packages/confit/src/specializer/frontend/subquery.rs`: aggregate binding.
- `packages/confit/tests/test_scalar_subquery.py`: aggregate boundaries.

## Out of scope

Changing the DOUBLE bound or serving DuckDB binder errors.
