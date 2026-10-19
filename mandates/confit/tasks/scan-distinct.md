# Compute DISTINCT scan aggregates

**Done when:** Scan count, sum, avg and bit_xor over DISTINCT values match DuckDB equality and NULL handling.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

n1045 refused 41 DISTINCT count/sum/avg items and two DISTINCT bit_xor items. Grouped DISTINCT already treats -0.0 and 0.0 as equal and all NaNs as one class.

## Wanted

Sort each key’s rows by value at build. Skip a kept value equal to the last kept value, not merely the preceding input row.

## Involved

- `packages/confit/src/duckdb/mod.rs`: `AggIn::class`, scan rows.
- `packages/confit/src/specializer/lower.rs`: fold lanes.
- `packages/confit/tests/test_scalar_subquery.py`: DISTINCT scans.

## Out of scope

Changing the 100-row scan bound.
