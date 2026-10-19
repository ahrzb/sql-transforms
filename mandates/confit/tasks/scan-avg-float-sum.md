# Compute scan avg and DOUBLE sum

**Done when:** Scan avg and DOUBLE sum have exact integer/decimal tests and bounded float comparisons.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The n1045 scan campaign refused 28 avg and 15 DOUBLE sum items. A scan uses at most 100 rows per request and currently runs only on the interpreter. Existing grouped avg follows DuckDB: double for SMALLINT/DECIMAL(<=4,s), x87 long double otherwise, x86_64 Linux only. Σ|v| past i128 refuses.

## Wanted

Reuse measured avg finalization and the [DOUBLE reduction bound](../../../proposals/closed/double-sum-bound.md). DOUBLE sum starts at +0.0 in table order; admit only whole output columns. Refuse NaN/infinity and sum|v| + B out of range.

## Involved

- `packages/confit/src/duckdb/avg.rs`: avg finalization.
- `packages/confit/src/duckdb/mod.rs`: `Acc::Float`.
- `packages/confit/src/specializer/lower.rs`: `emit_fold`, `many_fold`.
- `packages/confit/src/specializer/mod.rs`: `float_bounds`.
- `packages/confit/tests/test_scalar_subquery.py`: scan aggregates.

## Out of scope

Non-whole-output DOUBLE reductions or a new floating-point bound.
