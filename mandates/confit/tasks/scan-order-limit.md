# Compute ORDER BY LIMIT scans

**Done when:** Scan single-row items support measured ORDER BY/LIMIT/OFFSET with exact ties and traps.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Grouped row items already sort after evaluating each item, including rows LIMIT drops. NULLS LAST is default for ASC and DESC; NaN is greatest, -0.0 = 0.0, VARCHAR uses bytes. Different-bit ties at the retained row refuse.

## Wanted

Apply the same row ordering to scan-selected rows. Preserve keyed BIGINT offset+limit wrap: LIMIT NULL OFFSET 1 keeps no row. Record optimizer-on arg_min/arg_max’s n >= 1,000,000 divergence separately.

## Involved

- `packages/confit/src/specializer/plan.rs`: `RowOrder`, `AggPost::order`.
- `packages/confit/src/duckdb/mod.rs`: `materialize_rows`.
- `packages/confit/tests/test_scalar_subquery.py`: ordered row items.

## Out of scope

Aggregate ORDER BY/LIMIT, EXISTS/IN/ANY ordering, ORDER BY ALL, expression/outer order terms, LIMIT m,n or non-whole limits.
