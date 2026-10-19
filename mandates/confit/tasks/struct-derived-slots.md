# Carry struct slots through derived tables

**Done when:** Struct-valued columns survive a row-local derived table and CTE with parity tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Phase 1 carries scalar slots only. Wrapped-query tolerances in `fuzz/exclusions.py` allow struct/list columns in derived tables, about 13% of wrapped generated queries.

## Wanted

Carry struct slots through nested scopes and remove only the covered wrap tolerance.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: nested scopes.
- `packages/confit/tests/test_struct_outputs.py`: struct values.
- `packages/confit/fuzz/exclusions.py`: rewrite tolerances.

## Out of scope

List slots and new shape=many subqueries.
