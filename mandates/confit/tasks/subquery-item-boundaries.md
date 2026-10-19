# Serve scalar subquery field reads

**Done when:** Field and element reads from one static row have parity tests for NULL, no-row and multi-row cases.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Items still refuse SQL/UDF/model calls or nested subqueries, outer-query columns, whole table rows, field/element reads, bare SQLNULL, and types not served by min. EXISTS permits only *, constants or table columns; aggregate EXISTS, LIMIT, ORDER BY, DISTINCT and GROUP BY refuse.

## Wanted

Bind field and element reads through the single-row item path. Other item boundaries remain separate tasks.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: `subquery_value`, `exists_subquery`, scalar item binding.
- `packages/confit/tests/test_scalar_subquery.py`: item boundary tests.

## Out of scope

Outer-query columns, whole rows, SQLNULL items, nested calls and batch-dependent forms.
