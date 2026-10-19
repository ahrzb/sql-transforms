# Serve reusable row-local CTEs

**Done when:** A CTE referenced twice serves with correct scope and row-local evaluation, or retains a measured named batch-dependent refusal.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Other scalar/correlated/IN subqueries, a CTE referenced twice and set operations remain refused outside the served forms. FROM and JOIN ON subqueries, nested subqueries, request-table subqueries and an outer side reading a joined table remain named boundaries.

## Wanted

Measure a minimal CTE referenced twice. Implement the row-local form without duplicating observable trapping evaluation.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: scope and locality checks.
- `packages/confit/tests/test_scalar_subquery.py`: refusal witnesses.

## Out of scope

Set operations, nested subqueries, FROM/JOIN ON extensions and batch-dependent forms.
