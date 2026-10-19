# Serve remaining row-local subquery scopes

**Done when:** A nested or FROM-position row-local scope slice serves with tested name binding and trap timing.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Subqueries in FROM, JOIN ON and inside another subquery remain named boundaries outside already served derived forms. Outer sides reading joined tables and request-table subqueries also refuse.

## Wanted

Measure and implement one remaining row-local scope at a time. Preserve explicit batch-dependent refusals for the others.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: nested scopes and locality checks.
- `packages/confit/src/specializer/frontend/from.rs`: FROM and JOIN ON binding.

## Out of scope

New filter/many shapes; batch-dependent request-table forms.
