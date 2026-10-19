# Serve remaining scalar subquery output forms

**Done when:** Whole-row and bare-NULL items have exact schema and miss semantics, with named unsupported type boundaries.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Whole table rows (unless a same-named column resolves first), bare SQLNULL items and types not served by min refuse.

## Wanted

Implement measured whole-row and NULL item representation without pretending min supports an opaque value.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: scalar item binding.
- `packages/confit/src/specializer/plan.rs`: `AggPost::row`.

## Out of scope

Multi-row values or batch-dependent request-table subqueries.
