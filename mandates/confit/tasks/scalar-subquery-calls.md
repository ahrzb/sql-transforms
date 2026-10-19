# Serve calls in scalar subquery items

**Done when:** A measured SQL-function/UDF item slice serves with per-row and trap parity; model calls retain an explicit boundary.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Scalar subquery items calling SQL functions, UDFs or models still refuse. Static WHERE terms holding SQL calls or subqueries also refuse.

## Wanted

Measure static SQL-function items first and implement that slice. Retain separate refusal witnesses for UDF/model and nested-subquery terms until their evaluation order is established.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: item and static-term binding.
- `packages/confit/src/specializer/frontend/calls.rs`: function binding.

## Out of scope

Batch-dependent evaluation and unmeasured model execution.
