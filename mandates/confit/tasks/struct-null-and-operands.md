# Serve struct NULL tests and operands

**Done when:** Measured struct NULL tests, equality and coalesce preserve field traps and whole-value validity.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

IS NULL over struct_pack or struct CASE refuses because DuckDB evaluates fields first. Whole-struct operands such as `s = s` and `coalesce(s, t)` also refuse.

## Wanted

Preserve evaluation of trapping fields before the NULL test; implement measured whole-value semantics.

## Involved

- `packages/confit/tests/test_struct_outputs.py`: struct operand parity.
- `packages/confit/src/specializer/frontend/expr.rs`: NULL tests and operators.

## Out of scope

Unrelated query classes and changes to the oracle contract.
