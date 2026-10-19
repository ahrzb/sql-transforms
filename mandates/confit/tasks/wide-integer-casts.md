# Serve remaining wide numeric casts

**Done when:** DOUBLE-to-small-unsigned casts and HUGEINT-to-DECIMAL join coercions have exact parity tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

DOUBLE-to-UTINYINT..UINTEGER refuses: DuckDB checks range before rounding, so 255.5 wraps. A HUGEINT key against a DECIMAL build key also refuses.

## Wanted

Model each cast’s measured checks without routing through an inexact f64 integer representation.

## Involved

- `packages/confit/src/specializer/frontend/typing.rs`: cast binding.
- `packages/confit/tests/test_unsigned.py`: rounded casts.
- `packages/confit/tests/test_hugeint.py`: key coercion.

## Out of scope

Unrelated query classes and changes to the oracle contract.
