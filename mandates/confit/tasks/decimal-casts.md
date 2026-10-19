# Serve remaining DECIMAL input casts

**Done when:** Measured DOUBLE, VARCHAR and BOOLEAN casts into DECIMAL serve with exact rounding, parsing and traps.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

DECIMAL row columns and arithmetic serve. DOUBLE/VARCHAR/BOOLEAN-to-DECIMAL casts still refuse by name.

## Wanted

Match DuckDB’s double rounding and string parser before admitting these casts.

## Involved

- `packages/confit/src/specializer/frontend/decimal.rs`: DECIMAL coercion.
- `packages/confit/tests/test_decimal_expressions.py`: cast parity.

## Out of scope

Unrelated query classes and changes to the oracle contract.
