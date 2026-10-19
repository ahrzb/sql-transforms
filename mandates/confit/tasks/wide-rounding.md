# Serve wide round and trunc digits

**Done when:** UBIGINT/HUGEINT round and trunc with digits match DuckDB at boundaries.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The digits forms of `round` and `trunc` refuse for UBIGINT/HUGEINT.

## Wanted

Implement the exact wide kernels and their boundary traps.

## Involved

- `packages/confit/tests/test_hugeint.py`: wide rounding.
- `packages/confit/src/specializer/frontend/functions.rs`: builtin signatures.

## Out of scope

Unrelated query classes and changes to the oracle contract.
