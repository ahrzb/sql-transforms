# Serve unsigned negation and wide shifts

**Done when:** Unsigned negation and unsigned/HUGEINT shifts match DuckDB on both backends.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

UTINYINT..UINTEGER use i64; HUGEINT and UBIGINT use i128. Unsigned unary minus wraps on DuckDB but refuses in confit. Unsigned/HUGEINT shifts also refuse.

## Wanted

Implement the measured wrap, range and trap behaviour for those operators.

## Involved

- `packages/confit/tests/test_unsigned.py`: unsigned operations.
- `packages/confit/tests/test_hugeint.py`: wide operations.
- `packages/confit/src/specializer/lower.rs`: integer lowering.

## Out of scope

Unrelated query classes and changes to the oracle contract.
