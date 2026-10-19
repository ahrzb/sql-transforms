# Serve UHUGEINT

**Done when:** UHUGEINT literals and CAST targets round-trip with checked arithmetic on both backends.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

UHUGEINT is the remaining unserved integer width. Literals beyond HUGEINT and UHUGEINT CAST targets refuse.

## Wanted

Carry the unsigned 128-bit range without changing HUGEINT signed semantics.

## Involved

- `packages/confit/src/specializer/plan.rs`: integer types and literals.
- `packages/confit/tests/test_unsigned.py`: integer lane parity.

## Out of scope

Exact sum/product at decimal128(38,0), which belongs to aggregation.
