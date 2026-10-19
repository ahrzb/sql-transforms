# Serve BLOB values

**Done when:** BLOB columns and output values round-trip with parity tests and a named operation boundary.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

BLOB remains unserved.

## Wanted

Add the value lane without interpreting binary bytes as VARCHAR.

## Involved

- `packages/confit/src/specializer/plan.rs`: value types.
- `packages/confit/src/specializer/lower.rs`: value lowering.

## Out of scope

BLOB function expansion beyond the measured value boundary.
