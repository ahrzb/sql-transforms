# Serve float32 arithmetic

**Done when:** Float32 columns and operations preserve f32 rounding rather than widening to f64.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

float32 remains unserved and needs its own arithmetic lane.

## Wanted

Carry f32 arithmetic, casts and Arrow outputs with parity tests.

## Involved

- `packages/confit/src/specializer/plan.rs`: numeric types and operations.
- `packages/confit/src/specializer/exec/cranelift.rs`: numeric lane lowering.

## Out of scope

Temporal and BLOB lanes.
