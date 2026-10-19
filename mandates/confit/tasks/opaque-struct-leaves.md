# Carry opaque struct leaves

**Done when:** One measured opaque leaf type round-trips inside a struct with exact nested validity.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Opaque leaves remain refused; list leaves have their own task.

## Wanted

Extend nested output representation only for a supported opaque value lane.

## Involved

- `packages/confit/tests/test_struct_outputs.py`: nested output validity.
- `packages/confit/src/specializer/plan.rs`: struct/value representation.

## Out of scope

Inventing an opaque lane not supported at the row boundary.
