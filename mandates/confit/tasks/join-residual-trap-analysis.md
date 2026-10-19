# Measure static JOIN residual trap analysis

**Done when:** Measured nontrapping DOUBLE residuals serve without changing join evaluation order.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

may_trap treats all arithmetic as trapping. DuckDB DOUBLE division by zero returns ±inf. `(0.5e0 / s0.f) >= 0.75` refuses, as does its BETWEEN form (seed 9047919, formerly served). Switching to can_trap widens admission and needs an evaluation-order measurement.

## Wanted

Measure DuckDB 1.5.5 first. Use more precise trap analysis only where the measured join plan permits it.

## Involved

- `packages/confit/src/specializer/frontend/from.rs`: static residual admission.
- `packages/confit/src/specializer/plan.rs`: `may_trap`, `can_trap`.

## Out of scope

An unmeasured blanket replacement of may_trap.
