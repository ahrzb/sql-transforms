# Serve opaque temporal join keys

**Done when:** One measured temporal key type joins and round-trips without changing its bits.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Temporal columns, casts, literals and functions remain open. A key-only opaque lane can serve joins without implementing temporal arithmetic.

## Wanted

Implement a measured temporal key lane first. Retain named boundaries for temporal expression families until separate tasks define them.

## Involved

- `packages/confit/src/specializer/frontend/from.rs`: key typing.
- `packages/confit/src/specializer/plan.rs`: key representation.

## Out of scope

Implementing all temporal functions or TIMETZ parameter-table authoring.
