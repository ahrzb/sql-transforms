# Serve temporal values and casts

**Done when:** One measured temporal value type supports columns, literals and casts with exact output and trap tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Temporal columns, casts, literals and functions remain unserved beyond any separate opaque-key work.

## Wanted

Carry the first measured temporal value type. Keep unsupported temporal types named and split their remaining implementation by type.

## Involved

- `packages/confit/src/specializer/plan.rs`: value types.
- `packages/confit/src/specializer/frontend/typing.rs`: casts.

## Out of scope

Temporal functions and BLOB.
