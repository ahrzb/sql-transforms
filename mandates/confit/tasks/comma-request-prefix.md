# Evaluate request-only comma prefixes before probes

**Done when:** Measured request-only WHERE prefixes serve under supported map comma joins.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

A prefix before the key can read only the request row. DuckDB evaluates it over every pair with one answer for all static rows. Empty static input has the empty-static-trap-timing exclusion.

## Wanted

Measure DuckDB 1.5.5, then run the request-only prefix before the probe where permitted. Keep the exclusion explicit.

## Involved

- `packages/confit/src/specializer/frontend/from.rs`: `check_comma_order`.
- `packages/confit/src/specializer/lower.rs`: WHERE/probe ordering.

## Out of scope

New shape=many admission; the postponed proposal to scan all static rows under many.
