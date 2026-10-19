# Admit typed takes

**Done when:** Typed takes retain input/output type and index semantics in fit and serving tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Typed takes remain authoring admission headroom.

## Wanted

Implement the measured take form under the existing authoring contract.

## Involved

- `packages/sql-transform/`: typed take translation.
- `packages/sql-transform/spec/README.md`: authoring boundary.

## Out of scope

New composition laws.
