# Admit frozen static joins in composition

**Done when:** A frozen static-table composition example fits and serves under the agreed contract.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Static-table joins with frozen composition remain admission headroom.

## Wanted

Use the [authoring contract](../../../packages/sql-transform/spec/README.md) to implement the measured supported slice; ask before a rule change.

## Involved

- `packages/sql-transform/spec/composition.md`: frozen composition.
- `packages/sql-transform/`: fit/composition implementation.

## Out of scope

New composition laws.
