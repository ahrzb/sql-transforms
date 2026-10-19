# Batch the guard probes

**Done when:** the input guard's probing calls the twin a number of times that does not grow with features times special values, trap parity holds on the catalog fixtures, and whole warm `to_native` of the widest guarded fixture is measured before and after.
**State:** not started. **Next:** count probe calls for the widest guarded fixtures.

## Now

The guard probes each feature with special values separately (findings 65 and 84 in the [reports](../reports/)).

## Wanted

Batch special values across features, and probe rejected rows separately.

## Involved

- `packages/sql-transform/sql_transform/native/_registry.py`: `_probed`, the input guard

## Out of scope

Changing which inputs trap.
