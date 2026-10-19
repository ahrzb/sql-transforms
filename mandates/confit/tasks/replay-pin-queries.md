# Make pin queries replayable

**Done when:** The 816 unreplayable pin queries have typed executable inputs or explicit retained reasons they cannot replay.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

pins-drift.json records 460 untyped input_repr values and 356 prose-only tables.

## Wanted

Recover typed rows and static tables in bounded batches. Keep each converted batch replayable; split remaining batches before a session exceeds one review sitting.

## Involved

- `packages/confit/docs/specs/pins-drift.json`: missing typed inputs.
- `packages/confit/docs/specs/`: pin query records.

## Out of scope

Changing expected oracle semantics.
