# Match narrow overflow errors

**Done when:** Narrow multiplication and left-shift parity tests match DuckDB’s error class and text.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Both engines trap, but confit names the Arrow range. DuckDB says `Overflow in multiplication of INT8`, `Overflow in left shift`, or `Left-shift value 8 is out of range`. This remains open after the [first six sharded nights](../reports/2026-10-05-sharded-nightly-triage.md).

## Wanted

Preserve the trap and report the matching operation-specific error.

## Involved

- `packages/confit/tests/test_fuzz_smoke.py`: narrow-width campaign pins.
- `packages/confit/src/specializer/exec/`: arithmetic trap execution.

## Out of scope

Unrelated query classes and changes to the oracle contract.
