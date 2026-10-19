# Skip the ON clause for an all-NULL hash build

**Done when:** Parity tests distinguish hash joins from AnyJoin when all static keys are NULL.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

DuckDB’s hash LEFT join skips ON when every static key is NULL. Confit evaluates it. AnyJoin instead reads the static rows individually. This was not yet a campaign finding after the 2026-10-06 triage.

## Wanted

Use the measured plan-dependent evaluation behaviour without excusing a live divergence.

## Involved

- `packages/confit/src/specializer/frontend/from.rs`: join binding.
- `packages/confit/src/specializer/lower.rs`: join lowering.
- `packages/confit/tests/test_fuzz_smoke.py`: join parity pins.

## Out of scope

Unrelated query classes and changes to the oracle contract.
