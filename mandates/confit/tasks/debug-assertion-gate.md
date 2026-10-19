# Run pytest with debug assertions in the gate

**Done when:** The pull-request gate executes the confit pytest suite with Rust debug assertions enabled.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Release extensions omit lowering assertions. Nightly enables CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS; the PR gate does not. #471 broke a materializer assertion, causing 137 PANIC verdicts in nightly shard 0 on 2026-10-15; #478 then prevented metamorphic pool results. Historical cost: about 6 min build in its own target directory and 3 min tests on four workers.

## Wanted

Add an isolated assertion-enabled build and pytest pass. Until then, lowering/materializer changes need that pass before merge.

## Involved

- `scripts/gate.py`: build/test passes.
- `.github/workflows/nightly-campaign.yml`: assertion-enabled build.
- `packages/confit/tests/test_scalar_subquery.py`: invariant witness.

## Out of scope

Re-triaging the already fixed nightly classes.
