# Merge downstream cost tests

**Done when:** #516 (`claude/confit-native-cost-tests`) merges with CI green.
**State:** Claimed by the review-fixes session of 2026-10-17. #516 is open and its CI passed. **Next:** merge it.

## Now

Branch `claude/confit-native-cost-tests`, commit `c77c3d7`, adds the cost pytest marker, native Python cost tests, a Rust fsub count test, and a serial cost pass in the gate.

## Wanted

Open its pull request and merge with CI green.

## Involved

- `packages/confit/tests/test_native_costs.py`: native cost tests.
- `packages/confit/src/specializer/tests.rs`: Rust `fsub` count test.
- `packages/sql-transform/sql_transform/native/cost_test.py`: downstream cost checks.
- `scripts/gate.py`: serial cost pass.

## Out of scope

Changing unrelated specializer behaviour.
