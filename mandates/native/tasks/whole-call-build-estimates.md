# Whole-call build estimates

**Done when:** #520 (`claude/native-build-estimates`, stacked on #519) merges with Linux CI green.
**State:** #520 is open. It has master `abc055ce` and #519 merged in; on Windows the native package gives 1,457 passed and only the 8 failures of clean master there. Linux CI on `0a2ce4a9` fails only the two carried-bound tests of the notice `carried-bound-tests-fail-on-ci`. **Next:** after #519 merges, check that GitHub retargeted #520 to master, and read its CI.

## Now

On master, `MAX_BUILD_S` is copied in five modules, four estimates time only the trial build, and no entry must declare an estimate.

## Wanted

One `MAX_BUILD_S` in `_registry.py`; `BuildEstimate` with named terms; a `builds=` or `uncapped=` declaration on every entry, checked by a test; the four estimates converted to whole warm `to_native`; `benchmarks/native_estimates.py --check` and `--fit`.

## Involved

- `packages/sql-transform/sql_transform/native/_registry.py`: `MAX_BUILD_S`, `BuildEstimate`, `translates`
- the family modules with estimates
- `benchmarks/native_estimates.py`

## Out of scope

Refitting absolute coefficients to a host other than the Linux reference.
