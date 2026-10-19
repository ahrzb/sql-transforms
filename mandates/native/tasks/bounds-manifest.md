# Bounds manifest

**Done when:** #517 (`claude/native-bounds-manifest`) merges after #519 and #520, with `bounds.md` regenerated on that head; then #518 (`claude/native-ci-bounds-and-nightly`, the workflow files) merges with CI green.
**State:** #517 and #518 are open as drafts. #517 has master `abc055ce` merged in, and its guard reads `proposals/closed/` as the covering folder. #518 fails CI until #517 merges, because `scripts/check_bounds.py` is missing on master. **Next:** after #520 merges, merge master into #517, regenerate `bounds.md`, and mark both ready.

## Now

On master, bounds are code literals; a widening looks like any other edit.

## Wanted

A generated manifest whose widening fails CI unless the change touches `proposals/closed/`, and a nightly that runs 200 rotating seeds, the bounded families under five BLAS setups, and `native_estimates.py --check`.

## Involved

- `packages/sql-transform/sql_transform/native/coverage.py`
- `scripts/check_bounds.py`
- `.github/workflows/ci.yml`, `.github/workflows/nightly-native.yml`

## Out of scope

How ulp bounds are chosen ([evidence for an ulp bound](../../../questions/open/ulp-bound-evidence.md)).
