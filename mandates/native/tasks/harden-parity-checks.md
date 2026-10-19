# Harden the parity checks

**Done when:** #519 (`claude/native-check-blind-spots`) merges with Linux CI green, after the maintainer approves or rejects its changes to `packages/sql-transform/spec/native/catalog-contract.md`.
**State:** #519 is open, based on master `abc055ce`. On Windows, the native package gives 1,442 passed, 113 skipped, 5 xfailed, and only the 8 failures that clean master also has there. It adopts the four fixtures of #432, which is closed. It found real mismatches on rows where a sum overflows on one side only; under the interim rule of [where a sum overflows](../../../questions/open/where-a-sum-overflows.md), the gate leaves those rows out and strict `xfail` tests keep them. Linux CI passed on `0a9d4bc`, which also derives the overflow-risk sums from each entry (`sum_products`) after RBFSampler's weights were missed. CI's pytest time rose from about 614 s to about 1,100–1,230 s. **Next:** take the maintainer's answer on the contract changes and on the added gate time.

## Now

On master, `check` counts NULL-id rows as compared, accepts overflowed bounds where a long double is a double, and the kernel probe treats −0.0 and 0.0 as one value.

## Wanted

What #519 does: answered-row counting with a floor of 64, a richer row generator, checker overflow as `ParityError`, a width and depth ladder, a zero-sign kernel probe, `check` on the installed DuckDB, `NATIVE_SEED_BASE`, the `PolynomialFeatures` term limit, and linear bound discovery in nested `Pipeline`s.

## Involved

- `packages/sql-transform/sql_transform/native/_check.py`: `check`, `near`
- `packages/sql-transform/sql_transform/native/catalog_test.py`: `_fit_matrix`, `_rows`, `EDGES`, `SEEDS`, `test_an_entry_matches_its_twin`
- `packages/sql-transform/sql_transform/native/function.py`: the kernel probe
- `packages/sql-transform/sql_transform/native/polynomial.py`, `compose.py`
- `packages/sql-transform/spec/native/catalog-contract.md`

## Out of scope

Failing `check` on a zero sign under a bound above 0 ([zero signs under a bound](../../../questions/open/signed-zero-under-bound.md)); passing the overflow rows (held by its question).
