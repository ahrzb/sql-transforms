# Bound the uncapped entries

**Done when:** `PolynomialFeatures`, the encoders (`OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`), `KBinsDiscretizer` and compositions each declare a build estimate or a structural limit that keeps whole warm `to_native` under `MAX_BUILD_S`, with a check.
**Blocked by:** [whole-call build estimates](whole-call-build-estimates.md), which adds the `builds=` and `uncapped=` declarations.
**State:** not started. **Next:** time the encoders at 64, 128 and 256 features.

## Now

These entries declare `uncapped` reasons; nothing bounds their build time. The three encoders take 33–37 s in `to_native` at 256 features (measured 2026-10-17 on a Windows host, release build of master `40f6ab17`). Compositions have no aggregate build limit.

## Wanted

An estimate or a structural refusal for each, and an aggregate limit for a composition's whole build.

## Involved

- `packages/sql-transform/sql_transform/native/encode.py`, `polynomial.py`, `discretize.py`, `compose.py`
- `benchmarks/native_estimates.py`

## Out of scope

Changing the 7 s budget ([is the build budget right](../../../questions/open/build-budget.md)).
