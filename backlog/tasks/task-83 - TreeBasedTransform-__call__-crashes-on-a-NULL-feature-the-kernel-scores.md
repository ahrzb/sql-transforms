---
id: TASK-83
title: >-
  TreeBasedTransform.__call__ crashes on a NULL feature the kernel scores via the missing branch
status: To Do
assignee: []
created_date: '2026-08-11 13:00'
labels:
  - bug
  - parity
  - trees
  - fuzz
dependencies: []
documentation:
  - packages/sql-transform/sql_transform/_trees.py
type: bug
ordinal: 76000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The one-declaration-three-bindings rule (PR #95) breaks on NULL features for
estimators whose `predict` rejects NaN:

```text
trees = TreeBasedTransform(instances={0: GradientBoostingRegressor(...)}, ...)

confit kernel   trees(0, NULL) -> scores via missing_left    (rows served)
trees(0, None)  ValueError: Input X contains NaN.
                GradientBoostingRegressor does not accept missing values ...
```

`_as_grid_value` maps `None -> np.nan` and hands it to `est.predict`.
`DecisionTreeRegressor` (>= 1.3) accepts NaN at predict; `GradientBoosting`
and `RandomForest` raise. So the Python binding — the one DuckDB calls when
the object is registered as a UDF, and the one the marginalizer layer calls —
crashes on inputs the native kernel serves. Same declaration, different
answers, which is exactly what that PR's "one declaration, read the same way
by all three bindings" commit was for.

Found by the fuzzer 2026-08-11 (seed 3112, `SELECT trees(0, NULL)` — DuckDB
leg died with "Python exception" while confit returned rows). Reproduced by
hand with a minimal GBR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `t(iid, None)` returns the same value the kernel serves for a NULL
      feature, for dtr, rf and gbr instances alike
- [ ] #2 A test drives all three bindings (kernel, `__call__`, DuckDB-
      registered) over a NULL feature and asserts three-way agreement
- [ ] #3 sklearn remains the ground truth where sklearn CAN answer; where it
      cannot (NaN-rejecting predict), the packed missing_left traversal is the
      documented reference and `__call__` follows it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`__call__` cannot delegate NULL rows to `est.predict` for NaN-rejecting
estimators; it has to walk the packed nodes (which `_pack` already built,
with `missing_left` decided per node) or mask NULL rows and traverse only
those. Walking the packed table is the honest option — it is the same data
the kernel reads, so the two cannot drift.
<!-- SECTION:NOTES:END -->
