---
id: TASK-77
title: >-
  An integer feature above 2 to the 53 double-rounds and diverges from sklearn
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 70000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-65's threshold rewrite is exact for a DOUBLE feature - it reproduces
`float32(x) <= t` for every double. But `tree_predict` binds an integer feature
as `Ty::I64 => promote_f64(e)`, so the value compared against the rewritten
threshold is `float64(n)`, not `n`. sklearn's `_validate_X_predict` narrows the
int64 array to float32 in ONE step. Above 2**53 those are two roundings versus
one.

```text
n = 9007199791611905          (2**53 + 2**29 + 1)
sklearn  float32(n)          = 9007200328482816
engine   float32(float64(n)) = 9007199254740992     a whole float32 ULP (2**30) apart
engine  = 1.0    sklearn(int64) = 9.0    sklearn(float64) = 1.0
```

The last line is the diagnosis: the engine agrees with sklearn-given-float64 and
disagrees with sklearn-given-int64.

Exposure is narrow but sharply structured. Uniformly random large int64
essentially never hits (0/3000 measured); integers on a power-of-two lattice -
bucketed ids, values scaled by 2**k, epoch nanoseconds (~1.7e18) - hit ~20% of
rows. Deterministic per value, so an affected group is wrong on every request
forever rather than intermittently.

**Reproduced by hand**, not just relayed.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An integer feature matches sklearn bit-exactly, or is refused by name
      when it could exceed 2**53
- [ ] #2 `float64(n)` is exact below 2**53, so the fix must not disturb the
      overwhelmingly common small-integer case
- [ ] #3 The docs state the integer contract explicitly either way
- [ ] #4 Covers both backends
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Options: refuse integer features outright at the `tree_predict` call site
(harsh - small ints are common and correct today); narrow i64 -> f32 -> f64 in
one step at the boundary so the rounding matches sklearn's; or carry the integer
to the compare and do the f32 narrowing on the integer. The second is probably
smallest and keeps the kernel f64-only, which is what made TASK-65's fix clean.

Note this is a REAL gap in TASK-65, whose resolution note claims exactness
without qualifying it to float features. Correct that note as part of this.
<!-- SECTION:NOTES:END -->
