---
id: TASK-78
title: >-
  pack_trees ignores feature_names_in_ so a DataFrame-fitted model can bind the wrong columns
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 71000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**DISPUTED by the sweep's own verifiers** - not adjudicated by hand.

`pack_trees` checks `n_features_in_` against the number of names it is given,
but never looks at `feature_names_in_`. A regressor fitted on a DataFrame with
columns `["b", "a"]` and packed with `features=["a", "b"]` passes the count
check and binds every feature to the wrong column - a plausible-looking wrong
answer, the same failure shape `BaggingRegressor` and multi-output are already
refused for.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The claim is adjudicated by hand first
- [x] #2 If real: a mismatch between `feature_names_in_` and the given names is
      refused, naming both orders
- [x] #3 An estimator fitted on a bare ndarray (no `feature_names_in_`) still
      packs - that is the common case and must not regress
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`feature_names_in_` exists only when the estimator was fitted on something with
column names, so the check has to be conditional. Refusing on mismatch is
consistent with how the packer already treats every other silent-wrong-answer
hazard.
<!-- SECTION:NOTES:END -->

## Adjudication (2026-08-08): CONFIRMED, and not subtle

Reproduced by hand. A `RandomForestRegressor` fitted on a DataFrame with
columns `['b', 'a']`, packed as `features=['a', 'b']`:

```text
engine   [-2.7217, -1.5920, -4.9149,  2.2374,  3.0037]
sklearn  [ 0.8369,  0.2171,  3.4565, -0.9225, -2.4544]
```

It builds without complaint, every other check passes, and the numbers look
entirely plausible. That is the failure shape `BaggingRegressor` and
multi-output are already refused for.

## Resolution

`pack_trees` now requires `list(est.feature_names_in_) == features`, guarded by
`getattr(..., None)` so an ndarray-fitted estimator — which has no
`feature_names_in_` at all, and is the common case — packs exactly as before
(tested).

Strict equality rather than a permutation check, because renaming does not need
this knob: `features` is the model's declared feature ORDER, and the call site
already renames by keyword — `struct_pack(b := whatever_column, ...)`. So the
list handed to the packer should always mirror the fitted one, and the error
message says so.
