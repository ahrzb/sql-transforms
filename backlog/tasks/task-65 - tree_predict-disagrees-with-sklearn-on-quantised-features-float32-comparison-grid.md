---
id: TASK-65
title: >-
  tree_predict disagrees with sklearn on quantised features (float32 comparison grid)
status: To Do
assignee: []
created_date: '2026-08-07 22:40'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - >-
    docs/superpowers/specs/2026-08-07-confit-tree-ensemble-design.md
type: bug
ordinal: 58000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
sklearn narrows `X` to float32 (`DTYPE`) before traversal and keeps
`tree_.threshold` in float64, so it evaluates `float32(x) <= threshold`. The
kernel compares the raw f64 (`TreeEnsemble::leaf_value`). Every `x` whose
float32 rounding crosses a threshold takes the other branch.

This is not a rounding detail. The threshold is *exactly* the float32 midpoint
of the two neighbouring training values, which proves the splitter itself only
ever saw float32 — the narrowing is baked into where the split SITS. Our f64
compare is therefore not a more precise evaluation of the same model; it is a
different model.

```text
threshold        = 0.15000000223517418   # == mean(f32(0.1), f32(0.2))
float64 midpoint = 0.15000000000000002   # not this
x = 0.15   ->  kernel -1.0,  sklearn +1.0
```

Measured 2026-08-07: 2-decimal price grid, `RandomForestRegressor(30)`,
**157/3000 rows differ**, max delta 0.43 against a target range of −7.9..19.4 —
a whole-leaf jump. Narrowing the inputs to float32 first gives 0/3000, which
pins the cause exactly.

Continuous float64 draws hide it (the mismatch band is ~1 float32 ULP wide),
which is why the original gate was green — it passed by luck, not immunity.
Quantised features — prices, percentages, any decimal grid — hit it constantly.

Found by the 2026-08-07 adversarial sweep; independently reproduced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `_trees_test.py::test_quantised_features_match_sklearn` passes without
      the xfail marker
- [ ] #2 The choice among the three candidate fixes is made deliberately and
      recorded, NOT defaulted to the cheapest
- [ ] #3 Whatever is chosen holds for a float64-trained library too, or refuses
      it by name
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three candidates, none obviously right (spec has the full write-up):

1. **Rewrite thresholds at pack time** — map each threshold to the f64 value
   reproducing the f32 comparison. Build-time, zero per-row cost, kernel stays
   library-agnostic. Correct only for float32-trained models.
2. **Narrow in the kernel** — cast each feature to f32 before comparing.
   Correct for sklearn, wrong for float64-trained libraries, adds a cast to the
   hot path.
3. **Declare an f32 contract at the boundary** — refuse or narrow explicitly.

The reason this cannot be settled by picking the cheapest: the same two-table
layout is documented as the path for other libraries, and **XGBoost is float32
while LightGBM is float64**. The narrowing is a property of the ENTRY, not of
the kernel or the packer — which points at a per-entry flag in the model header
rather than any of the three as written. That is a layout change, hence a
decision rather than a patch.
<!-- SECTION:NOTES:END -->
