---
id: TASK-65
title: >-
  tree_predict disagrees with sklearn on quantised features (float32 comparison grid)
status: Done
assignee: []
created_date: '2026-08-07 22:40'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - >-
    packages/confit/docs/specs/2026-08-07-confit-tree-ensemble-design.md
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
- [x] #1 `_trees_test.py::test_quantised_features_match_sklearn` passes without
      the xfail marker
- [x] #2 The choice among the three candidate fixes is made deliberately and
      recorded, NOT defaulted to the cheapest
- [x] #3 Whatever is chosen holds for a float64-trained library too, or refuses
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

## Resolution (2026-08-07): candidate 1, and the layout premise was wrong

`_f32_grid_threshold` in `packages/sql-transform/sql_transform/_trees.py` maps
each threshold to the largest double still narrowing to the largest float32 at
or below it. Rounding to float32 is monotone, so `float32(x) <= t` remains a
single cutpoint over the doubles and the rewrite is **exact, not approximate**
— verified by walking f64 ULPs across the boundary of 18012 thresholds
(sklearn-shaped midpoints, exactly-f32 values, subnormals, both overflow ends)
for 0 disagreements in ~4.8M probes.

Exactness is what removes the need for the per-entry flag: a lossless rewrite
belongs to whoever packs, so a LightGBM packer just does not call it, and the
wire format keeps meaning "compare this double" for every library. No header
change, no kernel change, no cast on the row path (AC #3). Cost: the packed
`threshold` column deliberately no longer equals `tree_.threshold`.

Two things only running it revealed:

- **`t = +inf` is a real sklearn threshold** ("every non-missing value goes
  left"), not a sentinel. It passes through untouched. A first revision
  refused out-of-f32-range thresholds and broke all three missing-value tests.
- The overflow ends have no finite neighbour to average against; ±inf stands
  in at ±2**128, where float32 rounding actually tips.

Mutation-checked: a **one-ULP** perturbation of the rewrite still passes the
1500-row end-to-end parity test, and is caught only by the ULP-walking test —
which is why that test walks instead of samples. Identity (no rewrite)
reproduces the original defect at 69/1500.

Post-fix sweep: 4 families × 6 quantisation grids × both backends × 3000 rows
= 144000 rows, 0 mismatches.
<!-- SECTION:NOTES:END -->

## Correction (2026-08-08): the exactness claim above was under-qualified

The rewrite is exact for a **DOUBLE** feature — it answers `float32(x) <= t`
for whatever double it is handed, and the ~4.8M-probe ULP walk stands.

It was not exact for an **INTEGER** feature, and the note above did not say
so. An i64 bound through `promote_f64` reaches the compare as `float64(n)`,
which above `2**53` has already rounded, so the engine computed
`float32(float64(n))` where sklearn computes `float32(n)` — a whole float32
ULP apart. Found and fixed as TASK-77: integer features now convert with the
new `itof.f32` opcode, one rounding, and the claim of exactness holds for
both feature types.

Asserted without testing is how this got written. It is the second of two
such notes from this session that a later finding falsified — see TASK-68's
note, which TASK-73 corrected.
