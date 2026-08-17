---
id: TASK-77
title: >-
  An integer feature above 2 to the 53 double-rounds and diverges from sklearn
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - parity
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/
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
`packages/confit/tests/known_divergences/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An integer feature matches sklearn bit-exactly, or is refused by name
      when it could exceed 2**53
- [x] #2 `float64(n)` is exact below 2**53, so the fix must not disturb the
      overwhelmingly common small-integer case
- [x] #3 The docs state the integer contract explicitly either way
- [x] #4 Covers both backends
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

## Resolution (2026-08-08): the second option, and the design worry dissolved

The ticket's middle option — narrow i64 -> f32 -> f64 in ONE step at the
boundary — is what landed. `Inst::Itof` gains a `narrow` flag, printed
`itof.f32`; cranelift converts straight to F32 and widens back exactly,
the interpreter does `n as f32 as f64`. An integer `tree_predict` feature
binds through `SKind::IntToFloat32` instead of the ordinary `promote_f64`.

**The stated blocker turned out not to be one.** The handoff said this
"needs an IR op that narrows through float32, and TASK-65's whole design
point was 'no float32 anywhere in the engine' — a real design decision, not
a patch." On inspection that premise is about the TYPE SYSTEM: the engine
computes in exactly i64 / f64 / string / bool. `itof.f32` takes i64 and
yields f64; no lane, column or static is ever f32, exactly as
`ftoi.nearest` is a rounding mode rather than an integer type. Nothing had
to be traded, so no call was needed.

AC #2 is what makes this a fix rather than a trade: below `2**53`,
`float64(n)` is exact, so `float32(float64(n)) == float32(n)` and the new
node is a no-op. Every ordinary small-integer feature is byte-identical to
before, tested over 200 random rows against sklearn.

### One hole only building it revealed

The constant folder collapses `IntToFloat(Lit::I64)` at build time. A
LITERAL integer feature — `struct_pack(n := 9007199791611905)` — would have
kept the double rounding even with the runtime path fixed, because it never
reaches an instruction. `fold` now folds `IntToFloat32` through f32 too, and
that has its own test.

### Mutation-checked

- Reverting to `promote_f64`: 3 tests fail, both backends and the literal
  path.
- Keeping the fix but folding the literal with `i as f64`: exactly one test
  fails, the literal one — so the fold hole is independently guarded rather
  than incidentally covered.

TASK-65's resolution note claimed exactness without qualifying it to double
features; corrected there and in `docs/serving-fitted-models.md`.

## Follow-up (2026-08-08): the grid is DECLARED, not assumed

The fix above put the narrowing in the ENGINE, where — unlike TASK-65's
threshold rewrite, which lives in the packer and a packer can skip — it fired
for every model. That quietly made the wire format sklearn-specific, undoing
the property TASK-65 was careful to buy. Measured on a hand-built model with
un-rewritten thresholds:

```text
threshold = 16777216.5     n = 16777217   (2**24+1, exact as a double)
BIGINT feature -> 10.0     # narrowed to float32(n) = 16777216 -> left
DOUBLE feature -> 20.0     # exact: 16777217 > 16777216.5      -> right
```

The SQL type of the feature changed the answer for a model no sklearn packer
produced — and `docs/serving-fitted-models.md` explicitly anticipates such a
packer, while the kernel docstring claims the layout covers XGBoost and
LightGBM. So the trap was laid for whoever writes the second one.

A `models=` entry now declares `compare_grid: "float32" | "float64"`, and the
frontend picks the conversion from it. `pack_trees` emits `"float32"`.

Three decisions worth keeping:

- **Required, not defaulted.** The packer that would get this wrong is exactly
  the one that never thought about it; a default is the same trap with an
  extra step. It costs every existing caller one key — acceptable pre-1.0, and
  much cheaper now than once model tables are persisted.
- **Per SET, not per model.** `tree_predict('m', id, ..)` takes the id from a
  row, so it is a runtime value, while the conversion is a lowering decision
  made once. A per-model grid could only be honoured with a per-row branch.
  (My first instinct was to put it beside `agg`/`link` in the models table.
  That does not work.)
- **The choice was A vs B and B won on re-examination.** The argument for
  documenting instead ("a second packer is speculative") does not hold: the
  library-agnostic wire format was a property that already existed and was
  already documented, so restoring it is not speculative work. And the
  mitigation A offered — a pin — asserts OUR behaviour and would not stop a
  new packer falling in.

Mutation-checked: making the `F64` arm narrow too fails the IR-text test and
both end-to-end scoring tests, on both backends.
