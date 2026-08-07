---
id: TASK-66
title: >-
  Block-splitting expression inside a tree_predict argument lowers to an ill-formed program
status: To Do
assignee: []
created_date: '2026-08-07 22:40'
labels:
  - bug
  - lowering
dependencies: []
documentation:
  - >-
    docs/superpowers/specs/2026-08-07-confit-tree-ensemble-design.md
type: bug
ordinal: 59000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `SKind::TreePredict` lowering emits the id lane, then each feature lane,
then `Inst::Predict` — but it never pushes those lanes onto the `live` stack.
Every other multi-operand arm (`Arith`, `Cmp`, `Concat`, `Trim`, `Substr`,
`Str2/3`, `emit_extern`, `emit_probe`) does `live.push((la, ty))` before
evaluating the next operand precisely so the earlier register survives a block
split.

So the moment any feature lowers through a construct that splits the CFG, the
`Predict` instruction lands in a later block while `id` and the
already-emitted features still live in the old one. `ir::verify` catches it and
`prepare()` returns `PrepareError::Internal`.

Measured 2026-08-07 (nullable feature column; a non-nullable one folds away):

```text
COALESCE(x, 0.0)                        RAISED  internal specializer bug: …
CASE WHEN x > 0 THEN x ELSE 0.0 END     RAISED  internal specializer bug: …
NULLIF(x, 1.0)                          RAISED  internal specializer bug: …
CAST(s AS DOUBLE)                       RAISED  internal specializer bug: …
COALESCE(x, 0.0)  -- outside the call   OK
```

Message: `b1[0]: model id %v0 is not visible here: values cross blocks only as
branch args to block params`.

Build-time and loud, so **not a wrong answer** — but it rejects ordinary SQL
with an internal-bug message rather than serving it. Users writing a
`COALESCE` around a nullable feature will hit this immediately.

Found by the 2026-08-07 adversarial sweep (four independent agents); confirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `COALESCE`, `CASE`, `NULLIF` and a guarded `CAST` all build and serve
      correctly as `tree_predict` feature expressions
- [ ] #2 The same holds for the id expression, not just the features
- [ ] #3 A test covers a block split in the FIRST feature and in a LATER one —
      the stranding differs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Almost certainly a one-arm fix: push each emitted lane onto `live` as the other
multi-operand arms do. Worth checking whether the id needs the same treatment
separately, and whether `shape='many'` with a CASE/COALESCE over a joined
static column (reported as a separate symptom by the sweep) is the same root
cause or a distinct one.
<!-- SECTION:NOTES:END -->
