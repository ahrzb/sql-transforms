---
id: TASK-29
title: 'codegen: implement deferred SQL surface (containers/UNNEST, unary minus, ||)'
status: In Progress
assignee:
  - Ritchie
created_date: '2026-07-18 20:14'
updated_date: '2026-07-19 16:18'
labels:
  - codegen
  - feature
dependencies: []
priority: low
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The codegen engine defers SQL surface it doesn't implement yet -- shows as 16 skips on the codegen backend in the differential suite (native + DataFusion oracle cover and pass all of it). NOT bugs: codegen raises UnsupportedInCodegen and skips LOUDLY. Deferred surface (from the 16 skips): containers -- struct/list column projection, struct field access, struct/list construction (named_struct / array), struct/list comparison, UNNEST (~13 of 16); unary minus on a non-literal (-a); || string-concat operator. Enumerated in the codegen spec 'Deferred' section as intended fast-follows. BLOCKED-ON-FRAMING: gated on the codegen-engine framing decision (maintained/default vs opt-in) -- the pluggable-backend design paused mid-brainstorm + the two-engine ratification (decision-7) that's parked / not-short-term. If codegen becomes default -> real feature-completeness, priority rises; if opt-in -> stays low. Not a bug, not blocking anything today; tests/test_codegen_coverage.py asserts codegen skips ONLY this exact set, guarding against silent drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 each deferred case passes on the codegen backend (the 16-skip set shrinks)
- [ ] #2 tests/test_codegen_coverage.py updated as items land (it currently pins the exact skip set)
- [x] #3 PRECONDITION: codegen-engine framing decision made (default vs opt-in) before this is actively prioritized
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 15:43
---
QA (Fermi) confirmed the exact 16-skip inventory on current master (2026-07-19): unnest() x5, struct field access x3, struct/list-typed column x3, struct/list construction x1, struct/list comparison x1, named_struct() x1, unary-minus-on-non-literal x1, || operator x1. Split insight to unblock the framing gate: the TWO operator-level defers (unary minus on a non-literal, ||) are cheap scalar ops that are NOT truly framing-gated — same category as CASE (TASK-30), which shipped on codegen regardless of the default-vs-opt-in question. The other 14 (struct/list/UNNEST container surface) are the real framing-gated core. Not spinning a separate ticket yet (no demand for the 2 ops today; native covers them), but documented here so the easy wins can be pulled ahead of the framing decision if codegen-default is chosen or demand appears.
---

author: Iris (PM)
created: 2026-07-19 16:13
---
decision-7 RULED (2026-07-19): native is default, codegen is opt-in for now. So AC #3's precondition ('framing decision made') is satisfied — decided → opt-in → this stays LOW, a fast-follow only for someone who opts into codegen, not milestone work. Reclassified from 'blocked on an open question' to 'deprioritized by decision' (dropped the 'blocked' label). Revisit only if codegen gets promoted toward default.
---

author: Iris (PM)
created: 2026-07-19 16:18
---
Dispatched to Ritchie (2026-07-19). AmirHossein's call: Ritchie takes the codegen lane (TASK-29 then TASK-34), Wren takes the rest (m-1 spine). Precondition AC #3 checked — decision-7 is ruled (opt-in), so this is cleared to work. Goal is codegen feature-completeness for opt-in users; still Low importance vs the native default path, but actively worked now. Suggested order within the ticket: the 2 cheap operator defers (unary-minus-on-non-literal, ||) first, then the container surface (struct/list/named_struct/UNNEST).
---
<!-- COMMENTS:END -->
