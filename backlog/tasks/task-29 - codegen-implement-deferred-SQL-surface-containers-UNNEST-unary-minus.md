---
id: TASK-29
title: 'codegen: implement deferred SQL surface (containers/UNNEST, unary minus, ||)'
status: To Do
assignee: []
created_date: '2026-07-18 20:14'
labels:
  - codegen
  - feature
  - blocked
dependencies: []
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
- [ ] #3 PRECONDITION: codegen-engine framing decision made (default vs opt-in) before this is actively prioritized
<!-- AC:END -->
