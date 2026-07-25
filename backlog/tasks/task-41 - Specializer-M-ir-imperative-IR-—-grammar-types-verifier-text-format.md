---
id: TASK-41
title: 'Specializer M-ir: imperative IR — grammar, types, verifier, text format'
status: To Do
assignee: []
created_date: '2026-07-25 02:31'
labels: []
milestone: m-7
dependencies: []
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the specializer's imperative IR per §6 of docs/superpowers/specs/2026-07-25-sql-specializer-design.md: SSA over typed scalars with a separate null lane (T? vs T are distinct types; ops on T? only via the .opt instructions), StaticRef handles, no allocation vocabulary. Deliverables are the IR definitions, the verifier, and a round-trippable text format under src/specializer/ir/. This is the diagnostic surface for the whole engine and the loop's machine-checkable gate substrate — the boundary must be airtight before anything targets it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Verifier rejects: non-SSA defs, type mismatches, any arithmetic on a T? value not routed through the null-lane ops, unresolvable StaticRef ids, allocating constructs
- [ ] #2 Text format round-trips: parse(print(ir)) == ir on every test program, including a property/fuzz round-trip test
- [ ] #3 Hand-written IR programs covering every instruction exist as test fixtures
- [ ] #4 mise gate-specializer green
<!-- AC:END -->
