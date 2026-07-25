---
id: TASK-41
title: 'Specializer M-ir: imperative IR — grammar, types, verifier, text format'
status: In Progress
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-07-25 02:58'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze IR design decisions (recorded in design doc §6 update): implicit row cursors (no idx params), strict block-param SSA (cross-block value uses forbidden — no dominance analysis needed), acyclic CFG for v0, terminators emit/skip/trap/jump/brif, store-completeness dataflow at emit.
2. Implement src/specializer/ir/: core types + instructions (mod.rs), verifier (verify.rs), printer (print.rs), parser (parse.rs), shared fixtures (fixtures.rs).
3. Tests: per-instruction fixture programs (parse+verify+round-trip), negative verifier tests per rule, negative parser tests, deterministic seeded fuzz round-trip (hand-rolled xorshift generator, no new deps).
4. Adversarial workflow: fan-out attackers on verifier soundness + round-trip edge cases (float/string literals), plus design-conformance and simplification reviewers; fix confirmed findings.
5. Gate green; stacked PR onto claude/duckdb-native-interpreter-36d016.
<!-- SECTION:PLAN:END -->
