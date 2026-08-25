---
id: TASK-41
title: 'Specializer M-ir: imperative IR — grammar, types, verifier, text format'
status: Done
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-08-08 03:38'
labels: []
milestone: m-7
dependencies: []
documentation:
  - packages/confit/docs/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the specializer's imperative IR per §6 of packages/confit/docs/specs/2026-07-25-sql-specializer-design.md: SSA over typed scalars with a separate null lane (T? vs T are distinct types; ops on T? only via the .opt instructions), StaticRef handles, no allocation vocabulary. Deliverables are the IR definitions, the verifier, and a round-trippable text format under src/specializer/ir/. This is the diagnostic surface for the whole engine and the loop's machine-checkable gate substrate — the boundary must be airtight before anything targets it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verifier rejects: non-SSA defs, type mismatches, any arithmetic on a T? value not routed through the null-lane ops, unresolvable StaticRef ids, allocating constructs
- [x] #2 Text format round-trips: parse(print(ir)) == ir on every test program, including a property/fuzz round-trip test
- [x] #3 Hand-written IR programs covering every instruction exist as test fixtures
- [x] #4 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze IR design decisions (recorded in design doc §6 update): implicit row cursors (no idx params), strict block-param SSA (cross-block value uses forbidden — no dominance analysis needed), acyclic CFG for v0, terminators emit/skip/trap/jump/brif, store-completeness dataflow at emit.
2. Implement src/specializer/ir/: core types + instructions (mod.rs), verifier (verify.rs), printer (print.rs), parser (parse.rs), shared fixtures (fixtures.rs).
3. Tests: per-instruction fixture programs (parse+verify+round-trip), negative verifier tests per rule, negative parser tests, deterministic seeded fuzz round-trip (hand-rolled xorshift generator, no new deps).
4. Adversarial workflow: fan-out attackers on verifier soundness + round-trip edge cases (float/string literals), plus design-conformance and simplification reviewers; fix confirmed findings.
5. Gate green; stacked PR onto claude/duckdb-native-interpreter-36d016.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in src/specializer/ir/ (mod, verify, print, parse, fixtures, gen, tests). Key decisions vs the original sketch (design doc §6 updated): implicit row cursor, emit/skip/trap terminators carrying the store contract, strict block-param SSA (no dominance analysis), acyclic v0 CFG, presentation-only names (canonical %vN/bN). Adversarial pass (6 agents) found and I fixed: recursive-DFS process abort on deep CFGs (now iterative, 50k blocks OK), empty map static signatures and non-ident fn names verifying but printing unparseably, non-canonical NaN payloads breaking bitwise round-trip equality (NaN-class equality now), unreachable-island starvation masking join store errors (topo indegrees over reachable sources only), doc/grammar drift, dead uses() code. 45 cargo tests; every verifier rule and parser guard has a rejecting test; 300-seed fuzz round-trip.
<!-- SECTION:NOTES:END -->
