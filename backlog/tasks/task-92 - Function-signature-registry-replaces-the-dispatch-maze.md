---
id: TASK-92
title: >-
  Function signature registry replaces the dispatch maze
status: Done
assignee: []
created_date: '2026-08-13 21:30'
labels:
  - refactor
  - frontend
dependencies: []
documentation:
  - docs/superpowers/specs/2026-08-13-function-signature-registry-design.md
type: refactor
ordinal: 84000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Binder::function` is a ~1200-line `match name.as_str()` where every arm
hand-rolls arity, per-argument type checks, bare-NULL binding, and result
typing. m-8 phase 2 measured the cost: one typing change (integer widths)
touched ~12 unrelated arms, and TASK-82/86/79 were all signature facts
hand-coded inconsistently. Dispatch runs once at prepare — serve time is
compiled IR — so the maze buys nothing.

Split WHAT a function accepts/returns (one declarative Sig table: params
as ArgTy, result as a Ret rule, bare-NULL binding = the param's declared
type) from HOW its node is built (small per-arm builders keeping quirks:
budget refusal, desugars, lazy CASE construction). Design in the spec.

Prerequisite of the m-8 phase-2 PR: this lands first from master,
behavior-identical; task-79 rebases on top and re-expresses its function
typing as table rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Binder::function resolves arity/arg types/NULL binding/result type
      through the signature table; per-arm ad-hoc type checks are gone
- [ ] #2 Behavior-identical on its base: full pytest green with ZERO test
      edits, cargo test green, fuzz smoke + 5k campaign show no new verdict
      classes vs the base commit
- [ ] #3 A totality unit test: every builtin name has exactly one table row
      and one builder
- [ ] #4 task-79's function-width typing re-lands as table edits only
<!-- AC:END -->

## Notes

2026-08-13 hygiene: shipped on master — src/specializer/sig.rs (20 rows/57 aliases, CUSTOM_NAMES, partition-totality test). Closed by status sweep after the arrow-schema migration (PR #144 era).
