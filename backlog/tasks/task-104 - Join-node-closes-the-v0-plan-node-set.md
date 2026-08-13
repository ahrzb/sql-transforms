---
id: TASK-104
title: >-
  Join node — static joins bound, verified, printed (closes the v0 node set)
status: To Do
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies: []
type: feature
ordinal: 96000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The spec's v0 node set includes `Join(input, input, kind, [JoinKey])` with
`JoinKey.null_safe: bool` mandatory (D3); the plan today has only
Scan/Filter/Project (`packages/confit/src/dialect/plan.rs`). The corpus's
static-join statements (the specializer's current surface) are all
clean-unsupported at the L2 gate because of this.

Frontend fills null_safe from the source spelling (`=` → false,
`IS NOT DISTINCT FROM` → true); DuckDB printer round-trips both; Spark
prints `<=>` for null-safe (pinned in `pins-dialect/spark-ansi.json`);
BigQuery refuses null-safe until its expansion spelling is probed in the
phase-4 gate. Exhaustive matches everywhere — a printer that cannot answer
for Join must fail to compile, not default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Join in the node set: kind (inner/left at minimum, corpus-driven), keys with mandatory null_safe; verifier checks key types unify and ordinals resolve
- [ ] #2 canonical plan text + text round-trip cover Join
- [ ] #3 DuckDB frontend binds the corpus's join statements; L2 gate floor rises (record old → new in the commit message)
- [ ] #4 Spark printer emits `<=>` for null-safe keys; L3 gate floor rises
- [ ] #5 three-outcome accounting: no statement changes class downward, zero wrong answers
<!-- AC:END -->
