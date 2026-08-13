---
id: TASK-107
title: >-
  Differential fuzzer routes generated statements through parse→print (L2 invisibility fuzzing)
status: To Do
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies:
  - TASK-104
type: feature
ordinal: 99000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2's gate extension: teach the differential fuzzer
(`packages/confit/fuzz`) a mode where every generated statement also runs
through `parse → print_duck` and the reprint executes on DuckDB — L2
(invisibility) checked on generated data, not just the curated corpus.
Statements the frontend refuses count as clean-unsupported, never as
silent skips; a reprint that changes the oracle's answer is a finding
class of its own.

Same campaign discipline as the existing fuzzer: findings become
xfail-strict pins + a ticket, never inline fixes (the engine-bug process).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 fuzz mode exists: generate → parse → print_duck → execute both, compare bit-exact
- [ ] #2 three-outcome accounting per seed (match / clean-unsupported / divergence), divergences deduped by class
- [ ] #3 a 20k campaign runs; every divergence class is pinned xfail-strict with a ticket
<!-- AC:END -->
