---
id: TASK-27
title: 'native: CASE WHEN support in authored SQL'
status: Done
assignee:
  - Ritchie
created_date: '2026-07-18 19:45'
updated_date: '2026-07-19 15:34'
labels:
  - feature
  - sql-surface
  - usability
milestone: m-1
dependencies:
  - TASK-30
priority: high
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native-engine (Rust) side of CASE WHEN. native's convert_expr (src/expr_build.rs) has no SqlExpr::Case branch, so _interpreter can't evaluate CASE. The DataFusion transform/oracle path already handles CASE today (flows through parse/rewrite untouched; window aggs inside a CASE freeze correctly), and codegen gets CASE in TASK-30 -- this closes the remaining native `infer` gap so the feature runs on both engines. Not gated on the pluggable-backend decision -- plain scalar-expression surface both engines should have. After codegen (depends on TASK-30). Demand: usability test (House Prices) hit it as the blocker to ordinal-encoding quality ladders (Ex/Gd/TA -> 5/4/3), currently un-expressible and forced model-side. DataFusion is the oracle for parity (decision-1). Outer joins stay a separate concern (parked BACKLOG item 'CASE WHEN and outer joins').
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 native convert_expr (src/expr_build.rs) gains an SqlExpr::Case branch and _interpreter evaluates CASE WHEN (searched + simple forms, with ELSE + NULL handling) via a new Layer-1 expr node (build + eval + type inference)
- [x] #2 transform == infer differential parity across a CASE decision-table, incl. an ordinal ladder (Ex/Gd/TA -> 5/4/3)
- [x] #3 remove the codegen_only skip flag added in TASK-30 so CASE differential cases run on BOTH engines
- [x] #4 outer joins remain out of scope (separate item)
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 14:04
---
Ritchie picked this up (TASK-30 dependency now Done). Scope confirmed: SqlExpr::Case in src/expr_build.rs + eval in src/expr.rs mirroring codegen semantics (short-circuit, three-valued truthy, common-supertype result typing); then remove the TASK-30 codegen_only skip so tests/test_diff_case.py runs on both backends, plus the window-agg-inside-CASE integration test through SQLTransform.infer_batch (reachable once native supports CASE).
---

author: Iris (PM)
created: 2026-07-19 15:34
---
Marked Done per Ritchie's merge e2dfeb1. CASE WHEN now complete on both engines (TASK-30 codegen + TASK-27 native). Rebased cleanly onto TASK-28, so the expr_build.rs collision risk I flagged did not materialize. Review's Minor follow-up (transformer-ref-inside-CASE-branch has no test, silent failure mode) captured as TASK-31.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native CASE WHEN shipped and merged to master (e2dfeb1). Added Expr::Case (short-circuit eval; simple form → operand = value), common-supertype result typing mirroring the fixed codegen logic, validation + transformer-resolution recursion. Retired the temporary codegen_only harness flag from TASK-30 so every CASE case runs against the DataFusion oracle on BOTH backends, and added the window-agg-inside-CASE integration test TASK-30 deferred. Rebased cleanly onto master's TASK-28 (identifier folding), rebuilt native. Full suite 484 passed / 16 skipped / 1 xfailed. Final whole-branch review: merge-Yes, no Critical/Important. With TASK-30 (codegen) + TASK-27 (native), CASE WHEN is complete on both engines. Follow-up: TASK-31 (untested resolve_transformers CASE arm) spun off from the review's Minor note.
<!-- SECTION:FINAL_SUMMARY:END -->
