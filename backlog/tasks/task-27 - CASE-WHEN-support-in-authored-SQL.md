---
id: TASK-27
title: 'native: CASE WHEN support in authored SQL'
status: To Do
assignee:
  - Ritchie
created_date: '2026-07-18 19:45'
updated_date: '2026-07-19 00:17'
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
- [ ] #1 native convert_expr (src/expr_build.rs) gains an SqlExpr::Case branch and _interpreter evaluates CASE WHEN (searched + simple forms, with ELSE + NULL handling) via a new Layer-1 expr node (build + eval + type inference)
- [ ] #2 transform == infer differential parity across a CASE decision-table, incl. an ordinal ladder (Ex/Gd/TA -> 5/4/3)
- [ ] #3 remove the codegen_only skip flag added in TASK-30 so CASE differential cases run on BOTH engines
- [ ] #4 outer joins remain out of scope (separate item)
<!-- AC:END -->
