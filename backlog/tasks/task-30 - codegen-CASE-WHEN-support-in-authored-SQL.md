---
id: TASK-30
title: 'codegen: CASE WHEN support in authored SQL'
status: Done
assignee:
  - Ritchie
created_date: '2026-07-19 00:17'
updated_date: '2026-07-19 14:04'
labels:
  - feature
  - sql-surface
  - codegen
milestone: m-1
dependencies: []
references:
  - sql_transform/_codegen/
priority: high
ordinal: 30000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Codegen-engine side of CASE WHEN. Teach the codegen expression pipeline about exp.Case: a new IR node + _convert_expr + infer_type + validation + emission as a short-circuiting nested conditional. Self-contained to sql_transform/_codegen/. The DataFusion transform/oracle path already handles CASE today (flows through parse/rewrite untouched; window aggs inside a CASE freeze correctly), so this only closes the codegen `infer` gap. Not gated on the pluggable-backend decision -- plain scalar-expression surface both engines should have, independent of default-engine choice. Native counterpart is TASK-27 (picked up after this). Ritchie working it now (spec + plan in progress). DataFusion is the oracle for parity (decision-1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 codegen expr pipeline handles exp.Case (searched + simple forms, with ELSE + NULL handling): IR node + _convert_expr + infer_type + validation + emission as a short-circuiting nested conditional
- [x] #2 differential parity vs the DataFusion oracle across a CASE decision-table, incl. an ordinal ladder (Ex/Gd/TA -> 5/4/3)
- [x] #3 add a codegen_only skip flag so CASE differential cases run codegen-only until the native ticket (TASK-27) lands, after which they run on both backends
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 14:04
---
Marked Done per Ritchie's session report (merge a92554f). Native counterpart TASK-27 now In Progress; it will remove the codegen_only skip so CASE runs on both engines.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Codegen CASE WHEN shipped and merged to master (a92554f). Searched + simple forms, DataFusion-parity, short-circuiting emission. Suite: 458 passed / 26 skipped / 1 xfailed; whole-branch review verdict merge-Yes. AC #3's codegen_only skip is in place so CASE cases run codegen-only until TASK-27 (native) removes it.
<!-- SECTION:FINAL_SUMMARY:END -->
