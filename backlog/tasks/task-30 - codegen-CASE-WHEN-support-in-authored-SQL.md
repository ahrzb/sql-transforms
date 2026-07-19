---
id: TASK-30
title: 'codegen: CASE WHEN support in authored SQL'
status: In Progress
assignee:
  - Ritchie
created_date: '2026-07-19 00:17'
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
- [ ] #1 codegen expr pipeline handles exp.Case (searched + simple forms, with ELSE + NULL handling): IR node + _convert_expr + infer_type + validation + emission as a short-circuiting nested conditional
- [ ] #2 differential parity vs the DataFusion oracle across a CASE decision-table, incl. an ordinal ladder (Ex/Gd/TA -> 5/4/3)
- [ ] #3 add a codegen_only skip flag so CASE differential cases run codegen-only until the native ticket (TASK-27) lands, after which they run on both backends
<!-- AC:END -->
