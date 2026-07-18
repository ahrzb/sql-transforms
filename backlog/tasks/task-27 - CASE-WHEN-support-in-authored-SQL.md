---
id: TASK-27
title: CASE WHEN support in authored SQL
status: To Do
assignee: []
created_date: '2026-07-18 19:45'
labels:
  - feature
  - sql-surface
dependencies: []
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CASE WHEN is not supported in the native interpreter today: rewrite_sql only handles plain columns, binary ops, and window aggregates (SQL_SUPPORT.md), and the native InferFn has no CASE expr node. DataFusion transform already supports it (it IS DataFusion). Demand: usability test (House Prices) hit it as the blocker to ordinal-encoding quality ladders (Ex/Gd/TA -> 5/4/3), currently un-expressible and forced model-side. Scope THIS ticket to CASE WHEN only -- the parked BACKLOG item 'CASE WHEN and outer joins' also bundles outer joins, which stay a separate concern. DataFusion is the oracle for parity (decision-1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CASE WHEN (searched + simple forms, with ELSE and NULL handling) authorable in SQL; rewrite_sql carries it through (not just plain cols/binops/window-aggs)
- [ ] #2 native InferFn evaluates CASE WHEN via a new Layer-1 expr node (build + eval + type inference)
- [ ] #3 transform == infer differential parity across a decision-table of cases, incl. an ordinal ladder (Ex/Gd/TA -> 5/4/3)
- [ ] #4 outer joins remain out of scope (separate item)
<!-- AC:END -->
