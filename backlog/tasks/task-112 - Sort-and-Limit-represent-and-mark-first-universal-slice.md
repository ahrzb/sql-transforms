---
id: TASK-110
title: >-
  Sort + Limit nodes: represent + mark determinism (first universal-coverage slice)
status: To Do
assignee: []
created_date: '2026-08-13 22:30'
labels:
  - m-9
dependencies: []
type: feature
ordinal: 102000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First slice of the universal-representation amendment (epic spec,
2026-08-13): ORDER BY and LIMIT/OFFSET get plan nodes instead of frontend
refusals. SortKey already exists with mandatory direction + null order
(D3); the frontend fills DuckDB's pinned defaults. The verifier gains the
determinism VERDICT: a Limit whose input carries no total order, and any
top-level Sort, classify the plan nondeterministic with a named cause
instead of refusing. L2 gate: these statements flip clean-unsupported to
match (the oracle comparison for nondeterministic plans compares
multisets for Sort-only plans and needs a pinned strategy for LIMIT -
probe first). L3/spark refuses nondeterministic plans by name, floors
must not drop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Sort/Limit nodes with canonical text round-trip; SortKey reused, all fields mandatory
- [ ] #2 verifier emits the determinism verdict with named causes; no frontend refusal for ORDER BY/LIMIT
- [ ] #3 L2 gate handles nondeterministic plans with a probed comparison strategy; floor rises, 0 FAIL
- [ ] #4 L3 gate refuses nondeterministic plans by name; spark floor does not drop
<!-- AC:END -->
