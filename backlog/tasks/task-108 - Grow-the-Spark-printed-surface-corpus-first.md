---
id: TASK-108
title: >-
  Grow the Spark printed surface corpus-first — signatures, types, forced divergences
status: To Do
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies: []
type: feature
ordinal: 100000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ongoing phase-3 growth loop: the L3 gate matches 213 while L2 matches 235 —
the delta is constructs the plan represents but the Spark printer refuses.
Work the largest refusal families first, each with a side-by-side probe
before the spelling ships (the concat-coalesce and startswith precedents
from PR #140). Byte-vs-character families (edit distances) stay refused —
that divergence is semantic, not spelling.

Publish the printed/refused KPI per construct class in `docs/kpis.md`, like
550/678 is today for the specializer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 refusal families on the corpus enumerated by count (the worklist is data, in the task or commit)
- [ ] #2 each new spelling lands with its probe pinned in pins-dialect/
- [ ] #3 L3 floor rises; no L2 regression
- [ ] #4 kpis.md carries the per-dialect printed/refused ladder
<!-- AC:END -->
