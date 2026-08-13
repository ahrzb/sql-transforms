---
id: TASK-106
title: >-
  Aggregate + Distinct nodes, aggregate signatures carry their measured tier
status: To Do
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies:
  - TASK-104
type: feature
ordinal: 98000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of the dialect-logical-plan spec: `Aggregate(input, [key], [agg])`
and `Distinct(input)` — the fit plan's GROUP BY / DISTINCT steps. Each
aggregate call is a resolved signature id whose tier (exact vs ε) comes
from the measured table in `pins-dialect/aggregate-tiers.json`, never
assumed. `array_agg` without an inner ORDER BY is a verifier rejection (L4).

The `quantize(x, grid)` node ships here too — the explicit opt-in bridge
from ε back to exact (DuckDB spelling `CAST(CAST(x AS FLOAT) AS DOUBLE)`
for the f32 grid, per the spec; Spark spelling probed before it prints).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Aggregate/Distinct in the node set with verifier rules; canonical text round-trips
- [ ] #2 every admitted aggregate signature carries a tier from the pinned table; unpinned aggregate = refusal by name
- [ ] #3 quantize node exists with pinned per-dialect spellings; moving a value across tiers is observable in the plan text
- [ ] #4 L2/L3 gate floors rise; ε-tier columns compare within the provisional tolerance at the L3 gate
<!-- AC:END -->
