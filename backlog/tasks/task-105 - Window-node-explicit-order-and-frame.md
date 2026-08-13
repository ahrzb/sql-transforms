---
id: TASK-105
title: >-
  Window node — explicit total order, explicit frame, determinism verified
status: To Do
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies:
  - TASK-104
type: feature
ordinal: 97000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 of the dialect-logical-plan spec: `Window(input, [WindowDef])` with
partition keys, ORDER (direction AND null order mandatory per D3, filled
from the pinned DuckDB defaults in `pins-dialect/sort-window-defaults.json`),
and explicit frame bounds always (the frontend materializes DuckDB's default
frame; printers emit it, every dialect, every time).

Determinism (L4) is verified structurally: `first_value`/`lag`-family over a
non-total order is a constructor error, not a review item. Which window
functions land is corpus-driven — start from what the marginalizer's fixture
corpus actually uses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 WindowDef carries direction + null order + explicit frame, all mandatory; canonical text round-trips
- [ ] #2 verifier rejects order-dependent window functions over non-total orders, by name
- [ ] #3 L2 gate floor rises on the corpus's window statements
- [ ] #4 Spark printing per pinned spellings, or named refusals; L3 accounting updated
<!-- AC:END -->
