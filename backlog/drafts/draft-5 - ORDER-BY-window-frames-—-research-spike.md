---
id: DRAFT-5
title: ORDER BY / window frames — research spike
status: Draft
assignee: []
created_date: '2026-07-19 01:09'
labels:
  - sql-surface
  - spike
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
AGG(col) OVER (ORDER BY ...) and explicit ROWS/RANGE BETWEEN frames -- running sums, cumulative means, moving windows. Currently rejected with NotImplementedError (WindowAgg.has_order). FUNDAMENTALLY HARDER: order-dependent and stateful across rows, so they do NOT fit the 'freeze a value at fit, broadcast at inference' model that OVER () and PARTITION BY share -- inference would need streaming/sequence state. Treat as a research spike, not a small feature; decide whether it's even in scope for a row-at-a-time inference engine before investing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 decision: are order-dependent/stateful window frames in scope for a row-at-a-time inference engine, and if so what execution model
<!-- AC:END -->
