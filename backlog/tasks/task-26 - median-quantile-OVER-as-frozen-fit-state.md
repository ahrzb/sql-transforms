---
id: TASK-26
title: median / quantile OVER as frozen fit-state
status: To Do
assignee:
  - Ritchie
created_date: '2026-07-18 19:01'
updated_date: '2026-07-18 19:05'
labels:
  - feature
milestone: m-1
dependencies: []
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Only MEAN/SUM/COUNT/STDDEV freeze at fit as window-agg state; median (and general quantile) don't. Usability test had to impute a skewed column (LotFrontage) with neighborhood MEAN instead of the textbook median. Add median/quantile OVER (PARTITION BY) frozen state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 median/quantile OVER (PARTITION BY ...) freezes at fit + is looked up per row; transform==infer parity
<!-- AC:END -->
