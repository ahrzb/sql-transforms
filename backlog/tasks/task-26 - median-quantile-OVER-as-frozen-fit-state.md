---
id: TASK-26
title: median / quantile OVER as frozen fit-state
status: Done
assignee:
  - Ritchie
created_date: '2026-07-18 19:01'
updated_date: '2026-07-23 00:53'
labels:
  - feature
  - usability
milestone: m-1
dependencies: []
documentation:
  - doc-7 (Transformer execution model — fit-state)
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Only MEAN/SUM/COUNT/STDDEV freeze at fit as window-agg state; median (and general quantile) don't. Usability test had to impute a skewed column (LotFrontage) with neighborhood MEAN instead of the textbook median. Add median/quantile OVER (PARTITION BY) frozen state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 median/quantile OVER (PARTITION BY ...) freezes at fit + is looked up per row; transform==infer parity
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done & merged (worktree-task-26-median-quantile -> 6b12ad0). Suite 446 passed. (1) MEDIAN(x) OVER already flowed through fit->freeze->lookup (any 1-arg window fn) -- was a test-coverage gap, now pinned (partition/global/unseen-data). (2) Quantile percentile_cont(x,q)/approx_percentile_cont: the 2nd literal arg was silently dropped; fixed in fit-time path -- _sql.py captures the param + folds it into the state key (p25/p75 don't collide) + rejects non-literal; _state.py emits it in the extraction GROUP BY; _rewrite.py untouched. Authoring = plain DataFusion SQL: MEDIAN(x), percentile_cont(x, 0.25).
<!-- SECTION:NOTES:END -->
