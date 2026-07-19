---
id: TASK-13
title: 'Feature output: dense float64 (n,k) numpy mode'
status: To Do
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-19 01:15'
labels:
  - feature-output
milestone: m-1
dependencies: []
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dense float64 (n,k) matrix output on the columnar path for numeric feature sets (scalers/trees/numeric sklearn). Immediate win, no new engine. Part of the feature-output model (records/dense/sparse); see doc-10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 infer/transform can emit a dense float64 (n,k) matrix, sklearn-consumable
- [ ] #2 records mode (pydantic) unchanged
<!-- AC:END -->
