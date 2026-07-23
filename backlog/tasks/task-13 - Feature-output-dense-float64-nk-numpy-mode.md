---
id: TASK-13
title: 'Feature output: dense float64 (n,k) numpy mode'
status: Done
assignee:
  - '@Wren'
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 00:35'
labels:
  - feature-output
milestone: m-1
dependencies: []
priority: high
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dense float64 (n,k) matrix output on the columnar path for numeric feature sets (scalers/trees/numeric sklearn). Immediate win, no new engine. Part of the feature-output model (records/dense/sparse); see doc-10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 infer/transform can emit a dense float64 (n,k) matrix, sklearn-consumable
- [x] #2 records mode (pydantic) unchanged
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 00:35
---
Landed as f3e80cf on master (sql_transform/__init__.py + __init___test.py). AC#1: transform() -> float64 (n,k) via pyarrow cast (NULL->NaN), infer/infer_batch -> (k,)/(n,k); sklearn-consumability proven by test_dense_output_sklearn_consumable (LinearRegression.fit). AC#2: records stays default, guarded by test_records_output_is_default_and_unchanged. Verified against the diff. Suite 501 passed / 14 skipped. Ponytail ceiling noted in-code: dense infer routes through pydantic records; direct columnar-from-Rust is the follow-up if profiles demand.
---
<!-- COMMENTS:END -->
