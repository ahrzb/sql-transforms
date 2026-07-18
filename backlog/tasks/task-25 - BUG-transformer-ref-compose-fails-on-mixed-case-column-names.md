---
id: TASK-25
title: 'BUG: transformer-ref compose fails on mixed-case column names'
status: To Do
assignee:
  - Wren
created_date: '2026-07-18 19:01'
updated_date: '2026-07-18 19:05'
labels:
  - bug
  - transformer-refs
milestone: m-1
dependencies: []
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The t"...{ohe}(cols)..." compose (fitted transformer as UDF) fails at fit/transform on ANY non-lowercase column. Root cause VERIFIED: _named_struct (sql_transform/_transformer_ref.py:32) rebuilds column refs with unquoted exp.column(c); DataFusion folds MSZoning -> mszoning, then errors 'No field named __this__.mszoning'. Proven quoting-only: lowercasing the columns fixes it. SAME CLASS as the earlier identifier-quoting bug (c056ec3, fixed for composition-inline + PARTITION BY) -- the transformer-ref path didn't carry that fix. Found by usability test on House Prices (80-col CamelCase Kaggle set).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _named_struct quotes column refs (exp.column(c, quoted=True)); CamelCase compose works through both transform and infer
- [ ] #2 xfail-strict regression test with a CamelCase column (flips to pass on fix)
- [ ] #3 root-cause sweep: no other unquoted exp.column rebuilds left in the transformer-ref path
<!-- AC:END -->
