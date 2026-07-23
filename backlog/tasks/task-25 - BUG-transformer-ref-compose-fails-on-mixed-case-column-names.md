---
id: TASK-25
title: 'BUG: transformer-ref compose fails on mixed-case column names'
status: Done
assignee:
  - Wren
created_date: '2026-07-18 19:01'
updated_date: '2026-07-23 00:53'
labels:
  - bug
  - transformer-refs
  - usability
milestone: m-1
dependencies: []
documentation:
  - 'doc-8 (Composition — {transform}(col) references)'
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The t"...{ohe}(cols)..." compose (fitted transformer as UDF) fails at fit/transform on ANY non-lowercase column. Root cause VERIFIED: _named_struct (sql_transform/_transformer_ref.py:32) rebuilds column refs with unquoted exp.column(c); DataFusion folds MSZoning -> mszoning, then errors 'No field named __this__.mszoning'. Proven quoting-only: lowercasing the columns fixes it. SAME CLASS as the earlier identifier-quoting bug (c056ec3, fixed for composition-inline + PARTITION BY) -- the transformer-ref path didn't carry that fix. Found by usability test on House Prices (80-col CamelCase Kaggle set).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 _named_struct quotes column refs (exp.column(c, quoted=True)); CamelCase compose works through both transform and infer
- [x] #2 xfail-strict regression test with a CamelCase column (flips to pass on fix)
- [x] #3 root-cause sweep: no other unquoted exp.column rebuilds left in the transformer-ref path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done & merged (task-25-camelcase-ref -> 6ddbfcd). Fix: exp.column(c, quoted=True) at _transformer_ref.py:34. Regression test tests/test_transformer_ref.py::test_camelcase_columns_compose (xfailed on bug, xpass-strict on fix, marker removed -> plain passing). Root-cause sweep: only unquoted rebuild in the transformer-ref path. Suite 440 passed. Latent adjacent spot _compose.py:214 (fit_into_scope) flagged separately -- safe today (state keys lowercased by construction).
<!-- SECTION:NOTES:END -->
