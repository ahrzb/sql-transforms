---
id: TASK-5
title: 'Native-swap: native per-transformer (Tier 0)'
status: In Progress
assignee:
  - Wren
created_date: '2026-07-18 13:44'
updated_date: '2026-07-23 00:36'
labels:
  - sklearn
  - native-swap
milestone: m-1
dependencies: []
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Swap each fallback-backed transformer to a native engine impl, diffed against the fallback oracle, in tier order. Most Tier 0/1 map onto shipped machinery (window aggs, PARTITION BY, LookupJoin, struct/UNNEST). Full plan: doc-2 (sklearn transformer plan).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 StandardScaler native, differential parity vs sklearn fallback
- [ ] #2 SimpleImputer native + parity
- [ ] #3 OrdinalEncoder native + parity (unknown-category handling)
- [ ] #4 OneHotEncoder native + parity (multi-output)
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 00:36
---
Dispatched to Wren (2026-07-23), next on the m-1 spine after TASK-13. Tier order per doc-2; sklearn fallback is the parity oracle per AC wording, DataFusion for SQL semantics (decision-1). TASK-6 (ColumnTransformer assembly) unblocks once this + TASK-13 are done — TASK-13 already landed.
---
<!-- COMMENTS:END -->
