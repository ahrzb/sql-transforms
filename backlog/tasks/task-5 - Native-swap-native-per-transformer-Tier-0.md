---
id: TASK-5
title: 'Native-swap: native per-transformer (Tier 0)'
status: To Do
assignee: []
created_date: '2026-07-18 13:44'
updated_date: '2026-07-19 15:50'
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
