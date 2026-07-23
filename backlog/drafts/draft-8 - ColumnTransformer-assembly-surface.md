---
id: DRAFT-8
title: ColumnTransformer assembly surface
status: Draft
assignee: []
created_date: '2026-07-18 13:44'
updated_date: '2026-07-23 01:02'
labels:
  - sklearn
  - fallback
milestone: m-1
dependencies:
  - DRAFT-7
documentation:
  - doc-2 (sklearn transformer implementation plan)
  - 'doc-10 (Feature-output model — records, dense, sparse)'
priority: high
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Column routing + horizontal concat; the end-to-end assembly-parity oracle: bit-identical width + column order + values vs stock ColumnTransformer.transform(). Must include a multi-output transformer. Detail in doc-2 (sklearn strategy) / doc-10 (feature-output).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Column routing + horizontal concat, bit-identical vs stock ColumnTransformer
- [ ] #2 Includes a multi-output transformer (variable-width concat + feature-name expansion exercised)
- [ ] #3 Provenance/feature-names survive routing + concat in order
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 01:01
---
Moved to Draft (2026-07-23): ColumnTransformer assembly surface needs design work — not well-scoped yet. Depends on the native-transformer draft (was TASK-5). Re-scope before promoting.
---
<!-- COMMENTS:END -->
