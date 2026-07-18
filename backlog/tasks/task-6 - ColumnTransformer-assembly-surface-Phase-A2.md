---
id: TASK-6
title: ColumnTransformer assembly surface
status: To Do
assignee: []
created_date: '2026-07-18 13:44'
updated_date: '2026-07-18 23:50'
labels:
  - sklearn
  - fallback
milestone: m-1
dependencies: []
references:
  - docs/ROADMAP.md
  - docs/BACKLOG.md
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Column routing + horizontal concat; the end-to-end assembly-parity oracle: bit-identical width + column order + values vs stock ColumnTransformer.transform(). Must include a multi-output transformer. Detail in docs/ROADMAP.md (ColumnTransformer-glue slice) + docs/BACKLOG.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Column routing + horizontal concat, bit-identical vs stock ColumnTransformer
- [ ] #2 Includes a multi-output transformer (variable-width concat + feature-name expansion exercised)
- [ ] #3 Provenance/feature-names survive routing + concat in order
<!-- AC:END -->
