---
id: DRAFT-10
title: 'Feature output: tfidf / array multi-hot (opaque, needs explode)'
status: Draft
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 04:46'
labels:
  - feature-output
milestone: m-1
dependencies:
  - DRAFT-9
documentation:
  - 'doc-10 (Feature-output model — records, dense, sparse)'
  - doc-9 (Rich type system and UNNEST)
priority: low
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tfidf and array multi-hot need variable-expansion (explode), so they are the OPAQUE ones -- map onto the shipped opaque-transform mechanism (decision-3), corpus-gated. Distinct from the sparse-column / scalar-one-hot tasks (TASK-14/15) which are fixed-fanout and composable. Depends on opaque-transform (Part 1/2 shipped).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 tfidf + array multi-hot available via the opaque-transform path
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 04:46
---
Parked as Draft (2026-07-23): tfidf / array multi-hot is very low priority. Depends on the sparse-COO column work (DRAFT-9) — tfidf output is sparse and needs that materializer path before it's buildable. Re-promote once DRAFT-9 is scoped/done and there's demand.
---
<!-- COMMENTS:END -->
