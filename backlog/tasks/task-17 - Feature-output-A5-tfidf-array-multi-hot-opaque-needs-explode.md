---
id: TASK-17
title: 'Feature output A5: tfidf / array multi-hot (opaque, needs explode)'
status: To Do
assignee: []
created_date: '2026-07-18 15:52'
labels:
  - feature-output
milestone: m-1
dependencies: []
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tfidf and array multi-hot need variable-expansion (explode), so they are the OPAQUE ones -- map onto the shipped opaque-transform mechanism (decision-3), corpus-gated. Distinct from A2/A3 which are fixed-fanout and composable. Depends on opaque-transform (Part 1/2 shipped).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 tfidf + array multi-hot available via the opaque-transform path
<!-- AC:END -->
