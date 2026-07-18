---
id: TASK-10
title: 'Conformance harness S3: nullability-tolerant comparator'
status: To Do
assignee: []
created_date: '2026-07-18 15:37'
labels:
  - tests
  - conformance
milestone: m-5
dependencies: []
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DataFusion constant-folds literal exprs to not-null; our engines emit nullable. Comparator asserts values match while tolerating null-flag differences -- project law: only values must match (decision-2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 value-equal / null-flag-differing cases pass
- [ ] #2 genuine value divergence still fails
<!-- AC:END -->
