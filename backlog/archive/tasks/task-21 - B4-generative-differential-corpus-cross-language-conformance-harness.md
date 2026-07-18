---
id: TASK-21
title: B4 generative differential corpus + cross-language conformance harness
status: To Do
assignee: []
created_date: '2026-07-18 16:08'
labels:
  - epic-b
milestone: m-6
dependencies: []
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fuzz exprs/SQL -> DataFusion -> freeze expected; run in every runtime's CI. Doubles as producer-coverage sweep. FOUNDATIONAL -- before any runtime; the single point of failure for N reimplementations (coverage can't be hand-written). See doc-4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 generative corpus + harness runnable per-runtime in CI
<!-- AC:END -->
