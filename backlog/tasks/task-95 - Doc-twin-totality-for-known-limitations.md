---
id: TASK-95
title: >-
  Doc-twin totality for known-limitations
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: test
ordinal: 87000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The stale "narrow CASTs are rejected" claim survived weeks; §3 had no
executable twin at all. Same trick as sig.rs's totality test: every §4
table row names its twin test, a row without one fails the build — the
TASK-69 doctrine applied to prose. Approved 2026-08-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a known-limitations row without a named twin test fails a unit test
- [ ] #2 every existing row is linked to its twin (adding missing twins where none exist)
<!-- AC:END -->
