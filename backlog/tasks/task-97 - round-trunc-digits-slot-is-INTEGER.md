---
id: TASK-97
title: >-
  round/trunc digits slot is INTEGER
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: bug
ordinal: 89000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DuckDB refuses BIGINT digits (round(d, k) is a binder error there; we
bind) — audit-documented at the arm. With widths real it is one table
fact: digits accept INTEGER-or-narrower. Approved 2026-08-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 round(d, k)/trunc(d, k) with a BIGINT digits expression refuses like DuckDB
- [ ] #2 INTEGER-typed digits (literals, ::INTEGER) keep binding; live-oracle pins
<!-- AC:END -->

## Notes

2026-08-13 hygiene: branch origin/task-97 exists UNMERGED. The corpus round(a, b)-INTEGER row (case 622) now serves and matches via PR #144's row widths, so re-check what of this ticket's expression-side scope remains before picking the branch up.
