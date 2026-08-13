---
id: TASK-98
title: >-
  if() and ifnull() exist on DuckDB
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: feature
ordinal: 90000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both bind on DuckDB 1.5.5 (audited semantics: if(b, x, y) unifies x/y;
ifnull is 2-arg coalesce) and refuse here. Both desugar to machinery we
have (CASE / coalesce). Approved 2026-08-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 if()/ifnull() serve with DuckDB-equal values and schema, live-oracle pinned
- [ ] #2 signature-table rows + totality test cover both names
<!-- AC:END -->

## Notes

2026-08-13: RFC 1 delivered in chat (measured facts, options, recommendation); its ASK block awaits AmirHossein's answers. Lands on the arrow schema surface (PR #144 merged).
