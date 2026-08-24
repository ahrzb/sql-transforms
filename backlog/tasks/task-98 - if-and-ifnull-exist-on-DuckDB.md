---
id: TASK-98
title: >-
  if() and ifnull() exist on DuckDB
status: Done
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
- [x] #1 if()/ifnull() serve with DuckDB-equal values and schema, live-oracle pinned
- [x] #2 (adjusted: both names desugar to their AST twin BEFORE signature resolution -- CASE and coalesce -- so there are no rows of their own; the catalogue-guard test derives the names from the dispatch and covers both) signature-table rows + totality test cover both names
<!-- AC:END -->

## Notes

2026-08-13: RFC 1 delivered in chat (measured facts, options, recommendation); its ASK block awaits AmirHossein's answers. Lands on the arrow schema surface (PR #144 merged).

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done 2026-08-19. `if(c, a, b)` rebuilds `CASE WHEN c THEN a ELSE b END` as
AST and re-enters the binder (the `-a % b` rewrite precedent), so type
unification, SQLNULL channels, lazy arms and TASK-124's selection-context
conditions all apply verbatim. `ifnull` shares the coalesce arm with its
own arity-2 gate. BUILTIN_NAMES carries both, enforced by the
catalogue-guard test. Live-oracle rows (8 spellings incl. lazy-arm traps
and a NATURAL JOIN static) + arity refusals in
test_duckdb_wave5_structural.py. Suite 2950 passed.
<!-- SECTION:NOTES:END -->
