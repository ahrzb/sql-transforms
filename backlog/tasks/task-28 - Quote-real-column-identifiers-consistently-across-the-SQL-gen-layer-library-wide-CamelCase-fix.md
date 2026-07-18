---
id: TASK-28
title: >-
  Quote real-column identifiers consistently across the SQL-gen layer
  (library-wide CamelCase fix)
status: To Do
assignee: []
created_date: '2026-07-18 19:48'
updated_date: '2026-07-18 20:45'
labels:
  - bug
  - sql-surface
  - usability
milestone: m-1
dependencies: []
priority: high
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SEVERITY: high. Wren found (while investigating _compose.py:214) that CamelCase is broken LIBRARY-WIDE, not just the transformer-ref path TASK-25 fixed. Evidence (table with column 'Age'): SELECT Age AS x FROM __THIS__ -> FAILS 'No field named __this__.age' (plain passthrough!); SELECT Age / AVG(Age) OVER () -> FAILS; quoting ("Age") works; {a}(MixedCol) composition fails. So today the library REQUIRES users to double-quote every non-lowercase column -- bites nearly everywhere on real datasets (House Prices 80-col CamelCase). Unquoted rebuilds fold to lowercase in: rewrite_sql (__THIS__.col), build_state_tables (AVG(col) + GROUP BY / PARTITION BY keys), _compose.py:104/140. TASK-25 fixed only _named_struct. State-key columns (engine-internal, lowercased by state_key()) stay UNQUOTED -- the _compose.py:214 invariant, Wren verified (quoting THAT would create a latent mismatch; leave it + document with a comment). DESIGN DECISION embedded (needs AmirHossein's conscious yes): the fix makes authored identifiers case-sensitive-exact (match the Arrow schema, = what quoting does) vs. consistently folding everything to lowercase (case-insensitive). Wren + PM recommend case-sensitive-exact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 authored non-lowercase columns work WITHOUT user double-quoting across: plain passthrough, window-agg OVER, PARTITION BY, composition (frozen + unfit); transformer-ref already fixed (regression-guard)
- [ ] #2 state-key columns stay unquoted (214 invariant preserved + documented with a comment)
- [ ] #3 CamelCase test matrix green; transform == infer parity (decision-1)
- [ ] #4 DESIGN: authored identifiers are case-sensitive-exact vs Arrow schema -- per AmirHossein's ratification (see report)
<!-- AC:END -->
