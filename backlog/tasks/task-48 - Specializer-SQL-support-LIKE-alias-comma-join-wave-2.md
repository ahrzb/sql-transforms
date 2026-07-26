---
id: TASK-48
title: 'Specializer SQL support: LIKE, dynamic-table alias, comma-join rewrite (wave 2)'
status: In Progress
assignee: []
created_date: '2026-07-26 15:05'
updated_date: '2026-07-26 15:05'
labels: []
milestone: m-7
dependencies:
  - TASK-47
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
  - docs/superpowers/specs/2026-07-26-wave1-builtin-pins.md
type: feature
ordinal: 42000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dual-axis rung plus two structural cheap wins, ~120 corpus first-blockers in reach. (1) LIKE / NOT LIKE with % and _ wildcards and the ESCAPE clause — DuckDB semantics pinned FIRST (case sensitivity, codepoint vs byte matching for _, escape edge cases, NULL propagation, degenerate patterns); a compiled-pattern op on both backends via one shared matcher; SIMILAR TO / regexp reject by name unless the pins show LIKE-only covers the corpus head. Closes the workload ladder's last gap (title extraction, device normalization patterns). (2) Alias on the dynamic table — scope plumbing in the frontend binder (30 cases). (3) Comma-join rewrite: FROM t, dim WHERE <equi-conjuncts> rewrites to the INNER equi-join the engine already serves, with non-equi/cross shapes rejecting cleanly by name (up to 50 join-form cases share this first blocker).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 LIKE semantics pinned by measurement (duck_check tests recorded before implementation: wildcards, ESCAPE, unicode, NULL, degenerate patterns) and both backends agree via a shared matcher (differential extended)
- [ ] #2 Dynamic-table alias binds (qualified refs via alias, original name behavior measured and mirrored)
- [ ] #3 Comma-join with equi-WHERE rewrites to the served INNER join; non-equi shapes reject by name
- [ ] #4 Corpus replay: wave-2 first-blocker cases flip to match or a named deeper blocker; zero FAILs; tally recorded here
- [ ] #5 mise gate-specializer green
<!-- AC:END -->
