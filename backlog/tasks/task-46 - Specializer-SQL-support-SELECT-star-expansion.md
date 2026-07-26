---
id: TASK-46
title: 'Specializer SQL support: SELECT * star expansion'
status: To Do
assignee: []
created_date: '2026-07-26 11:42'
labels: []
milestone: m-7
dependencies:
  - TASK-45
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The single biggest corpus rung: 128 of 625 clean-unsupported corpus cases have `SELECT *` (or `tbl.*`) as their first blocker. Expand the star at bind time in the frontend against DuckDB's measured semantics — column order and naming for the row table alone and under joins (including duplicate-name handling), `tbl.*` qualified forms, and whatever star modifiers the corpus actually uses (EXCLUDE/REPLACE are measured-first: support only what the corpus needs, reject the rest cleanly by name). Pure frontend work — no IR, backend, or boundary changes. Clearing it also de-masks second blockers currently hidden behind star for the builtins wave (TASK-47).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Star semantics pinned by measurement against DuckDB 1.5.5 (column order, duplicate-name policy, qualified tbl.*, modifier handling) and recorded as duck_check tests
- [ ] #2 Corpus replay: star-first-blocker cases flip to match or to a NAMED deeper blocker; zero FAILs; match count and the new first-blocker tally recorded here
- [ ] #3 Unsupported star forms (if any remain) reject with a clean "unsupported: ..." naming the form
- [ ] #4 mise gate-specializer green
<!-- AC:END -->
