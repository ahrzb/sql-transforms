---
id: TASK-94
title: >-
  The fuzzer reads the signature table
status: To Do
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: refactor
ordinal: 86000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
gen.py's SIGS dict is a hand-maintained shadow of sig.rs (and its
BUILTIN_NAMES copy is dead code — the audited "wildcard production" never
existed). Export the Rust table (build-time JSON or an introspection fn)
and derive the generator's productions from it, so a new table row is
patrolled automatically — no phase-5 decimal row can be forgotten.
Approved 2026-08-13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 gen.py carries no hand-written signature list; productions derive from the exported table
- [ ] #2 a table row added in Rust is exercised by the next campaign without gen.py edits
<!-- AC:END -->
