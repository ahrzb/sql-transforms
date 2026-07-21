---
id: TASK-4
title: Codegen-only parity bugs
status: Done
assignee:
  - 'Developer: Codegen'
created_date: '2026-07-18 13:44'
updated_date: '2026-07-19 01:15'
labels:
  - codegen
  - parity
dependencies: []
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two divergences on the codegen path only (Rust already matches the oracle here). NOTE: whether codegen is a maintained/default engine is a pending framing call by AmirHossein -- decision-7 (two-engine framing — ruled 2026-07-19: native default, codegen opt-in); these are recorded regardless.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 float->string for |x| < 1e-4 matches DF (0.00001 not 1e-05; 1e-6 not 1e-06)
- [x] #2 Integer arithmetic overflow wraps like DF/Rust instead of Python bigint
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both done & merged into codegen (131fa0b): #1 float->string small-value, #2 i64 overflow wrap. Codegen matches DataFusion oracle.
<!-- SECTION:NOTES:END -->
