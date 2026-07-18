---
id: TASK-16
title: 'Feature output A4: type-directed assembler (dense + sparse in one SELECT)'
status: To Do
assignee: []
created_date: '2026-07-18 15:52'
labels:
  - feature-output
milestone: m-1
dependencies: []
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One SELECT compiles to dense(+)sparse via type-directed decompose+assemble: split the projection by output type -> route -> hstack -> materialize. sklearn ColumnTransformer is the INTERNAL assembly target (not a user API) -- users write SQL, we compile the ColumnTransformer. Depends on A2 (sparse column) + A1 (dense).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 one SELECT yields a mixed dense+sparse feature set, materialized correctly
- [ ] #2 ColumnTransformer used internally for hstack/densify, not exposed
<!-- AC:END -->
