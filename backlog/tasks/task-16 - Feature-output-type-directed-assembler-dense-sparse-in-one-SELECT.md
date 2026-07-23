---
id: TASK-16
title: 'Feature output: type-directed assembler (dense + sparse in one SELECT)'
status: To Do
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 00:57'
labels:
  - feature-output
milestone: m-1
dependencies:
  - TASK-14
documentation:
  - 'doc-10 (Feature-output model — records, dense, sparse)'
priority: medium
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One SELECT compiles to dense(+)sparse via type-directed decompose+assemble: split the projection by output type -> route -> hstack -> materialize. sklearn ColumnTransformer is the INTERNAL assembly target (not a user API) -- users write SQL, we compile the ColumnTransformer. Depends on the sparse-column (TASK-14) + dense-matrix (TASK-13) tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 one SELECT yields a mixed dense+sparse feature set, materialized correctly
- [ ] #2 ColumnTransformer used internally for hstack/densify, not exposed
<!-- AC:END -->
