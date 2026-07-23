---
id: TASK-16
title: 'Feature output: type-directed assembler (dense + sparse in one SELECT)'
status: To Do
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 14:32'
labels:
  - feature-output
milestone: m-1
dependencies:
  - DRAFT-9
documentation:
  - 'doc-10 (Feature-output model — records, dense, sparse)'
priority: medium
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
A realistic feature set mixes shapes. Some columns are plain numbers, one is a high-cardinality one-hot:

    SELECT
      (age - AVG(age) OVER ()) / STDDEV(age) OVER ()  AS age_z,     -- dense scalar
      lot_area / AVG(lot_area) OVER ()                AS lot_norm,  -- dense scalar
      {ohe}(neighborhood)                             AS nbhd       -- 800-wide, sparse
    FROM __THIS__

The user wrote ONE query and expects ONE feature matrix. But the pieces have genuinely different physical representations: two float64 scalars and an 800-wide mostly-zero block. Forcing everything dense blows up memory; forcing everything sparse wastes it on the scalars. Today the user has to know which is which and assemble by hand, which means they also own the column-ordering problem — and getting that wrong silently misfeeds the model.

WHAT THIS TICKET DOES
Make one SELECT compile to a correctly-assembled mixed dense+sparse feature set, automatically, by TYPE-DIRECTED decompose and assemble: split the projection by output type, route each part to the right representation, hstack, materialize.

Key design point: sklearn's ColumnTransformer is used as the INTERNAL assembly target — the thing we compile TO for hstack/densify — not as a user-facing API. The user writes SQL; we compile the ColumnTransformer for them. That inversion is the whole point (project goal 1): the SQL is the authoring surface, sklearn is an implementation detail.

BLOCKED
Depends on DRAFT-9 (sparse COO struct column) — there is no sparse representation to route to until that exists, and DRAFT-9 is currently parked as an unscoped draft. The dense half (TASK-13) has already landed. So this cannot finish until DRAFT-9 is scoped, promoted, and done.

Context: doc-10 (feature-output model — records, dense, sparse).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 one SELECT yields a mixed dense+sparse feature set, materialized correctly
- [ ] #2 ColumnTransformer used internally for hstack/densify, not exposed
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 01:02
---
Dependency retargeted: the sparse-COO prerequisite (was TASK-14) is now DRAFT-9 — parked as a draft pending design. So TASK-16 is effectively blocked on unscoped work; it can't finish until DRAFT-9 is scoped, promoted, and done.
---
<!-- COMMENTS:END -->
