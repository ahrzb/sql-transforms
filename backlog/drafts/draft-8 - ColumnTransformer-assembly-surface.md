---
id: DRAFT-8
title: ColumnTransformer assembly surface
status: Draft
assignee: []
created_date: '2026-07-18 13:44'
updated_date: '2026-07-23 14:31'
labels:
  - sklearn
  - fallback
milestone: m-1
dependencies:
  - DRAFT-7
documentation:
  - doc-2 (sklearn transformer implementation plan)
  - 'doc-10 (Feature-output model — records, dense, sparse)'
priority: high
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
A real preprocessing job is never one transformer — it is different treatment per column group:

    numeric   = ["age", "income", "lot_area"]        -> impute median, then scale
    categorical = ["neighborhood", "house_style"]    -> one-hot
    ordinal   = ["quality"]                          -> Ex/Gd/TA -> 5/4/3

In sklearn you express that with a ColumnTransformer and get one matrix out, with a known column order. Today in this library there is no equivalent assembly surface: the user writes per-column SQL and is left to stitch the outputs together themselves, and the column ORDER of the assembled result is whatever falls out — which is the one thing a model cannot tolerate. Feed the model columns in a different order than it was trained on and you get silently wrong predictions, no error.

The multi-output case is where it really bites: one-hot on a 25-category column contributes 25 columns whose width depends on what was seen at fit time. Users need the assembled width and order to be pinned by the fitted artifact, not recomputed per batch.

WHAT THIS TICKET DOES
Column routing + horizontal concat, with the parity bar being bit-identical output vs a stock sklearn ColumnTransformer.transform() — same width, same column order, same values. Must include at least one multi-output transformer so variable-width concat and feature-name expansion are actually exercised. Feature names/provenance must survive routing and concat in order, so users can map a column index back to what produced it.

WHY IT IS A DRAFT
Needs design work before it is dispatchable (AmirHossein, 2026-07-23), and it sits on top of the native-transformer draft (DRAFT-7). Open questions: what the user-facing authoring surface for column routing even looks like in SQL (this is the goal-1 ergonomics question, not just an implementation), how remainder/passthrough columns behave, and how this relates to the type-directed assembler (TASK-16) which solves the adjacent dense+sparse problem. Re-scope before promoting.

Context: doc-2 (sklearn transformer implementation plan), doc-10 (feature-output model).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Column routing + horizontal concat, bit-identical vs stock ColumnTransformer
- [ ] #2 Includes a multi-output transformer (variable-width concat + feature-name expansion exercised)
- [ ] #3 Provenance/feature-names survive routing + concat in order
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 01:01
---
Moved to Draft (2026-07-23): ColumnTransformer assembly surface needs design work — not well-scoped yet. Depends on the native-transformer draft (was TASK-5). Re-scope before promoting.
---
<!-- COMMENTS:END -->
