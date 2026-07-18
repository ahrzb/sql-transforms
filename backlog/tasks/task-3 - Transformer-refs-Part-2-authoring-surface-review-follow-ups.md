---
id: TASK-3
title: Transformer-refs (Part-2 authoring surface) review follow-ups
status: To Do
assignee: []
created_date: '2026-07-18 13:44'
updated_date: '2026-07-18 15:19'
labels:
  - python
  - transformer-refs
milestone: m-1
dependencies: []
references:
  - docs/BACKLOG.md
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-ups from the whole-branch review (ready-to-merge, no Critical/Important). Detail in docs/BACKLOG.md 'Opaque transform support' Part 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Single-ref path runs .transform() twice at fit; reuse the _derive_schemas probe, skip _materialize when no outer consumer
- [ ] #2 Friendly pre-check errors for aggregate-over-output and unfitted-transformer paths
- [ ] #3 Negative/contract tests: mixed leaf+nested args, aggregate-over-output, column vs feature_names_in_ mismatch, unfitted ref; + regression for transformer + PARTITION BY input-col
- [ ] #4 Confirmatory 3+ level nesting test (low value)
<!-- AC:END -->
