---
id: TASK-14
title: >-
  Feature output: sparse COO struct column + CSR materializer + width-invariant
  assert
status: To Do
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-18 23:35'
labels:
  - feature-output
milestone: m-1
dependencies: []
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Per-row sparse feature = Arrow struct<indices: list<int32>, values: list<float64>> (COO). 1:1 with rows, so it mixes with dense scalar columns in ONE SELECT (sparseness lives in the cell, not row cardinality). Materialize to scipy CSR: concat rows' arrays -> data/indices, per-row lengths -> indptr. Dimension N + unseen-key handling come from the FITTED transform (self-contained artifact), NOT the cell/type. N pins shape=(n,N); a width-invariant assert catches silent batch-width drift (a model-misalignment bug).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 COO struct column type + CSR and dense materializers
- [ ] #2 N/unseen-key sourced from the fitted transform; shape pinned (n,N)
- [ ] #3 width-invariant assert fails on batch-width drift
<!-- AC:END -->
