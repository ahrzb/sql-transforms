---
id: DRAFT-9
title: >-
  Feature output: sparse COO struct column + CSR materializer + width-invariant
  assert
status: Draft
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 14:31'
labels:
  - feature-output
milestone: m-1
dependencies: []
documentation:
  - 'doc-10 (Feature-output model — records, dense, sparse)'
priority: low
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You one-hot encode a high-cardinality column — say `neighborhood` with 800 distinct values, or a product SKU with 50k:

    SELECT {ohe}(neighborhood) AS nbhd FROM __THIS__

Dense output means every row carries 800 (or 50,000) float64 columns, nearly all zeros. That is the difference between a feature matrix that fits in memory and one that does not, and between a fast model fit and an unusable one. sklearn's own OneHotEncoder returns a scipy sparse matrix by default for exactly this reason — users coming from sklearn will expect sparse and be surprised when a 50k-category encode tries to materialize dense.

Second, subtler failure this ticket guards against: BATCH-WIDTH DRIFT. If the encoded width is derived from the data in each batch rather than pinned by the fitted artifact, a serving batch that happens not to contain some category produces a NARROWER matrix. The model then receives misaligned columns — feature 300 means something different than it did at training. No error, just wrong predictions. That is why the width-invariant assert is an acceptance criterion and not a nice-to-have.

WHAT THIS TICKET DOES
Represent a per-row sparse feature as an Arrow struct<indices: list<int32>, values: list<float64>> (COO). Because it stays 1:1 with rows, it mixes with ordinary dense scalar columns in ONE SELECT — the sparseness lives in the cell, not in the row cardinality, so nothing has to explode. Materialize to scipy CSR by concatenating the per-row arrays into data/indices and using per-row lengths as indptr.

Dimension N and unknown-category handling come from the FITTED transform (a self-contained artifact), NOT from the cell or the type. N pins shape=(n, N), and a width-invariant assert fails loudly if a batch ever disagrees.

WHY IT IS A DRAFT
Super-low priority AND needs design work (AmirHossein, 2026-07-23). No demand today. Blocks TASK-16 (type-directed assembler) and DRAFT-10 (tfidf), both of which need the sparse cell type to exist first — so if either of those ever becomes wanted, this gets scoped first.

Context: doc-10 (feature-output model — records, dense, sparse).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 COO struct column type + CSR and dense materializers
- [ ] #2 N/unseen-key sourced from the fitted transform; shape pinned (n,N)
- [ ] #3 width-invariant assert fails on batch-width drift
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 01:01
---
Moved to Draft (2026-07-23): sparse COO column is super-low priority AND needs design work. Parked as a draft until scoped.
---
<!-- COMMENTS:END -->
