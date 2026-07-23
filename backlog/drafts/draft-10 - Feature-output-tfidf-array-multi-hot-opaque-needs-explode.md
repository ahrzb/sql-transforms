---
id: DRAFT-10
title: 'Feature output: tfidf / array multi-hot (opaque, needs explode)'
status: Draft
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 14:31'
labels:
  - feature-output
milestone: m-1
dependencies:
  - DRAFT-9
documentation:
  - 'doc-10 (Feature-output model — records, dense, sparse)'
  - doc-9 (Rich type system and UNNEST)
priority: low
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You have a free-text column — a listing description, a support ticket body, a product title — and you want it as features:

    tfidf = TfidfVectorizer().fit(train["description"])
    SELECT {tfidf}(description) AS text_feats FROM __THIS__

One input row produces a wide sparse vector over the learned vocabulary (thousands of terms). Same for array multi-hot: a `tags` column holding ["garage", "pool", "corner_lot"] expands to one indicator per known tag.

This is the shape the current engine cannot express. Every other transformer we compile is FIXED FAN-OUT — one input value maps to a known number of output columns decided at fit time, and the expression stays per-row. tfidf needs variable expansion (an explode), so it does not fit the scalar/window-agg surface the fast path is built on.

Practical consequence for the user: text features are the one preprocessing step that will NOT fuse into the single native serving expression. They run through the opaque callout, which means a Python interpreter in the loop and no Go/Java/WASM serving for a pipeline containing them.

WHAT THIS TICKET DOES
Route tfidf and array multi-hot onto the shipped opaque-transform mechanism (decision-3), corpus-gated. Deliberately NOT trying to make them native — accepting the opaque path is the correct trade for a fundamentally variable-width operation.

Distinct from DRAFT-9 (sparse COO column) and TASK-15 (scalar one-hot), which are fixed-fanout and composable — those stay on the fast path. This is the escape hatch for the ones that genuinely cannot.

WHY IT IS A DRAFT
Very low priority (AmirHossein, 2026-07-23) and depends on DRAFT-9: tfidf output is sparse, so it needs the sparse-column materializer path to exist before it is buildable. Re-promote only if text features get actual demand.

Context: doc-10 (feature-output model), doc-9 (rich type system and UNNEST).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 tfidf + array multi-hot available via the opaque-transform path
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 04:46
---
Parked as Draft (2026-07-23): tfidf / array multi-hot is very low priority. Depends on the sparse-COO column work (DRAFT-9) — tfidf output is sparse and needs that materializer path before it's buildable. Re-promote once DRAFT-9 is scoped/done and there's demand.
---
<!-- COMMENTS:END -->
