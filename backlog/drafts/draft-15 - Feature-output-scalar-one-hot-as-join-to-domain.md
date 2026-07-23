---
id: DRAFT-15
title: 'Feature output: scalar one-hot as join-to-domain'
status: Draft
assignee: []
created_date: '2026-07-18 15:52'
updated_date: '2026-07-23 16:28'
labels:
  - feature-output
dependencies: []
documentation:
  - 'doc-10 (Feature-output model â€” records, dense, sparse)'
priority: medium
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
You want to one-hot a categorical column â€” the single most common preprocessing step there is:

    SELECT {ohe}(house_style) AS style FROM __THIS__

Today this goes through the opaque sklearn callout: a Python boundary crossing per serving call, and a step that cannot run on the Go/Java/WASM runtimes. For an operation that is conceptually just "is this value equal to each known category," that is a lot of machinery.

The user-visible payoff of this ticket is that a one-hot stops being a black box and becomes part of the fused per-row expression â€” so single-row serving gets fast, and the pipeline stays portable.

WHAT THIS TICKET DOES
Compile a SCALAR one-hot into a JOIN against the fitted category domain table, riding the existing static_tables / lookup mechanism the engine already has (the same machinery TargetEncoder-style PARTITION BY work uses).

Why a join and not an explode: this is the FIXED-FANOUT case. The category list is frozen at fit time, so the output width is known and the operation stays 1:1 with rows â€” no row-cardinality change, nothing to explode, and it composes with everything else in the same SELECT. That is what makes it cheap and why it is separable from the hard cases.

Unknown categories at serving (a `house_style` never seen during fit) resolve through the join as a miss, which is the natural place to express sklearn's handle_unknown behavior.

SCOPE BOUNDARY
This is deliberately the EASY half of one-hot. The variable-expansion cases â€” tfidf, array multi-hot â€” need an explode and are parked separately (DRAFT-10). The sparse representation for high-cardinality columns is also separate (DRAFT-9). This ticket is the composable, fast-path slice that needs neither.

Independent: nothing blocks it, and it does not depend on the drafted spine.

Context: doc-10 (feature-output model â€” records, dense, sparse).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 scalar one-hot compiles to a join-to-domain, no explode
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 16:28
---
Moved to Draft (2026-07-23, AmirHossein). This is an OPTIMIZED SKLEARN TRANSFORM implementation â€” compiling one-hot down to a join-to-domain instead of the opaque callout. That class of work is getting its own milestone, which does NOT exist yet by explicit decision. Parked here until that milestone is created and this is scoped into it. Sibling drafts in the same class: DRAFT-7 (native per-transformer swap).
---
<!-- COMMENTS:END -->

