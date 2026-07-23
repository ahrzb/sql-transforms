---
id: TASK-3
title: Transformer-refs (Part-2 authoring surface) review follow-ups
status: In Progress
assignee:
  - Wren
created_date: '2026-07-18 13:44'
updated_date: '2026-07-23 14:32'
labels:
  - python
  - transformer-refs
milestone: m-1
dependencies: []
documentation:
  - 'doc-8 (Composition — {transform}(col) references)'
  - doc-7 (Transformer execution model)
priority: medium
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
These are the papercuts on the shipped {transformer}(col) authoring surface — each one is a real thing a user runs into:

1. SILENT DOUBLE WORK. A user fits a transform with a single transformer ref and it runs sklearn's .transform() TWICE at fit time — once to derive the output schema, once for real. On an expensive transformer over a large training frame, they pay double for no reason and nothing tells them why fit is slow.

2. CRYPTIC ERRORS ON PREDICTABLE MISTAKES. Two mistakes users will absolutely make:
       SELECT AVG({scaler}(age)) OVER () FROM __THIS__   -- aggregating over a transformer's output
       SELECT {scaler}(age) FROM __THIS__                -- where scaler was never .fit()
   Today these surface as whatever the engine happens to throw deep in the stack, not as "you can't aggregate over a transformer output" / "this transformer isn't fitted."

3. THE feature_names_in_ FOOTGUN. This is the nastiest one. A transformer-ref needs feature_names_in_ to bind columns. But sklearn only sets it when the transformer was fit with NAMED columns — i.e. a DataFrame. Fit a OneHotEncoder on a numpy array and it silently has no feature_names_in_, so the ref fails at a confusing place. The user's mistake happened much earlier (in their sklearn code, not ours) and nothing connects the two. Options: document it loudly, or accept an explicit names argument so the footgun stops existing.

4. STRUCT OUTPUT IS A SURPRISE AT THE SKLEARN HANDOFF. Transformer-ref output is ONE Arrow struct column, not N flat columns. A user who wants to hand the result to a model has to flatten it, and nothing in the README shows that step — so their first attempt to feed our output into .fit() fails on shape.

WHAT THIS TICKET DOES
Fix 1 by reusing the _derive_schemas probe and skipping _materialize when nothing consumes the output. Fix 2 with friendly pre-check errors. Fix 3 by documenting the contract and considering an explicit names arg (API change — brainstorm with PM before building). Fix 4 with a README note/example showing the flatten step; this is near-term DX that the feature-output dense/assembler work (TASK-16) supersedes long-term.

Plus the negative/contract tests that pin all of it, and a regression test for transformer + PARTITION BY input-col.

Origin: follow-ups from the whole-branch review (which was ready-to-merge, no Critical/Important findings). Split rationale: decision-3 (opaque-transform split, Part 2).

Context: doc-8 (composition — {transform}(col) references), doc-7 (transformer execution model).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Single-ref path runs .transform() twice at fit; reuse the _derive_schemas probe, skip _materialize when no outer consumer
- [ ] #2 Friendly pre-check errors for aggregate-over-output and unfitted-transformer paths
- [ ] #3 Negative/contract tests: mixed leaf+nested args, aggregate-over-output, column vs feature_names_in_ mismatch, unfitted ref; + regression for transformer + PARTITION BY input-col
- [ ] #4 Confirmatory 3+ level nesting test (low value)
- [ ] #5 Document the feature_names_in_ contract: transformer-ref needs it; OneHotEncoder sets it only when fit with named columns (a DataFrame) -- else hand-assign obj.feature_names_in_ = names. Consider accepting an explicit names arg to remove the footgun.
- [ ] #6 README note/example: transformer-ref output is a single Arrow struct column; show the flatten step for the sklearn handoff (near-term DX; the feature-output model's dense output / assembler task TASK-16 supersedes it long-term).
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 04:54
---
PRE-AUTHORIZED (2026-07-23): AmirHossein green-lit dispatching TASK-3 to Wren after TASK-2 completes. PM dispatches on TASK-2 done (verify diff, close 2, then hand Wren TASK-3 with the superpowers reminder). No further approval needed.
---

author: Iris (PM)
created: 2026-07-23 13:09
---
Dispatched to Wren (2026-07-23) per the standing pre-authorization, on TASK-2 completion. 6 ACs — mostly DX/guardrails on the shipped authoring surface. Note AC#6 is a README/docs item and AC#4 is self-described low value; Wren to flag if any AC looks not-worth-it rather than padding it out.
---
<!-- COMMENTS:END -->
