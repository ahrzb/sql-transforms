---
id: TASK-3
title: Transformer-refs (Part-2 authoring surface) review follow-ups
status: Done
assignee:
  - Wren
created_date: '2026-07-18 13:44'
updated_date: '2026-07-25 01:23'
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

author: Claude (TASK-40)
created: 2026-07-25 01:23
---
CLOSED BY TASK-40 (2026-07-25). The five fixes that were completed landed on master (commits d47d2df..a83b742: bind names on both ref paths, probe in CALL order, nested refs declare the struct they receive, probe in fitted order + engine tolerance, shared _rows_equal).

ACs #1-#6 are now VOID, not deferred: TASK-40 removes the {transformer}(col) authoring surface entirely, so the double-.transform() probe, the feature_names_in_ footgun, the aggregate-over-output error, and the struct-output README note all describe code that no longer exists. Re-open a fresh ticket against the new surface if transformer support returns.

Important: this ticket carried a standing PRE-AUTHORIZATION to dispatch to Wren. Closing it retires that authorization -- do not dispatch.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Partially completed, then superseded by TASK-40 (remove transformer support).

LANDED (on master): probe/binding correctness fixes for the transformer-ref path -- bind names on both ref paths with an actionable error when unsettable; probe in CALL order so the UDF signature matches the named_struct the SQL builds; nested refs declare the struct they actually receive; probe in fitted order and compare engines with a tolerance; test cleanup to the shared _rows_equal helper.

NOT DONE, AND NOW VOID: ACs #1-#6. All six target the {transformer}(col) authoring surface, which TASK-40 deletes from both the Python front-end and the native interpreter. They are not deferred work -- the code they describe is gone.
<!-- SECTION:FINAL_SUMMARY:END -->
