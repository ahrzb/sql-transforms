---
id: TASK-3
title: Transformer-refs (Part-2 authoring surface) review follow-ups
status: Done
assignee:
  - Wren
created_date: '2026-07-18 13:44'
updated_date: '2026-07-24 14:18'
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
- [x] #1 Single-ref path runs .transform() twice at fit; reuse the _derive_schemas probe, skip _materialize when no outer consumer
- [x] #2 Friendly pre-check errors for aggregate-over-output and unfitted-transformer paths
- [x] #3 Negative/contract tests: mixed leaf+nested args, aggregate-over-output, column vs feature_names_in_ mismatch, unfitted ref; + regression for transformer + PARTITION BY input-col
- [x] #4 Confirmatory 3+ level nesting test (low value)
- [x] #5 Document the feature_names_in_ contract: transformer-ref needs it; OneHotEncoder sets it only when fit with named columns (a DataFrame) -- else hand-assign obj.feature_names_in_ = names. Consider accepting an explicit names arg to remove the footgun.
- [x] #6 README note/example: transformer-ref output is a single Arrow struct column; show the flatten step for the sklearn handoff (near-term DX; the feature-output model's dense output / assembler task TASK-16 supersedes it long-term).
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

author: Iris (PM)
created: 2026-07-24 02:00
---
Wren delivered as PR #16 (12 commits, 7 files, no src/*.rs, no codegen files — scope verified). All 6 ACs claimed met; suite 561 passed / 5 skipped / 5 xfailed. NOT closing yet — the PR is open and approval/merge is AmirHossein's. Will verify ACs against the merged diff before marking Done.

Two AC assumptions the work overturned, both worth recording:
- AC#5's documented workaround (obj.feature_names_in_ = ["a","b"]) was BROKEN AS WRITTEN — native calls .tolist(), so a plain list raises. Rather than document a working incantation, Wren removed the need for it: names are synthesised from the call site onto a copy.copy(). Better outcome than the AC asked for.
- AC#4 (3-level nesting) was NOT the low-value confirmatory test the ticket assumed. After AC#1 it is the only shape where a ref is both consumed and a consumer — exactly what the new materialisation logic keys on. I had flagged AC#4/#6 as possible busywork and invited Wren to drop them; he correctly pushed back on evidence. Ticket assumption was wrong, not the work.

The review loop caught THREE engine-divergence bugs, all one signature: an input-binding order/identity assumption differing between engines, invisible because every existing test used the aligned case. Twice the fix introduced the next one. Recurring worst shape: fit() accepts, DataFusion refuses to plan, native happily returns rows. Closed by enumerating every order/name/identity decision across both engines + 80 randomised trials (depth 1-4, random call-order permutations, mixed DataFrame/ndarray fit) — 0 divergences, 0 mutated user objects. 12 fixes mutation-checked individually.

5 pre-existing findings spun out: DRAFT-17 (exact-float-equality latent flake, High) and DRAFT-18 (4 error-quality gaps, Medium).
---

author: Iris (PM)
created: 2026-07-24 14:18
---
CLOSED against the merged diff (2026-07-24). PR #16 merged at 14:17Z, master a83b742. Verified in the merged tree, not off the report:
- AC#2 friendly errors: test_aggregate_over_transformer_output_raises, test_unsettable_feature_names_gives_actionable_error
- AC#3 negative/contract + PARTITION BY regression: test_ndarray_fit_arity_mismatch_raises, test_named_fit_column_mismatch_still_raises, test_transformer_alongside_partitioned_window_agg
- AC#4 (the one I wrongly called low-value): test_three_level_nesting_parity present and load-bearing
- AC#5 reframed: test_unsettable_feature_names_gives_actionable_error — the footgun is removed (names synthesised onto a copy.copy()) rather than documented
- AC#6 README: transformer-ref/struct-column/flatten content present
- The divergence class is pinned by test_nested_outer_fitted_in_permuted_order_parity, the shape the 3 review-caught bugs shared.

25 tests in test_transformer_ref.py. This is the ticket TASK-35 was spun out of — TASK-35 makes that whole permuted-order divergence class UNREPRESENTABLE rather than tested-against, and is now In Progress with Wren.
---
<!-- COMMENTS:END -->
