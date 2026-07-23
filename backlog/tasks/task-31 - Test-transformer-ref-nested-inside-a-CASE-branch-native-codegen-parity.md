---
id: TASK-31
title: Test transformer ref nested inside a CASE branch (native + codegen parity)
status: Done
assignee:
  - Ritchie
created_date: '2026-07-19 15:35'
updated_date: '2026-07-23 00:53'
labels:
  - test
  - sql-surface
milestone: m-1
dependencies: []
references:
  - src/expr_build.rs
documentation:
  - 'doc-8 (Composition — {transform}(col) references)'
priority: medium
type: task
ordinal: 31000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up flagged by the TASK-27 final review (Minor, non-blocking). The resolve_transformers CASE arm — a transformer call nested inside a CASE branch (e.g. `CASE WHEN ... THEN {t}(col) ... END`) — is correct by inspection but has NO test, and its failure mode is silent: if the arm stopped recursing, the call would just never be rewritten to Expr::Transform (no error, wrong result). Its own small unit of work because it needs the transformer-callout fixtures rather than the plain CASE decision-table. Depends on nothing beyond the shipped CASE + transformer-ref machinery (TASK-27, TASK-30).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A transformer ref inside a CASE THEN (and at least one non-first branch) is exercised and asserts transform == infer_batch parity (DataFusion oracle, decision-1)
- [x] #2 Covers both engines (native + codegen), since resolve_transformers recursion is the shared concern
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 15:41
---
Dispatched to Ritchie now (bumped Low→Medium). AmirHossein escalated: testing is important, no test = not done — the silent-failure arm can't ship untested. Bar: the test must genuinely FAIL if resolve_transformers stopped recursing, i.e. place the transformer ref in a NON-first CASE branch (not just THEN of the first WHEN), asserting transform == infer_batch. Worktree-isolated, verified green before merge.
---

author: Iris (PM)
created: 2026-07-19 15:53
---
Done per Ritchie's merge cb8c350. Mutation-verified (broke the recursion, saw the test go red on non-first branches, restored) — meets the 'must catch the regression' bar exactly. Answering Ritchie's codegen question: your call is correct — transformers are native-only, so a native parity test + a codegen defers-loudly (UnsupportedInCodegen) assertion is the right shape of 'both engines'. Nothing to change; don't fabricate codegen computation for a native-only feature. AC #2 accepted under that corrected premise.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged to master (cb8c350). tests/test_transformer_case.py: a transformer ref (__tfm_0__) in a NON-first WHEN arm AND the ELSE of a CASE, asserting transform == infer_batch vs the DataFusion oracle, with rows hitting each branch (g=1 plain arm, g=2 transformer arm, g=3 transformer ELSE). Regression-catch VERIFIED by mutation: temporarily broke resolve_transformers to recurse only the first arm → test failed with 'Unknown function: __tfm_0__' on the g=2/g=3 rows → restored + reverified green, so it genuinely fails if the arm stops recursing into every branch. Full suite 486 passed / 16 skipped / 1 xfailed. Scope correction on 'both engines': transformers are a NATIVE-ONLY feature (codegen has no transformer support; SQLTransform runs on the native InferFn), so resolve_transformers is native-only — the AC's 'shared concern' premise was off. Correct resolution: native gets the real parity test; codegen is covered by asserting the same construct DEFERS loudly (UnsupportedInCodegen) rather than silently mishandling it. PM-accepted as the right call — don't pretend codegen computes a native-only feature.
<!-- SECTION:FINAL_SUMMARY:END -->
