---
id: DRAFT-18
title: >-
  Transformer-ref misuse should fail early and name what the user wrote (4
  error-quality gaps)
status: Draft
assignee: []
created_date: '2026-07-24 02:00'
labels:
  - transformer-refs
  - usability
  - error-messages
dependencies: []
references:
  - 'PR #16'
  - sql_transform/_transformer_ref.py
  - sql_transform/_compose.py
documentation:
  - 'doc-8 (Composition — {transform}(col) references)'
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four pre-existing error-quality gaps on the transformer-ref surface, all found by Wren during TASK-3 (2026-07-23). None blocked that ticket; all are the same class, in the same code area, and would likely be one sitting's work — hence one ticket rather than four.

THE COMMON THREAD: when a user misuses a transformer ref, the failure either arrives too late (run time instead of build time) or names something the user never wrote. Both leave the user staring at an error that does not point at their SQL.

THE FOUR CASES

1. DUPLICATE COLUMN, ndarray-fit transformer

       SELECT {sc}(age, age) AS out FROM __THIS__     -- sc fitted on an ndarray

   Errors late, and the message names nothing the user wrote. They see internal vocabulary, not "you passed `age` twice."

2. TRANSFORMER WITHOUT get_feature_names_out

       SELECT {FunctionTransformer(...)}(x) FROM __THIS__

   A raw AttributeError escapes from inside fit(). FunctionTransformer is a perfectly ordinary thing to reach for, so this is not an exotic path. The user gets a Python internals traceback instead of "this transformer does not expose output feature names; the ref surface needs them."

3. WINDOW-AGG GUARD MESSAGE IS WRONG

       SELECT AVG(income) OVER (PARTITION BY {sc}(age)) FROM __THIS__

   Rejecting this is CORRECT. The message misdescribes why — it describes a different construct than the one written. So the user is told "no" for a reason that does not match their query.

4. NON-WINDOW AGGREGATE OVER A REF FAILS AT RUN TIME, NOT BUILD TIME

       SELECT AVG({sc}(x)) FROM __THIS__

   Fails on both engines, but at run time. This is statically detectable at build time, and build-time rejection is the established pattern for this surface (see the TASK-2 projection guard, which rejects out-of-projection refs at build time precisely so the engines cannot disagree later).

WHY THIS BELONGS IN THE CURRENT MILESTONE
m-1's goal is opaque transforms that are bulletproof and hard to misuse. Every one of these is a case where the surface accepts or mishandles a mistake instead of naming it. None of them make anything faster — they make the thing that already works harder to get wrong, which is exactly the milestone's stated test.

NOT INCLUDED: the exact-float-equality test-infra flake Wren also reported. That one is its own draft (different risk, different fix, rated higher).

DRAFT pending AmirHossein's review of scope/priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 {sc}(age, age) on an ndarray-fit transformer fails with a message naming the duplicated column the user wrote
- [ ] #2 A transformer lacking get_feature_names_out is rejected with an explanatory error, not a raw AttributeError from inside fit()
- [ ] #3 The window-agg guard's message accurately describes the construct actually written (AVG(x) OVER (PARTITION BY {ref})), not a different one
- [ ] #4 AVG({ref}(x)) is rejected at BUILD time rather than failing at run time, consistent with the TASK-2 projection-guard pattern
- [ ] #5 Each new guard has a test that fails if the guard is removed (mutation-checked, not just present)
<!-- AC:END -->
