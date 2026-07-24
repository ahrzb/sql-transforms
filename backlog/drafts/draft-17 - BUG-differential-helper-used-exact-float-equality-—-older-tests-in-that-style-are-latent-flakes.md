---
id: DRAFT-17
title: >-
  BUG differential helper used exact float equality — older tests in that style
  are latent flakes
status: Draft
assignee: []
created_date: '2026-07-24 01:59'
labels:
  - test-infra
  - parity
  - bug
  - flake
dependencies: []
references:
  - 'PR #16'
  - tests/test_transformer_ref.py
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT GOES WRONG
The `_both_engines` differential helper compared engine outputs with EXACT float equality. The batch path and the row-at-a-time path are not bit-identical — an ordinary non-collinear fixture differs by ~2e-16 between them. Existing tests passed only because their fixtures happened to land on values where the two paths agreed exactly.

So the test suite has been asserting "these engines agree" using a comparison that can fail on a fixture change no one would think twice about — adding a row, nudging a value, changing a column. The failure would look like a real parity regression and cost someone a debugging session before they realized the comparison itself was wrong.

WHY THIS IS RATED HIGH DESPITE BEING "JUST TESTS"
Our entire correctness story is differential testing against the DataFusion oracle (decision-1). A comparison primitive that passes on fixture luck weakens every assertion built on it, in both directions:
- FALSE ALARM: an innocuous fixture edit trips a 2e-16 difference and reads as a parity bug.
- FALSE CONFIDENCE, the worse one: a helper that is wrong about how to compare is a helper nobody should trust to have been catching real divergence.

Found by Wren during TASK-3 (2026-07-23). He fixed the instance in PR #16 because he was already adding tests to that file, but the fix was NOT swept across the codebase — any older test written in the same exact-equality style is still a latent flake.

SCOPE
This is the sweep TASK-3 deliberately did not do. Find every exact float comparison in the differential/parity test helpers and tests, and make them use an appropriate tolerance. The point is not to fix one file; it is to establish that no parity assertion anywhere is passing by fixture luck.

Worth deciding as part of this: what the project's standard float comparison for parity assertions IS, so the next person does not reinvent it. A single shared helper is better than a tolerance argument sprinkled at call sites.

IN SCOPE FOR THE CURRENT MILESTONE: this is squarely "bulletproof opaque transforms that actually work" — it is about whether our correctness net has holes, not about making anything faster.

DRAFT pending AmirHossein's review of scope/priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every exact float comparison in differential/parity test helpers and tests is identified (a sweep, not a single-file fix)
- [ ] #2 A single shared float-comparison helper with a documented tolerance is used for parity assertions, rather than per-call-site tolerances
- [ ] #3 Verified the sweep is real: a deliberately perturbed fixture (values changed but semantics identical) does not flip any parity test
- [ ] #4 Any test found to have been passing on fixture luck is called out, since it means that assertion was not actually proving what it claimed
<!-- AC:END -->
