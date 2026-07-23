---
id: TASK-2
title: Opaque transform Part-1 review follow-ups
status: Done
assignee:
  - Wren
created_date: '2026-07-18 13:44'
updated_date: '2026-07-23 13:09'
labels:
  - rust
  - parity
  - opaque
milestone: m-1
dependencies: []
documentation:
  - 'doc-7 (Transformer execution model — UDF/UDAF, macros, composition)'
priority: medium
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-ups from the Part-1 (engine capability) review. Split rationale + deferred direction: decision-3 (opaque-transform split).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Out-of-projection transformer calls: build-time guard/consistent resolve so Rust and DF agree
- [x] #2 Add a single-field 1-in/1-out transformer differential parity case (oracle y[:,i] assumes 2-D)
- [x] #3 Enforce or reconcile out_schema vs the transform natural output dtype
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-23 04:49
---
Dispatched to Wren (2026-07-23), AmirHossein's explicit go. Opaque-transform Part-1 engine-correctness hardening (3 ACs). Reminded Wren to use the superpowers skills (brainstorm → TDD with regression-catching tests → verification-before-completion, own worktree). Parity bugs → xfail-strict + ticket request, no inline fixes.
---

author: Iris (PM)
created: 2026-07-23 13:09
---
Verified against the diff (0cda202 + 2c9be9b), not the report. AC#1: _sql.py::require_in_projection raises a build-time ValueError; the reachable clause set was measured (QUALIFY/DISTINCT ON/CLUSTER BY/SORT BY/WINDOW survive parse_and_validate) rather than assumed — engines demonstrably disagreed before (fit accepted, transform raised, infer silently ignored). AC#2: test_single_field_one_in_one_out_parity, mutation-checked (a squeeze() breaks only this test, all 8 existing 2-in/2-out cases stay green). AC#3: _transformer_udf.py::check_out_schema_natural ENFORCES rather than reconciles — correct per decision-1, since reconciling means silent coercion. Scope extension to _compose.py ACCEPTED: the ref path had the identical hole, so guarding only the callout path would have left a sibling caller broken — root-cause fix, not creep. Wren reports 527 passed / 9 skipped / 4 xfailed; I verified the diff and tests, did not re-run the suite. No native parity bug found, so no ticket request.
---
<!-- COMMENTS:END -->
