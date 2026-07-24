---
id: DRAFT-21
title: >-
  native: dispatch the array(...) function alias (make_array's twin, left out of
  TASK-37)
status: Draft
assignee: []
created_date: '2026-07-24 21:19'
labels:
  - native
  - parity
  - containers
  - sql-surface
dependencies: []
references:
  - src/expr_build.rs
  - 'PR #20'
  - TASK-37
documentation:
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: low
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS
    SELECT array(lat, lon) AS coords FROM __THIS__
DataFusion accepts array(...) as a builtin alias for make_array(...). Native raises 'Unknown function: array' — works on transform(), fails on infer(). Same loud-failure shape as the make_array gap TASK-37 just closed; array() is simply the alias TASK-37 deliberately did not pick up.

WHY IT'S A SEPARATE DRAFT, NOT PART OF TASK-37
During TASK-37 Wren measured array() and found it is NOT free — it needs its own dispatch line in convert_function plus a parity test. Per the standing rule (do not add untested surface for completeness), it was left out of TASK-37 rather than bundled in unmeasured. Trivial-but-not-free, by Wren's own estimate.

WHY LOW / DRAFT
Loud failure (raises, not silent), the make_array spelling and the bracket literal [a, b] both already work as alternatives, and there is no demonstrated demand for the array() spelling specifically. Parked as a draft; promote if demand appears or if a 'native container parity, finish the long tail' pass is scoped.

Fix is small and known: mirror the make_array dispatch TASK-37 added, keyed on the 'array' function name, routing to the same Expr::List construction path. Add a parity test against the oracle.

DRAFT pending AmirHossein's review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native convert_function dispatches array(...) to the same Expr::List path make_array uses
- [ ] #2 A differential parity test for array(a, b) passes on both engines vs the DataFusion oracle
- [ ] #3 Confirm array() and make_array() produce identical results (they are aliases in the oracle)
<!-- AC:END -->
