---
id: TASK-74
title: >-
  A trap-free CASE in a one-sided ON residual is refused as containing trapping ops
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - frontend
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 67000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`scan_residual` unconditionally sets `total = false` for `SKind::Case` without
looking at the arms, and `bind_residual` then rejects any one-sided residual
that is not `total`.

```sql
ON k = id AND (CASE WHEN n > 1 THEN 1 ELSE 0 END) = 1
  -> unsupported: JOIN ON condition '...' (single-side residual with trapping
     ops: DuckDB's scan-pushed evaluation order differs)
```

Every arm is an integer literal. There is no arithmetic, no cast, no division
anywhere in the expression. COALESCE and NULLIF desugar to CASE, so they are
refused the same way. Two-sided residuals take the permissive path.

Loud, not wrong - but it names a trapping op the expression does not contain,
and it rejects ordinary SQL.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A one-sided ON residual whose CASE arms are all total builds and
      matches DuckDB
- [x] #2 A residual whose arms genuinely DO trap is still refused
- [x] #3 COALESCE and NULLIF covered, being CASE in disguise
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The recursion into arms already exists; per the finder only the unconditional
`*total = false` needs to go, so that a CASE is total exactly when all its arms
are. Verify that claim against the source before trusting it - and be careful
that the guard exists for a real reason (DuckDB's scan-pushed evaluation order),
so the fix must keep refusing genuinely trapping arms.
<!-- SECTION:NOTES:END -->

## Resolution (2026-08-08)

The finder's claim — "only the unconditional `*total = false` needs to go" —
checked out, but the fix went one step further because TASK-75 needed the same
property and two copies of a trapping-op allowlist would have drifted.

`scan_residual` no longer computes `total` at all. Trap-freeness is now
`plan::may_trap`, one definition, called from `bind_residual` here and from
`FB::kleene` in lowering. `scan_residual` keeps only the questions that are
genuinely about scope: does the residual touch this join, another one, and was
every node classifiable.

A CASE is trap-free exactly when every arm and the default are — lowering
branches, so an arm that is not taken never runs, and if nothing can trap then
evaluation ORDER is unobservable, which is the only thing the wave-4 rule was
protecting.

AC #2 is a real test, not an assumption: a CASE that traps in an arm, in the
condition, or in the ELSE is still refused, as is bare arithmetic.
