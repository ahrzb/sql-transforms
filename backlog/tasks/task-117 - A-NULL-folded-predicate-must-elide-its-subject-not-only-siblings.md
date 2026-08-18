---
id: TASK-117
title: >-
  A NULL-folded predicate must elide its subject, not only the NULL's siblings
status: Done
assignee: []
created_date: '2026-08-16 00:00'
labels:
  - m-8
dependencies: []
type: bug
ordinal: 102000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
```sql
SELECT 1 AS o FROM __THIS__
WHERE CAST(s AS DOUBLE) BETWEEN 61.591e0 AND NULL     -- s = 'abc'
```

DuckDB returns no rows: the predicate is statically NULL, so the subject is
never evaluated. We evaluate the subject first and trap with "could not cast
VARCHAR to DOUBLE".

TASK-85 closed the adjacent case and its ACs say so precisely — "a strict
operator with a statically-NULL operand folds to NULL at build time ... so
its SIBLINGS never evaluate". Here the trapping expression is not a sibling
of the NULL; it is the thing being compared. The fold has to take the whole
comparison, subject included.

This is the last live member of the 2026-08-13 triage's §2 (seed 1667).
Re-measured 2026-08-16: the other two members of that section (7560, 11473)
and §4's 19788 now all match, so TASK-85 and TASK-88 really are closed — the
triage's scoreboard simply listed three tickets as To Do that were Done, and
this one residual hid behind that.

Pinned xfail-strict as `test_a_null_folded_predicate_elides_its_subject_too`
in test_open_divergences.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a predicate that folds to a constant NULL elides its SUBJECT as well
      as the NULL's siblings, so a trapping subject never runs
- [ ] #2 BETWEEN is covered specifically, since it desugars to two
      comparisons and the subject appears in both
- [ ] #3 `plan::may_trap` stays the single definition of "can this trap"
      (TASK-74/75's rule, restated by TASK-85 AC #2)
- [ ] #4 a predicate with NO NULL to fold keeps its trap — the guard must not
      become a blanket suppression
- [ ] #5 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RETIRED, not implemented. The fold this ticket asked for was written on
2026-08-16 and deleted the next day, because it reproduces DuckDB's
`statistics_propagation` optimizer pass rather than DuckDB.

The oracle is now DuckDB with `PRAGMA disable_optimizer`, and there the
subject IS evaluated:

    WHERE CAST(s AS DOUBLE) BETWEEN 61.591e0 AND NULL   -- s = 'abc'
    optimizer off: Conversion Error
    optimizer on:  []

So the divergence this ticket recorded does not exist against the oracle, and
eliding the filter would have been the bug. The measurement that made the
optimizer-on reading unusable is pinned in
known_divergences/test_trap_elision.py: its answer depends on the column's
stored null statistics, so the same query over the same visible rows answers
differently by insert history.

Nothing to build. The xfail pin is deleted with this ticket.
<!-- SECTION:NOTES:END -->
