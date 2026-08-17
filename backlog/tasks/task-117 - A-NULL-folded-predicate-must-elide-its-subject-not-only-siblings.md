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
- [x] #1 a predicate that folds to a constant NULL elides its SUBJECT as well
      as the NULL's siblings, so a trapping subject never runs
- [x] #2 BETWEEN is covered specifically, since it desugars to two
      comparisons and the subject appears in both
- [x] #3 `plan::may_trap` stays the single definition of "can this trap"
      (TASK-74/75's rule, restated by TASK-85 AC #2)
- [x] #4 a predicate with NO NULL to fold keeps its trap — the guard must not
      become a blanket suppression
- [x] #5 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decided at the FILTER, not at the operator, which is why BETWEEN needed no
special case (AC #2 falls out): a bound WHERE predicate with a
statically-NULL TOP-LEVEL CONJUNCT is replaced with constant FALSE, so every
other conjunct -- including the trapping subject BETWEEN leaves behind after
TASK-85 folds one of its two comparisons -- is dropped before lowering.

Measured on DuckDB 1.5.5 before implementing, and the shape of the rule comes
from the measurements rather than from the ticket:

  WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND NULL   rows=[]
  WHERE (CAST(s AS DOUBLE) > 1) AND NULL            rows=[]
  WHERE NULL AND (CAST(s AS DOUBLE) > 1)            rows=[]
  WHERE (CAST(s AS DOUBLE) > 1) OR NULL             TRAP
  SELECT (CAST(s AS DOUBLE) > 1) AND NULL           TRAP

So it is AND-only and FILTER-only -- the same split TASK-87 face C found for
the dead-range BETWEEN. `null_conjunct` stops recursing at anything that is
not an AND, so `(A AND NULL) OR B` keeps its trap.

AC #3 holds by construction: nothing here touches `plan::may_trap`. AC #4 is
pinned by four controls in the same parametrize (OR, a live range, a bare
comparison, and the projection form), all asserted against the live oracle.

Pin moved to known_divergences/test_trap_elision.py, directly under the proof
that bounds the class, and that proof's "why we still fix this one" paragraph
updated to past tense. The stopping rule it states is unchanged and still
governs: a SECOND fold-visibility mismatch this mechanism does not cover means
refusing at build instead of chasing DuckDB's folder.
<!-- SECTION:NOTES:END -->
