---
id: TASK-131
title: >-
  A NULL arm pins a CASE's width floor at INTEGER on DuckDB
status: Done
assignee: []
created_date: '2026-08-19 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 116000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Surfaced when TASK-130's oracle fix took the schema blindfold off seed
12745 (the schema-name mismatch used to return early, hiding the width
compare):

```sql
SELECT (CASE WHEN TRUE THEN NULL WHEN TRUE THEN c0 ELSE -22 END) AS o
-- c0 SMALLINT.   duck: INTEGER      ours: int16
```

A bare-NULL arm contributes SQLNULL -- INTEGER -- to DuckDB's CASE result
unification, so the result floor is int32 even though every VALUE arm fits
int16. Our unification adopts the NULL into the value arms' width instead.
Same family as the SQLNULL channels of TASK-101/102/103 (bind-time width
facts), one node higher.

Not measured yet: whether the floor applies through COALESCE/NULLIF
desugars, whether a TYPED null (CAST(NULL AS SMALLINT)) floors differently
(it should keep its width per the earlier unary measurements), and what
happens at int8. Measure first, then re-key the CASE unification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 measure the rule: bare NULL vs typed NULL arms, all widths,
      COALESCE/NULLIF desugars, nested CASE
- [x] #2 the CASE unification reproduces it; seed 12745 classifies clean
- [x] #3 live-oracle schema pins for the measured grid
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured 2026-08-24 (grid in test_integer_widths.py) and source-verified
against the v1.5.5 checkout. The ticket's premise was WRONG in an
instructive way: a bare NULL arm does NOT floor anything by itself
(CASE WHEN.. i16 ELSE NULL is SMALLINT). The real rule, from
bind_case_expression.cpp + types.cpp:

- CASE types by folding ELSE first, then THEN arms in order, through
  TryGetMaxLogicalType.
- Max(SQLNULL, X) = NormalizeType(X), and NormalizeType strips an
  INTEGER_LITERAL to its base type (INTEGER for -22). A literal that
  meets a NULL loses its shrink-to-fit BEFORE the column can absorb it;
  that is the whole "floor". Order-dependent, hence the seed 12745 shape
  (NULL, c0, ELSE -22 -> INTEGER) vs (c0, NULL, ELSE -22 -> SMALLINT).
- COALESCE folds the same rule but seeds from its first argument;
  coalesce(NULL, -22, i16) and coalesce(-22, NULL, i16) floor,
  coalesce(NULL, i16, -22) does not.

Fix: the CASE fold's NULL arm now clears the accumulator's literal hint
instead of skipping (frontend.rs case()); the coalesce fold keeps NULL
args as SQLNULL fold events (hint stripped before/right after the seed)
while still dropping them from evaluation. 30-expression live-oracle grid
pinned; seed 12745 classifies AGREE; the open-divergences pin flipped
strict and was deleted.
<!-- SECTION:NOTES:END -->
