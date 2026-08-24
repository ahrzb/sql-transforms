---
id: TASK-131
title: >-
  A NULL arm pins a CASE's width floor at INTEGER on DuckDB
status: To Do
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
- [ ] #1 measure the rule: bare NULL vs typed NULL arms, all widths,
      COALESCE/NULLIF desugars, nested CASE
- [ ] #2 the CASE unification reproduces it; seed 12745 classifies clean
- [ ] #3 live-oracle schema pins for the measured grid
<!-- AC:END -->
