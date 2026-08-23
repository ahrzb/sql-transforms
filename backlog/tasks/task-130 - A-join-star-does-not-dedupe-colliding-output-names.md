---
id: TASK-130
title: >-
  A join star does not dedupe colliding output names
status: To Do
assignee: []
created_date: '2026-08-19 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 115000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 4000-seed campaign after TASK-124's grammar widening shifted
the rng bands (seed 380) -- PRE-EXISTING, newly reached; nothing on the
selection-context branch touches star naming.

```sql
-- row __THIS__(c0, c1, c2), static s0(c0, c1)
SELECT * FROM __THIS__ LEFT JOIN s0 ON (c2 = s0.c0)
-- DuckDB output names: c0, c1, c2, c0_1, c1_1
-- ours:                c0, c1, c2, c0,   c1
```

DuckDB dedupes colliding output names positionally with `_1` (presumably
`_2`, ... on further collisions -- measure). We emit the duplicate names,
which is a column set DuckDB never produces (severity 2 by name, though the
VALUES are all present and ordered correctly).

Dict-shaped `infer_rows` output makes duplicate names actively lossy: the
second `c0` key overwrites the first in the row dict. `infer_arrow` carries
both but with the wrong (duplicate) names.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 measure DuckDB's dedup rule properly (three-way collisions,
      qualified stars, USING-merged keys, EXCLUDE interaction) before
      implementing -- the `_1` suffix is one observation, not the rule
- [ ] #2 a join star's output names match DuckDB's exactly, or the collision
      refuses by name -- never duplicate keys in a row dict
- [ ] #3 the campaign's seed-380 class is gone
<!-- AC:END -->
