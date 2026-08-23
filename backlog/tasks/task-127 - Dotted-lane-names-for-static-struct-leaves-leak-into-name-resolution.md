---
id: TASK-127
title: >-
  Dotted lane names for static struct leaves leak into name resolution
status: To Do
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 112000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-116 serves a static table's struct leaves by flattening them into
`parent.leaf` lane names. The encoding works for the case it was built for
(`SELECT s.w.mean`) and leaks everywhere else a NAME is resolved. All
severity-4 (we refuse where DuckDB serves), all measured 2026-08-18:

```sql
-- static s(id BIGINT, w STRUCT(mean DOUBLE, sd DOUBLE), z BIGINT)

SELECT s.* EXCLUDE (w) FROM __THIS__ JOIN s ON s.id = __THIS__.k
-- oracle: ['id','z']  [(1, 7)]
-- ours:   bind error: column "w" in EXCLUDE list not found in FROM clause
--         (`w` is no longer a lane name, so it cannot be excluded)

SELECT w.mean FROM ...   -- unqualified
-- oracle: 1.5
-- ours:   unknown table 'w'
```

And the encoding is ambiguous by construction. With a struct `w{mean}` AND a
literal column named `"w.mean"` in the same static table:

```sql
SELECT s.w.mean    -- oracle 1.5    ours: Could not find key "mean" in struct
SELECT s."w.mean"  -- oracle 99.0   ours: ambiguous column 'w.mean'
```

`flatten_static` guards against a `.` inside a FIELD name but not against a
flattened leaf colliding with a sibling column's real name.

Separate from TASK-121 (an ambiguous struct-path HEAD binding instead of
refusing) and from TASK-125 (star expansion), though all three trace to the
same flattening. Severity 4 throughout: everything here refuses rather than
serving a wrong value, so it is a cost, not a correctness hole.

Found by an independent review of the divergence-sweep branch, 2026-08-18.
The EXCLUDE case was reproduced by hand; the other two are relayed from that
review and NOT independently re-measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `EXCLUDE (w)` over a static struct column works, or refuses naming
      the struct rather than claiming the column does not exist
      (CLOSED BY TASK-125, 2026-08-19: the star's Opaque entry restores the
      name, so EXCLUDE takes it out and the rest serves -- pinned passing in
      test_arrow_schema_api.py)
- [ ] #2 an unqualified `w.mean` resolves like DuckDB, or refuses with a
      message that names the real problem
- [ ] #3 a flattened leaf colliding with a real sibling column name is
      detected -- at build, by name, not by a struct-key lookup failure
- [ ] #4 decide whether the dotted lane NAME is the right encoding at all,
      or whether the lane should carry a structured path and the dotted
      spelling stay a display detail; #3 is only cheap under the second
- [ ] #5 re-measure #2 and #3 before building -- they are relayed, not
      confirmed here
<!-- AC:END -->
