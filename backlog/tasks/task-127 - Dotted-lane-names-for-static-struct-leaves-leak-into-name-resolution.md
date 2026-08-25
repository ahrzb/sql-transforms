---
id: TASK-127
title: >-
  Dotted lane names for static struct leaves leak into name resolution
status: Done
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - m-8
  - parity
dependencies:
  - TASK-132
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
- [x] #2 an unqualified `w.mean` resolves like DuckDB, or refuses with a
      message that names the real problem
      (CLOSED 2026-08-25: bare heads see the whole scope, qualified heads
      backtrack exactly where DuckDB's binder does, ambiguity refuses
      before any field is examined -- live-oracle pinned, spec
      packages/confit/docs/specs/2026-08-25-task-127-remainders-design.md)
- [x] #3 a flattened leaf colliding with a real sibling column name is
      detected -- at build, by name, not by a struct-key lookup failure
      (CLOSED 2026-08-25: vacuous on the static side -- under TASK-132
      both spellings are different references and SERVE, so there is
      nothing left to collide; on the row side the stale IR name-
      uniqueness invariant was the real bug, turning the collision into
      "internal specializer bug" for every query over the table. The
      in-side duplicate check moved to build, over plain row columns
      only, where a name is still an identifier; a genuine duplicate
      plain column refuses by name, and the collision table serves both
      spellings like the static side)
- [x] #4 decide whether the dotted lane NAME is the right encoding at all,
      or whether the lane should carry a structured path and the dotted
      spelling stay a display detail; #3 is only cheap under the second
      (DECIDED 2026-08-25: structured path, RFC
      packages/confit/docs/rfcs/2026-08-19-static-struct-lane-encoding.md alternative A;
      the refactor is TASK-132, and #2/#3 land on top of it)
- [x] #5 re-measure #2 and #3 before building -- they are relayed, not
      confirmed here
      (RE-MEASURED 2026-08-19, recorded in the RFC: DuckDB serves both
      collision spellings and the unqualified reference; our refusals
      and wrong-reason messages confirmed)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed per the spec (packages/confit/docs/specs/2026-08-25-task-127-
remainders-design.md), on top of TASK-132's structured paths. The
unqualified ladder now mirrors DuckDB's source-pinned order: a
qualified head backtracks to the next rung exactly when the relation
matched but its column half missed (never past an ambiguity), and a
bare head collects candidates over the whole scope, refusing ambiguous
heads before any field is examined. The row-side collision fix removed
the stale IR in-side name-uniqueness check (display names stopped
being identifiers in TASK-132) and re-homed a real duplicate-plain-
column check at build, by name.

Also in this change, on AmirHossein's word: the two message-only
divergences align with DuckDB's wording (EXCLUDE names the scope it
searched, with DuckDB's own qualified/unqualified split; the
not-a-struct refusal enumerates struct, union, map, or json).

Out of scope, discovered and pinned: NATURAL JOIN drops non-scalar
shared columns from the key set and serves rows DuckDB does not
(severity 2). xfail-strict pinned in test_open_divergences.py, two
legs; TASK-133 owns the fix, direction decided: support, not refuse.
<!-- SECTION:NOTES:END -->
