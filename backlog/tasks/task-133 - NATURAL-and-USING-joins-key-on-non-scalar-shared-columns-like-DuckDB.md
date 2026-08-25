---
id: TASK-133
title: >-
  NATURAL and USING joins key on non-scalar shared columns like DuckDB
status: To Do
assignee: []
created_date: '2026-08-25 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: bug
ordinal: 118000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Severity 2, found and verified during the TASK-127 measurement pass
(2026-08-25). The NATURAL arm iterates only the scalar column list, so
every shared column the engine cannot serve as a scalar -- struct heads
AND opaque columns like TIMESTAMP -- is silently dropped from the key
set, and we emit rows DuckDB never produces:

```sql
-- row (id BIGINT, w STRUCT(mean DOUBLE)), static s(id, w, z), ids equal, w NOT equal
SELECT z FROM __THIS__ NATURAL JOIN s
-- DuckDB: []          (joins on id AND w)
-- ours:   [{'o': 7}]  (joins on id alone)
```

Reproduces identically with a shared TIMESTAMP instead of the struct,
so it is every opaque column, not a struct thing. Predates TASK-132.

Direction decided by AmirHossein, 2026-08-25: SUPPORT these joins --
key on the non-scalar shared columns like DuckDB does -- do not refuse
them. DuckDB has no type check in its join binder (bind_joinref.cpp
intersects name sets and emits ordinary comparisons; struct falls
through TryBindComparison), so `USING (w)` with a struct key also
serves there while we refuse it with a false "does not exist on right
side of join!". Same machinery, same fix.

Measure first: DuckDB's join comparison semantics for nested types are
NOT plain scalar equality composed field-wise -- NULL handling inside
nested comparisons differs from top-level `=` -- and our KeyBits key
encoding is scalar-only today. Pin the oracle's behavior for NULL
fields, NULL structs, and nested structs before designing the key
encoding extension.

An xfail-strict pin for the severity-2 lands with TASK-127
(packages/confit/tests/test_open_divergences.py); flip it when this
fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the join-key semantics for struct and opaque shared columns
      are measured against DuckDB first (NULL fields, NULL structs,
      nested structs, and at least one non-struct opaque type), with a
      recorded matrix, before any key-encoding design
- [ ] #2 NATURAL JOIN keys on ALL shared columns; the TASK-127 pin
      (struct leg and TIMESTAMP leg) flips from xfail to passing
- [ ] #3 `USING (w)` with a struct column serves like DuckDB, and the
      false "does not exist on right side of join!" refusal is gone
- [ ] #4 a shared non-scalar column the key encoding still cannot carry
      (if any remain) refuses by name at build -- never silently drops
      from the key set
<!-- AC:END -->
