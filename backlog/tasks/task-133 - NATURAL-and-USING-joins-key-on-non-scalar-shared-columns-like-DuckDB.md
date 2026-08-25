---
id: TASK-133
title: >-
  NATURAL and USING joins key on non-scalar shared columns like DuckDB
status: Done
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
- [x] #1 the join-key semantics for struct and opaque shared columns
      are measured against DuckDB first (NULL fields, NULL structs,
      nested structs, and at least one non-struct opaque type), with a
      recorded matrix, before any key-encoding design
      (docs/superpowers/specs/2026-08-25-task-133-join-keys-design.md;
      discriminator cells independently re-verified)
- [x] #2 NATURAL JOIN keys on ALL shared columns; the TASK-127 pin
      (struct leg and TIMESTAMP leg) flips from xfail to passing
      (AMENDED by the split decision, AmirHossein 2026-08-25: STRUCT
      keys serve and that leg flipped and moved to real pins; OPAQUE
      scalar keys refuse by name until TASK-134 adds key-only lanes,
      so the TIMESTAMP leg is REWRITTEN as a named-refusal pin rather
      than flipped. The severity-2 -- wrong rows from silently dropped
      keys -- is dead in both arms.)
- [x] #3 `USING (w)` with a struct column serves like DuckDB, and the
      false "does not exist on right side of join!" refusal is gone
- [x] #4 a shared non-scalar column the key encoding still cannot carry
      (if any remain) refuses by name at build -- never silently drops
      from the key set
      (opaque columns, mismatched field sets, unlaneable fields, and
      struct keys over a duplicate-key static all refuse by name)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed per the spec plus the decisions section. A struct join key
expands at bind time into composite keys mirroring DuckDB's own
row-matcher structure: a PLAIN presence key for the top-level struct
(NULL struct never matches), a NOT-DISTINCT presence key per nested
node (presence is part of the key -- {inner: NULL} vs
{inner: {val: NULL}} is a miss despite identical leaf tuples), and a
NOT-DISTINCT key per leaf, paired by field path case-insensitively.
Presence lanes are minted lazily in the frontend, only on struct-keyed
joins. The NATURAL arm's silent skip of unkeyable shared columns (the
severity-2) is deleted; both arms refuse by name for what the encoding
cannot carry. Mutation-checked: dropping the nested presence key fails
exactly the discriminator cells; INDF at top level fails exactly the
NULL-struct cells.

Follow-ups split out on AmirHossein's word: TASK-134 (key-only lanes
so opaque scalar keys serve), TASK-135 (the fan-out loop learns
NOT-DISTINCT comparison so struct keys serve over duplicate-key
statics).

Gate: full root suite release AND debug, 3216 passed, 3 xfailed (the
two TASK-133 legs are gone -- one flipped to real pins, one rewritten
as a named refusal), cargo the 5 known pre-existing failures, 107
live-oracle pins in test_join_keys.py.
<!-- SECTION:NOTES:END -->
