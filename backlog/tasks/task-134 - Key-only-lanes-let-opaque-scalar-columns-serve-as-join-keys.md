---
id: TASK-134
title: >-
  Key-only lanes let opaque scalar columns serve as join keys
status: To Do
assignee: []
created_date: '2026-08-25 00:00'
labels:
  - m-8
  - parity
dependencies:
  - TASK-133
type: feature
ordinal: 119000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The second half of the TASK-133 split (decided by AmirHossein,
2026-08-25): TASK-133 made NATURAL/USING joins serve STRUCT keys and
refuse OPAQUE scalar keys by name -- "shared join column 't' has a
non-scalar type on row table '__THIS__', so this engine cannot key on
it". DuckDB serves those joins (no type check in its join binder), so
every such refusal is a severity-4 cost this task removes.

The lift is a new lane KIND: servable-as-key, not-as-value. A
TIMESTAMP shared column can be compared for join equality (its Arrow
physical representation is an exact integer) even though the engine
refuses to serve its VALUE. That kind must thread through schema.rs
(the opaque classification), in_cols, star expansion (a key-only lane
never appears in output), the IR verifier, arrow ingest, and both
marshallers -- plus a per-type equality proof for each admitted type:
timestamp units are exact ints, date32 is an exact int, float32
injects into f64, uint64 needs a bit-reinterpretation to compare in a
signed lane, timezone-carrying types need their own answer or a named
refusal.

The TASK-133 named-refusal pin for the TIMESTAMP leg (packages/confit/
tests/test_join_keys.py) flips to a serving pin when this lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 each admitted opaque type gets a measured equality matrix
      against optimizer-off DuckDB (equal / unequal / NULL one side /
      NULL both sides, NATURAL and USING and explicit ON) before its
      lane lands, and each type's comparison is argued exact or the
      type refuses by name
- [ ] #2 a NATURAL or USING join over a shared TIMESTAMP column serves
      like DuckDB; the TASK-133 named-refusal pin flips to a serving
      live-oracle pin
- [ ] #3 a key-only lane never appears in star expansion, output
      schema, or value resolution -- projecting the column still
      refuses with the existing non-scalar message
- [ ] #4 types with no exact comparison story (if any remain) keep a
      refusal that names the column and the reason
<!-- AC:END -->
