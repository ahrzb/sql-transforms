---
id: TASK-135
title: >-
  The fan-out loop learns NOT-DISTINCT comparison
status: To Do
assignee: []
created_date: '2026-08-25 00:00'
labels:
  - m-8
  - parity
dependencies:
  - TASK-133
type: feature
ordinal: 120000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The third TASK-133 decision (AmirHossein, 2026-08-25): struct join
keys serve under the map and filter shapes (unique static keys) and
REFUSE BY NAME under the fan-out loop -- shape='many', a static whose
key values repeat so one probe row emits several output rows --
because that loop form only implements plain equality
(specializer/lower.rs refuses NOT-DISTINCT keys under 'many', a gap
that predates TASK-133).

Struct keys lower to NOT-DISTINCT presence and leaf keys, so a
struct-keyed NATURAL/USING join over a duplicate-key static refuses
today where DuckDB fans out. Teaching the stage-B multiplicity loop
the NOT-DISTINCT comparison form lifts both this refusal and the
pre-existing one for explicit IS NOT DISTINCT FROM keys under 'many'.

Measure first: pin DuckDB's fan-out multiset for a duplicate STRUCT
key (including interior-NULL rows that match under NOT-DISTINCT and
would not under plain equality) before touching the loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 DuckDB's fan-out multiset for duplicate struct keys is
      measured and pinned first, including interior-NULL rows where
      NOT-DISTINCT and plain equality disagree
- [ ] #2 a struct-keyed NATURAL/USING join over a duplicate-key static
      serves the measured multiset under shape='many'; the TASK-133
      named refusal for that shape is gone
- [ ] #3 explicit IS NOT DISTINCT FROM join keys serve under
      shape='many' (the pre-existing lower.rs refusal lifts with the
      same loop change)
- [ ] #4 the map and filter shapes are untouched -- every TASK-133 pin
      stays green
<!-- AC:END -->
