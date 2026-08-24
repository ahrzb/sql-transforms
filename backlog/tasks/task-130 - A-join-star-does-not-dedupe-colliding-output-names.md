---
id: TASK-130
title: >-
  A join star does not dedupe colliding output names
status: Done
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
- [x] #1 measure DuckDB's dedup rule properly (three-way collisions,
      qualified stars, USING-merged keys, EXCLUDE interaction) before
      implementing -- the `_1` suffix is one observation, not the rule
- [x] #2 a join star's output names match DuckDB's exactly, or the collision
      refuses by name -- never duplicate keys in a row dict
- [x] #3 the campaign's seed-380 class is gone
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-08-19 as a FUZZER fix plus a corrected record -- the ticket as
filed had the direction BACKWARDS (my misread of a schema-delta message:
the deduped names were OURS, the duplicates DuckDB's).

AC #1's proper measurement: DuckDB's raw cursor AND its top-level arrow
export both keep DUPLICATE output names (['c0','c2','c0','c1']); the
`<name>_N` rename is what DuckDB does at subquery/CTE/CTAS boundaries and
in .df() -- which is exactly the wave-5 client contract this engine
adopted (pins-wave5/dup-names-client-contract.json,
frontend.rs::dedup_output_names), because dict-shaped infer_rows output
cannot hold two 'c0' keys losslessly.

So the engine is right by decided contract, and the finding class was the
campaign's schema leg not KNOWING the contract. TWO oracle fixes: the
schema compare normalizes DuckDB's names through the same rule
(_dedup_names, mirrored constant for constant), and duck's arrow table is
RENAMED through it on entry -- pyarrow's to_pylist silently collapses
duplicate dict keys, so without the rename the value compare was garbage
for every dup-name star (duck rows lost their row-side values entirely).

Re-run: 380/4160/9097 clean; 12745 was a REAL width divergence the schema
mismatch had been hiding (bare-NULL CASE arm floors DuckDB's unification
at INTEGER; ours adopts int16) -- pinned xfail-strict and ticketed as
TASK-131, not fixed here. The kept dedup divergence is pinned in
known_divergences/test_arrow_boundary.py with the remeasure guard on
DuckDB's side and the lossless-dict assertion on ours.
<!-- SECTION:NOTES:END -->
