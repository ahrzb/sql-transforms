---
id: DRAFT-19
title: >-
  BUG both engines: UNNEST output column naming diverges from DataFusion — bare
  unnest silently drops a column
status: Draft
assignee: []
created_date: '2026-07-24 02:19'
labels:
  - parity
  - bug
  - containers
  - native
  - codegen
dependencies: []
references:
  - 'PR #17'
  - src/plan.rs
  - sql_transform/_codegen/plan.py
  - tests/test_diff_types.py
documentation:
  - doc-9 (Rich type system and UNNEST — status and deferred edges)
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two confirmed naming divergences from the DataFusion oracle, BOTH ENGINES, found by Ritchie during TASK-29 Phase C (2026-07-24). One fix, two cases: they share a root cause (UNNEST output display-name logic) and the same pair of sites.

IMPORTANT: this is NOT a codegen bug. Codegen mirrored native and inherited native's mistake. Fixing codegen alone would SPLIT the engines, so this must land as one change across both.

CASE 2 IS THE SERIOUS ONE — SILENT COLUMN LOSS

    SELECT id AS unnest, unnest(l) FROM t

DataFusion names the bare-unnest output column `UNNEST(t.l)`. Both our engines name it the literal string `unnest`. So a user who has a column aliased `unnest` — or simply writes that alias — collides with the generated name, and the projection SILENTLY DROPS the `id` column. No error, no warning; a column the user explicitly selected just is not in the output.

That is data loss driven by a name collision the user cannot see, which is why this is rated High despite "naming" sounding cosmetic.

    Site:  column_name in src/plan.rs (native), mirrored placeholder in codegen
    Tests: test_unnest_bare_list_column_name_diverges,
           test_unnest_bare_list_alias_collision_drops_column

CASE 1 — MISNAMED STRUCT FIELD-ACCESS UNNEST

    unnest() of a struct-typed FIELD ACCESS

    DataFusion:   t.s[inner].x     (bracket notation for the intermediate hop)
    Both engines: t.s.inner.x

Wrong output column name. No data loss, but it means a user selecting by name against the oracle's naming gets a column that is not there, and any downstream code keyed on column names diverges between our engines and DataFusion.

    Site:  unnest_display_name in src/plan.rs, _unnest_display_name in sql_transform/_codegen/plan.py
    Test:  test_unnest_struct_field_access_expands_columns

CURRENT STATE
Both pinned as xfail-strict in PR #17 and deliberately NOT fixed, per the standing native-bug process. Ritchie's call to pin rather than fix was correct — fixing the codegen half in a codegen ticket would have created the very engine split the differential harness exists to prevent.

HOW THEY WERE FOUND (worth noting)
Both surfaced from adding oracle coverage for behavior that was previously "verified" only by code-reading. Same lesson as DRAFT-16: reading the code said it was fine; asking the oracle said otherwise. Evidence for the validate-don't-assume practice.

DRAFT pending AmirHossein's review of scope/priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Bare unnest(<list>) output column is named as DataFusion names it (UNNEST(t.l)), on BOTH engines in one change
- [ ] #2 The alias-collision case no longer drops a column: SELECT id AS unnest, unnest(l) FROM t returns both columns
- [ ] #3 unnest() of a struct-typed field access uses DataFusion's bracket notation for the intermediate hop (t.s[inner].x) on both engines
- [ ] #4 The xfail-strict markers on test_unnest_bare_list_column_name_diverges, test_unnest_bare_list_alias_collision_drops_column and test_unnest_struct_field_access_expands_columns are removed and the tests pass on both engines
- [ ] #5 Root-cause sweep: confirm no other generated/derived column name is emitted as a bare literal that could collide with a user alias
<!-- AC:END -->
