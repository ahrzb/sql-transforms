---
id: TASK-46
title: 'Specializer SQL support: SELECT * star expansion'
status: Done
assignee: []
created_date: '2026-07-26 11:42'
updated_date: '2026-07-26 12:20'
labels: []
milestone: m-7
dependencies:
  - TASK-45
documentation:
  - packages/confit/docs/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The single biggest corpus rung: 128 of 625 clean-unsupported corpus cases have `SELECT *` (or `tbl.*`) as their first blocker. Expand the star at bind time in the frontend against DuckDB's measured semantics — column order and naming for the row table alone and under joins (including duplicate-name handling), `tbl.*` qualified forms, and whatever star modifiers the corpus actually uses (EXCLUDE/REPLACE are measured-first: support only what the corpus needs, reject the rest cleanly by name). Pure frontend work — no IR, backend, or boundary changes. Clearing it also de-masks second blockers currently hidden behind star for the builtins wave (TASK-47).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Star semantics pinned by measurement against DuckDB 1.5.5 (column order, duplicate-name policy, qualified tbl.*, modifier handling) and recorded as duck_check tests
- [x] #2 Corpus replay: star-first-blocker cases flip to match or to a NAMED deeper blocker; zero FAILs; match count and the new first-blocker tally recorded here
- [x] #3 Unsupported star forms (if any remain) reject with a clean "unsupported: ..." naming the form
- [x] #4 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26 after the measurement spike; all facts below are measured on DuckDB 1.5.5).
Corpus star census: 156 plain `SELECT *`, 26 `tbl.*`, 16 `* EXCLUDE`, 13 `COLUMNS(...)`, 3 `* LIKE`, 2 `* REPLACE`, 2 mixed. Wave scope: plain `*`, `tbl.*`, `EXCLUDE` (~198 mentions); COLUMNS()/REPLACE/LIKE/RENAME/EXCEPT/ILIKE reject by name. All forms already parse under sqlparser 0.62 and funnel to the single rejection at frontend.rs:125 — one expansion site.
Measured pins: expansion order is FROM order then declared column order per table; `SELECT * FROM a JOIN b ON a.k=b.k` emits DUPLICATE column names ('k','x','k','y') — unrepresentable in a pydantic output model, and any star item covering a joined static table necessarily includes its join-key column, whose projection our engine already cleanly rejects; therefore star items that cover a static table reject cleanly naming the key column (covers the duplicate-name shape too). `__THIS__.*` under a join is fine (row columns only). EXCLUDE: paren and bare forms; case-insensitive; unknown name = binder error mirroring DuckDB ("Column \"nope\" in EXCLUDE list not found"); duplicate list entry = error ("Duplicate entry"); unqualified EXCLUDE removes every same-named column.
1. Binder::expand_star(qualifier, options) -> Vec<(name, SExpr)> using the existing SKind::Col / static_lane constructors; wire into the projection loop (mixed `*, expr` items fall out of item-position expansion); existing duplicate-output-name check stays as the final guard.
2. duck_check pins for every measured behavior above incl. the EXCLUDE edges (exclude-all, EXCLUDE K case-folding) and clean-unsupported messages for the rejected forms.
3. Corpus replay re-tally into this ticket (AC #2): flips + newly-surfaced second blockers.
4. Gate green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed in one pass: Binder::expand_star (frontend.rs) expands `*` and `tbl.*` at the single projection site using the existing SKind::Col / static_lane constructors — no IR/backend/boundary changes, exactly as scoped. EXCLUDE handles bare + paren forms case-insensitively (sqlparser 0.62 carries entries as ObjectName; single-part idents only, qualified entries reject by name); unknown-column and duplicate-entry EXCLUDE mirror DuckDB's binder errors; exclude-all falls through to the existing empty-SELECT bind error (same shape as DuckDB's "SELECT list is empty after resolving * expressions"). Star items covering a joined static table reject cleanly naming the join-key column (DuckDB includes the key AND emits duplicate output names there — both unrepresentable; `__THIS__.*` under a join works). ILIKE/EXCEPT/REPLACE/RENAME/COLUMNS() reject by name. CORPUS RESULT (AC #2): 53 -> 172 match of 678 (+119, a 3.2x coverage jump), 0 FAILs, 506 clean-unsupported. New first-blocker head: BETWEEN 31, comma join 30, dynamic-table alias 30, table-function FROM 24, then the builtin catalogue (array_slice 23, contains 18, damerau_levenshtein 17, ...) — the TASK-47 target list confirmed. Gate green: cargo 129 + pytest 627 (13 xfail incl. the pre-existing substr const-fold pin from master).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged as PR #30 (2026-07-26). Star expansion at the frontend's single projection site tripled corpus coverage: 53 -> 172 match of 678, zero FAILs. All measured edges pinned (order, EXCLUDE case-folding + binder errors, star-over-join rejection naming the join key, macro forms rejected by name). Gate green cargo 129 / pytest 627. Unblocked TASK-47's target list: post-star first-blocker head is BETWEEN 31, comma join 30, dynamic-table alias 30, then the builtin catalogue.
<!-- SECTION:FINAL_SUMMARY:END -->
