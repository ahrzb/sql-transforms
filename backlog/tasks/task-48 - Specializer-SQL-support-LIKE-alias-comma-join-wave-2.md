---
id: TASK-48
title: >-
  Specializer SQL support: LIKE, dynamic-table alias, comma-join rewrite (wave
  2)
status: Done
assignee: []
created_date: '2026-07-26 15:05'
updated_date: '2026-07-26 16:41'
labels: []
milestone: m-7
dependencies:
  - TASK-47
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
  - docs/superpowers/specs/2026-07-26-wave1-builtin-pins.md
type: feature
ordinal: 42000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dual-axis rung plus two structural cheap wins, ~120 corpus first-blockers in reach. (1) LIKE / NOT LIKE with % and _ wildcards and the ESCAPE clause — DuckDB semantics pinned FIRST (case sensitivity, codepoint vs byte matching for _, escape edge cases, NULL propagation, degenerate patterns); a compiled-pattern op on both backends via one shared matcher; SIMILAR TO / regexp reject by name unless the pins show LIKE-only covers the corpus head. Closes the workload ladder's last gap (title extraction, device normalization patterns). (2) Alias on the dynamic table — scope plumbing in the frontend binder (30 cases). (3) Comma-join rewrite: FROM t, dim WHERE <equi-conjuncts> rewrites to the INNER equi-join the engine already serves, with non-equi/cross shapes rejecting cleanly by name (up to 50 join-form cases share this first blocker).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LIKE semantics pinned by measurement (duck_check tests recorded before implementation: wildcards, ESCAPE, unicode, NULL, degenerate patterns) and both backends agree via a shared matcher (differential extended)
- [x] #2 Dynamic-table alias binds (qualified refs via alias, original name behavior measured and mirrored)
- [x] #3 RE-SCOPED by census (recorded below): the corpus comma-join cases are SELF-joins of the dynamic table (FROM integers i1, integers i2 ...), not dynamic-x-static shapes — a rewrite would flip ~0 cases. Comma-join stays cleanly unsupported; dynamic self-join is a distinct future feature (needs row-table-as-static materialization).
- [x] #4 Corpus replay: wave-2 first-blocker cases flip to match or a named deeper blocker; zero FAILs; tally recorded here
- [x] #5 mise gate-specializer green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Census before code (2026-07-26): the 30 alias-first-blocker cases are mostly alias + a DEEPER blocker (self-joins, USING, rowid, column-renaming aliases); the ~30 comma-join cases are nearly all SELF cross-products of the dynamic table — the planned equi-WHERE rewrite would flip approximately zero, so it was dropped from scope honestly rather than shipped as dead code. Alias landed: the alias REPLACES the original name (measured: qualified refs through the original are DuckDB binder errors) — implemented by making the alias the binder's this_name, which yields exactly that scoping for qualified refs, star expansion, and error messages; AS/bare forms identical; column-renaming alias t(a,b) rejects by name. LIKE landed: byte-based O(n*m) two-pointer matcher with codepoint _, leftmost-first backtracking reproducing the DATA-DEPENDENT dangling-escape error rows, full ESCAPE semantics (any single byte, doubled = literal, never self-matching, '' = none, NULL = NULL), NOT LIKE, and ILIKE via double simple-casemap fold (exhaustively verified vs DuckDB lower()). The corpus caught a live divergence during replay: DuckDB's ILIKE result for NUL-containing rows depends on SIBLING rows (stats-selected kernels; the generic one NUL-truncates) — irreproducible row-locally, recorded as the repo's first documented oracle divergence (_KNOWN_DIVERGENT_SOURCES) with the measurement in the pins. SIMILAR TO rejects by name (DuckDB binds it to regexp_full_match). CORPUS: 240 -> 265 match of 678, zero FAILs, gate green.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wave 2 complete: LIKE/NOT LIKE/ILIKE with full ESCAPE semantics on both backends (pins-first; the matcher reproduces even DuckDB's data-dependent error rows while avoiding its measured O(n^k) blowup), dynamic-table alias (alias replaces the original name), comma-join honestly descoped by census (corpus cases are self-joins). Corpus 240 -> 265/678, zero FAILs; first documented oracle divergence recorded with proof of row-local irreproducibility (stats-dependent ILIKE kernels).
<!-- SECTION:FINAL_SUMMARY:END -->
