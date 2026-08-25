---
id: TASK-52
title: >-
  Specializer wave 5 — structural + dialect sweep (slices, subscripts,
  operators, star forms, dup-name contract, binder tail)
status: Done
assignee: []
created_date: '2026-07-26 21:46'
updated_date: '2026-07-26 23:10'
labels: []
milestone: m-7
dependencies:
  - TASK-50
priority: high
type: feature
ordinal: 46000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave-5 census (census_wave5, 2026-07-26, master 13a3a7e: 395 match / 281 unsupported / 2 known-divergent, zero FAILs) decomposed the structural tail. This wave takes the buckets adjacent to shipped machinery — no new subsystems:

- VARCHAR slices s[a:b] (26 cases): extends wave-3 subscripts
- subscript forms still unbound (~11): dynamic / negative / huge indices
- parser-only operator mappings (~15 of 32 parse errors): ^@ (starts_with), GLOB, star LIKE/ILIKE/SIMILAR/NOT, * REPLACE, * RENAME, alias-prefix colon
- duplicate output column names (29): pin DuckDB's own Python-client surface for duplicates and mirror it
- binder tail (~20): NULL <op> NULL typing, lateral aliases, t(a,b) renaming aliases, NATURAL JOIN, schema-qualified driving table, bitwise << >>, BETWEEN/IN mixed-type casts, qualified EXCLUDE

Out of wave: regexp family (wave B), lists/structs (wave C), stage-B multiplicity (gated). Pins-first: measure DuckDB before implementing; wave-3 over-generalization precedent applies — every pin claim needs an executed query recorded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pins-first: wave-5 pins spec (md + JSON) committed before implementation, incl. a sqlparser-capability spike deciding parse strategy per dialect form
- [x] #2 VARCHAR slices s[a:b] serve with DuckDB semantics (bounds, negatives, NULL/dynamic bounds, out-of-range, non-ASCII)
- [x] #3 Extended subscripts serve: dynamic, negative, out-of-range indices per pins
- [x] #4 Operator/star dialect forms serve or reclassify cleanly: ^@, GLOB, << >>, * LIKE/ILIKE/SIMILAR, * REPLACE, * RENAME, qualified EXCLUDE, alias-prefix colon
- [x] #5 Duplicate output column names: contract pinned to DuckDB Python-client behavior, implemented across output modes
- [x] #6 Binder tail per pins: NULL <op> NULL typing, lateral aliases, t(a,b), NATURAL JOIN, schema-qualified driving relation, BETWEEN/IN mixed-type semantics
- [x] #7 Corpus replay: three-outcome contract holds, zero FAILs, match count reported (ceiling ~130 over 395)
- [x] #8 Gate green both backends, clippy clean, serving-bench parity gate passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stages (each lands with tests + corpus replay green):
1. Parser groundwork: GenericDialect switch (regression net = cargo test + 678-corpus + py gate), colon-alias pre-rewrite (silent JsonAccess misparse!), GLOB pre-rewrite (LIKE + __glob_pat marker), star-filter capture (LIKE/NOT LIKE/GLOB; SIMILAR TO stays parse-error clean).
2. Slices s[a:b] + extended subscripts: shared codepoint kernel, both backends, verbatim runtime range traps (asymmetric ±2^32 window), NULL-first.
3. Bitwise << >> & | + xor() + ~: exact error ladder (<< five-step), NULL masks errors row-wise; ^ stays POWER.
4. ^@ -> starts_with kernel; dedicated GLOB byte matcher (classes/escapes/dead-match quirks per pins).
5. Star forms: name filters, qualified EXCLUDE (USING unmerge!), REPLACE, RENAME (silent-ignore), dup-name rename algorithm (DuckDB boundary rename: left-to-right, case-insensitive, own-name _N) across synthesized/dict/slot-fill/supplied surfaces.
6. Binder tail: lateral aliases (real column wins), t AS u(x,y), NATURAL JOIN (hard error on no common cols), main. qualifier only, mixed-type comparison casts (string->int half-away rounding), NULL-op-NULL typing (BIGINT/DOUBLE/BOOLEAN).
7. Census + bench parity + close-out.
Spec: packages/confit/docs/specs/2026-07-26-wave5-structural-pins.md (pins committed cc28fc0, before implementation).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corpus arc through the wave (census_wave5, all zero-FAIL): 395 -> 397 (colon alias) -> 427 (slices+subscripts) -> 431 (bitwise) -> 446 (^@+GLOB) -> 477 (star forms + dup-name rename) -> 484 (binder tail). +89 of the ~130 ceiling; the remainder is gated by dynamic self-joins, wave-B regexp (SIMILAR TO star, COLUMNS), lists/structs, and the USING-unmerge corner.

Two replay FAILs caught and fixed during stage 6 (NULL ^@ NULL typing; exec-time IN-list conversion on empty input) — the three-outcome contract did its job; corrections recorded in the spec addendum.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wave 5 shipped: the structural + dialect sweep, corpus 395 -> 484 bit-exact of 678 (+89, zero wrong answers, zero replay FAILs throughout). Pins-first: an 8-agent measurement fleet (7 DuckDB areas + a sqlparser capability spike) landed as packages/confit/docs/specs/2026-07-26-wave5-structural-pins.md + pins-wave5/*.json BEFORE implementation, with a post-landing addendum recording two corpus-caught corrections (NULL ^@ NULL typing; exec-time IN-list conversion — an empty input succeeds in DuckDB, so non-numeric literals stay clean-unsupported).

Landed in seven stages: (1a) frontend switched to GenericDialect — measured strict superset, byte-identical corpus; (1b) colon prefix alias k: expr via token pre-rewrite (sqlparser misparses it SILENTLY as a Snowflake JSON path — no error to catch); (2) bracket subscripts s[i] and slices s[a:b] bound onto the wave-3 kernels with chained access, open-bound desugar, and the asymmetric ±2^32 runtime trap window on both backends; (3) bitwise << >> & | + xor() as new i64 BinOps with DuckDB's five-step << trap ladder, >> total, and a source-order re-association pass because DuckDB puts all four in ONE flat left-assoc tier while sqlparser tiers them (fold shares the duck_shl kernel so folding and trapping can never disagree); (4) ^@ -> the starts_with kernel + a dedicated byte-level GLOB matcher (one-byte ?, byte-set classes where only ! negates, literal-if-first ']'/'-' rules, dead patterns match nothing) reached via a LIKE + __glob_pat marker rewrite; (5) star forms — name filters * LIKE/ILIKE/NOT LIKE/GLOB over column names, qualified EXCLUDE, * REPLACE, * RENAME — plus the duplicate-name contract: dedup_output_names applies DuckDB's own boundary rename (left-to-right, own-case _N, case-insensitive incl. candidates), verified against .df(); (6) binder tail — lateral aliases with real-column-wins shadowing in SELECT and WHERE, NULL <op> NULL typing by operator, main.tbl qualifier, NATURAL JOIN desugared to USING with the pinned hard error, t AS u(x,y) via a Cow column view, mixed BETWEEN/IN literal casts with half-away rounding.

Gate: 148 Rust + 553 py green (interp backend identical except the deliberate backend-identity guards), clippy clean on wave files, serving-bench parity gate passed on all four scenarios with the typed path at 1.4-2.0x over handcrafted python in 16/16 cells — 27 new surface forms cost the hot path nothing. Remaining unsupported head after the wave: aggregation 45 (out of scope), table-fns 24 (out of scope), regexp ~25+SIMILAR/COLUMNS (wave B), lists/structs 25 (wave C), dynamic self-joins 10, small named tails.
<!-- SECTION:FINAL_SUMMARY:END -->
