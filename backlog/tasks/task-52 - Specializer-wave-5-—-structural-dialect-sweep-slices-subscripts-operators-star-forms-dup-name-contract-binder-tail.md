---
id: TASK-52
title: >-
  Specializer wave 5 — structural + dialect sweep (slices, subscripts,
  operators, star forms, dup-name contract, binder tail)
status: In Progress
assignee: []
created_date: '2026-07-26 21:46'
updated_date: '2026-07-26 22:02'
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
- [ ] #2 VARCHAR slices s[a:b] serve with DuckDB semantics (bounds, negatives, NULL/dynamic bounds, out-of-range, non-ASCII)
- [ ] #3 Extended subscripts serve: dynamic, negative, out-of-range indices per pins
- [ ] #4 Operator/star dialect forms serve or reclassify cleanly: ^@, GLOB, << >>, * LIKE/ILIKE/SIMILAR, * REPLACE, * RENAME, qualified EXCLUDE, alias-prefix colon
- [ ] #5 Duplicate output column names: contract pinned to DuckDB Python-client behavior, implemented across output modes
- [ ] #6 Binder tail per pins: NULL <op> NULL typing, lateral aliases, t(a,b), NATURAL JOIN, schema-qualified driving relation, BETWEEN/IN mixed-type semantics
- [ ] #7 Corpus replay: three-outcome contract holds, zero FAILs, match count reported (ceiling ~130 over 395)
- [ ] #8 Gate green both backends, clippy clean, serving-bench parity gate passes
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
Spec: docs/superpowers/specs/2026-07-26-wave5-structural-pins.md (pins committed cc28fc0, before implementation).
<!-- SECTION:PLAN:END -->
