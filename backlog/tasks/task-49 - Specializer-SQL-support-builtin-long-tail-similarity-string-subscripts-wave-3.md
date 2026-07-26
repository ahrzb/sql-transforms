---
id: TASK-49
title: >-
  Specializer SQL support: builtin long tail + similarity + string subscripts
  (wave 3)
status: Done
assignee: []
created_date: '2026-07-26 16:45'
updated_date: '2026-07-26 18:15'
labels: []
milestone: m-7
dependencies:
  - TASK-48
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
  - docs/superpowers/specs/2026-07-26-wave1-builtin-pins.md
type: feature
ordinal: 43000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Census-measured wave (2026-07-26 replay: 265/678 match, zero FAILs). Pure expression kernels riding the wave-1/2 machinery — pins fleet FIRST, shared semantic fn per op, both backends, differential + trap agreement, corpus zero-FAIL gate. Catalogue by measured first-blocker: SIMILARITY (39 cases): levenshtein/editdist3, damerau_levenshtein, jaccard, hamming/mismatches. STRING TAIL (~35): repeat, concat_ws, lpad/rpad, replace, reverse, translate, unicode/ord, strip_accents (oracle-extracted table, casemap playbook), ucase/lcase aliases, bit_length. STRING SUBSCRIPTS (39): array_extract/array_slice/list_slice on VARCHAR (the blocked sources are function/string/*, not lists). MATH TAIL (~10): add/subtract/multiply/divide aliases, mod, fmod, fdiv, nextafter. REJECT BY NAME: aggregates (sum/count/geomean), regexp_* (RE2, own wave), columns(), list-typed subscripts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pins measured BEFORE implementation for all six families (similarity; pad/repeat/replace/translate/reverse/concat_ws; unicode/bit_length/case-aliases; VARCHAR subscripts; math aliases+fmod/fdiv/mod/nextafter; strip_accents oracle table) — JSON pins + spec addendum committed
- [x] #2 Every op is one shared semantic fn used verbatim by both backends; differential fuzz extended incl. trap agreement
- [x] #3 Aggregates and regexp family reject cleanly by name (no behavior change, messages recorded)
- [x] #4 Corpus replay: flips recorded here, zero FAILs, known-divergent list unchanged or justified
- [x] #5 mise gate-specializer green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pins committed (2622a14) BEFORE implementation. Fleet: 6 agents, 528k tokens, 300+ probes. Headline refutations: similarity is BYTE-based and damerau is the UNRESTRICTED DL variant (not OSA); fmod is FLOORED (divisor sign) while mod/% is C-fmod (dividend sign), // on doubles is PLAIN division with NULL-on-zero; rpad truncation keeps the PREFIX like lpad; hamming('','') is an ERROR; unicode('')=-1 but ascii('')=0 (sole ascii/unicode divergence); VARCHAR out-of-range subscripts give '' where LIST gives NULL and NULL is never an open bound; strip_accents = oracle map (4460 cps) + Hangul compose + context-dependent NUL truncation, DuckDB tables lag Unicode 16 by 57 cps. SCOPE CHANGE: reverse DESCOPED (measured grapheme-cluster/UAX-29 semantics incl. RI pairing = real segmentation machinery for 3 corpus cases; rejects by name, pins retained). concat/concat_ws restricted to VARCHAR args (DuckDB implicit float rendering not modeled). nextafter f64 only (FLOAT overload is f32; f32 columns already reject).

IMPLEMENTATION COMPLETE. Corpus 265 -> 383 of 678, zero FAILs (deterministic, replayed twice); gate green (738 passed, 13 xfailed; cargo 129). New IR: StrOp2 += Levenshtein/Damerau/Jaccard/Hamming, StrOp3 (Replace/Translate), StrOp2i (Repeat/Extract), Spad, Sslice, Sord, StrOp1 += StripAccents, BinOp += Ffloordiv/Ffloormod/Fnextafter, ArithOp::IDiv (// operator + divide()); concat_ws/concat desugar onto Case/Or/Concat with the has-prior-non-null fold — no new IR. Shared semantic fns verbatim in both backends; 500-seed differential incl. trap agreement green. NULL-pre-empts-trap masking: Jaccard/Hamming mask BOTH operands to 'a' under the combined flag ('' is IN their trap domain — the Flogb pattern); Spad masks len->0. Overflow trap texts now DuckDB-verbatim with operand values (test-fleet finding, fixed in-wave; abs text measured too). f32 base tables classify clean-unsupported in the replay (nextafter's f32-grid was the wave's only FAIL class). 57 new oracle tests in 4 files (tests/test_duckdb_wave3_*.py), drafted by a 4-agent fleet, all green incl. NaN-bit assertions (fmod fff8 vs % 7ff8) and the data-dependent Insufficient-padding trap split.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wave 3 complete: 26 builtins across six measured families — similarity (byte-based levenshtein/editdist3, unrestricted damerau, byte-set jaccard, hamming/mismatches with verbatim trap texts), string builders (repeat, lpad/rpad with the data-dependent empty-pad trap, replace, translate, concat_ws/concat via a pure desugar), VARCHAR subscripts (array_extract/list_extract, array_slice/list_slice — '' out-of-range, NULL never an open bound), inspection (unicode/ord/ascii, bit_length, ucase/lcase), math tail (add/subtract/multiply/divide/mod aliases, the // operator, floor-pair fdiv/fmod, bit-exact nextafter), and strip_accents from an oracle-extracted 4460-codepoint table + Hangul jamo composition. reverse descoped by measurement (UAX-29 grapheme clusters for 3 corpus cases). Overflow trap texts brought to DuckDB's verbatim, operand-interpolated forms. Corpus 265 -> 383 of 678, zero FAILs; f32 base tables now classify honestly as clean-unsupported. Gate green: cargo 129, pytest 738 + 13 xfail, 500-seed cross-backend differential incl. trap agreement, 57 new oracle tests.
<!-- SECTION:FINAL_SUMMARY:END -->
