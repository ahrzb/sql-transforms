---
id: TASK-49
title: >-
  Specializer SQL support: builtin long tail + similarity + string subscripts
  (wave 3)
status: In Progress
assignee: []
created_date: '2026-07-26 16:45'
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
- [ ] #1 Pins measured BEFORE implementation for all six families (similarity; pad/repeat/replace/translate/reverse/concat_ws; unicode/bit_length/case-aliases; VARCHAR subscripts; math aliases+fmod/fdiv/mod/nextafter; strip_accents oracle table) — JSON pins + spec addendum committed
- [ ] #2 Every op is one shared semantic fn used verbatim by both backends; differential fuzz extended incl. trap agreement
- [ ] #3 Aggregates and regexp family reject cleanly by name (no behavior change, messages recorded)
- [ ] #4 Corpus replay: flips recorded here, zero FAILs, known-divergent list unchanged or justified
- [ ] #5 mise gate-specializer green
<!-- AC:END -->
