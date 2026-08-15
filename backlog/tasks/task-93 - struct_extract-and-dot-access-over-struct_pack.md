---
id: TASK-93
title: >-
  struct_extract and dot access over struct_pack
status: Done
assignee: []
created_date: '2026-08-13 23:55'
labels:
  - frontend
  - parity
dependencies: []
type: feature
ordinal: 85000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`struct_extract(struct_pack(a := e, ...), 'a')` and the dot form
`(struct_pack(a := e)).a` bind on DuckDB and refuse here (audit
2026-08-13: the recognizer is projection-loop-only). Requested by
AmirHossein 2026-08-13.

Pure bind-time desugar: extracting a field of a just-packed struct IS
binding that field's expression — no engine machinery. Follow DuckDB's
case-insensitive field matching (duplicate-field refusal already exists
on the pack side) and its missing-field error wording. `struct_extract`
over a wide EXTERN keeps its existing arm.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Both spellings serve with DuckDB-equal values and schema (live
      oracle test), including a NULL field and nested struct_pack
- [x] #2 Missing / ambiguous field names refuse with DuckDB's error class
- [x] #3 The audit comments claiming engine-stricter behavior for these
      shapes come out in the same PR
<!-- AC:END -->
