---
id: TASK-132
title: >-
  Static struct lanes carry a structured path; the dotted name becomes display
status: Done
assignee: []
created_date: '2026-08-25 00:00'
labels:
  - m-8
  - parity
dependencies: []
type: refactor
ordinal: 117000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The accepted lane-encoding RFC
(packages/confit/docs/rfcs/2026-08-19-static-struct-lane-encoding.md, alternative A,
2026-08-25): a static table's struct leaves are currently flattened into
lanes NAMED by their dotted spelling ("w.mean"), which destroys the
leaf-vs-literal-column distinction at catalog build and leaks into every
name-resolution feature (the TASK-127 family). The lane gets a
STRUCTURED PATH instead -- head name plus field path as data -- and the
dotted string survives only as display.

The reference model is DuckDB's own, source-verified in the RFC: parts
vector in, outermost-first consumption, table.column before
column.field, leftover parts as struct extracts resolved to ordinals,
head-name ambiguity decided before fields are examined (already our
TASK-121 rule). Name lookups never see dots.

Spec first (the refactor touches every consumer of
`StaticTable::cols[..].name`: resolution, star, EXCLUDE, aliases,
dedup, output schema, and the row-side flatten shares the convention --
audit it in the same pass). TASK-127's remaining items land on top of
this and are deliberately NOT in scope here beyond not making them
worse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 a spec in packages/confit/docs/specs/ maps every consumer of the
      lane name to its behavior under the structured path, including the
      row-side flatten, before any code changes
- [x] #2 the collision table from the RFC serves BOTH spellings like
      DuckDB: `s.w.mean` is the leaf, `s."w.mean"` is the literal
      column, live-oracle pinned
- [x] #3 no name-resolution path constructs or splits a dotted string;
      the dotted spelling appears only in display surfaces (output
      schema field names, error text), each listed in the spec
- [x] #4 every existing green behavior over static structs survives:
      star, EXCLUDE, aliases, dedup, TASK-121 ambiguity trio, TASK-125
      star pins -- full suite plus a 4k campaign clean
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed per the spec (packages/confit/docs/specs/2026-08-25-static-struct-
lane-path-design.md). StaticTable carries the row side's StructCol tree;
one shared walk_fields serves both sides message-for-message; dotted
names are display-only; the data paths carry segment paths. Four flips,
live-oracle pinned: the collision table serves both spellings, a quoted
dotted identifier no longer binds a leaf, a plain column named 'a.b'
serves (it was misread as a struct walk in the data path), and the
non-ASCII prefix-scan panic is a named refusal.

Gate: full suite release AND debug (2986 passed, no debug_assert
deviations), 4k campaign = baseline classes seed-for-seed. The one new
campaign item (seed 2813 TIMEOUT) attributed: the seed spends ~28s in
DUCKDB materializing a 2 GiB lpad against a 30s budget - our side
REFUSES it instantly via the TASK-88 string-budget limit; inherently
marginal under the campaign budget, unrelated to this change. Follow-up
option: raise the campaign timeout or bound the lpad-count generator.
<!-- SECTION:NOTES:END -->
