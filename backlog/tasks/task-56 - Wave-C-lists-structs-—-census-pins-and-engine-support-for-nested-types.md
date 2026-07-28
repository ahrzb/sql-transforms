---
id: TASK-56
title: >-
  Wave A structural tails: structs-as-lanes + dialect/regex/ingest tails (re-cut
  from lists/structs after census)
status: In Progress
assignee: []
created_date: '2026-07-27 23:15'
updated_date: '2026-07-28 00:25'
labels:
  - specializer
  - wave-c
dependencies: []
type: feature
ordinal: 50000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Census (in notes below) dissolved the 'lists/structs ~25 cases' premise: the true nested pool is 6 servable STRUCT cases + 1 LIST case double-blocked by self-join, and zero corpus cases need regexp_extract_all/split_to_array. User approved the re-cut ('A makes sense', 2026-07-28): a structural-tails wave of ~17 servable cases, corpus 511 -> ~528, with NO new runtime types.

Scope:
- Structs-as-lanes (6 cases): struct row columns flatten to scalar lanes at bind time (fixed shape). Serves a.* struct-star expansion incl. EXCLUDE/REPLACE on struct fields, and nested multi-part field access (t.t.t.t through 8 parts). All outputs are scalars. Pins needed: expansion order/naming, NULL-struct vs NULL-field semantics, multi-part resolution precedence, case sensitivity of field names.
- FROM-position colon alias 'FROM b : a' (4 cases): token pre-rewrite like the SELECT-item colon alias from wave 5. Pin the semantics (alias:table order, qualified refs).
- reverse() (3 cases): lift the wave-3 descope by implementing true UAX-29 extended grapheme cluster reversal via the unicode-segmentation crate. Pin DuckDB reverse against the crate on adversarial inputs (regional indicators, ZWJ emoji, combining marks, Hangul, CRLF) BEFORE trusting it. Remove the limitations-doc row + flip its twin test in the same commit.
- Lazy non-scalar rejection (2 cases): row-model columns with unmappable types reject only when REFERENCED (incl. via *), not at ingest. Unreferenced timestamp columns stop blocking scalar-only queries.
- COLUMNS(* REPLACE ...) (2 cases): extend COLUMNS argument handling to the star-with-REPLACE form (pin naming/order over NATURAL JOIN).
- Constant-NULL regex pattern (1 case): regexp_matches(c0, CAST(NULL AS STRING)) — pin DuckDB's NULL-pattern behavior per regexp function + SIMILAR TO, serve it.
- Paren-less '* REPLACE expr AS col' (1 case): token pre-rewrite to the parenthesized form sqlparser accepts.

Same rhythm as every wave: pins fleet -> spec in docs/superpowers/specs/ -> staged implementation with the corpus gate green at every stage -> full gate on release build -> PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Case-level census committed with the spec (buckets of all 167 non-matches; true struct/list pool named)
- [ ] #2 DuckDB pins spec for every scope item (struct expansion/NULL semantics, FROM colon alias, reverse graphemes vs unicode-segmentation, COLUMNS(* REPLACE), NULL regex patterns) with raw pin JSONs
- [ ] #3 Struct-star + nested field access serve bit-identical to DuckDB (oracle tests; structs flattened to scalar lanes, no new runtime types)
- [ ] #4 FROM-position colon alias, paren-less * REPLACE, COLUMNS(* REPLACE), constant-NULL regex pattern serve with oracle tests
- [ ] #5 reverse() serves with UAX-29 grapheme semantics; limitations-doc row removed + twin test flipped in the same commit
- [ ] #6 Unreferenced non-scalar row columns no longer reject; referenced ones keep the named error (tests both ways)
- [ ] #7 Corpus replay: match count strictly above 511, zero FAILs; full gate green (cargo + pytest incl. twin + fuzzer) on release build
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CENSUS RESULT (pre-pins, changes the wave's premise): the '~25 lists/structs cases' estimate dissolves at case level. Actual buckets of the 167 non-matches: aggregation/COUNT(*) ~40+, stage-B 27, table-fns 24, BIT-type casts 13 (one source file; targets are mostly narrow-width casts we reject anyway), rowid 8, parse-tail 7 (4x FROM-position colon alias 'FROM b : a', 1x paren-less * REPLACE, 1x COLUMNS lambda, 1x other), f32 5, schema-qualified static collisions (s1.t1 vs s2.t1) 4, TIMESTAMP col 3 (2 of which never reference the timestamp column - blocked only by eager non-scalar rejection), reverse() 3, HUGEINT 2, COLUMNS(* REPLACE) 2, non-constant regex 6 (1 is actually a CONSTANT NULL pattern), known divergences 2. TRUE lists/structs: STRUCT 6 + LIST 1 (and the LIST case is double-blocked by self-join). Zero corpus cases need regexp_extract_all/split_to_array.

KEY DESIGN INSIGHT for the struct pool: all 6 struct cases have SCALAR outputs (a.* explodes the struct; t.t.t.t is INT). Structs are fixed-shape, so struct row columns can flatten to scalar lanes at bind time - no new runtime type, no list machinery, both backends work unchanged. Scope will be re-cut after the user picks from the reshaped menu.
<!-- SECTION:NOTES:END -->
