---
id: TASK-53
title: >-
  Specializer wave B — regexp family (regexp_* functions, ~/!~/SIMILAR TO,
  deferred star forms)
status: In Progress
assignee: []
created_date: '2026-07-26 23:50'
labels: []
milestone: m-7
dependencies:
  - TASK-52
priority: high
type: feature
ordinal: 47000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-wave-5 pool (census at master ba8bf98: 484 match / 192 unsupported / 2 known-divergent): the regexp bucket is the largest coherent workable slice (~25-30 cases + unlocks). Scope:

- regexp_matches / regexp_full_match / regexp_extract / regexp_replace (scalar forms; list/table-returning forms classify clean as non-scalar)
- operators ~ and !~ (DuckDB binds them to regexp_full_match), SIMILAR TO on values
- deferred wave-5 star forms: * SIMILAR TO 're' (unanchored name search + the pinned NOT asymmetry) and COLUMNS('re')

Engine decision: Rust `regex` crate (RE2-lineage, pure Rust; NOT in the tree yet — new direct dependency). DuckDB uses RE2 — the pins fleet must run a DIFFERENTIAL battery (duckdb vs rust-regex side by side) to pin exactly where they disagree (\d/\w/\s Unicode-ness, (?i) fold scope, empty matches in replace, error classes for invalid patterns); divergent corners classify clean-unsupported rather than serving wrong answers.

Pins-first; wave-3 over-generalization precedent applies — every claim needs an executed query/program recorded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pins-first: wave-B pins spec (md + JSON) committed before implementation, incl. the duckdb-vs-rust-regex differential battery and the crate decision
- [ ] #2 regexp_matches / regexp_full_match / regexp_extract / regexp_replace serve per pins (options strings, group semantics, backrefs, NULL handling); list/table forms classify clean
- [ ] #3 Operators ~ / !~ and SIMILAR TO on values serve per pins (anchoring semantics measured, not assumed)
- [ ] #4 Deferred star forms serve: * SIMILAR TO name filter (incl. the pinned NOT asymmetry) and COLUMNS('re') expansion
- [ ] #5 Patterns whose semantics differ between RE2 and rust-regex classify clean-unsupported (guard at bind time from the pinned divergence list) — never a wrong answer
- [ ] #6 Corpus replay: three-outcome contract holds, zero FAILs, match count reported
- [ ] #7 Gate green both backends, clippy clean, serving-bench parity gate passes
<!-- AC:END -->
