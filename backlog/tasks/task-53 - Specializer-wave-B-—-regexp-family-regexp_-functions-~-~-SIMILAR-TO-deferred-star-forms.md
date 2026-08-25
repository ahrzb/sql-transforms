---
id: TASK-53
title: >-
  Specializer wave B — regexp family (regexp_* functions, ~/!~/SIMILAR TO,
  deferred star forms)
status: Done
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:35'
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
- [x] #1 Pins-first: wave-B pins spec (md + JSON) committed before implementation, incl. the duckdb-vs-rust-regex differential battery and the crate decision
- [x] #2 regexp_matches / regexp_full_match / regexp_extract / regexp_replace serve per pins (options strings, group semantics, backrefs, NULL handling); list/table forms classify clean
- [x] #3 Operators ~ / !~ and SIMILAR TO on values serve per pins (anchoring semantics measured, not assumed)
- [x] #4 Deferred star forms serve: * SIMILAR TO name filter (incl. the pinned NOT asymmetry) and COLUMNS('re') expansion
- [x] #5 Patterns whose semantics differ between RE2 and rust-regex classify clean-unsupported (guard at bind time from the pinned divergence list) — never a wrong answer
- [x] #6 Corpus replay: three-outcome contract holds, zero FAILs, match count reported
- [x] #7 Gate green both backends, clippy clean, serving-bench parity gate passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stages (each lands with tests + corpus replay green):
1. Regex infrastructure: regex = "1" dep (RegexBuilder::octal(true), default Unicode — unicode(false) measured to BREAK (?i) parity); translation module (perl-class rewrite \d->(?-u:\d) etc incl. in-class variants, reject list: \B, (?<>), dup group names, bounds>1000, stacked quantifiers, \u, \Q\E; options parser c/i/l/s live, m/n/p no-ops, g replace-only; replacement translation \N->${N}, $->$$, with the out-of-range->identity and bad-escape global/non-global asymmetry resolved at BIND time); Program-level regex table; IR ops ReMatch/ReExtract/ReReplace + print/parse/verify + both backends.
2. Functions + operators: regexp_matches (search) / regexp_full_match / regexp_extract (''-on-no-match, flat 0..9 group check) / regexp_replace; ~ / !~ = full match (NOT Postgres search!); SIMILAR TO = raw-pattern full match (no % translation); NULL rules incl. the options-arg asymmetry.
3. Star forms: * SIMILAR TO (unanchored search) / NOT SIMILAR TO (NOT full match — independent predicates) via rewrite.rs marker codes; COLUMNS('re') interception + declared-order expansion.
4. Census + bench parity + close-out + PR.
Spec: packages/confit/docs/specs/2026-07-27-waveB-regexp-pins.md (pins committed b003122 before implementation).
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wave B shipped: the regexp family, corpus 484 -> 505 bit-exact of 678 (zero wrong answers, zero replay FAILs). Pins-first: 6-agent fleet (5 DuckDB areas + the RE2-vs-rust-regex DIFFERENTIAL battery) committed as packages/confit/docs/specs/2026-07-27-waveB-regexp-pins.md + pins-waveB/*.json before implementation.

The engine is the rust `regex` crate (new direct dependency) behind the measured parity recipe: RegexBuilder::octal(true) + default Unicode mode (unicode(false) was measured to BREAK (?i) folding parity — the 'obvious' fix is the wrong one), a bind-time Perl-class rewrite in retrans.rs (\d -> (?-u:\d) etc, in-class variants included) closing the whole RE2-ASCII-vs-rust-Unicode gap, a reject list for irreconcilables (\B unservable in DuckDB itself, (?<name>), duplicate group names, bounds > 1000, stacked quantifiers a*+ — silently WRONG in rust, not just error-shaped different, \u, \Q\E), and replacement-template translation with RE2's invalid-rewrite quirks resolved at bind (out-of-range backref = identity; global bad escape = consume-with-prefix). With that applied all 98 differential battery entries were byte-identical or identically-rejected.

Infrastructure: Program grew a prepare-time regex table (print/parse round-trips it; verify checks indices and rewrite presence); ReMatch/ReExtract/ReReplace on both backends (interp closures own their compiled Regex; cranelift helpers read a CompiledRe table through Cx). Semantics per pins: regexp_matches = unanchored SEARCH, regexp_full_match / ~ / !~ / SIMILAR TO = FULL match (~ is NOT the Postgres search — the DuckDB binder error names regexp_full_match; SIMILAR TO translates NO wildcards), regexp_extract '' on no-match with the flat 0..9 group check and NULL-group -> '', regexp_replace backslash backrefs with $-literals and the NULL-options -> NULL asymmetry, options alphabet c/i/l/s live with m/n/p as measured no-ops, constant patterns compile at prepare (pinned eagerness). Star forms: * SIMILAR TO = unanchored name search, * NOT SIMILAR TO = NOT full-match (independent predicates — pinned non-complement asymmetry), bare COLUMNS('re')/COLUMNS(*) expand in declared order with alias stamping into the wave-5 dedup. List-valued forms (extract_all, split_to_array) and COLUMNS-in-expression stay clean-unsupported.

Gate: 154 Rust + 557 py green, interp backend identical (548 minus the backend-identity guards), clippy clean on wave files, serving-bench parity gate passed with spec in the usual 1.4-2x band. Remaining pool after the wave: aggregation 45 + table-fns 24 (out of scope), lists/structs 25 (wave C), dynamic self-joins 10, small tails.
<!-- SECTION:FINAL_SUMMARY:END -->
