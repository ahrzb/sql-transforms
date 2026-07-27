---
id: TASK-54
title: >-
  Specializer standing differential regexp fuzzer — grammar-biased duckdb vs
  engine parity
status: Done
assignee: []
created_date: '2026-07-27 12:00'
updated_date: '2026-07-27 14:30'
labels: []
milestone: m-7
dependencies:
  - TASK-53
priority: medium
type: chore
ordinal: 48000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave B (TASK-53) shipped the regexp family on a bind-time RE2→rust-regex
translation with a measured reject list (pins:
docs/superpowers/specs/2026-07-27-waveB-regexp-pins.md). The one-time
differential battery had 98 entries; the residual risk is untested constructs
slipping through translate_pattern's pass-through path and serving wrong
answers.

Build a STANDING differential fuzzer: a pytest (bounded to ~2-5s, in the
normal gate) that generates random patterns from a grammar biased toward the
divergence-prone axes — Perl classes in/out of char classes, inline flags,
alternation, bounded repetition incl. the 1000 cap edge, escapes
(octal/hex/\Q\E/\u), Unicode properties, char-class edge shapes (POSIX
elements, rust set-notation `&&`/`--`/nested `[..]`), options strings,
replacement templates — and runs each against BOTH duckdb
(regexp_matches/extract/replace) and the engine (DuckDBInferFn on a tiny
table).

Per-case contract: identical rows, OR the engine rejected at build time
(conservative), OR both engines errored. duck-errors-while-engine-serves is
always a failure. Deterministic fixed seed, env-overridable
(REGEXP_FUZZ_SEED / REGEXP_FUZZ_N) for exploratory deep runs; failure
messages carry seed + case index + SQL for direct reproduction.

Repo process: any divergence found → the construct goes on the retrans.rs
reject list + a pin note in the wave-B spec addendum (pins discipline,
decision: never a wrong answer).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fuzzer test committed (tests/test_duckdb_regexp_fuzz.py): deterministic default seed, REGEXP_FUZZ_SEED/REGEXP_FUZZ_N overrides, bounded to ~2-5s in the normal gate
- [x] #2 Generator covers the divergence-prone axes: Perl classes in/out of classes, inline flags, alternation, bounded repetition (1000 cap edge), escapes, Unicode properties, char-class edge shapes, options strings, replacement templates
- [x] #3 Contract asserted per case: identical multiset of rows, or engine build-time rejection, or both error; duck-ok-engine-wrong and duck-err-engine-serves both fail with a reproducible message
- [x] #4 Exploratory deep run (≥20k cases across seeds) executed before landing; every divergence found lands on the retrans.rs reject list with a pin note in the spec addendum
- [x] #5 Gate green (cargo + pytest), clippy clean if Rust touched
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Standing fuzzer landed (tests/test_duckdb_regexp_fuzz.py, N=250 default ~4s
in the normal gate, seed fixed at 20260727, REGEXP_FUZZ_SEED/REGEXP_FUZZ_N
overrides, failure messages carry seed+case+SQL). First deep run found 12
divergence classes the 98-entry battery missed — including three silent
wrong-answer shapes (class set-notation `--`/`&&`/`~~`, spaced bounds
`{1, 3}`, nested-class `[`) and one DuckDB self-inconsistency (anchor-only
patterns: row path literal-optimizes `$\z` to string equality while the
constant fold matches). All fixed in retrans.rs (reject list + POSIX tracker
fix + rewrite MaxSubmatch pre-scan) and pinned:
docs/superpowers/specs/pins-waveB/fuzzer-task54.json + spec addendum.
Re-swept to ZERO divergences over 40k cases / 8 seeds. Gate: 155 Rust + 802
py green, clippy clean on retrans.rs.
<!-- SECTION:FINAL_SUMMARY:END -->
