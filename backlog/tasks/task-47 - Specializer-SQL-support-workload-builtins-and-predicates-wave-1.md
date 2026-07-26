---
id: TASK-47
title: 'Specializer SQL support: workload builtins & predicates wave 1'
status: In Progress
assignee: []
created_date: '2026-07-26 11:42'
updated_date: '2026-07-26 12:20'
labels: []
milestone: m-7
dependencies:
  - TASK-46
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
  - docs/superpowers/specs/2026-07-26-stretch4-builtin-pins.md
type: feature
ordinal: 41000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the workload ladder measured by the serving-bench scenarios (benchmarks/serving_scenarios/, each module's compromises list) plus the overlapping corpus predicates. Ranked by how many famous-solution pipelines hit the wall: (1) ln/log/log2/log10/log1p/exp — blocked features in all four scenarios (log-fare, log1p amount, log sales, skew fixes); (2) true floor/ceil/trunc — CAST rounds half-even, so decade bins / cents / week buckets are inexpressible; (3) instr/position/strpos + contains/starts_with/ends_with — title extraction, email-domain and device parsing; (4) IN (...) and BETWEEN predicates — also 72+ corpus first-blocker cases; (5) pow/sqrt (fractional) — Box-Cox and sqrt skew features; (6) sin/cos — cyclical hour/month encodings; (7) least/greatest — clamp ergonomics. Every function lands via the measured-pin discipline (builtin-pins spec): pin DuckDB 1.5.5 semantics with duck_check tests FIRST (edge cases: domain errors, NULL propagation, -0.0/NaN/inf, int/float overloads), then lower, then implement on BOTH backends via shared semantic functions. Float-y functions must match DuckDB bit-exactly or trap cleanly — the differential decides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each shipped function/predicate has measured DuckDB pins recorded as duck_check tests before its implementation landed (domain edges, NULL, special floats)
- [x] #2 Interpreter and cranelift agree byte-identically on all new ops (shared semantic fns; 500-seed differential extended to cover them)
- [x] #3 The four serving scenarios' compromises lists re-audited: every gap this wave claims to close is exercised by an upgraded scenario feature or a new duck_check
- [x] #4 Corpus replay: predicate/function first-blocker cases flip to match or to a named deeper blocker; zero FAILs; new tally recorded here
- [x] #5 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26). Measurement BEFORE implementation, per the builtin-pins discipline; a parallel measurement fleet pins each family against DuckDB 1.5.5 first, and lowering decisions are finalized from those pins.
1. Measurement fleet (6 families, parallel): log/ln/log2/log10/exp; floor/ceil/trunc + round(x, digits); pow/sqrt (+ the ^ operator); sin/cos; string search (instr/position/strpos/contains/starts_with/ends_with/length); predicates (BETWEEN incl. NULL bounds and NOT, IN-list incl. NULL-element three-valued logic and NOT IN, least/greatest NULL policy). Each family delivers: measured pin table (edges: 0/negative domains, NULL, NaN, +-0.0, +-inf, int vs float overloads, return types, error messages), draft duck_check tests, and a lowering proposal (frontend desugar vs new IR op with the exact Rust semantics fn).
2. Frontend desugars first (no IR changes where the pins allow): BETWEEN -> >= AND <= under Kleene; IN-list -> OR chain of equalities (three-valued logic makes this exact); NOT variants; least/greatest per measured NULL policy (CASE-chain if NULL-ignoring is expressible, else an IR op). Corpus predicates flip here.
3. New IR ops for the rest: math unaries/binaries and string-search ops, implemented ONCE as shared semantic functions used by both backends (interp closures + cranelift helpers), fuzz generator extended so the 500-seed differential covers every new op; catalogue entries with the pinned edge/trap behavior.
4. Re-audit the four serving scenarios' compromises lists (AC #3): upgrade scenario features the wave unblocks (log1p amount/fare/sales, floor decade bins, instr title extraction, IN-list flags), keeping three-way parity green; corpus re-tally into this ticket (AC #4); gate green (AC #5).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Progress (2026-07-26, branch claude/specializer-builtins-wave1). Landed, each oracle-pinned before implementation and green on both backends: (1) BETWEEN/IN as exact K3 desugars with whole-construct f64 unification (corpus 172->179; remaining BETWEEN cases moved to NAMED deeper blockers — COUNT aggregation shapes). (2) Fourteen math ops (ln/log/log2/log10/exp/sqrt/cbrt/sin/cos/tan/floor/ceil/trunc/pow + log(b,x)) — shared semantic fns, safe-masked NULL lowering (log family's trap domain includes the type default; Flogb masks under the COMBINED flag since NULL pre-empts every domain check), fuzz-covered incl. trap agreement; the ^ operator stays cleanly unsupported (sqlparser parses it below *, DuckDB binds it above — measured tree divergence), ** does not parse. (3) String search (instr/strpos/position incl. the needle-first SQL form, contains, starts_with/ends_with + prefix/suffix, length/char_length/len, strlen) — 1-based codepoint positions, empty-needle-matches-all, NULL-strict; the corpus REFUTED the fleet's contains-NULL blanket-error pin, re-measured to the real rule (NULL needle binds iff a non-literal Str haystack anchors resolution). (4) least/greatest as CASE+duck-order composition (NULL-ignoring, first-arg ties, NaN-above-inf) — no IR op. CORPUS: 53 (start of day) -> 172 (star) -> 179 (predicates) -> 237 match of 678, zero FAILs, gate green (cargo 129 / pytest 667). COMPLETE: round/trunc-with-digits landed with the oracle-extracted pow10 table (gen_pow10.py, ulp witnesses k=23/k=126 checked at generation; wrapping i64 semantics; round/trunc non-finite asymmetry preserved) — corpus 240/678, zero FAILs. Scenario re-audit done by a parity-gated 4-agent fleet: 47 famous-solution features restored (titanic 24->31, ames 42->54, fraud 41->57, rossmann 44->56 outputs) — ln(1+x) everywhere, decade buckets, Deotte cents, IN-set flags, domain parsing via starts_with/ends_with/instr, sin/cos cyclical encodings, least/greatest clamps; three-way parity green on all four. Final gate: cargo 129 / pytest 671 / corpus 240 match, 0 FAIL.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wave 1 complete on claude/specializer-builtins-wave1. Shipped, all measured-first: BETWEEN/IN/least/greatest as exact compositions (zero IR); 14 math ops + log(b,x) + round/trunc-with-digits (oracle pow10 table) + 6 string-search ops on both backends via shared semantic fns, fuzz-covered incl. trap agreement. Corpus 53 -> 240 match of 678 across the day (star 172, predicates 179, builtins 240), zero wrong answers at every step. Two measured rejections ship as named unsupported: the ^ operator (sqlparser precedence contradicts DuckDB) and ** (does not parse). One fleet pin was refuted by the corpus and re-measured (contains-NULL overload anchoring). The four serving scenarios regained 47 real famous-solution features under the standing three-way parity gate.
<!-- SECTION:FINAL_SUMMARY:END -->
