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
- [ ] #1 Each shipped function/predicate has measured DuckDB pins recorded as duck_check tests before its implementation landed (domain edges, NULL, special floats)
- [ ] #2 Interpreter and cranelift agree byte-identically on all new ops (shared semantic fns; 500-seed differential extended to cover them)
- [ ] #3 The four serving scenarios' compromises lists re-audited: every gap this wave claims to close is exercised by an upgraded scenario feature or a new duck_check
- [ ] #4 Corpus replay: predicate/function first-blocker cases flip to match or to a named deeper blocker; zero FAILs; new tally recorded here
- [ ] #5 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26). Measurement BEFORE implementation, per the builtin-pins discipline; a parallel measurement fleet pins each family against DuckDB 1.5.5 first, and lowering decisions are finalized from those pins.
1. Measurement fleet (6 families, parallel): log/ln/log2/log10/exp; floor/ceil/trunc + round(x, digits); pow/sqrt (+ the ^ operator); sin/cos; string search (instr/position/strpos/contains/starts_with/ends_with/length); predicates (BETWEEN incl. NULL bounds and NOT, IN-list incl. NULL-element three-valued logic and NOT IN, least/greatest NULL policy). Each family delivers: measured pin table (edges: 0/negative domains, NULL, NaN, +-0.0, +-inf, int vs float overloads, return types, error messages), draft duck_check tests, and a lowering proposal (frontend desugar vs new IR op with the exact Rust semantics fn).
2. Frontend desugars first (no IR changes where the pins allow): BETWEEN -> >= AND <= under Kleene; IN-list -> OR chain of equalities (three-valued logic makes this exact); NOT variants; least/greatest per measured NULL policy (CASE-chain if NULL-ignoring is expressible, else an IR op). Corpus predicates flip here.
3. New IR ops for the rest: math unaries/binaries and string-search ops, implemented ONCE as shared semantic functions used by both backends (interp closures + cranelift helpers), fuzz generator extended so the 500-seed differential covers every new op; catalogue entries with the pinned edge/trap behavior.
4. Re-audit the four serving scenarios' compromises lists (AC #3): upgrade scenario features the wave unblocks (log1p amount/fare/sales, floor decade bins, instr title extraction, IN-list flags), keeping three-way parity green; corpus re-tally into this ticket (AC #4); gate green (AC #5).
<!-- SECTION:PLAN:END -->
