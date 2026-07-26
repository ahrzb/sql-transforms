---
id: TASK-47
title: 'Specializer SQL support: workload builtins & predicates wave 1'
status: To Do
assignee: []
created_date: '2026-07-26 11:42'
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
