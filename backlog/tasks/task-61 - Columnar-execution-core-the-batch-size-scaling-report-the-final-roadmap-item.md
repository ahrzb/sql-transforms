---
id: TASK-61
title: >-
  Columnar execution core + the batch-size scaling report (the final roadmap
  item)
status: Done
assignee: []
created_date: '2026-07-28 03:22'
updated_date: '2026-07-28 03:56'
labels:
  - specializer
  - columnar
  - performance
dependencies: []
type: feature
ordinal: 55000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-approved finale: a vectorized execution core, then a REPORT with the batch-size-vs-performance scaling plot comparing columnar-ours, row-path-ours, duckdb-on-arrow, and handcrafted python.

Design (mask-based vectorization of the existing IR — no new IR): exec/columnar.rs compiles a verified Program into column-at-a-time kernels. The acyclic non-'many' CFG flattens to straight-line masked execution: blocks process in topo order, each with an active-row MASK; Brif splits the mask by the cond lane; block params merge predecessor arg lanes under edge masks; kernels compute whole lanes (branch-free, wasted lanes are masked); trap-capable kernels check only ACTIVE rows and report the first failing row (row-order parity with the row loop); Emit/Skip blocks mark per-row emission — gathering emitted rows in ROW ORDER reproduces the row loop's output exactly (max one emit per row without 'many').

Coverage grows kernel by kernel behind a clean compile-time rejection: anything unimplemented (incl. all stage-B multiplicity) -> the caller's existing row-path fallback, so every commit ships. infer_arrow prefers the columnar fn when it compiles; row APIs keep the row backends; fn.backend reports 'columnar'. Differential from day one: columnar vs interp over the gen.rs random-program fuzz (skip-and-count rejects) AND the serving scenarios via the existing infer_arrow==infer_rows suite.

Report at the end: bench all four engines across n = 1..262144, matplotlib plot (log-log), delivered as a doc + updated artifact edition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 exec/columnar.rs executes the four serving scenarios bit-identically to the interpreter (differential over gen.rs programs + the scenario suite)
- [x] #2 Traps fire for the first ACTIVE failing row only (masked rows never trap); WHERE/CASE/joins (1:1) covered
- [ ] #3 infer_arrow uses the columnar core when it compiles, falls back cleanly otherwise; fn.backend reports it
- [x] #4 Bench: columnar vs row vs duckdb-on-arrow vs python twin across batch sizes with the scaling PLOT; report delivered
- [x] #5 Full gates green; PR opened
- [x] #6 (amended #3) infer_arrow uses the columnar core under SPECIALIZER_COLUMNAR=1 — opt-in, not default: the measured v1 core is at row-core compute parity with a small-batch allocation cost, so default-on would regress (honest deviation, recorded in the PR + report)
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shipped as PR #48. The columnar core (exec/columnar.rs, +1942 lines, built by a worktree agent under TDD): mask-based vectorization of the existing IR, all 37 kernels, exact trap row-order parity, stage-B rejected to the row path; verified by a 500-seed interp differential + fixtures + the python infer_arrow==infer_rows suite through the core (cargo 177, pytest 622+13).

Measured verdict (benchmarks/scaling_results.json, 4 engines x 4 scenarios x n=1..262144): the v1 core computes at ROW-CORE PARITY (kernels call the same scalar helpers per row) — house ~1.35x better, titanic/fraud ~tie, store behind, plus a per-call lane-allocation cost at tiny n. Wired OPT-IN (SPECIALIZER_COLUMNAR=1) to avoid regressing the default; arrow_backend reports the engine. Next lever identified: true vectorized kernels + lane reuse.

The scaling report (the user's headline ask) is published as a mobile-friendly artifact with the dependency-free log-log per-row plot: https://claude.ai/code/artifact/40dbeb1c-f102-489e-9e65-d57d48385846 — regime map: serving calls (1-1k rows) ours by 2-3 orders (DuckDB pays ~7-12ms per query); DuckDB takes over at ~4-16k rows/call; the middle band within ~1.5x. This closes the entire user-approved roadmap: shape contract -> stage B (corpus 550/678) -> arrow boundary -> columnar foundation + report.
<!-- SECTION:FINAL_SUMMARY:END -->
