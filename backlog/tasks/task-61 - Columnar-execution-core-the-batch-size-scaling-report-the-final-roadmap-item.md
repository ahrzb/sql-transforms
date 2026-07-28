---
id: TASK-61
title: >-
  Columnar execution core + the batch-size scaling report (the final roadmap
  item)
status: In Progress
assignee: []
created_date: '2026-07-28 03:22'
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
- [ ] #1 exec/columnar.rs executes the four serving scenarios bit-identically to the interpreter (differential over gen.rs programs + the scenario suite)
- [ ] #2 Traps fire for the first ACTIVE failing row only (masked rows never trap); WHERE/CASE/joins (1:1) covered
- [ ] #3 infer_arrow uses the columnar core when it compiles, falls back cleanly otherwise; fn.backend reports it
- [ ] #4 Bench: columnar vs row vs duckdb-on-arrow vs python twin across batch sizes with the scaling PLOT; report delivered
- [ ] #5 Full gates green; PR opened
<!-- AC:END -->
