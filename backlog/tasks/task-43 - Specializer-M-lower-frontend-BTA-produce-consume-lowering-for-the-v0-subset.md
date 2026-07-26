---
id: TASK-43
title: >-
  Specializer M-lower: frontend + BTA + produce/consume lowering for the v0
  subset
status: In Progress
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-07-25 23:53'
labels: []
milestone: m-7
dependencies:
  - TASK-42
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
  - docs/superpowers/specs/2026-07-25-sql-specializer-loop-execution-design.md
type: feature
ordinal: 37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The load-bearing milestone (design doc §5): sqlparser(DuckDB dialect) frontend to relational IR; binding-time analysis that taints __THIS__ and evaluates every all-static subtree in DuckDB at prepare time, materializing Const structures (scalar / dense array / perfect hash / inline); produce/consume lowering of the dynamic frontier to imperative IR. v0 subset per design doc §4. Differential oracle is DuckDB (python pkg); the mined corpus at tests/corpus/duckdb_mined.jsonl replays under the three-outcome contract (match / clean-unsupported / FAIL) documented in scripts/mine_duckdb_corpus.py. Wide-mechanical: run per-operator fan-out via workflows per the loop-execution doc §2.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 v0-subset queries prepare end-to-end: sql + static tables -> verified imperative IR running on the interpreter backend
- [ ] #2 All-static subtrees are evaluated at prepare time: a static-tables-only query lowers to a constant emitter with no probe/filter ops in its IR
- [ ] #3 Differential suite vs DuckDB green on hand-written v0 cases; engine-vs-oracle disagreement follows the xfail-strict + ticket protocol
- [ ] #4 Corpus replay reports match / clean-unsupported / FAIL counts; zero FAILs; every unsupported rejection is a clean build-time error naming the construct
- [ ] #5 mise gate-specializer green (corpus replay wired into it)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (~5-6 stretches estimated, recorded 2026-07-26):
1. Frontend spine: sqlparser (DuckDB dialect) -> relational IR (scan/project/filter over __THIS__), reusing shape lessons from src/datafusion; end-to-end sql -> imperative IR -> interpreter for arithmetic projections.
2. 3VL lowering: SQL NULL semantics compiled to flag algebra (comparisons, AND/OR Kleene logic, CASE, IS NULL); type + nullability derivation; the correctness core.
3. BTA + statics: taint __THIS__, equi-joins to static tables lower to probes, scalar folding; Python materialization through the DuckDBInferFn shell (pa.Table -> StaticData).
4. IR builtin extensions (workflow fan-out per op): upper/lower/trim/substr/abs/round/concat + :: casts + COALESCE/NULLIF as lowerings; each new instruction lands across verifier/printer/parser/interp with pin tests.
5.-6. Differential suite vs duckdb-python (pytest backend id "specialized") + corpus replay under the three-outcome contract; grind divergences to zero FAILs; xfail-strict + ticket for genuine oracle disagreements.
<!-- SECTION:PLAN:END -->
