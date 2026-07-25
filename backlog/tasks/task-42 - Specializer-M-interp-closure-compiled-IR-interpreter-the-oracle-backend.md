---
id: TASK-42
title: 'Specializer M-interp: closure-compiled IR interpreter (the oracle backend)'
status: In Progress
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-07-25 18:42'
labels: []
milestone: m-7
dependencies:
  - TASK-41
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the interpreter backend over the imperative IR (design doc §7): one pre-traversal builds a closure tree, then execution is plain dispatch. This backend is the differential-testing oracle for every future codegen backend and the fallback for uncovered ops — correctness and coverage over speed, never optimized. Depends on M-ir for the IR definitions and verifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every IR instruction executes; the M-ir fixture programs all produce expected outputs
- [ ] #2 Runs only verifier-accepted IR; rejects unverified programs
- [ ] #3 No allocation during execution (arena-only for varlen), asserted by a test
- [ ] #4 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Runtime substrate in src/specializer/exec/: Batch (typed columns + validity), Arena (bump String, spans as (u32,u32) StrRef), StaticData (Scalar / Map keyed by bit-hashed key tuples) type-checked against Program.statics at compile.
2. interp.rs: compile(program, statics) runs verify() first (unverified IR is rejected — AC#2), then one pre-traversal builds per-block closure lists + terminator thunks; Frame of Copy registers indexed by Value id, reused across rows; stores push straight into pre-capacitied out builders (verifier's exactly-once contract keeps columns row-aligned).
3. Semantics pins (documented in interp.rs, oracle-differential at M-lower): checked integer arithmetic traps on overflow; idiv/irem trap on 0 and MIN/-1; fdiv IEEE; fcmp IEEE-ordered (NaN compares false); ftoi.trunc toward zero, ftoi.round half-away-from-zero (DuckDB CAST), both trap out of i64 range; stoi/stof exact parse.
4. Tests: all 5 M-ir fixtures executed against hand-computed inputs/statics/outputs; unverified-program rejection; counting global allocator asserts zero heap allocs on a warmed second run (arena/regs/builders reused); executor fuzz over gen::gen_program seeds with random inputs (no panics, |out| <= |in|, deterministic).
5. Adversarial workflow (semantics vs spec, alloc discipline, trap/edge paths, fuzz), fix confirmed findings; gate green; stacked PR onto claude/duckdb-native-interpreter-36d016.
<!-- SECTION:PLAN:END -->
