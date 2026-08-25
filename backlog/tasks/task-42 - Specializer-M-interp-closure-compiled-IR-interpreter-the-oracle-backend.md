---
id: TASK-42
title: 'Specializer M-interp: closure-compiled IR interpreter (the oracle backend)'
status: Done
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-08-08 03:38'
labels: []
milestone: m-7
dependencies:
  - TASK-41
documentation:
  - packages/confit/docs/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the interpreter backend over the imperative IR (design doc §7): one pre-traversal builds a closure tree, then execution is plain dispatch. This backend is the differential-testing oracle for every future codegen backend and the fallback for uncovered ops — correctness and coverage over speed, never optimized. Depends on M-ir for the IR definitions and verifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every IR instruction executes; the M-ir fixture programs all produce expected outputs
- [x] #2 Runs only verifier-accepted IR; rejects unverified programs
- [x] #3 No allocation during execution (arena-only for varlen), asserted by a test
- [x] #4 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Runtime substrate in src/specializer/exec/: Batch (typed columns + validity), Arena (bump String, spans as (u32,u32) StrRef), StaticData (Scalar / Map keyed by bit-hashed key tuples) type-checked against Program.statics at compile.
2. interp.rs: compile(program, statics) runs verify() first (unverified IR is rejected — AC#2), then one pre-traversal builds per-block closure lists + terminator thunks; Frame of Copy registers indexed by Value id, reused across rows; stores push straight into pre-capacitied out builders (verifier's exactly-once contract keeps columns row-aligned).
3. Semantics pins (documented in interp.rs, oracle-differential at M-lower): checked integer arithmetic traps on overflow; idiv/irem trap on 0 and MIN/-1; fdiv IEEE; fcmp IEEE-ordered (NaN compares false); ftoi.trunc toward zero, ftoi.round half-away-from-zero (DuckDB CAST), both trap out of i64 range; stoi/stof exact parse.
4. Tests: all 5 M-ir fixtures executed against hand-computed inputs/statics/outputs; unverified-program rejection; counting global allocator asserts zero heap allocs on a warmed second run (arena/regs/builders reused); executor fuzz over gen::gen_program seeds with random inputs (no panics, |out| <= |in|, deterministic).
5. Adversarial workflow (semantics vs spec, alloc discipline, trap/edge paths, fuzz), fix confirmed findings; gate green; stacked PR onto claude/duckdb-native-interpreter-36d016.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in src/specializer/exec/ (mod.rs substrate + interp.rs + tests.rs). compile() verifies first; closures built in one pre-traversal; terminators interpreted as a small enum in the row loop; register slots densely remapped at compile. Semantics pins documented and every one tested. Adversarial pass (6 agents) found and I fixed: u32 arena-span wrap silently corrupting output past 4 GiB (offsets now usize), store.opt false-flag storing the live register instead of the pinned type default, unvalidated validity-lane length, sparse value ids inflating the register frame, and an overbroad zero-allocation claim (restated: warmth is per content profile, growth one-time+monotone). RunState.emitted added. cargo 65 tests; alloc-free steady state asserted via thread-local counting global allocator; 150-seed executor fuzz deterministic.
<!-- SECTION:NOTES:END -->
