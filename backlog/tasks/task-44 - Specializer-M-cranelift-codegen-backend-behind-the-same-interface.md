---
id: TASK-44
title: 'Specializer M-cranelift: codegen backend behind the same interface'
status: In Progress
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-07-26 06:05'
labels: []
milestone: m-7
dependencies:
  - TASK-43
documentation:
  - docs/superpowers/specs/2026-07-25-sql-specializer-design.md
type: feature
ordinal: 38000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cranelift-jit backend for the imperative IR (design doc §7; cranelift-jit 0.126 spike-verified on x86_64-pc-windows-msvc 2026-07-25). StaticRef handles resolve to absolute addresses of prepare-time structures owned by the compiled artifact. Interpreter-vs-cranelift differential on random IR programs plus the full corpus; first ns/call numbers per the measurement discipline (design doc §10).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every v0-subset prepared query compiles under cranelift and agrees with the interpreter backend on the corpus and on randomized inputs
- [ ] #2 Uncovered ops fall back to the interpreter backend rather than failing prepare
- [ ] #3 p50/p99 ns/call reported at n in {1, 8, 64, 1024} against the interpreter control and the existing native + codegen engines
- [ ] #4 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26, design doc §7 + §10):
1. ABI + scalar spine: add cranelift-jit 0.126 (+module/codegen/frontend); exec/cranelift.rs. One JIT'd fn per program, called per row: extern "C" fn(ctx: *mut RowCtx) -> i64 (0 = emit, 1 = skip, 2+k = trap k; trap messages in a side table). RowCtx carries column base pointers, row index, output sink, arena. Coverage-first op strategy: inline CLIF only where trivially safe (const, float arith, cmp via duck_fcmp helper or inline, select, conversions); EVERYTHING nontrivial (checked int arith, strings, probes, loads/stores) via extern "C" helpers shared with the interpreter's semantics — correctness and full coverage first, inlining hot ops is a later measured optimization. ponytail: helper-call backend that agrees beats an inline backend that diverges.
2. CFG: IR blocks/params map 1:1 to CLIF blocks/params (the IR was shaped for this); Brif/Jump/Emit/Skip/Trap terms. compile_cranelift(p, statics) -> Result<CompiledFn, Unsupported>; DuckDBInferFn tries cranelift, falls back to interpreter (AC #2), exposes which backend ran for tests.
3. Differential: gen.rs random-IR fuzz interpreter-vs-cranelift (same seeds, byte-identical outputs incl. NaN/-0.0/arena strings); corpus replay + duck_check suite through the cranelift backend; gate green.
4. Bench per §10: baseline the boundary with a no-op f first, then p50/p99 ns/call at n in {1,8,64,1024}, interpreter as control, next to the existing native + codegen engines; committed as a script, numbers into the task.
<!-- SECTION:PLAN:END -->
