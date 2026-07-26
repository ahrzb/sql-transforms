---
id: TASK-44
title: 'Specializer M-cranelift: codegen backend behind the same interface'
status: Done
assignee: []
created_date: '2026-07-25 02:31'
updated_date: '2026-07-26 07:55'
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
- [x] #1 Every v0-subset prepared query compiles under cranelift and agrees with the interpreter backend on the corpus and on randomized inputs
- [x] #2 Uncovered ops fall back to the interpreter backend rather than failing prepare
- [x] #3 p50/p99 ns/call reported at n in {1, 8, 64, 1024} against the interpreter control and the existing native + codegen engines
- [x] #4 mise gate-specializer green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stretch plan (recorded 2026-07-26, design doc §7 + §10):
1. ABI + scalar spine: add cranelift-jit 0.126 (+module/codegen/frontend); exec/cranelift.rs. One JIT'd fn per program, called per row: extern "C" fn(ctx: *mut RowCtx) -> i64 (0 = emit, 1 = skip, 2+k = trap k; trap messages in a side table). RowCtx carries column base pointers, row index, output sink, arena. Coverage-first op strategy: inline CLIF only where trivially safe (const, float arith, cmp via duck_fcmp helper or inline, select, conversions); EVERYTHING nontrivial (checked int arith, strings, probes, loads/stores) via extern "C" helpers shared with the interpreter's semantics — correctness and full coverage first, inlining hot ops is a later measured optimization. ponytail: helper-call backend that agrees beats an inline backend that diverges.
2. CFG: IR blocks/params map 1:1 to CLIF blocks/params (the IR was shaped for this); Brif/Jump/Emit/Skip/Trap terms. compile_cranelift(p, statics) -> Result<CompiledFn, Unsupported>; DuckDBInferFn tries cranelift, falls back to interpreter (AC #2), exposes which backend ran for tests.
3. Differential: gen.rs random-IR fuzz interpreter-vs-cranelift (same seeds, byte-identical outputs incl. NaN/-0.0/arena strings); corpus replay + duck_check suite through the cranelift backend; gate green.
4. Bench per §10: baseline the boundary with a no-op f first, then p50/p99 ns/call at n in {1,8,64,1024}, interpreter as control, next to the existing native + codegen engines; committed as a script, numbers into the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All four stretches landed in one session (commits 6fd137b/0362b37/222138c on claude/specializer-m-cranelift, PR #27). Backend shape: one JIT'd fn per program, called per row — extern "C" fn(*mut Cx) -> i64 (0 emit / 1 skip / 2 helper trap / 3+k term trap); Cx is repr(C), the generated code reads only trap_flag at offset 0 (checked after every fallible helper). Coverage-first: consts/float arith/int cmp/logic/select/itof/fabs inline in CLIF; everything nontrivial calls extern helpers delegating to the interpreter's own semantic fns (casemap, substr_window, duck_fcmp, DuckF64, arena) — backends cannot drift where they share code. Strings = (off,len) i64 pairs; IR blocks map 1:1 to CLIF blocks+params. CraneliftFn owns a compiled InterpFn (checks, statics, fallback). The 500-seed random-IR differential caught two real bugs pre-landing: load.opt and sload.opt must normalize payloads to type defaults under a false flag. DuckDBInferFn compiles cranelift-first with interpreter fallback (AC #2) + SPECIALIZER_FORCE_INTERP knob + .backend getter; the whole pytest suite (607 + corpus 53/625/0) now exercises the JIT. Bench (scripts/bench_specializer.py, §10 discipline): pydantic boundary dominates end-to-end — noop passthrough within ~20% of a full join; cranelift==interp through Python; both beat native/codegen ~1.5-2x at larger n. Raw compute isolation (backend_compute_bench, ignored test): cranelift 13.2 ns/row vs interp 37.6 ns/row = 2.8x — the JIT win is real and waiting behind the boundary (M-boundary's job). MILESTONE COMPLETE pending review; hard stop before M-boundary.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M-cranelift delivered on claude/specializer-m-cranelift (PR #27): cranelift-jit 0.126 backend with FULL IR instruction coverage behind the same interface. Per-row extern "C" ABI; coverage-first op strategy (inline CLIF where trivially safe, shared-semantics extern helpers elsewhere — the two backends physically share the semantic functions). 500-seed interpreter-vs-cranelift random-IR differential green (caught two real payload-normalization bugs before landing); whole pytest suite incl. 678-case corpus replay (53 match / 625 clean-unsupported / 0 FAIL) runs on the JIT; interpreter fallback + force knob wired (AC #2). Bench per design doc §10 at n in {1,8,64,1024} vs interp control + native + codegen: boundary dominates end-to-end (JIT==interp through Python, both ~1.5-2x ahead of the existing engines); raw compute isolated: 13.2 vs 37.6 ns/row = 2.8x for cranelift. Gate green (cargo + 607 pytest, 12 xfail). Next milestone (M-boundary, generated row marshaller) is where the compute win becomes end-to-end visible.
<!-- SECTION:FINAL_SUMMARY:END -->
