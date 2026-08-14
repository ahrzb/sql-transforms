---
id: TASK-111
title: >-
  Lower shape="many" in cranelift and retire the second eval loop
status: To Do
assignee: []
labels:
  - m-8
dependencies: []
type: task
ordinal: 97000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`shape="many"` (join multiplicity, stage B) is the ONE thing that only the
interpreter can execute. `cranelift::compile_ext` refuses multiplicity
programs up front — `MultiMap`/`BatchMap` statics, `Term::EmitTo`,
`Inst::ProbeRange`/`ProbeRead` — so the caller's fallback runs them:

```python
DuckDBInferFn(join_sql, ..., shape="many").backend   # "interpreter"
DuckDBInferFn(scalar_sql, ...).backend               # "cranelift"
```

Everything else already compiles. Measured 2026-08-14: of 137
fuzz-generated programs that built, 137 chose cranelift; cranelift's
instruction dispatch is an exhaustive `match` over `Inst`, so there is no
coverage gap to fall back FOR. (The old "an uncovered op must not fail
prepare" rationale in the comments was wrong; corrected in the same commit
that added this ticket.)

Why it was never written: the JIT'd row function is shaped one row in ->
0 or 1 row out. `many` needs a loop with a back-edge inside the compiled
function plus output buffers that grow at runtime. The interpreter gets
this for free — emitting N rows is a jump in its eval loop:

```rust
CTerm::EmitTo { to, moves } => { emitted += 1; do_moves(ctx.regs, moves); bi = *to; }
```

Cranelift compiles loops fine; this is unwritten, not blocked.

## Why bother

1. `many` queries currently run on the slow backend. Proxy from the
   2026-08-13 serving bench (interp vs cranelift on the scalar path):
   titanic n=1024 4.77M vs 3.63M ns, house_prices 7.5M vs 5.1M — roughly
   25-35% left on the table for every multiplicity query.
2. It retires a genuine duplicate: interp.rs and cranelift.rs each carry
   40 `Inst::` arms of the same semantics. Every new instruction is
   written twice today.

Note what does NOT go away, and must not be deleted with it:
`exec/kernels.rs` (the DuckDB-exact scalar semantics BOTH backends call)
and the compile front half in interp.rs (verify + prepare statics +
regexes, which cranelift runs first). Those were split out of interp.rs
precisely so this ticket's scope is unambiguous.

## Decide first, before coding

- **Is `many` worth compiling at all?** The serving bench has no `many`
  scenario, so the 25-35% number above is a proxy from scalar queries. Add
  a multiplicity scenario to benchmarks/ and measure the real gap first —
  if `many` is rare in serving, the honest answer may be to keep the eval
  loop deliberately and delete this ticket.
- **Does the eval loop have a second life as the WASM executor?** The
  cranelift JIT cannot run inside a wasm sandbox, so a Rust->wasm build of
  the engine would need an interpreter. UNVERIFIED for this crate — check
  before retiring anything.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a `many` scenario exists in benchmarks/ and the real interp-vs-cranelift gap is measured, BEFORE any lowering work
- [ ] #2 EmitTo / ProbeRange / ProbeRead / MultiMap / BatchMap lower in cranelift; the multiplicity refusal in compile_ext is deleted
- [ ] #3 a `many` build reports backend == "cranelift" and serves values byte-identical to the interpreter (differential over the existing stage-B suite)
- [ ] #4 the row-out contract handles N rows per input row without a per-row allocation on the hot path
- [ ] #5 20k campaign: no new divergence classes, `many`-tagged cases included
- [ ] #6 decide explicitly, on the WASM question, whether the eval loop is deleted or kept; if kept, its module doc says why in one line
<!-- AC:END -->
