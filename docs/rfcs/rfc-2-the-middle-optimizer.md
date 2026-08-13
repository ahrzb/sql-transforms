# RFC-2: The layer between relational optimization and codegen

**Status:** proposed. **Date:** 2026-08-13. From AmirHossein's question:
"confit operates on a small subset of SQL and is a dataflow system, so
there is a different kind of optimization problem — neither SQL-ish nor
LLVM-ish. Is there something in between?"

Research: five parallel literature sweeps (analytics dataflow IRs;
equality saturation; algorithm/schedule separation; incremental dataflow
and partial evaluation; the LMS query-compiler lineage). Claims marked
**[verified]** were reproduced against this repo; everything else is
relayed from the sweeps and carries its citation.

## Context

```
today:  SQL --specializer::frontend--> SExpr --> SSA IR --> { interp | cranelift }
                                                             ^^^^^^^^^^^^^^^^^^^^
        nothing chooses between these two, and nothing optimizes the IR
        beyond fold.rs (constant folding, trap-preserving)
```

Measured, from this repo's own reports: inference is **boundary-bound**
(FFI/pydantic dominates); the expression kernel is **memory-bound** as
trees grow (3.2 -> 19.4 ns/node); serving batches are **< 5k rows**;
large-batch columnar was ceded to DuckDB. Correctness is C2: bit-identical
to DuckDB or a named refusal, no third behavior — including trap message
text, which embeds operands in source order (`interp.rs:1655`).

**[verified] The JIT's integer core is invisible to Cranelift's optimizer.**
Float and bitwise ops are native CLIF; every integer arithmetic op is an
opaque side-effecting libcall plus a load-and-branch, per op, per row:

```rust
// cranelift.rs:1500-1504          — opaque: no fold, no GVN, no DCE, no strength-reduction
BinOp::Iadd => call_h(b, module, "h_iadd", &[cxp, x, y]).unwrap(),
BinOp::Fadd => b.ins().fadd(x, y),                    // native: full ægraph mid-end
// cranelift.rs:1397-1402          — emitted after every Iadd|Isub|Imul|Idiv|Irem|Ishl|Flogb
let flag = b.ins().load(types::I8, MemFlags::trusted(), cxp, 0);
b.ins().brif(flag, trap_exit, &[], cont, &[]);
```

**[verified]** `Ty::int_range()` already exists (`ir/mod.rs:188`) but only
`fold.rs`/`frontend.rs` consume it, for literal fitting. `ir/mod.rs`
documents op totality in **prose** (15 TOTAL annotations); no code reads it.

## Problems

1. **Nothing proves trap-freedom**, so every integer op pays a call + load
   + branch it usually does not need, and no batching/vectorization is
   legal (a trap must fire on the first offending row, in row order, with
   DuckDB's exact message).
2. **Nothing chooses the execution strategy.** Two engines exist; the
   choice is static. Compile cost is unamortizable at small batches
   (Umbra: "not a lot of time spent in execution to amortize... code").
3. **The dominant cost is the FFI boundary, and no optimizer touches it.**
4. **No cheap correctness oracle for fast paths.** New execution paths
   need a second engine to differentially test against.

## Choices

### A. Totality + interval analysis over the SSA IR (the keystone)

```rust
impl Inst { pub fn is_total(&self) -> bool { /* transcribe the 15 TOTAL doc comments */ } }
fn interval(v: Value) -> Interval;   // seeded from Ty::int_range(), which already exists

// consumer 1: emit native CLIF and delete the guard
if fits_i64(interval(a) + interval(b)) { b.ins().iadd(x, y) }   // no call, no load, no brif
else                                   { call_h(.., "h_iadd", ..); trap_check(b) }
// consumer 2: legality gate for EVERYTHING in B and C below
```

**Pros**
- Solves problem 1, and is the legality gate that makes B possible at all.
- Removes a call + load + branch per integer op per row from a walk we
  measured as memory-bound. Standalone win: no new concepts, no new layer.
- Runs before the interp/codegen split, so BOTH engines benefit and the
  cranelift ≡ interpreter differential still holds by construction.
- Inputs already exist: `Ty::int_range()` [verified], and 15 TOTAL doc
  comments to transcribe. m-8 phase 2 made the widths real; nothing
  consumes them yet.

**Cons**
- A new analysis pass is new surface to verify — and a wrong interval is a
  silently missing trap, i.e. a C2 violation, the worst failure class here.
- Rule validation has no prior art to copy: nobody validates rewrites
  against an *executable* oracle including error text (Alive2, WeTune and
  SQLSolver all verify against written-down models). We would build it.
- Pays nothing on string-shaped or float-shaped workloads.

### B. A schedule side-table (Halide's split, 1-D)

```rust
pub struct Sched { grain: Grain, engine: Engine, nulls: NullRep }
pub enum Grain  { Row, Batch(u16) }          // fused per-row | vectorized per-op
pub enum Engine { Interp, Jit }
// the IR never changes; the schedule is annotation. Nullability is already
// pure i1 algebra, so batched it is ONE mask word per 64 rows, not a vector.

#[test] fn schedules_agree() {              // <- the free oracle
    for p in corpus() { let gold = run(p, Sched::reference());
        for s in [batch(1), batch(1024), jit_all()] { assert_bits_eq!(run(p, s), gold) } } }
```

**Pros**
- Solves problem 4 for free, and it is the sweeps' best single find: all
  schedules of one program must agree bit-for-bit, so the fuzzer gains a
  differential gate **without a second engine**.
- Solves problem 2: the interp/JIT choice becomes data, picked per call
  from the batch size actually seen.
- Batched, the null lane is ONE mask word per 64 rows instead of 64
  branches — nullability is already pure i1 algebra, so this falls out.
- The IR never changes; a schedule is annotation. Cheap to revert.
- Precedent: Vectorwise micro-adaptivity (SIGMOD 2013), Weld adaptive
  predication (up to 3.75x over rule-based, VLDB 2018), InkFuse (ICDE 2024).

**Cons**
- Worthless before A — every legality guard is A's `is_total`.
- Every knob is a new way for the two engines to diverge. (The oracle
  catches it, but only if the corpus covers that shape.)
- Strings (`StrOp*`, `Slike`, regex) do not widen: the win is confined to
  the numeric + i1 + CASE subset.
- Grain/engine thresholds are machine-specific numbers that will rot; they
  need a benchmark to defend them, not a constant someone once measured.

### C. Batch the FFI boundary (not an optimizer at all)

```python
# Weld's actual answer to its own headline problem was a lazy API, not fusion:
h1 = t.transform_lazy(rows_a); h2 = t.transform_lazy(rows_b)
evaluate([h1, h2])          # ONE crossing
```

**Pros**
- Attacks problem 3 — the cost this repo's own measurements call dominant.
- Highest value per line in this RFC; needs no IR change and no analysis.
- Independent of A and B: it can land in any order.

**Cons**
- It is a user-facing API change: RED LINE, needs explicit approval with
  concrete before/after usage cases before anything is written.
- May already be covered by the Arrow-everywhere migration's batch
  tickets — check before scheduling, or two people build one thing.
- Deferred execution makes errors surface at `evaluate()` rather than at
  the call that caused them; the trap message must still name the row.

### D. Do nothing new; keep folding constants

**Pros**
- Zero risk, zero work, zero new surface to keep bit-identical.
- The engine is correct today; none of problems 1-4 is a correctness bug.

**Cons**
- The JIT keeps paying a call + load + branch per integer op, per row,
  forever — and that is measurable today, not a hypothetical.
- Problem 4 stands: every future fast path needs a second engine to test
  against, which is what makes fast paths expensive to add.

### Rejected, with reasons (so nobody re-proposes them)

```text
MLIR              C++-only passes + drags in LLVM (Cranelift was chosen to avoid it);
                  its dialects target tensor loop nests confit does not have.
                  Domain precedent is the opposite: Umbra hand-rolled its own IR.
polyhedral        needs an affine loop nest; confit has one 1-D row loop.
equality sat.     as a SEARCH: no fuel. Trap-carrying, message-exact algebra gates
                  commutativity/associativity/cancellation into near-determinism.
                  Adopt egg only when two concrete rules fight over phase order.
Diospyros         searches lane PACKING; confit's batch is N rows x one identical
                  tree — uniform, statically known. Nothing to search.
DBSP/differential hundreds of microseconds per single-record update — confit's whole
                  budget. Two nuggets kept: bilinear-with-zero-delta and the chain
                  rule (see RFC-1 / fit-serve).
Lift/RISE         headline rules reassociate reductions: illegal against DuckDB.
Weld/Voodoo       dormant 2020 / dead 2016. Read the source, do not depend.
LegoBase/DBLAB/   Scala + a modified compiler (Rep[T] staging does not exist in Rust),
LB2               all dead, and all assume 0.3-2.7 s compile budgets per query.
```

## What the research says about RFC-1

Two sweeps converged independently: **keep exactly one IR.** DBLAB's
many-IR argument is driven by a pass cross-product confit does not have
(its own rule: add a level only when two lowerings exist between the same
pair); LB2's zero-IR argument rests on staging Rust lacks, and measured
299 ms/query codegen ("three queries per second"). Umbra — the only live
branch — landed on single-pass lowering into **its own** IR plus a fast
non-LLVM backend plus an interpreter tier, i.e. confit's architecture
already. This is evidence for RFC-1's **choice A** (`confit_lower` into the
existing SSA IR, no third representation).

> **ASK(order):** recommendation is **A -> B -> C**, each independently
> measurable, with A first because it is the legality gate for B *and* a
> standalone JIT win that needs no new concepts. C is the biggest single
> number but is an API change gated on your approval. Agree, or put C first?

> **ASK(scope):** A on the specializer's `SExpr`/SSA only, never on the
> dialect logical plan (where DECIMAL(p,s) and cross-dialect printing make
> a wrong rewrite hardest to detect)? Recommendation: yes, specializer only.

> **ASK(rules):** should rule validation be a CI gate that pushes both
> sides of every guarded rewrite through DuckDB at the guard's boundary
> values (Ruler-style), before any rule lands? Recommendation: yes — the
> mutation-check lesson applies (a gate shaped like the rule cannot see a
> wrong premise).

## Caveats on sourcing

Academic PDFs resisted text extraction in the research environment; the
Kersten (VLDB 2018) per-query figures and the Vectorwise granularity
number came from secondary summaries and are **unverified** — confirm
before quoting them anywhere load-bearing. The Cranelift and `int_range`
findings above are [verified] against this repo directly.
