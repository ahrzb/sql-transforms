# RFC-1: Where confit meets the dialect logical plan

**Status:** proposed. **Date:** 2026-08-13. **Author:** the m-9 session,
from AmirHossein's question: "confit might reject many of these — should
we decouple the two? For example introduce a physical plan for confit."

## Context

Two representations of SQL coexist, each behind its own gate:

- **The dialect logical plan** (`packages/confit/src/dialect/`, milestone
  m-9): bound, typed, verified, multiset-semantics; DuckDB frontend and
  DuckDB/Spark/BigQuery printers. Gates: L2 oracle-invisibility 288/678
  corpus match, L3 live-Spark 260/678, both zero-FAIL. Direction decided
  2026-08-13: representation goes **universal** — almost any DuckDB query
  gets a plan; determinism becomes a verifier *verdict* (represent + mark),
  and every consumer enforces its own policy by named refusal.
- **The specializer** (`packages/confit/src/specializer/`): confit's
  serving frontend — its own sqlparser binding, lowering to an owned SSA
  IR (`specializer/ir/`: verifier, canonical text, round-trip), then
  interpreter/Cranelift execution. Gate: corpus replay 550/678 prepared,
  three-outcome, plus the differential fuzzer.
- The JSON-AST marginalizer is a third consumer, out of scope here (its
  re-hosting is separately gated on the marginalize law).

The epic spec deliberately kept `specializer/` untouched and named its
re-hosting a future epic. Universal coverage forces the question now:
once the logical plan admits nearly everything, confit accepts a small
subset of it, and that boundary needs a home.

## Problems

1. **Two frontends drift.** The dialect binder and the specializer binder
   answer the same questions (auto-naming, literal typing, coercions)
   twice; every pin lands twice or diverges silently. The 288 vs 550
   corpus numbers are not comparable because the frontends differ.
2. **Confit's refusals have no principled seat.** Today they are frontend
   refusals. Under universal representation the frontend stops refusing,
   so confit's "I don't serve this" must live somewhere explicit — or it
   degenerates into crashes past the boundary (the C5 class).
3. **The fit plan has no home.** The epic's goal is pushing fit-time
   computation to the warehouse. Somewhere, one query's plan must split:
   which part runs remotely at fit (printed via a dialect printer), which
   part lowers to the row kernel for serving, and where the fitted
   statics (params tables) materialize between them. No current layer
   owns that split.

## Choices

**A. Lower to the existing SSA IR.** The SSA IR *is* confit's physical
plan; refusal is one function with printer discipline:

```rust
// dialect plan in, specializer program out; Unsupported is a named refusal
pub fn confit_lower(p: &Rel, cat: &Catalog) -> Result<ir::Program, DialectError>;

// today                                   // after A
sql --specializer::frontend--> ir::Program  sql --dialect::parse--> Rel
                                            Rel --confit_lower----> ir::Program
// gate: for all corpus stmts: replay(old path) == replay(new path), bit-identical
```

**Pros**
- Solves problems 1 and 2 with zero new representations.
- The refusal boundary is ONE function with the same three-outcome
  discipline as a printer — a shape the repo already knows how to gate.
- Supported by RFC-2's research: DBLAB's many-IR case needs a pass
  cross-product confit lacks; Umbra (the only live branch of that
  lineage) landed on exactly this architecture.

**Cons**
- Does NOT address problem 3. The fit split later wedges into either the
  logical plan (an optimizer pass the epic forswore) or a bolted-on
  layer — i.e. B arrives anyway, later and under pressure.
- Migrating the frontend is real risk; it is gated by bit-identical
  replay, but the gate only proves what the corpus covers.

**B. A confit-owned physical plan.** A middle layer that also owns the
fit/serve split (problem 3); SSA IR becomes the backend of `serve` only:

```rust
pub struct FitPlan {
    pub remote:  Rel,          // runs on the warehouse at fit (printed via a dialect printer)
    pub statics: Vec<Table>,   // what fit materializes (params tables)
    pub serve:   PhysPlan,     // row-kernel side, lowered to ir::Program
}
pub fn plan_fit(p: &Rel, cat: &Catalog) -> Result<FitPlan, DialectError>;
pub fn lower_serve(p: &PhysPlan) -> Result<ir::Program, DialectError>;
// PhysPlan = a third representation: verifier + canonical text + round-trip owed (ir/ recipe)
```

**Pros**
- Problem 3 gets its home before it is urgent, instead of being wedged in
  later under deadline pressure.
- A's lowering becomes a two-step (logical -> phys -> SSA) whose middle is
  independently testable.
- One place owns "what runs on the warehouse vs the row kernel", which is
  the epic's actual goal.

**Cons**
- A third representation owing the full ir/ recipe: verifier, canonical
  text, round-trip. That is the expensive part, and it is all new surface.
- Designed NOW against a fit surface (Window/Aggregate) the logical plan
  does not have until TASK-105/106 — high risk of designing against guesses.
- KPI risk: serving latency is a control, so the extra layer must compile
  away to nothing measurable, and that has to be proven, not assumed.
- RFC-2's research argues against extra IR levels for an engine this size.

**C. Coexistence (status quo).** No new code; the shared corpus is the
only bridge:

```text
sql --dialect::parse--> Rel --print--> spark/bq        (m-9, universal)
sql --specializer::frontend--> ir::Program --> serve   (untouched)
```

**Pros**
- Costs nothing now and starves nothing; both paths are gated today.
- Keeps the m-9 epic and the serving engine fully decoupled while the
  logical plan is still growing fast.

**Cons**
- Problem 1 compounds with every pin: each new scalar function, width
  rule, and auto-name case lands twice or diverges silently.
- Problem 2 arrives anyway the moment any confit consumer reads dialect
  plans — deferring the decision does not remove it.

> **ASK(shape):** A now (with B's interface sketched but unbuilt), B now,
> or C until TASK-106 lands? Recommendation: **A after TASK-105/106**
> — lower from a logical plan that already has Window/Aggregate, so the
> lowering is designed once against the real fit surface; sketch B's
> `plan_fit` signature in the spec as the named future seam, build it
> when warehouse pushdown is scheduled.

> **ASK(gates):** is corpus-replay bit-identity (550/678, no class
> changes) sufficient to retire the specializer frontend, or must the
> differential fuzzer certify the lowering with a fresh 20k campaign
> first? Recommendation: both — the fuzzer found what replay could not,
> every time it ran.

> **ASK(statics):** when the fit split lands, do fitted statics
> materialize as catalog tables the logical plan Scans (statics are just
> tables; printers need nothing new), or as a dedicated plan node?
> Recommendation: catalog tables — the marginalizer already treats them
> that way, and Scan is the one node every consumer supports.
