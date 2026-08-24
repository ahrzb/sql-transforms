# RFC: what happens to the bind-time constant refusals

Status: REOPENED 2026-08-24. Alternative A was accepted earlier the same
day on a premise ("DuckDB's binder errors at bind") that a source-level
check then REFUTED; the decision needs re-making on the corrected facts
below. A remains what is currently shipped.

## Context

TASK-99 (the trap slice) carries an acceptance criterion, written
2026-08-14, that says:

> the AST-side constant evaluators and interim emit refusals are deleted
> in the same PR

Call it the deletion criterion. Its premise was that once runtime traps
landed, the bind-time constant machinery (`eval_i32_literal`, the
constant-cast narrow-range refusal) would be interim scaffolding with
nothing left to do.

Two pieces of it did complete: `eval_i128_literal` was deleted by the
oracle-conformance sweep (`ab5a897`), and the "constant cast not
implemented" refusal was deleted in TASK-99 itself -- the decimal parser
serves those casts now. This RFC is about the two survivors:
`eval_i32_literal` and the constant-cast narrow-range refusal.

### What DuckDB actually does (measured + source-verified, 2026-08-24)

An earlier version of this RFC claimed "DuckDB's binder evaluates
constant integer arithmetic and errors at bind". That was a phase
confusion -- the measurement ran prepare and execute in one call. The
checked facts, per the v1.5.5 source and phase-separated probes (both
optimizer modes unless noted):

- `PREPARE s AS SELECT 2147483647 + 1` SUCCEEDS; the error comes from
  `EXECUTE`, raised per-row by `AddOperatorOverflowCheck`
  (`add.hpp`) inside the projection executor. Same for `2000000000 * 2`
  and for `CAST(300 AS TINYINT)` (cast raises in the vector cast loop).
- Nothing in DuckDB's binder or planner folds constant arithmetic. The
  only constant folder is the optimizer's `ConstantFoldingRule`, and it
  deliberately SWALLOWS evaluation failure
  (`ExpressionExecutor::TryEvaluateScalar`'s `catch (...)` returns
  false), leaving the expression in the plan to trap at runtime.
- Zero rows, zero errors: `SELECT 2147483647 + 1 WHERE FALSE` serves
  `[]` in both optimizer modes; an empty input table likewise. The
  expression only traps when a row actually reaches it.
- `CREATE VIEW` and `EXPLAIN` over the overflowing expression succeed.
- One optimizer-mode split: `... LIMIT 0` errors with the optimizer off
  (the dummy scan's one row hits the projection under the limit) and
  serves `[]` with it on (the plan collapses to EMPTY_RESULT).

So: DuckDB defers these expressions to runtime, full stop. A build-time
refusal on our side is STRICTER than DuckDB -- it refuses queries DuckDB
answers (`WHERE FALSE`, empty batches) and rejects at build what DuckDB
happily prepares. On the severity ladder that is rung 4, refuse where
DuckDB serves: the acceptable-cost rung, but a real cost, not parity.

### The timing correspondence

`DuckDBInferFn(...)` construction is bind, plan, and compile in one
moment, and holds everything DuckDB's prepare holds -- rows arrive later
for both engines. So we COULD match either phase: refuse at build
(stricter than DuckDB) or trap at infer time on the row that reaches the
expression (exactly DuckDB). There is no information preventing either.

One genuine asymmetry: for a CONSTANT query (no row tables), the engine
freezes the whole answer at build. There is no "first row" to defer to;
under runtime-trap semantics the build would have to bake a
trap-on-call program. That is coherent but is a design consequence B
has to own.

### Why `fold` cannot simply do the refusing (the shape of the duplication)

Both constant evaluators run at build. `fold`'s arithmetic returns
`Option<Lit>` where `None` means "would trap -- do not fold", which
deliberately conflates "guarded arm, leave for runtime" with "unguarded
constant". `fold` has no error channel and no guardedness context, so
the refusal lives in a separate raw-AST walk (`eval_i32_literal`) that
re-parses number text with i32 range rules. Two walkers know what `+`
does to constants; a semantics change must be remembered in both.

## Alternatives

### A. Keep the build-time refusals, documented as deliberate strictness

What currently ships. The honest framing after the source check: NOT
binder parity -- a chosen severity-4 divergence.

Pros:
- Fail-fast ergonomics: a query that must trap on every row that
  reaches the expression is told "this can never work" at pack time,
  not per row in production.
- Under the fuzzer's refusal-absorb rule (a refusal is acceptable where
  the oracle traps), the common case -- rows flow -- classifies clean.
- Zero code movement.

Cons:
- Refuses queries DuckDB serves: `WHERE FALSE` shapes, always-empty
  inputs. A campaign that generates those shapes will (correctly) log
  refuse-where-oracle-serves findings.
- The "parity" justification is gone; the RFC's own earlier argument
  for A was wrong.
- Keeps the two-walker duplication (see Context).

### B. Delete the refusals; trap at infer time like DuckDB does

The deletion criterion as written -- now revealed to be the EXACT-parity
option, not the divergence it was earlier painted as.

Pros:
- Bit-for-bit phase parity: builds what DuckDB prepares, traps on the
  first row that reaches the expression, serves `[]` on zero rows.
- Closes the `WHERE FALSE` / empty-batch divergence class entirely.
- Deletes the duplicated AST walk for free.

Cons:
- Loses prepare-time fail-fast: a query that can never serve a row
  builds successfully and fails per-batch in production.
- Constant queries (no row tables) need a baked trap-on-call program --
  new machinery, and the constant path's frozen-answer story gets a
  special case.
- TASK-84's agree-refuse tests and several comments assume the current
  behavior; they move with the change.

### C. Unify: make `fold` fallible and route constant overflow through it

Mechanism cleanup, orthogonal to the A/B choice but only meaningful if
refusal behavior (A) stays: one folder that distinguishes "folded",
"leave for runtime" (guarded), and "refuse the build".

Pros:
- Removes the two-walker duplication; one place knows what `+` does.
- Same observable behavior as A.

Cons:
- Threads the guardedness context through `fold` and changes every call
  site's return handling, for zero behavior change.
- Moot if B is chosen.

## Recommendation

This is now a genuine trade -- strict-at-build ergonomics (A) versus
exact phase parity (B) -- and the earlier "A because parity" argument is
dead. Weakly held: A remains defensible as a deliberate, documented
severity-4 strictness (fail-fast is worth something in a serving
engine), but B is what the contract's plain reading ("match optimizer-
off DuckDB or refuse by name") points at. AmirHossein's call.
