# RFC: what happens to the bind-time constant refusals

Status: ACCEPTED, alternative A -- AmirHossein, 2026-08-24. (Applied
provisionally in the TASK-99 PR during the 2026-08-19 overnight run;
now the decision of record. C remains available as a follow-up ticket
if the duplication bites.)

## Context

TASK-99 (the trap slice) carries an acceptance criterion, written
2026-08-14, that says:

> the AST-side constant evaluators and interim emit refusals are deleted
> in the same PR

Call it the deletion criterion. Its premise was that once runtime traps
landed, the bind-time constant machinery (`eval_i32_literal`, the
constant-cast narrow-range refusal) would be interim scaffolding with
nothing left to do.

Two facts changed under that premise before TASK-99 was built:

1. DuckDB's binder itself evaluates constant integer arithmetic and
   errors AT BIND, before any row exists (measured, optimizer off:
   `SELECT 2147483647 + 1` fails at prepare). Refusing at build is not
   scaffolding; it is what the oracle does.
2. The dual-read fuzzer now OBSERVES the difference: if we serve a
   runtime trap where DuckDB refuses at bind, the oracle classifies it
   "confit builds what DuckDB refuses", so every constant-overflow
   spelling becomes a standing finding.

Two pieces of the deletion criterion did complete: `eval_i128_literal`
was deleted by the oracle-conformance sweep (`ab5a897`) because the i128
comparison fold it fed was an optimizer emulation, and the
"constant cast not implemented" refusal was deleted in TASK-99 itself --
the decimal parser serves those casts now. This RFC is about the two
survivors.

### The timing correspondence (why bind parity is even possible)

There is no information gap between DuckDB's bind and our build. In this
engine, `DuckDBInferFn(...)` construction is bind, plan, and compile in
one moment, and at that moment we hold everything DuckDB's binder holds
at prepare: the SQL text, every schema, every constant. Rows are the
only thing that arrives later -- for BOTH engines:

```
DuckDB   PREPARE errors on 2147483647 + 1     EXECUTE traps per row
ours     DuckDBInferFn(...) refuses           infer_rows traps per row
```

A constant refusal depends only on the constants, so mirroring DuckDB's
bind error at our build is an exact correspondence, not an
approximation. (We actually know MORE at build than DuckDB's binder
does -- the static tables' full contents -- but constant refusals never
use that.)

### Why `fold` cannot simply do the refusing (the shape of the duplication)

Both constant evaluators run at build; the duplication is not
bind-vs-runtime, it is two BUILD-time walkers. `fold`'s arithmetic
returns `Option<Lit>` where `None` means "the interpreter would trap on
this -- do not fold" (fold.rs). That `None` deliberately conflates two
situations:

- `2147483647 + 1` at top level: "does not fold" should be a BIND ERROR
  (DuckDB's binder refuses it).
- the same expression inside a guarded position (a CASE arm that may
  never be taken): "does not fold" is CORRECT -- leave it as runtime
  code that traps only if a row reaches it. Turning this into a build
  error would refuse queries DuckDB serves.

`fold` has no error channel and no guardedness context, so the
context-aware refusal lives in a separate raw-AST walk
(`eval_i32_literal`) that re-parses number text with i32 range rules.
Both walkers know that `+` is checked addition; a semantics change
(TASK-122's `%` was one) must be remembered in both. That is the drift
risk named under alternative A.

## Alternatives

### A. Keep the bind-time refusals, reword their comments to "binder parity"

What TASK-99 shipped: `eval_i32_literal` and the constant-cast
narrow-range refusal stay, documented as reproducing DuckDB's own
bind-time behavior rather than as backstops.

Pros:
- Matches the oracle exactly: both engines refuse at prepare time on
  `2000000000 + 2000000000`, `2147483647 + 1`, `2000000000 * 2`
  (measured; this agreement is also what closes TASK-84).
- Keeps the campaign green -- no standing builds-what-DuckDB-refuses
  class.
- Better ergonomics than the alternative: the user hears "this can never
  work" at prepare time instead of a trap on every row.

Cons:
- Two constant-evaluation mechanisms live side by side (the AST walk and
  `fold`), which is duplication with drift risk.
- The ticket's deletion criterion is not honored as written; it needed a
  re-scope note.

### B. Delete as the criterion says, serve runtime traps instead

Pros:
- Honors the ticket verbatim; less code; one less AST walk to maintain.

Cons:
- Every constant-overflow spelling becomes a permanent fuzzer finding,
  which either pollutes campaigns or forces an allowlist in the oracle.
- Strictly worse user ergonomics for the same information (a per-row
  trap instead of a prepare-time error).
- Diverges from measured DuckDB behavior -- the thing the project
  contract forbids.

### C. Unify: make `fold` fallible and route constant overflow through it

Delete the parallel AST walk but keep the bind-time refusal behavior, by
letting the one constant folder distinguish "folded", "leave for
runtime" (guarded), and "refuse the build" (unguarded constant that
DuckDB bind-errors).

Pros:
- Removes the duplication that is A's only real cost; one mechanism, one
  place that knows what `+` does to constants.
- Same observable behavior as A.

Cons:
- The guardedness context (`in_guarded`) must thread into `fold`, and
  every `fold` call site changes its return-type handling -- a refactor
  for zero behavior change.
- Not overnight work; needs its own ticket and red tests.

## Recommendation

A now (it is what shipped); C as a follow-up ticket if the duplication
bites in practice. B trades a green campaign and binder parity for a
ticket checkbox and is not recommended. Overruling to B is one commit.
