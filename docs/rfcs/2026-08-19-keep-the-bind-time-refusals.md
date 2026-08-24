# RFC: what happens to the bind-time constant refusals

Status: alternative A applied provisionally in the TASK-99 PR (decided
overnight 2026-08-19, AmirHossein asleep). Overruling is one commit.

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
letting the one constant folder surface errors.

Pros:
- Removes the duplication that is A's only real cost; one mechanism.
- Same observable behavior as A.

Cons:
- A refactor of every `fold` call site for zero behavior change.
- Not overnight work; needs its own ticket and red tests.

## Recommendation

A now (it is what shipped); C as a follow-up ticket if the duplication
bites in practice. B trades a green campaign and binder parity for a
ticket checkbox and is not recommended. Overruling to B is one commit.
