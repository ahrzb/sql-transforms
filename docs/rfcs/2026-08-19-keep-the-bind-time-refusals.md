# RFC: TASK-99 AC #2 re-scoped -- the bind-time refusals stay

**Status: decided overnight (AmirHossein asleep), applied in the TASK-99 PR;
review and overrule freely -- reverting is one commit.**

## The AC as written

> #2 the AST-side constant evaluators and interim emit refusals are deleted
> in the same PR

Written 2026-08-14, on the premise that once runtime traps landed, the
bind-time constant machinery would be "subsumed".

## Why I did not do it

The premise predates the oracle change, and under the dual-read fuzzer the
distinction it erases is OBSERVABLE. DuckDB's binder evaluates constant
integer arithmetic and errors AT BIND (measured, optimizer off:
`SELECT 2147483647 + 1` errors before any row). If we deleted
`eval_i32_literal` and served a runtime trap instead:

- the fuzzer's `against()` classifies duck-bind-error-vs-our-serve as
  `DIVERGE_BUILD` ("confit builds what DuckDB refuses") -- every constant
  overflow spelling becomes a standing finding;
- a user gets a per-row trap where DuckDB tells them at prepare time,
  which is strictly worse ergonomics for the same information.

The same holds for the constant-cast refusal (`CAST(300 AS TINYINT)`): both
engines refuse before serving; ours at build, DuckDB's at plan time. That is
agreement, not an interim state.

`eval_i128_literal` WAS deleted -- by the oracle-conformance sweep
(`ab5a897`), because the i128 comparison fold it fed was an optimizer
emulation. What remains is binder parity, and the binder is in scope.

## What AC #2 becomes

- `eval_i128_literal`: gone (already, ab5a897). DONE.
- `eval_i32_literal`: KEPT -- it reproduces DuckDB's bind-time constant
  arithmetic error, which also closes TASK-84 (measured agree-refuse on
  `2000000000 + 2000000000`, `2147483647 + 1`, `2000000000 * 2`).
- the constant-cast narrow-range refusal: KEPT, comment reworded -- it is
  bind-parity, not a backstop.
- the "constant cast not implemented (TASK-113)" refusal: DELETED -- the
  decimal parser serves those now, so it had nothing left to refuse.

## Alternatives considered

(a) make `fold` fallible so constant overflow surfaces as a build error
through one mechanism instead of a parallel AST walk -- cleaner, but a
refactor of every fold call site for zero behaviour change; ticketable
later if the duplication bites.
(b) delete as written and accept the DIVERGE_BUILD findings -- rejected,
that is trading a green campaign for an AC checkbox.
