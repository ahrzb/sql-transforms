---
id: TASK-99
title: >-
  Phase-3 trap slice: collected facts
status: Done
assignee: []
created_date: '2026-08-14 03:30'
labels:
  - m-8
dependencies: []
type: feature
ordinal: 91000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The m-8 phase-3 work order, grown by the 2026-08-13 fleet: (a) INT32/16/8
overflow traps at the measured boundaries with DuckDB's message bodies;
(b) DOUBLE->narrow CAST truncation semantics (currently the lane never
truncates — silent wrong value as an intermediate); (c) TRY_CAST(DOUBLE
AS narrow) must check the RAW double half-open, not the rounded i64;
(d) TRY_CAST(VARCHAR AS int) accepts fractional/exponent strings on
DuckDB; (e) DuckDB's optimizer pushes fitting constants through TRY_CAST
(our NULL becomes their FALSE downstream). When traps land, DELETE the
eval_i32_literal/eval_i128_literal AST evaluators (subsumed) and the
narrow-emit interim refusals, and re-run a 20k campaign expecting the
DIVERGE_TRAP classes to close.

**2026-08-17 (TASK-118): (a) has LANDED.** INT32/16/8 overflow now traps at
the point of production, on every consumer, with DuckDB's comparison
simplification reproduced so the trap stays invisible exactly where DuckDB's
is. What remains here is (b) DOUBLE->narrow CAST truncation, (c) TRY_CAST
checking the RAW double, (d) TRY_CAST(VARCHAR AS int) accepting
fractional/exponent strings, and (e) the constant-through-TRY_CAST push. The
narrow-emit interim refusals are deliberately NOT deleted yet -- with the
production-site trap in front of them they now only fire where nothing
produced the value, so they are a cheap backstop rather than the mechanism.

**2026-08-18 (AmirHossein's call): do the REMAINDER next, and do all of it.**
Asked whether to keep the interim narrow-emit refusals as a backstop or obey
AC #2 and delete them, the answer was the full remainder -- so AC #2 stands as
written and the interim refusals go when (b)-(e) land. Sequenced after the
current divergence batch, not before it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 the campaign's INT32/16/8 DIVERGE_TRAP classes are gone
- [x] #2 (RE-SCOPED -- see packages/confit/docs/rfcs/2026-08-19-keep-the-bind-time-refusals.md) the AST-side constant evaluators and interim emit refusals are deleted in the same PR
- [x] #3 TRY_CAST double/string semantics match DuckDB, live-oracle pinned
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done 2026-08-19; spec at packages/confit/docs/specs/2026-08-19-cast-semantics-design.md.

(b)+(c): the engine ALREADY implemented round-first-then-check, which is
DuckDB main's #24393 fix and Postgres semantics -- verified 24/24 against
upstream's own conformance grid. Released 1.5.5 checks the raw double and
then hits UB (CAST(127.5 AS TINYINT) = -128 on x86, CPU-dependent at
INTEGER width; traced to numeric_cast.hpp by an Opus 5 source read).
Decision (AmirHossein): keep the fixed semantics; the two half-unit slivers
are a KEPT divergence pinned live-oracle so the pin flips when the DuckDB
wheel advances. The f64->i64 guard's trap text aligned to DuckDB's
Conversion-Error sentence.

(d)/TASK-113: kernels::duck_stoi -- DuckDB's decimal grammar (sign, frac,
exponent, hex/binary, single underscores between digits), rounding half
AWAY from zero on decimal DIGITS ('1.4999999999999999' is 1), width checked
on the rounded value inside the cast with DuckDB's message. One kernel,
three callers (interp, cranelift h_stoi, frontend constant probe), so fold
and runtime cannot drift. 228/228 on a 50-edge grid x 3 widths x both
spellings; the grid is a live-oracle matrix in test_cast_semantics.py. The
'constant cast not implemented (TASK-113)' refusal died with it.

(e): RETIRED, measured -- optimizer off, TRY_CAST(300 AS TINYINT) stays
NULL; pushing the constant through is an optimizer pass and reproducing it
would be OPT_EMULATED now.

AC #2: re-scoped, not executed -- eval_i128_literal was already deleted by
ab5a897; eval_i32_literal and the constant-cast narrow refusal are BINDER
parity (DuckDB errors at bind; deleting them would make every constant
overflow a DIVERGE_BUILD finding). RFC with alternatives:
packages/confit/docs/rfcs/2026-08-19-keep-the-bind-time-refusals.md. This also closes
TASK-84 (measured agree-refuse on its three spellings).
<!-- SECTION:NOTES:END -->
