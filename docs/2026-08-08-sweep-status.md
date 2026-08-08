# Adversarial sweep 2026-08-08 — status and handoff

Where the nine confirmed findings stand, what is fixed, and what the next
person needs to know that is not obvious from the tickets.

Everything here is pinned by a test. The tickets are `backlog/tasks/task-69`
through `task-78`; the pins are `packages/confit/tests/test_known_divergences.py`
(module docstring carries the sweep's provenance) and two in
`packages/sql-transform/sql_transform/_trees_test.py`.

## Done

| ticket | what it was | resolution |
|---|---|---|
| **TASK-69** | `QUALIFY`, `FETCH FIRST`, `SELECT TOP` parsed and silently dropped — every row emitted, and `shape='map'` still certified it | refuse, as a **class** |
| **TASK-73** | a CFG split inside a join's own ON residual recursed until the process died (`0xC00000FD`) | re-seed the scalar probe cache per block + re-entry guard |

TASK-69's fix is worth understanding before touching the frontend:
`refuse_unhandled_query` and `refuse_unhandled_select` destructure their
sqlparser AST node **exhaustively, with no `..` pattern**, so a clause added to
sqlparser breaks the build rather than the answers. It caught `Select::flavor`
during implementation, which a by-hand audit of the same field list had missed.
Do not add `..` to those patterns to silence a future compile error — that
reintroduces the entire bug class.

TASK-73 also **corrected TASK-68's resolution note**, which claimed the
scalar-join cache hole was harmless. It was asserted without testing and was
wrong.

## Open, in the order I would take them

### TASK-75 — WHERE's AND/OR is branchless (crash)

`WHERE k = 0 AND <trapping expr>` evaluates the right side on every row even
though the guard excludes all of them; DuckDB short-circuits and returns `[]`.
`fn kleene` emits both operands unconditionally, `fn case` right below it
branches.

The tension is real and this is not a one-liner: branchless flag algebra is
what makes Kleene three-valued NULL semantics cheap and correct, and
short-circuiting only matters when an operand can TRAP. The likely shape is to
branch only when the right operand contains a trapping op — and the frontend
already computes a `total`-style property for residuals, which **TASK-74 is
about to touch**. Do those two together or they will fight.

### TASK-77 — integer feature above 2**53 (wrong answer)

A real gap in TASK-65. `tree_predict` binds an integer feature through
`promote_f64`, so the f32-grid compare sees `float32(float64(n))` — two
roundings — while sklearn narrows `int64 -> float32` in one.

**The arithmetic, worked out, so nobody has to redo it:**

- below `2**24`: f32 is exact, no divergence;
- `2**24 .. 2**53`: `float64(n)` is exact, so `float32(float64(n)) ==
  float32(n)` — the current code is already right here;
- above `2**53`: `float64(n)` itself rounds, and the second rounding lands
  elsewhere. Only this band is broken.

So the fix is to feed `float64(float32(n))` instead of `float64(n)`: widening
f32->f64 is exact, so the kernel's narrowing becomes the identity and the
compare is `float32(n) <= t`, which is sklearn's. It is provably a no-op below
`2**53`, which is what makes it safe.

**The catch:** that needs an IR op that narrows through f32, and TASK-65's
whole design point was "no float32 anywhere in the engine". Adding one is a
real design decision (new opcode, both backends, verifier, text round-trip) —
not a patch. The alternative, refusing integer features outright, would break
the common small-integer case that is correct today. This needs a call, not a
guess.

### TASK-74 — trap-free CASE in a one-sided ON residual (over-refusal)

`scan_residual` clears `total` for any `SKind::Case` without inspecting the
arms, so a CASE whose every arm is an integer literal is refused naming a
trapping op it does not contain. Loud, not wrong. Interacts with TASK-75 —
both want "does this expression contain a trapping op".

### TASK-70 / 71 / 72 — relayed, not personally reproduced

- **TASK-70** `CAST(DOUBLE AS BIGINT)` is half-away-from-zero, DuckDB is
  half-to-even. Careful: the `round()` BUILTIN is *correctly* half-away-from-zero
  and its wave-1 pin must not regress. Cranelift's `nearest` is already
  half-to-even; the interpreter wants `f64::round_ties_even()`.
- **TASK-71** `infer_arrow` never applies `output_model`. Note the ticket allows
  refusing by name, which is much cheaper than pushing pydantic validation
  through the columnar path and arguably more honest. The existing arrow
  differential only ever builds fns *without* an `output_model` — fixing the
  test shape matters as much as fixing the code.
- **TASK-72** `arrow::emit` hard-codes `pa.large_string()` where DuckDB emits
  `pa.string()`. Not cosmetic: `pa.concat_tables` against DuckDB output raises.
  Changing it means 32-bit offsets, so it is not a one-line type swap.

### TASK-76 / 78 — DISPUTED, adjudicate before writing code

Both were split by the sweep's own verifiers. Reproduce or refute **in
writing** first. TASK-76 (DAG-shaped node table not refused) may turn out to be
a spec error rather than a code bug — in which case correct the spec. TASK-78
(`pack_trees` ignores `feature_names_in_`) must stay conditional: the attribute
only exists when the estimator was fitted on named data, and ndarray-fitted
models must keep packing.

## Six findings were dropped over the verification cap

The sweep produced 18 raw findings and verified 12. The six below the cut were
never verified and are **not** recorded in tickets. They are in the workflow
output if wanted; treat them as unproven.

## How to run things

- `uv run pytest -q` from the repo root is the gate.
- The venv holds a **release** build. Anything that depends on
  `debug_assertions` (the alignment traps of TASK-67) needs
  `uv run maturin develop` without `--release` first, and remember to put it back.
- Two pins run in a subprocess on purpose: their failure mode is process death,
  not an exception.
- Every scoring test must assert `fn.backend` — the cranelift compile error is
  discarded and the interpreter fallback is silent.
