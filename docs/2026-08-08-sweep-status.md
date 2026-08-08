# Adversarial sweep 2026-08-08 — closed

All nine confirmed findings are resolved. TASK-69 through TASK-78 are `Done`.
This is the account of what each turned out to be, kept because several of the
conclusions are not what the tickets predicted.

The pins live in `packages/confit/tests/test_known_divergences.py` (whose
module docstring carries the sweep's provenance) and
`packages/sql-transform/sql_transform/_trees_test.py`. As each finding was
fixed its `xfail` marker came off and the section above it became the account
of the fix, so those files read as history rather than as a list of
complaints.

## What each one was

| ticket | resolution |
|---|---|
| **69** | `QUALIFY` / `FETCH FIRST` / `SELECT TOP` parsed and silently dropped — refused, as a **class** |
| **70** | `CAST(DOUBLE AS BIGINT)` rounded half away from zero; DuckDB rounds half to even |
| **71** | `infer_arrow` ignored a supplied `output_model` — now refuses by name |
| **72** | `infer_arrow` emitted `large_string` where DuckDB emits `string` |
| **73** | a CFG split inside a join's own ON residual recursed until the process died |
| **74** | a trap-free `CASE` in a one-sided ON residual was refused for trapping ops it did not contain |
| **75** | `WHERE`'s AND/OR was branchless, so a guarded trapping expression still trapped |
| **76** | **not a code bug** — the SPEC overstated the refusal list; spec corrected |
| **77** | an integer feature above `2**53` double-rounded against sklearn |
| **78** | `pack_trees` ignored `feature_names_in_`, binding the wrong columns |

## The four things worth carrying forward

**Fix the class, not the instance.** TASK-69's `refuse_unhandled_query` and
`refuse_unhandled_select` destructure their sqlparser AST node exhaustively,
with **no `..` pattern**, so a clause added to sqlparser breaks the build
rather than the answers. It caught `Select::flavor` during implementation,
which a by-hand audit of the same field list had missed. Do not add `..` to
those patterns to silence a future compile error — that reintroduces the whole
bug class.

**One definition, or they drift.** TASK-74 and TASK-75 both needed "can this
expression trap". It is `plan::may_trap`, called from `bind_residual` and from
`FB::kleene`. `scan_residual` no longer answers that question at all; it keeps
only the scope questions that are genuinely its own.

**Measure a DOUBLE cast with a DOUBLE.** Two existing Rust pins asserted the
wrong rounding for `CAST(DOUBLE AS BIGINT)`, and both were written from a
DuckDB query on a bare `-2.5` literal — which DuckDB types `DECIMAL(2,1)`.
They measured the decimal cast and pinned its answer onto the double one.
Three roundings are reachable from this SQL and only one is the cast:

```text
CAST(DOUBLE AS BIGINT)   half to even        -2.5 -> -2
CAST(DECIMAL AS BIGINT)  half away from zero -2.5 -> -3
round(DOUBLE)            half away from zero -2.5 -> -3.0
```

**Run the Python suite against a DEBUG build too.** The TASK-75 fix passed the
entire suite in release and panicked in debug on `BETWEEN` and `IN`:
`non-nullable column with a flag lane`. The invariants holding the lowering
together are `debug_assert!`s, which release compiles out, so a green release
bar says nothing about them. `uv run maturin develop` (no `--release`), run
`pytest`, then put the release build back.

## Two of the sweep's conclusions were wrong, in opposite directions

Both DISPUTED findings were adjudicated by hand before any code, and they went
opposite ways — which is the argument for adjudicating rather than assuming.

**TASK-78 was real and not subtle.** A forest fitted on columns `['b','a']`
and packed as `['a','b']` builds without complaint and scores `-2.72` where
sklearn says `0.84`.

**TASK-76 was a defect in the spec, not the code.** A node with two parents is
not a cycle. Children are already forced to strictly follow their parent,
which rules out cycles by construction and is what makes traversal terminate
without a depth counter; under that rule a shared child is an ordinary
decision DAG that yields exactly what the table names. The spec's bullet
conflated three properties, and `unreachable` — one of them — really is
checked. Every other refusal the spec claims is now exercised by construction
rather than assumed; all nine hold.

## TASK-77's "design call" dissolved

The earlier handoff said TASK-77 needed a decision because the fix wants an IR
op that narrows through float32, contradicting TASK-65's "no float32 anywhere
in the engine". That premise is about the **type system** — the engine
computes in exactly `i64` / `f64` / string / bool. `itof.f32` takes i64 and
yields f64; no lane, column or static is ever f32, exactly as `ftoi.nearest`
is a rounding mode and not an integer type. Nothing had to be traded.

It is also provably a no-op below `2**53`, where `float64(n)` is exact — which
is what made it a fix rather than a trade against the common small-integer
case.

## Still open

**An integer-WIDTH divergence in `infer_arrow`'s output schema**, found while
fixing TASK-72 by widening its scenario sweep from "the string column" to "the
whole output schema". DuckDB types a bare integer literal `INTEGER`, so
`CASE WHEN .. THEN 1 ELSE 0 END` comes back `int32` for it and `int64` for us;
`pa.concat_tables` raises, exactly as it did for TASK-72. It hits the
`titanic` serving scenario's `multi_cabin` column, so it is not hypothetical
SQL.

Deliberately not folded into TASK-72 — different type, different cause. It is
the arrow-visible face of the documented "narrow integer widths don't exist"
limitation. Pinned xfail-strict, noted in `docs/known-limitations.md`, and it
**needs a ticket**. The scenario sweep allows that one widening and nothing
else, so it cannot quietly grow.

## Six findings were dropped over the verification cap

The sweep produced 18 raw findings and verified 12. The six below the cut were
never verified and are **not** recorded in tickets. They are in the workflow
output if wanted; treat them as unproven.

## How to run things

- `uv run pytest -q` from the repo root is the gate: **879 passed, 3 xfailed**.
- `cargo test --release` in `packages/confit`: **223 passed**.
- Run the Python suite against both build profiles — see above.
- Two pins run in a subprocess on purpose: their failure mode is process
  death, not an exception.
- Every scoring test must assert `fn.backend` — the cranelift compile error is
  discarded and the interpreter fallback is silent.
