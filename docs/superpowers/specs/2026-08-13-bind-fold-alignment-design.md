# Bind-fold alignment: pure UDFs and `||`'s SQLNULL collapse

**Date:** 2026-08-13 · **Tickets:** TASK-101, TASK-102 · **Status:** decided
(AmirHossein: "do what duckdb does", bug-for-bug), pinned xfail-strict in
`packages/confit/tests/test_integer_widths.py`, implementation after the
PR #126/#127 review queue.

Both items are the same DuckDB mechanism seen from two sides: **the binder
constant-folds subexpressions, and a fold result of NULL is an untyped
SQLNULL constant** (INTEGER at every boundary, like `SELECT NULL`). Every
fact below is measured on DuckDB 1.5.5, `DESCRIBE` agreeing with execution
— bind-time, not optimizer phases.

## Measured mechanism

| probe | DuckDB | why |
|---|---|---|
| `i + NULL`, `f + NULL`, `s LIKE NULL`, `-(NULL)`, `length(NULL)`, `upper(NULL)` | BIGINT / DOUBLE / BOOLEAN / BIGINT / BIGINT / VARCHAR | promotion: NULL adopts the signature type — no collapse |
| `s \|\| NULL`, `s \|\| CAST(NULL AS VARCHAR)`, `upper(NULL) \|\| 'a'`, `nullif('a','a') \|\| 'b'`, `(CASE WHEN 1=0 THEN 'x' END) \|\| 'a'` | INTEGER | concat-specific: a foldable constant-NULL operand collapses the whole `\|\|` to SQLNULL |
| `CASE WHEN 1=0 THEN s END \|\| 'a'` | VARCHAR | column inside → operand not foldable |
| `(-1) % scalar_udf(1, 2)` | BIGINT, udf NOT executed at bind | `%` does not fold children |
| `(struct_udf(1, NULL)).f1`, udf returns None for NULL args | INTEGER, udf EXECUTED at bind (call counter) | field access folds its operand; None → SQLNULL |
| same, udf returns `{'f1': 99}` for NULL args | BIGINT, value 99 | special null handling honored through the fold — real result used, never assumed |
| same, `side_effects=True` | BIGINT, udf NOT executed at bind | purity flag gates the fold; default False = pure |
| raising pure udf under `.f1`, constant args | error at bind | fold errors are bind errors |

## TASK-101 — pure-by-default UDF bind fold

API (the RED-LINE-approved surface): protocol objects gain one optional
attribute, DuckDB's own name and default.

```python
class MyUdf:                 # unchanged: pure by default, like DuckDB
    name, takes, returns = ...
    def __call__(self, a, b): ...

class LoggingUdf:
    side_effects = True      # never executed at build; opts out of the fold
```

Rules, in order:

1. Build may execute a udf iff `getattr(u, "side_effects", False)` is
   False AND every SQL argument folds to a constant AND the call sits in
   a fold context DuckDB folds (field access measured; probe `||`
   operands / cmp / arith operands before coding — arith measured NOT
   folding).
2. The udf runs with the constant args as-is, `None` included (our
   protocol is special-style; `duckdb/mod.rs` passes `py.None()`
   through). Its real result is used: `None` → the SQLNULL channel
   (`null_of(int32)` at the extracted field); a value → constants of the
   declared field types.
3. A raised exception during the fold is a BUILD error (error-phase
   parity — the fuzzer classifies build vs run).
4. `side_effects=True` calls are opaque exactly as today.

Non-goals: DuckDB's DEFAULT null-handling registration mode (our protocol
has no equivalent and the fuzzer registers `special`); folding outside
DuckDB's measured fold contexts (would move error phases and execution
counts off-oracle); determinism enforcement (pure-by-default bakes one
build-time sample of a non-deterministic udf into the plan — same trap
DuckDB ships; docs row it).

## TASK-102 — `||` with a foldable NULL operand

Fold-pass arm: after folding a concat node's children, if any operand is
a NULL literal, replace the node with `null_of(int32)` — value-correct
(`||` propagates NULL to every row), and our fold's constant coverage is
the same bottom-up closure as DuckDB's binder foldability (constant CASE,
`nullif('a','a')`, cast-NULL all fold; column-bearing operands don't).
Deletes the §5 "NULL || NULL types as VARCHAR" row (superseded by this
decision; it also understated the divergence).

Interaction: once TASK-101 lands, a pure udf operand of `||` may fold to
NULL and should collapse the same way iff DuckDB folds there — that is
the fold-context probe both tickets share.

## Test plan

- Flip both xfail-strict pins; add the 5-spelling `||` battery and the
  foldability-boundary pin (column CASE stays VARCHAR), all live-oracle.
- Contract pins: 99-for-NULL udf folds to 99; `side_effects=True` never
  executes at build; raising udf is a build error.
- 20k campaign: seed-601418 class gone, no new classes.
