# Confit

SQL specialized once, served bit-exact.

Confit is a partial evaluator for row-at-a-time serving. Fixed SQL plus static
tables frozen at fit time are compiled, once, into a native function whose only
remaining input is the request row — the way a confit is cooked slowly once,
sealed, and then merely brought up to heat at service.

## The contract

For any SQL you hand it, exactly one of two things happens:

1. it serves **bit-for-bit identical to DuckDB**, or
2. it **refuses at build time**, naming the construct.

There is no third mode. Nothing is approximated, silently dropped, or
"close enough" at inference. An engine that is 99% compatible does not fail on
1% of queries — it silently corrupts some fraction of rows on queries it appears
to support, and the damage surfaces weeks later as model skew. A build-time
refusal costs one engineer one minute.

```python
from confit import DuckDBInferFn

fn = DuckDBInferFn(sql, row_model, statics={"dim": arrow_table}, shape="map")
fn.infer_rows(rows)     # row objects in, row objects out
fn.infer_arrow(table)   # pa.Table in, pa.Table out
```

## Row shapes

The shape is a build-time proof about output multiplicity, not a runtime check:

| shape | guarantee | notes |
|---|---|---|
| `map` | exactly one row out per row in | statically proven; rejects `WHERE` and inner joins |
| `filter` | 0 or 1 rows out (default) | |
| `many` | 0..N rows out | the only shape under which join multiplicity will build |

A serving stack that assumes row alignment gets a build-time error rather than a
silently misaligned batch.

## Where it wins

Per-call, against DuckDB handed a pre-built Arrow table (release build, p50,
titanic scenario — 10 input columns, 31 output columns):

| rows/call | DuckDB | Confit | |
|---|---|---|---|
| 1 | 6.58 ms | 3.3 µs | 2055× |
| 64 | 6.75 ms | 206 µs | 33× |
| 1024 | 7.75 ms | 3.42 ms | 2.3× |
| 16k–262k | — | — | DuckDB wins 3–5× |

DuckDB pays roughly 5.5–12 ms of per-query cost on every call regardless of
size; Confit pays it once at build. The crossover is around 2–3k rows per call.
Above it, large-batch analytics is DuckDB's job and deliberately ceded to it —
Confit's regime is serving.

## Correctness

- **550 of 678** statements mined from DuckDB's own test suite replay
  bit-exact, with **zero wrong answers** at every point in the project's
  history; the remainder are clean, named build-time rejections.
- Every semantic is implemented from *measured* behavior — queries executed
  against DuckDB 1.5.5 and recorded verbatim — never from documentation or
  intuition.
- `docs/known-limitations.md` has an executable twin: lifting a limitation
  breaks a test, so the document cannot drift.
- A standing differential fuzzer keeps auditing the regex translation layer.

## Backends

One IR, verified by a six-rule verifier that makes whole bug classes
unrepresentable (there is no nullable SSA type, so a three-valued-logic bug has
nothing to be expressed on). Two backends share their semantic functions, so
they cannot drift: a closure-compiled interpreter (the oracle) and a Cranelift
JIT, checked against each other by a random-program differential.
