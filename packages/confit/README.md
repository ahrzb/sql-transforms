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
import pyarrow as pa
from confit import DuckDBInferFn

fn = DuckDBInferFn(
    sql,
    row_tables={"__THIS__": pa.schema([("a", pa.float64()), ("b", pa.int32())])},
    static_tables={"dim": arrow_table},
    shape="map",
)
fn.infer_rows(rows)     # dict-or-object rows in, dict rows out
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

## Functions

A query calls functions passed in `udfs=`. `confit.functions` names the kinds:

| class | defined by | how the engine serves it |
|---|---|---|
| `SqlFunction` | a SQL expression over its parameters | substitutes the arguments into the body, as DuckDB's macro does |
| `ExternFunction` | a Python callable | calls it (or folds a call over constants at bind) |
| `Ensemble(ExternFunction)` | the reference walk of its packed tree tables | scores the tables natively, bit-equal to the walk |

A function's definition is what the DuckDB oracle runs (`f.register(con)`);
a subclass only adds what the engine may know, never a different meaning.
How a call is evaluated is the engine's choice, not part of the contract.

```python
from confit import ExternFunction, Ensemble, SqlFunction, sql as S

scale = SqlFunction("scale", pa.schema([("x", pa.float64())]), pa.float64(),
                    lambda x: (x - S.lit(3.5)) * S.lit(2.0))

double = ExternFunction("double", pa.schema([("x", pa.float64())]), pa.float64(),
                        lambda x: None if x is None else (2 * x,))

score = Ensemble("score",
                 takes=pa.schema([("price", pa.float64()), ("sqft", pa.float64())]),
                 nodes=nodes, models=models, compare_grid="float32")

fn = DuckDBInferFn(
    "SELECT scale(t.price) AS z, double(t.price) AS p2, "
    "score(p.est, t.price, t.sqft) AS s "
    "FROM __THIS__ AS t LEFT JOIN params AS p ON t.country = p.country",
    row_tables={"__THIS__": row_schema},
    static_tables={"params": params},
    udfs=[scale, double, score],
)
```

An `Ensemble` takes the model id first. `nodes` holds every split and leaf of
every tree and `models` one header per model (base, `sum`/`mean`,
`identity`/`sigmoid`); `compare_grid` is the threshold grid (`"float32"` or
`"float64"`). The classes spell the structural protocol the engine reads, so
any object with `name`, `takes`, `returns`, `__call__` (and `instances`,
`tree_tables()` for a tree model) is accepted too.

### Building SQL

`confit.sql` builds SQL as a tree, in the shape of DuckDB's Python expression
API, and renders it losslessly: a constant carries its type
(`lit(0.1)` renders `CAST('0.1' AS DOUBLE)`), so the text means the value it
was built from. `to_duckdb()` gives the same expression as a DuckDB object.

```python
from confit import sql as S

a, x = S.col("a"), S.col("x")
q = (S.select(S.case(a.isnull(), S.lit(0.0)).otherwise(x * 2.0).alias("z"))
      .from_("__THIS__")
      .where(a > 0))
DuckDBInferFn(q.sql(), row_tables=..., static_tables=...)
```

Confit knows no ML library. A packer supplies the tables: sql-transform's
`TreeBasedTransform` packs sklearn's `DecisionTreeRegressor`,
`RandomForestRegressor`, `ExtraTreesRegressor` and
`GradientBoostingRegressor`, and its own tests hold sklearn parity at `==` on
raw doubles. The dependency runs one way — sql-transform builds on confit,
never the reverse (`tests/test_package_boundary.py`). See
[docs/serving-fitted-models.md](../../docs/serving-fitted-models.md).

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

- **539 of 678** statements (as of 2026-09-26, recorded in
  [`docs/reports/corpus-counts.json`](docs/reports/corpus-counts.json))
  mined from DuckDB's own test suite replay
  bit-exact, with **zero wrong answers**; the remainder are clean, named build-time rejections. The count
  is floored by `tests/test_corpus_replay.py` (`MATCH_FLOOR`), so it can
  grow but cannot shrink unnoticed.
- Every semantic is implemented from *measured* behavior — queries executed
  against DuckDB 1.5.5 and recorded verbatim — never from documentation or
  intuition.
- `packages/confit/docs/known-limitations.md` has an executable twin: lifting a limitation
  breaks a test, so the document cannot drift.
- A standing differential fuzzer keeps auditing the regex translation layer.

## Backends

One IR, verified by a six-rule verifier that makes whole bug classes
unrepresentable (there is no nullable SSA type, so a three-valued-logic bug has
nothing to be expressed on). Two backends share their semantic functions, so
they cannot drift: a closure-compiled interpreter (the oracle) and a Cranelift
JIT, checked against each other by a random-program differential.
