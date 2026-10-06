# Confit

Confit builds SQL once and serves it bit-exact, one request at a time.

Confit is a partial evaluator for row-at-a-time serving. A build takes fixed
SQL and its static tables, and compiles them once into a native function. The
only input that remains for the function is the request row.

## The contract

For each SQL statement, one of two things happens:

1. Confit serves it **bit-exact with the oracle**: DuckDB 1.5.5 with the
   optimizer off.
2. The build **refuses** it, and the error names the construct.

The contract forbids a third mode: a build that succeeds and a function that
answers differently from the oracle. Confit does not approximate a value and
does not drop a clause. An engine that gives correct answers on 99% of queries
does not fail on the other 1%. It gives wrong values for some rows of queries that
it appears to support, and the error shows later as skew in a model. A
refusal at build time shows the problem before the first request.

```python
import pyarrow as pa
from confit import DuckDBInferFn

dim = pa.table({"country": ["NL", "DE"], "rate": [0.21, 0.19]})
row = pa.schema([("country", pa.string()), ("price", pa.float64())])

fn = DuckDBInferFn(
    "SELECT t.price * (1 + d.rate) AS gross "
    "FROM __THIS__ t LEFT JOIN dim d ON t.country = d.country",
    row_tables={"__THIS__": row},
    static_tables={"dim": dim},
    shape="map",
)
rows = [{"country": "NL", "price": 100.0}, {"country": "FR", "price": 50.0}]
assert fn.infer_rows(rows) == [{"gross": 121.0}, {"gross": None}]  # dict rows
assert fn.infer_arrow(pa.Table.from_pylist(rows, schema=row)).num_rows == 2  # Arrow
```

`row_tables` declares the request table and its Arrow schema.
`static_tables` holds the Arrow tables that the build freezes into the
function.

## Shapes

The shape is the number of output rows for each request. The build proves the
shape. Confit does not check it at run time.

| Shape | Output rows for each request | Notes |
|---|---|---|
| `map` | Exactly one | The build refuses `WHERE` and inner joins, because they can drop a row. |
| `filter` (default) | Zero or one | |
| `many` | Zero or more | The only shape that accepts a join that can multiply rows |

If a serving stack expects one output row for each input row, a wrong query
fails at build time. It does not give a misaligned batch. For example, an
inner join under `shape="map"` refuses:

```python
try:
    DuckDBInferFn(
        "SELECT t.price FROM __THIS__ t JOIN dim d ON t.country = d.country",
        row_tables={"__THIS__": row}, static_tables={"dim": dim}, shape="map")
except ValueError as refusal:
    print(refusal)
# unsupported: shape='map': INNER JOIN 'dim' drops rows on a key miss (use LEFT JOIN)
```

## Functions

A query can call the functions that you give in `udfs=`. `confit.functions`
defines these kinds:

| Class | Defined by | How Confit serves it |
|---|---|---|
| `SqlFunction` | A SQL expression over its parameters | Substitutes the arguments into the body, like a DuckDB macro. A value that the body reads more than once expands once for each call. |
| `ExternFunction` | A Python callable | Calls it. If all arguments are constants, it may compute the call once at build time. |
| `Ensemble(ExternFunction)` | Packed tree tables and their reference walk | Scores the tables natively, bit-exact with the walk |

The definition of a function is what the oracle runs (`f.register(con)`). A
subclass only adds what the engine may know. It never gives a different
meaning. Confit chooses how it evaluates a call. That choice is not part of
the contract.

```python
from confit import ExternFunction, SqlFunction, sql as S

scale = SqlFunction("scale", pa.schema([("x", pa.float64())]), pa.float64(),
                    lambda x: (x - S.lit(3.5)) * S.lit(2.0))
double = ExternFunction("double", pa.schema([("x", pa.float64())]), pa.float64(),
                        lambda x: None if x is None else (2 * x,))

fn = DuckDBInferFn(
    "SELECT scale(t.price) AS z, double(t.price) AS p2 FROM __THIS__ t",
    row_tables={"__THIS__": row},
    static_tables={},
    udfs=[scale, double],
)
assert fn.infer_rows([{"country": "NL", "price": 4.0}]) == [{"z": 1.0, "p2": 8.0}]
```

An `Ensemble` takes the model id as its first argument. It has these parts:

- `nodes` holds each split and leaf of each tree.
- `models` holds one header for each model: the base value, `sum` or `mean`,
  and `identity` or `sigmoid`.
- `compare_grid` is the grid of the thresholds: `"float32"` or `"float64"`.

Confit does not need these classes. It accepts any object with `name`,
`takes`, `returns` and `__call__`. A tree model also needs `instances` and
`tree_tables()`.

Confit does not know any ML library. A packer supplies the tables.
sql-transform's `TreeBasedTransform` packs these scikit-learn models:
`DecisionTreeRegressor`, `RandomForestRegressor`, `ExtraTreesRegressor` and
`GradientBoostingRegressor`. Its tests require scikit-learn parity at `==` on
the raw doubles. sql-transform builds on Confit, and Confit never imports
sql-transform (`tests/test_package_boundary.py`). See
[serving fitted models](../../docs/serving-fitted-models.md).

## Building SQL

`confit.sql` builds SQL as a tree, in the shape of DuckDB's Python expression
API. It renders the tree without loss of information. A constant carries its
type: `lit(0.1)` renders `CAST('0.1' AS DOUBLE)`, so the text means the value
that you gave. `to_duckdb()` gives the same expression as a DuckDB object.

```python
price = S.col("price")
q = (S.select(S.case(price.isnull(), S.lit(0.0)).otherwise(price * 2.0).alias("z"))
      .from_("__THIS__")
      .where(price > 0))
fn = DuckDBInferFn(q.sql(), row_tables={"__THIS__": row}, static_tables={})
```

## Performance

This table compares one call of Confit with one DuckDB query over a prebuilt
Arrow table. The measurements are medians (p50) from a release build, on the
titanic scenario (`benchmarks/serving_scenarios/titanic.py`): 10 input columns
and 31 output columns. The
[performance report](docs/reports/performance-report.md) gives the method and
the other scenarios.

| Rows for each call | DuckDB | Confit | Ratio |
|---|---|---|---|
| 1 | 6.58 ms | 3.3 µs | Confit 2055× faster |
| 64 | 6.75 ms | 206 µs | Confit 33× faster |
| 1024 | 7.75 ms | 3.42 ms | Confit 2.3× faster |
| 16k–262k | — | — | DuckDB 3–5× faster |

DuckDB pays about 5.5–12 ms of fixed cost on each query, whatever the number
of rows. Confit pays that cost once, at build. Both take the same time for a
call of about 2–3k rows. Above that size, DuckDB is the correct tool for batch
analytics. Confit is for serving.

## Correctness

- The corpus has 678 SQL statements from DuckDB's own test suite. Confit
  replays **543 of 678** bit-exact with **zero wrong answers**, and refuses the
  others with a named error (2026-10-05,
  [`docs/reports/corpus-counts.json`](docs/reports/corpus-counts.json)).
  `MATCH_FLOOR` in `tests/test_corpus_replay.py` sets a minimum for this
  count, so the count cannot decrease without a test failure.
- Each behavior comes from a measurement: a query that ran on DuckDB 1.5.5 and
  a record of its exact answer. No behavior comes from documentation alone.
- [Known limitations](docs/known-limitations.md) has an executable twin
  (`tests/test_known_limitations.py`). If a change removes a limitation, a test
  fails, so the document stays correct.
- A differential fuzz test (`tests/test_duckdb_regexp_fuzz.py`) generates
  random regular expressions. It requires that Confit gives the same rows as
  DuckDB, or refuses the pattern. The nightly campaign runs the random SQL
  generator against the oracle.

## Backends

All code goes through one intermediate representation (IR). A verifier with
six rules checks each program before it runs. The IR has no nullable SSA
(static single assignment) type, so a bug in three-valued logic (TRUE, FALSE
and NULL) cannot be written in the IR.

Two backends execute a verified program: a closure-compiled interpreter and
Cranelift, a just-in-time (JIT) compiler to native code. The two backends
share their semantic functions, so their meaning cannot drift. A
random-program differential test checks the two backends against each other.

## Documentation

- [Serving contract](docs/specs/serving-contract.md): the API, shapes, types,
  UDFs and admission.
- [Oracle](docs/oracle/README.md): the reference, the comparison rules, and
  the permitted numerical bounds.
- [Known limitations](docs/known-limitations.md): the constructs that Confit
  refuses.
- [The architecture of Confit](docs/reports/confit-architecture.md)
