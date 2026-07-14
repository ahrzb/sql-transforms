# Phase 2: Rust SQL Interpreter — Design Spec

## Goal

A Rust-based SQL interpreter (PyO3) that takes a DataFusion SQL query, pre-indexes static tables into lookup structures, and evaluates the query for row-based dict inputs. Output matches DataFusion semantics exactly — testable by comparing against DataFusion batch results.

This is Phase 2 of the SQL Transform system. Phase 1 (Python) produces a reduced SQL + learned state. Phase 2 (Rust) takes that output and executes it for row inputs.

## Architecture

```
Python                          Rust
──────                          ────

InferFn(sql,
        row_tables,      ──►   parse SQL (DataFusion)
        static_tables)         walk plan
                               optimize:
                                 static tables → lookup indices
                                 JOIN detection + validation
                               store execution plan

fn.infer(tables)        ──►    execute plan per row
                               walk plan tree
                               eval expressions
                               return rows

                    ◄──        list[dict]
```

## Public API

```python
from sql_transform._interpreter import InferFn

fn = InferFn(
    "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id",
    row_tables=["data"],
    static_tables={"ref": pa.table({"id": [1, 2], "y": [10, 20]})},
)

result = fn.infer({"data": [{"id": 1, "x": 5}]})
# → [{"x": 5, "y": 10}]
```

- `InferFn(sql, row_tables, static_tables)` — parse SQL, optimize, validate. Raises `ValueError` at build time for unsupported patterns.
- `fn.infer(tables: dict[str, list[dict]]) -> list[dict]` — execute for given row inputs. Row dict values can be any Python object (int, float, str, bool, None, nested dicts).

### Static table indexing

At build time, each static pyarrow table is converted to a columnar lookup index:

```
pyarrow table:               lookup index:
┌────┬───────┬────────────┐   {(city_id,): {"city": "Paris", "population": 2_161_000},
│ id │ city  │ population │    (2,):      {"city": "Tehran", "population": 8_694_000}}
├────┼───────┼────────────┤
│  1 │ Paris │  2_161_000 │
│  2 │ Tehran│  8_694_000 │
└────┴───────┴────────────┘
```

For a JOIN `ON data.city_id = ref.id`, at inference:
```
ref_row = ref_index[(row["city_id"],)]
result["city"] = ref_row["city"]
result["population"] = ref_row["population"]
```

The index key is a tuple of the ON condition's right-side columns (in order). Multi-key joins produce multi-element key tuples.

## Supported Plan Nodes

| Plan node | Support |
|---|---|
| `Projection` | SELECT expressions evaluated per row |
| `Filter` | WHERE predicate evaluated per row |
| `TableScan` (row table) | Returns input dict rows |
| `TableScan` (static table) | Replaced by lookup index during optimization |
| `Join` (row × row, inner, equality ON) | Cartesian product + ON filter |
| `Join` (row × static, inner, equality ON) | Replaced by `LookupJoin` during optimization |
| `CrossJoin` (row × row, no ON) | Cartesian product |
| `SubqueryAlias` | Transparent passthrough |

### Join limitations (build-time ValueError)

| Pattern | Supported |
|---|---|
| `FROM row_tbl` | Yes |
| `FROM row_tbl, row_tbl` | Yes |
| `JOIN row_tbl ON row.col = row.col` | Yes |
| `JOIN static_tbl ON row.col = static.col` | Yes (→ lookup index) |
| Multi-key ON (`a.k1=b.k1 AND a.k2=b.k2`) | Yes |
| Non-equality ON (`>`, `<`, `!=`) | No |
| `LEFT / RIGHT / FULL OUTER JOIN` | No |
| `JOIN static_tbl JOIN static_tbl` | No |
| Self-join (`data JOIN data`) | No |

## Supported Expressions

| Category | Operators |
|---|---|
| Column reference | `col`, `table.col` |
| Literals | int, float, string, bool, null |
| Arithmetic | `+`, `-`, `*`, `/`, `%` |
| Comparison | `=`, `!=`, `<`, `>`, `<=`, `>=` |
| Logic | `AND`, `OR`, `NOT` |
| Built-in functions | `UPPER`, `LOWER`, `CONCAT`, `SUBSTR`, `TRIM`, `ABS`, `ROUND`, `CAST`, `COALESCE`, `NULLIF` |

Full DataFusion feature parity deferred to future work.

## Repo Layout

```
sql-transform/
├── pyproject.toml              # [tool.maturin] config
├── Cargo.toml                  # Rust workspace
├── src/                        # Rust source
│   ├── lib.rs                  # PyO3 module entry, InferFn class
│   ├── plan.rs                 # plan walking, optimization passes
│   ├── expr.rs                 # expression evaluator
│   └── lookup.rs               # static table → columnar index
├── sql_transform/
│   ├── __init__.py             # re-exports InferFn
│   └── _interpreter.pyi        # type stubs
└── tests/
    └── test_interpreter.py     # Python-side tests
```

Build: `maturin develop` or `pip install -e .`

## Testing Strategy

Every test compares interpreter output against DataFusion batch output:

```python
def test_simple_select():
    sql = "SELECT age FROM data"
    
    # Expected: DataFusion batch
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [30]}, name="data")
    expected = ctx.sql(sql).collect()[0].to_pylist()
    
    # Actual: interpreter
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    
    assert actual == expected
```

Test matrix:

| Test | SQL pattern |
|---|---|
| Column pass-through | `SELECT col FROM data` |
| Multiple columns | `SELECT a, b FROM data` |
| Literal | `SELECT 42 AS x FROM data` |
| Arithmetic | `SELECT a + b * c AS x FROM data` |
| Built-in UPPER | `SELECT UPPER(s) AS x FROM data` |
| Built-in CONCAT | `SELECT CONCAT(a, '-', b) AS x FROM data` |
| Built-in CAST | `SELECT CAST(n AS VARCHAR) AS x FROM data` |
| Cross join (two row tables) | `SELECT a.x, b.y FROM a, b` |
| Inner join (two row tables) | `SELECT a.x, b.y FROM a JOIN b ON a.id = b.id` |
| Inner join (multi-key) | `...JOIN b ON a.k1 = b.k1 AND a.k2 = b.k2` |
| Inner join (row + static) | `...JOIN ref ON data.id = ref.id` |
| WHERE filter | `SELECT x FROM data WHERE x > 5` |
| Multi-row | Multiple input rows per table |
| Error: LEFT JOIN | Raises ValueError at build time |
| Error: non-equality ON | Raises ValueError at build time |
| Reusable fn | Same InferFn called twice with different inputs |

## Edge Cases

- **Missing key in lookup:** `KeyError` with descriptive message including the key value and table name
- **Empty row list:** Returns empty `[]`
- **Null handling:** `None` in input rows is treated as SQL NULL. Comparison with NULL → NULL (not true/false). `COALESCE` and `NULLIF` handle NULL correctly.
- **Type coercion:** Arithmetic follows Python type rules (int/int → float, etc.). String concatenation via CONCAT, not `+`.
- **Column name conflicts:** When two tables have same-named columns after a join, the plan's column references include table qualifiers. The interpreter resolves them correctly via the plan tree structure, not by name alone.