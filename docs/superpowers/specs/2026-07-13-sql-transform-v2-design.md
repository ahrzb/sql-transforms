# SQL Transform v2 Design Spec

## Goal

Extend `SQLTransform` with two new concerns, split into a two-phase architecture:

1. **Sklearn transformers callable from SQL** — `SELECT svd(tfidf(text)) AS embedding FROM data`
2. **JOINs as inference-time lookups** — `FROM data JOIN ref ON data.id = ref.id` materialized at fit time

## Architecture: Two Phases

```
                        raw SQL + training data
                               │
              ┌────────────────┴────────────────┐
              │         PHASE 1: fit()           │
              │  - parse SQL (sqlglot)           │
              │  - fit transformers              │
              │  - extract window agg values     │
              │  - materialize JOIN lookups      │
              │  - produce reduced SQL + state   │
              │  - get reduced plan (DataFusion) │
              └────────────────┬────────────────┘
                               │
                    state dict + reduced logical plan
                               │
              ┌────────────────┴────────────────┐
              │       PHASE 2: interpret()       │
              │  - walk plan expressions         │
              │  - evaluate for single row       │
              │  - resolve UDFs via state        │
              │  - return {alias: value} dict    │
              └────────────────┬────────────────┘
                               │
                         row result dict
```

**Phase 1** learns from training data and emits a reduced DataFusion logical plan + state dict. The reduced plan has no window expressions, no JOINs — only table scans, arithmetic, built-in functions, and UDF calls.

**Phase 2** is a row-by-row plan interpreter. It walks the logical plan tree and evaluates each expression node for a single input row. No exec(), no code generation. Testable in isolation: compare interpret() output against DataFusion batch result on 1-row tables.

## Public API

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfm = (
    SQLTransform("""
        SELECT
            svd(tfidf(text)) AS embedding,
            age / MEAN(age) OVER () AS age_norm,
            ref.avg_temp
        FROM data
        JOIN ref ON data.id = ref.id
    """)
    .register_transformer("tfidf", TfidfVectorizer(max_features=500))
    .register_transformer("svd", TruncatedSVD(n_components=50))
    .add_table("ref", ref_table)
)

tfm.fit(train)              # Phase 1: learn state, produce reduced plan
out = tfm.transform(test)   # Phase 2: interpret per row
row = tfm._infer({"text": "hello", "age": 30, "id": 1})
```

### Builder methods

- `SQLTransform(sql: str)` — stores raw SQL string
- `register_transformer(name: str, transformer) -> Self` — registers an already-instantiated sklearn transformer. `name` matches function name used in SQL.
- `add_table(name: str, table: pa.Table) -> Self` — registers a pyarrow table for JOIN references.
- `from_file(path: str) -> SQLTransform` — classmethod, reads SQL from file.

### fit() flow (Phase 1)

```
1. Parse SQL with sqlglot → extract transformer call chains, build dependency DAG
2. Pre-scan training data → fit transformers in DAG order
3. Strip transformer calls from SQL → clean SQL with `NULL AS alias` placeholders
4. Run clean SQL through DataFusion with all registered tables
5. Walk logical plan (typed AST, no regex):
   a. Extract window agg values (constants, partition lookups)
   b. Extract JOIN conditions → build lookup dicts from right-side tables
6. Build reduced SQL:
   a. Window aggs → literal constants (age / 30.0)
   b. JOIN columns → UDF calls (ref_lookup(id, 'temp'))
   c. Transformers → UDF calls (tfidf_udf(text))
7. Register UDFs (fitted transformers, lookup functions) in DataFusion session
8. Run reduced SQL through DataFusion → get reduced logical plan
9. Store state + reduced plan on self
```

### transform() / _infer() flow (Phase 2)

```
1. For each input row:
   a. Call interpret(reduced_plan, row, state)
   b. Collect result dict
2. transform(): aggregate dicts → pyarrow Table
3. _infer(): return single dict
```

### Error conditions

- `transform()` or `_infer()` before `fit()` → `RuntimeError`
- Registered transformer name not found in SQL → warning
- JOIN lookup misses on a row key → `KeyError` (strict 1-1 assumption)
- Multi-row JOIN match per data row → `ValueError` at fit time
- Non-equality JOIN ON condition → `ValueError` at fit time

## SQL Format

```sql
SELECT
    -- Plain column pass-through
    col,

    -- Window aggregates (constant or partitioned)
    col / MEAN(col) OVER () AS norm,
    MEAN(target) OVER (PARTITION BY city) AS city_enc,

    -- Transformer calls (nested composition supported)
    svd(tfidf(text)) AS embedding,

    -- JOIN column references
    ref.col

FROM data
[JOIN registered_table ON data.key = registered_table.key [AND ...]]
```

Constraints:
- `FROM data` references main table passed to `fit()`.
- JOIN tables must be registered via `add_table()` before `fit()`.
- JOINs must be 1-1 between `data` and each right-side table.
- Transformer names must not collide with DataFusion built-in function names.

## Internal Components

### `_interpreter.py` (replaces `_codegen.py`)

Row-by-row DataFusion logical plan interpreter. The core of Phase 2.

```python
def interpret(plan: LogicalPlan, row: dict, state: dict) -> dict:
    """Evaluate a logical plan for a single input row.

    plan: reduced logical plan (no windows, no joins)
    row: {"col": val, ...} from the FROM table
    state: learned state dict (UDF implementations, pre-computed values)

    Returns: {"alias": value, ...} for each projection
    """
```

Walks the plan tree top-down:

| Node type | Interpretation |
|---|---|
| `Projection` | Evaluate each projection expression → `{alias: value}` |
| `TableScan` | Return the input `row` dict |
| `Column(name)` | `row[name]` |
| `Literal(value)` | `value` |
| `BinaryExpr(op, left, right)` | `interpret(left) OP interpret(right)` |
| `Alias(expr, name)` | `{name: interpret(expr)}` |
| Built-in function call | Map DataFusion function → Python equivalent (UPPER, CONCAT, SUBSTR, ...) |
| UDF call | `state["udf_name"](args)` — fitted transformer or lookup function |

**Built-in function mapping:** DataFusion functions mapped to Python stdlib. Initial set: UPPER, LOWER, CONCAT, SUBSTR, TRIM, ABS, ROUND, CAST, NULLIF, COALESCE. Add more as needed.

**UDF resolution:** State keys match UDF names. For fitted transformers: `state["tfidf_udf"]` is a callable that wraps `transformer.transform()`. For JOIN lookups: `state["ref_lookup"]` is a callable `(key, col) -> value`.

**Testing strategy:** For each expression type, create a SQL query, run through DataFusion on 1-row table, run through interpret() with same row, assert equality. Covers: column ref, literal, arithmetic, built-in function, UDF, alias, projection.

### `_state.py` (rewrite)

Typed AST walk of DataFusion logical plan. No regex.

```python
def extract_state(plan: LogicalPlan, ctx: SessionContext, table_name: str) -> dict:
    """Walk logical plan, extract window agg values and JOIN lookups."""
```

**Window aggs:** Walk `Projection.input()` → `Window.input()` → iterate `WindowExpr`. Extract fn name, column, partition_by columns. Run separate DataFusion queries for values.

**JOIN lookups:** Walk plan for `Join` nodes where right side matches a registered table. Extract ON equality keys. Build lookup dict `{(key_tuple): {col: val, ...}}` from registered pyarrow table.

**State shape:**

```python
{
    "age_norm": 30.0,
    "city_enc": {"lookup": {"tehran": 3.5}, "partition_col": "city"},
    "ref": {"lookup": {(1,): {"avg_temp": 22.5}}, "keys": ["id"]},
}
```

### `_transformers.py` (new)

sqlglot-based parsing and fitting of sklearn transformer calls.

```python
def extract_transformer_calls(sql: str, registered: set[str]) -> dict:
    """Parse SQL with sqlglot, find transformer calls, build DAG.

    Returns: {
        "embedding": TransformNode(name="svd", args=[
            TransformNode(name="tfidf", args=["text"])
        ])
    }
    """

def fit_transformers(nodes: dict, table: pa.Table) -> dict:
    """Fit transformers in topological order. Returns fitted instances dict."""

def strip_transformer_calls(sql: str, registered: set[str]) -> str:
    """Remove transformer calls from SQL, replace with placeholders."""
```

### `__init__.py` (update)

```python
class SQLTransform:
    def __init__(self, sql: str) -> None
    def register_transformer(self, name: str, transformer) -> Self
    def add_table(self, name: str, table: pa.Table) -> Self
    def fit(self, table: pa.Table) -> Self      # Phase 1
    def transform(self, table: pa.Table) -> pa.Table  # Phase 2
    def _infer(self, row: dict) -> dict         # Phase 2 single row
    @classmethod
    def from_file(cls, path: str) -> SQLTransform
```

Internal state after fit():
- `self._state`: combined state dict (agg values + lookup dicts + fitted transformers)
- `self._reduced_plan`: DataFusion logical plan from reduced SQL
- `self._udf_registry`: dict of `{name: callable}` for UDFs in reduced plan

### Reduced SQL Format

Phase 1 transforms the original SQL into a reduced form:

```
Original:
  SELECT svd(tfidf(text)) AS emb, age / MEAN(age) OVER () AS norm, ref.temp
  FROM data JOIN ref ON data.id = ref.id

Reduced:
  SELECT svd_udf(tfidf_udf(text)) AS emb, age / 30.0 AS norm, ref_lookup(id, 'temp') AS temp
  FROM data
```

- Window aggs → literal values from state
- Transformer calls → UDF names (e.g., `tfidf` → `tfidf_udf`)
- JOIN columns → UDF calls that do dict lookups (e.g., `ref.col` → `ref_lookup(key, 'col')`)
- The reduced SQL is valid DataFusion SQL; it is parsed once to produce the `_reduced_plan` that Phase 2 interprets.

## Module Map

| File | v2 role |
|---|---|
| `_interpreter.py` | NEW: walk plan expressions, evaluate per row, map built-in functions |
| `_state.py` | REWRITE: typed AST walk, extract window aggs + JOIN lookups |
| `_transformers.py` | NEW: sqlglot parse, DAG build, fit, strip calls |
| `__init__.py` | UPDATE: two-phase fit(), interpreter-backed transform/_infer |

## Dependency Changes

- **Add sqlglot** for transformer call parsing in `_transformers.py`.
- DataFusion >=46.0.0, PyArrow >=19.0 (unchanged).

## Testing Strategy

### `_interpreter.py` tests — the most critical

Each test: write SQL → run DataFusion on single-row table → run interpret() on same row → compare.

| Test | SQL |
|---|---|
| Column pass-through | `SELECT col FROM data` |
| Literal | `SELECT 42 AS x FROM data` |
| Arithmetic | `SELECT col / 2 AS x FROM data` |
| Built-in UPPER | `SELECT UPPER(s) AS x FROM data` |
| Built-in CONCAT | `SELECT CONCAT(a, b) AS x FROM data` |
| Multiple columns | `SELECT a, b, a + b AS c FROM data` |
| UDF call | `SELECT my_udf(col) AS x FROM data` (with registered UDF) |
| Mixed | `SELECT UPPER(s) AS up, col / 2 AS half FROM data` |

### `_state.py` tests

| Test | Covers |
|---|---|
| Constant window agg | Walk plan, extract scalar |
| Partitioned window agg | Walk plan, extract lookup |
| Multi-window-agg | Two window aggs in one query |
| Single-key JOIN | Extract 1-key lookup from JOIN plan |
| Multi-key JOIN | Extract multi-key lookup from JOIN plan |
| Mixed window+join | Both in one query |

### `_transformers.py` tests

| Test | Covers |
|---|---|
| Parse simple call | `tfidf(text) AS bow` → one node |
| Parse nested call | `svd(tfidf(text)) AS emb` → two-node chain |
| Parse multiple calls | Two independent transformer chains |
| DAG ordering | Verify topological sort |
| SQL stripping | Input with calls → valid DataFusion SQL |
| Fit single transformer | Fit TfidfVectorizer on text column |
| Fit nested chain | tfidf → transform → fit SVD on output |

### `__init__.py` tests

| Test | Covers |
|---|---|
| Full pipeline: all three sources | Window agg + transformer + JOIN in one query |
| Unseen JOIN key | KeyError on missing lookup |
| Transformer on unseen data | Transform with pre-fitted state |
| fit() returns self | Sklearn pattern |
| from_file() | Load SQL from file |
| Error before fit | RuntimeError |
| Nested transformer composition | `svd(tfidf(text))` → verify output shape |
| Single-row _infer | Match batch transform row |

### Integration test

End-to-end: `SELECT svd(tfidf(text)) AS emb, age / MEAN(age) OVER() AS norm, ref.temp FROM data JOIN ref ON data.id = ref.id` — fit on training (4 rows), transform on test batch (2 rows), _infer single row. Verify all three output columns match expected values.