# SQL Transform v2 Design Spec

## Goal

Extend `SQLTransform` to support:

1. **Sklearn transformers callable from SQL** — `SELECT svd(tfidf(text)) AS embedding FROM data`
2. **JOINs as inference-time lookups** — `FROM data JOIN ref ON data.id = ref.id` materialized at fit time, used as lookup dicts at inference

## Architecture

```
                     SQL string
                         │
          ┌──────────────┴──────────────┐
          │  sqlglot parse              │
          │  - extract transformer calls│
          │  - build dependency DAG     │
          │  - strip calls → clean SQL  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │  DataFusion parse + execute │
          │  - window aggs (plan walk)  │
          │  - JOIN handling (plan walk)│
          └──────────────┬──────────────┘
                         │
          ┌──────────────┴──────────────┐
          │  State extraction + codegen │
          │  - window agg values        │
          │  - JOIN lookup dicts        │
          │  - fitted transformers      │
          │  - generate Python infer fn │
          └──────────────┬──────────────┘
                         │
                    callable: (row) -> dict
```

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

tfm.fit(train)              # fits transformers, extracts state
out = tfm.transform(test)   # pure Python, uses fitted state
row = tfm._infer({"text": "hello", "age": 30, "id": 1})  # single row
```

### Builder methods

- `SQLTransform(sql: str)` — constructor, stores SQL string
- `register_transformer(name: str, transformer) -> Self` — registers an already-instantiated sklearn transformer. The `name` must match the function name used in SQL (e.g., `"tfidf"` → `tfidf(text)`).
- `add_table(name: str, table: pa.Table) -> Self` — registers a pyarrow table referenced in SQL JOINs.

### From file

```python
tfm = SQLTransform.from_file("features.sql")
```

### Error conditions

- `transform()` or `_infer()` before `fit()` → `RuntimeError`
- Registered transformer name not found in SQL → warning, no-op
- JOIN right-side lookup misses on a row key → `KeyError` (strict 1-1 assumption)

## SQL Format

### Allowed patterns

```sql
SELECT
    -- Plain column pass-through
    col,

    -- Window aggregates (constant or partitioned)
    col / MEAN(col) OVER () AS norm,
    MEAN(target) OVER (PARTITION BY city) AS city_enc,

    -- Transformer calls (nested composition supported)
    svd(tfidf(text)) AS embedding,
    countvec(text) AS bow,

    -- JOIN column references (FROM data JOIN ref ON ...)
    ref.col

FROM data
[JOIN registered_table ON data.key = registered_table.key [AND ...]]
```

### Constraints

- `FROM data` must reference the main table (passed to `fit(table)`).
- All JOIN right-side tables must be registered via `add_table()` before `fit()`.
- JOINs must be 1-1 between `data` and each right-side table. Multi-row matches per data row are a `ValueError` at fit time.
- Transformer calls use comma-separated args (no Python kwargs in SQL — kwargs passed at `register_transformer` time).
- Transformer names must not collide with DataFusion built-in function names.

## Internal Components

### `_transformers.py` (new)

Parses SQL text via sqlglot to extract transformer call chains.

```
Input:  "SELECT svd(tfidf(text)) AS embedding FROM data"
Output: {"embedding": TransformNode(name="svd", args=[TransformNode(name="tfidf", args=["text"])])}
```

**Parse flow:**
1. Parse SQL with sqlglot
2. Walk SELECT expressions, find function calls matching registered transformer names
3. Build a tree of `TransformNode` objects (dependency DAG)
4. Strip transformer calls from SQL AST, replace with `1 AS alias` placeholders
5. Generate cleaned SQL string for DataFusion

**Fit flow:**
1. Topological sort transformer nodes (leaf-first: tfidf before svd)
2. For each node in order:
   a. If leaf (raw column arg): collect column values from training data
   b. If internal (another transformer arg): transform using previous node's output
   c. Fit transformer on collected/transformed values
3. Store fitted transformers in state dict

### `_state.py` (rewrite)

Replaces regex-based `display_indent()` parsing with typed AST walking.

```
Input:  DataFusion logical plan (plan.to_variant())
Output: state dict
```

**Window aggs:** Walk `Projection` → `Window` → `WindowExpr` → extract `fn`, `col`, `partition_by`. Execute separate queries for constants/lookups. No more regex.

**JOIN lookups:** Walk plan for `Join` nodes. For each join on `data`:
1. Get right-side table name
2. Walk ON condition — must be `AND` of equality expressions (`data.col = ref.col`). Complex conditions (non-equality) → `ValueError` at fit time.
3. Extract left keys (data cols) and right keys (ref cols) from equalities
4. Materialize right table from registered pyarrow table: `{(key_tuple): {col: val, ...}}`
5. Store in state: `{"lookup": dict, "keys": ["col1", "col2"]}`

**State shape:**

```python
{
    # Window aggs (unchanged structure)
    "age_norm": 30.0,
    "city_enc": {"lookup": {"tehran": 3.5}, "partition_col": "city"},

    # JOIN lookups
    "ref": {
        "lookup": {
            (1,): {"avg_temp": 22.5, "population": 1000},
            (2,): {"avg_temp": 18.0, "population": 2000},
        },
        "keys": ["id"],
    },

    # Fitted transformers
    "tfidf": fitted_TfidfVectorizer,
    "svd": fitted_TruncatedSVD,
}
```

### `_codegen.py` (rewrite)

Walks DataFusion plan projections via typed AST (no regex column detection).
Merges in transformer metadata for non-SQL columns.

**Expression sources, in priority order:**

| Plan expression | Codegen output |
|---|---|
| `Column` with table qualifier (`ref.col`) | `_state["ref"]["lookup"][keys]["col"]` |
| `Column` without qualifier (`col`) | `row["col"]` |
| `Alias` referencing window agg | `_state["alias"]` or `_state["alias"]["lookup"][row["part_col"]]` |
| BinaryExpr (arithmetic) | `(left_expr op right_expr)` |
| Transformer (from metametadata, not plan) | `_state["tfm"].transform([arg])[0]` (nested via walrus) |

**Nested transformer codegen:**

```sql
svd(tfidf(text)) AS embedding
```

```python
(_t0 := _state["tfidf"].transform([row["text"]])[0],
 _state["svd"].transform([_t0])[0])[1]
```

Each inner call is bound to a temp variable via walrus, outer call uses it.

**Multi-value outputs:** Transformers return whatever `.transform()` returns (sparse vector, array). DataFusion column type: `List` or `FixedSizeList`. No automatic column explosion — user pipes to another transformer or indexes manually.

### `__init__.py` (update)

New builder methods + fit/transform integration.

```python
class SQLTransform:
    def __init__(self, sql: str) -> None
    def register_transformer(self, name: str, transformer) -> Self
    def add_table(self, name: str, table: pa.Table) -> Self
    def fit(self, table: pa.Table) -> Self
    def transform(self, table: pa.Table) -> pa.Table
    def _infer(self, row: dict) -> dict
    @classmethod
    def from_file(cls, path: str) -> SQLTransform
```

**fit() flow:**
1. Parse SQL with sqlglot → extract transformer calls, build DAG
2. Clean SQL → strip transformer calls, replace with placeholders
3. Pre-scan tables for transformer fitting (in DAG order)
4. Register cleaned SQL tables in DataFusion session (main + add_table'd tables)
5. Execute cleaned SQL → get logical plan
6. Walk plan → extract window agg values + JOIN lookups (typed AST)
7. Merge all state: window aggs + JOIN lookups + fitted transformers
8. Generate inference function via codegen

**transform() flow:**
1. Assert fit() called
2. Iterate rows, call inference function per row
3. Return pyarrow Table from output rows

## Dependency Changes

- **Add sqlglot** back for transformer call parsing (`pyproject.toml`).

## Module Map

| File | v1 behavior | v2 behavior |
|---|---|---|
| `_state.py` | regex on `display_indent()` → window agg values | typed AST walk → window agg values + JOIN lookups |
| `_codegen.py` | regex-based column detection, plan walk for expressions | typed AST walk for all columns + merged transformer metadata |
| `_transformers.py` | N/A | sqlglot parse → DAG → fit → provide metadata for codegen |
| `__init__.py` | `fit(pa.Table)` + `transform(pa.Table)` | +`register_transformer`, +`add_table`, multi-phase fit |

## Testing Strategy

### Unit tests

| Module | Tests |
|---|---|
| `_transformers.py` | Parse simple call, parse nested call, parse multiple independent calls, DAG ordering, SQL cleaning with placeholders, fit single transformer, fit nested chain, arg collection from column |
| `_state.py` | Window agg (constant), window agg (partitioned), multi-window-agg, single-key JOIN lookup, multi-key JOIN lookup, mixed window+join |
| `_codegen.py` | Column pass-through, window agg constant, window agg partitioned, single-key JOIN column, multi-key JOIN column, single transformer, nested transformer, mixed all three, simple arithmetic |
| `__init__.py` | Full pipeline with all three sources, transformer composition, JOIN lookup on unseen data, fit returns self, from_file, error on fit-before-transform |

### Integration test

End-to-end: `SELECT svd(tfidf(text)) AS emb, age / MEAN(age) OVER() AS norm, ref.temp FROM data JOIN ref ON data.id = ref.id` — fit on training, transform on test batch, _infer single row. Verify all three output columns.