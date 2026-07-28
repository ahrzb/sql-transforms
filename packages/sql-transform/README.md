# sql-transform

The authoring surface for SQL feature transforms — **not implemented**.

```python
from sql_transform import SQLTransform

t = SQLTransform(sql)   # NotImplementedError
```

Every method raises `NotImplementedError`. The signatures, defaults and return
types are all that remain, and they are the contract a rebuild has to satisfy:

| method | intended behaviour |
|---|---|
| `SQLTransform(sql)` | accept SQL (or a t-string) over `__THIS__` |
| `from_file(path)` | same, read from a file |
| `fit(table, this_model=None)` | freeze window-aggregate state into static tables, rewrite the SQL to reference them, specialize via Confit; returns self |
| `infer(row)` / `infer_batch(rows)` | serve rows through the specialized function |
| `backend` / `boundary` | report which engine and boundary the fitted function uses |

The serving half already exists and is unaffected: [Confit](../confit) takes SQL
plus frozen static tables and partially evaluates them into a native function,
bit-exact with DuckDB or refusing at build time. What was removed here is the
fit half, the DataFusion batch engine, the codegen backend, and transform
composition.

Tests in this package assert only that the surface exists and refuses honestly —
they fail if a method is dropped, renamed, or quietly starts returning something.
