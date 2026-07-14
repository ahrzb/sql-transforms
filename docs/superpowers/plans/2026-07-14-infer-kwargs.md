# infer() Kwargs Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `InferFn.infer()` accept table data as keyword arguments (`fn.infer(data=[...])`) in addition to the existing positional dict, merging the two when both are given.

**Architecture:** `infer()`'s signature grows an optional `tables` positional (was required) plus a `**kwargs` catch-all of the same per-table shape (`list[BaseModel]`). Both are merged into one `HashMap<String, Vec<Py<PyAny>>>` before doing exactly what `infer()` already does today — no other code path changes.

**Tech Stack:** Same Rust/pyo3 0.29 stack as the rest of `sql_transform._interpreter`.

## Global Constraints

- Each kwarg's value must be a `list[BaseModel]` — the same shape as the dict form's values, not single-instance sugar (per the design spec's explicit Non-Goal)
- Merge rule: `{**tables, **kwargs}` — kwargs win on a key collision, no error path for "both given"
- No change to `InferFn.__init__`, `output_model`, error taxonomy, or row/output conversion — this is additive to `infer()`'s calling convention only
- Known pyo3 0.29 API drift patterns already established in this codebase: `.cast()` not `.downcast()`, `Python::attach` not `Python::with_gil`. If the brief's exact code doesn't compile against the installed pyo3 due to a small API-shape difference, investigate the actual installed crate rather than guessing, and fix minimally.

---

### Task 1: `infer()` kwargs support

**Files:**
- Modify: `src/lib.rs:160-199` (the `infer` method)
- Modify: `tests/test_interpreter.py`
- Modify: `sql_transform/_interpreter.pyi`

**Interfaces:**
- Consumes: nothing new — `Value::from_pyobject`, `plan::execute`, `self.row_table_columns`, `self.output_model` are all unchanged (see `src/lib.rs:160-199` for current code)
- Produces: `infer(self, tables: dict[str, list[BaseModel]] | None = None, **kwargs: list[BaseModel]) -> list[BaseModel]` — the merged-dict behavior described above

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interpreter.py`:

```python
def test_infer_accepts_kwargs_instead_of_dict():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    via_dict = _as_dicts(fn.infer({"data": [Data(age=1), Data(age=2)]}))
    via_kwargs = _as_dicts(fn.infer(data=[Data(age=1), Data(age=2)]))
    assert via_dict == via_kwargs == [{"age": 1}, {"age": 2}]


def test_infer_accepts_kwargs_for_multi_table_join():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    result = fn.infer(a=[A(id=1, x=10)], b=[B(id=1, y=20)])
    assert _as_dicts(result) == [{"x": 10, "y": 20}]


def test_infer_merges_positional_dict_and_kwargs_kwargs_win():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    # kwargs "data" overrides the positional dict's "data" entry
    result = fn.infer({"data": [Data(age=999)]}, data=[Data(age=1), Data(age=2)])
    assert _as_dicts(result) == [{"age": 1}, {"age": 2}]
```

(`_as_dicts` and `A`/`B`/`Data` are already defined earlier in `tests/test_interpreter.py` from prior tasks — reuse them, don't redefine.)

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k "kwargs"
```

Expected: FAIL — `infer()` currently requires a single positional `tables: dict` argument and rejects keyword arguments for table names (`TypeError: infer() got an unexpected keyword argument 'data'`).

- [ ] **Step 3: Update `infer()` in `src/lib.rs`**

Replace the `infer` method (currently `src/lib.rs:160-199`) with:

```rust
    #[pyo3(signature = (tables=None, **kwargs))]
    fn infer(
        &self,
        py: Python<'_>,
        tables: Option<HashMap<String, Vec<Py<PyAny>>>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let mut merged: HashMap<String, Vec<Py<PyAny>>> = tables.unwrap_or_default();
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                let rows: Vec<Py<PyAny>> = v.extract()?;
                merged.insert(key, rows);
            }
        }

        let empty: Vec<String> = Vec::new();
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &merged {
            let columns = self.row_table_columns.get(table).unwrap_or(&empty);
            let mut out_rows = Vec::with_capacity(rows.len());
            for row_obj in rows {
                let bound = row_obj.bind(py);
                let mut row: HashMap<String, Value> = HashMap::new();
                for col in columns {
                    let attr = bound.getattr(col.as_str()).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{table}' is missing attribute '{col}': {e}"
                        ))
                    })?;
                    row.insert(col.clone(), Value::from_pyobject(&attr)?);
                }
                out_rows.push(row);
            }
            value_tables.insert(table.clone(), out_rows);
        }

        let result_rows = plan::execute(&self.plan, &value_tables, &self.lookups)?;

        let output_model = self.output_model.bind(py);
        let mut out = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, v.to_pyobject(py)?)?;
            }
            let instance = output_model.call_method1("model_validate", (dict,))?;
            out.push(instance.unbind());
        }
        Ok(out)
    }
```

Note: `#[pyo3(signature = ...)]` must go directly above `fn infer`, same placement pattern as the existing `#[new] #[pyo3(signature = (sql, row_tables, static_tables, output_model=None))]` above `fn new` a few lines up in the same file — follow that precedent.

- [ ] **Step 4: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS (27 prior + 3 new = 30).

- [ ] **Step 5: Update the type stub**

Read `sql_transform/_interpreter.pyi`. Change the `infer` method signature from:

```python
    def infer(self, tables: dict[str, list[BaseModel]]) -> list[BaseModel]: ...
```

to:

```python
    def infer(
        self, tables: dict[str, list[BaseModel]] | None = None, **kwargs: list[BaseModel]
    ) -> list[BaseModel]: ...
```

- [ ] **Step 6: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/lib.rs tests/test_interpreter.py sql_transform/_interpreter.pyi
git commit -m "feat: accept table data as kwargs in InferFn.infer()"
```
