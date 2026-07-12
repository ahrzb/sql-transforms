# SQL Transform MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `SQLTransform` — a sklearn-compatible transformer that runs DataFusion SQL at fit time, extracts learned state (window aggregate constants/lookups), and generates a pure-Python function for single-row inference.

**Architecture:** Three modules. `_state.py` walks the DataFusion logical plan and extracts aggregate values via separate queries. `_codegen.py` walks the logical plan's projection expressions and generates Python code with baked-in constants. `__init__.py` exposes `SQLTransform` with `.fit()` (pyarrow Table → self) and `.transform()` (pyarrow Table → pyarrow Table). Sklearn integration deferred to v2.

**Tech Stack:** DataFusion >=46.0.0, PyArrow >=19.0, Python >=3.13

## Global Constraints

- datafusion>=46.0.0, pyarrow>=19.0
- No sqlglot — DataFusion handles all SQL parsing via logical plan
- No sklearn in MVP — window aggregates only
- Public API: `SQLTransform(sql).fit(table).transform(table)`
- Tests use pyarrow tables (no pandas/polars required)
- Ruff lint + format passes before each commit

---

### Task 1: Remove sqlglot, add test infra

**Files:**
- Modify: `pyproject.toml:1-37`
- Modify: `sql_transform/__init__.py`
- Create: `.gitignore` (verify `__pycache__/` etc.)

- [ ] **Step 1: Drop sqlglot from pyproject.toml**

Use Edit tool to change line 10 from `"sqlglot>=26.16.2",` to nothing (remove the line and trailing comma on line 9).

Also add `[tool.pytest.ini_options]` section since we lost it during cleanup:

```toml
[tool.pytest.ini_options]
python_files = ["*_test.py"]
```

After edit, `pyproject.toml` dependencies should be:

```toml
dependencies = [
    "datafusion>=46.0.0",
    "pyarrow>=19.0",
]
```

- [ ] **Step 2: Rebuild venv**

```bash
uv sync
```

- [ ] **Step 3: Set up test directory**

Create `sql_transform/__init___test.py` with a smoke test:

```python
"""Tests for SQLTransform."""

import pyarrow as pa


def test_import():
    from sql_transform import SQLTransform
    assert SQLTransform is not None
```

Create it as `sql_transform/__init___test.py` — pytest picks it up via `*_test.py` pattern.

- [ ] **Step 4: Run test — fails (SQLTransform not defined)**

```bash
uv run pytest sql_transform/__init___test.py -v
```
Expected: `ImportError` or `AttributeError` — `SQLTransform` not in `__init__.py`.

- [ ] **Step 5: Add SQLTransform stub to `__init__.py`**

```python
class SQLTransform:
    pass
```

- [ ] **Step 6: Run test — passes**

```bash
uv run pytest sql_transform/__init___test.py -v
```
Expected: PASS.

- [ ] **Step 7: Run ruff, commit**

```bash
uv run ruff check --fix .
uv run ruff format .
git add -A
git commit -m "chore: drop sqlglot, add test infra"
```

---

### Task 2: State extraction from logical plan

**Files:**
- Create: `sql_transform/_state.py`
- Create: `sql_transform/_state_test.py`

**Interfaces:**
- Consumes: Nothing (self-contained module)
- Produces: `extract_state(plan, ctx, table_name: str) -> dict` — returns dict of:
  - For constant window aggs: `{alias_name: scalar_value}` (e.g., `{"age_norm": 30.0}`)
  - For partitioned window aggs: `{alias_name: {"lookup": {key: val, ...}, "partition_col": "city"}}`

- [ ] **Step 1: Write initial test file**

Create `sql_transform/_state_test.py`:

```python
"""Tests for state extraction from DataFusion logical plans."""

import pyarrow as pa
import datafusion

from sql_transform._state import extract_state


def test_extract_constant_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = "SELECT age / MEAN(age) OVER () AS age_norm FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "age_norm" in state
    assert state["age_norm"] == 30.0


def test_extract_partitioned_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
        name="data",
    )

    sql = (
        "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "city_enc" in state
    assert state["city_enc"] == {"lookup": {"a": 2.0, "b": 3.0}, "partition_col": "city"}


def test_multiple_window_aggs():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"age": [25, 30, 35], "score": [10, 20, 30]},
        name="data",
    )

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state["age_norm"] == 30.0
    assert state["score_norm"] == 60.0
```

- [ ] **Step 2: Run tests — fail (module not found)**

```bash
uv run pytest sql_transform/_state_test.py -v
```
Expected: `ModuleNotFoundError: No module named 'sql_transform._state'`

- [ ] **Step 3: Implement `_state.py`**

Create `sql_transform/_state.py`:

```python
"""Extract learned state from DataFusion logical plans.

Walks the plan tree to find window aggregates, then executes separate
queries to extract constant values (no PARTITION BY) or lookup dicts
(PARTITION BY). Returns a dict keyed by output column alias.
"""

from __future__ import annotations

import datafusion
from datafusion.expr import Alias, BinaryExpr, Column, Projection, Window


def extract_state(
    plan: datafusion.plan.LogicalPlan,
    ctx: datafusion.SessionContext,
    table_name: str,
) -> dict[str, float | dict[str, float]]:
    proj = plan.to_variant()

    # Map from projection alias name -> (agg_fn_name, agg_col, partition_cols)
    window_info: dict[str, tuple[str, str, list[str]]] = {}

    # Find all window aggregates referenced in projections
    _walk_projection(proj, window_info)

    # For each window aggregate, execute a separate query to get the value
    state: dict[str, float | dict[str, float]] = {}
    for out_alias, (fn_name, col_name, partition_cols) in window_info.items():
        if not partition_cols:
            # No partition -> single constant
            sql = f"SELECT {fn_name}({col_name}) FROM {table_name}"
            result = ctx.sql(sql).collect()
            value = result[0].column(0)[0].as_py()
            state[out_alias] = float(value)
        else:
            # Partitioned -> lookup dict
            part_cols = ", ".join(partition_cols)
            sql = (
                f"SELECT {part_cols}, {fn_name}({col_name}) "
                f"FROM {table_name} GROUP BY {part_cols}"
            )
            result = ctx.sql(sql).collect()
            batches = result[0]
            if len(partition_cols) == 1:
                keys = batches.column(0).to_pylist()
                vals = batches.column(1).to_pylist()
                state[out_alias] = {
                    "lookup": dict(zip(keys, vals)),
                    "partition_col": partition_cols[0],
                }
            else:
                # Multi-column partition -> tuple keys
                lookup: dict[tuple, float] = {}
                n_parts = len(partition_cols)
                for i in range(batches.num_rows):
                    key = tuple(
                        batches.column(j)[i].as_py() for j in range(n_parts)
                    )
                    lookup[key] = float(batches.column(n_parts)[i].as_py())
                state[out_alias] = {
                    "lookup": lookup,
                    "partition_col": partition_cols,
                }

    return state


def _walk_projection(
    proj: Projection, window_info: dict[str, tuple[str, str, list[str]]]
) -> None:
    """Walk projections to find which aliases reference window aggregates."""
    inp = proj.input().to_variant()

    # Collect window aggregate function names and their args
    if isinstance(inp, Window):
        for raw_we in inp.window_expr():
            we = raw_we.to_variant()
            fn_name = we.fun.name  # e.g., "avg" or "sum"
            args = we.params.args()
            if not args:
                continue
            col_name = args[0].to_variant().name
            partitions = [
                p.to_variant().name for p in we.params.partition_by()
            ]

            # The window agg result gets aliased internally by DataFusion.
            # We need to find which output alias references it.
            # Walk projections to match.
            for raw_p in proj.projections():
                alias = raw_p.to_variant()
                out_name = alias.alias()
                _find_window_ref_in_expr(
                    raw_expr=alias.expr(),
                    fn_name=fn_name,
                    col_name=col_name,
                    partitions=partitions,
                    out_name=out_name,
                    window_info=window_info,
                )


def _find_window_ref_in_expr(
    raw_expr,
    fn_name: str,
    col_name: str,
    partitions: list[str],
    out_name: str,
    window_info: dict,
) -> bool:
    """Recursively search an expression tree for a window agg reference.

    A window aggregate appears as an Alias wrapping a Column with a
    DataFusion-generated name like "avg(data.age) ROWS BETWEEN ...".
    We identify it by: the Alias name contains the function name and
    column name, and it wraps a Column (not another Alias/BinaryExpr).
    """
    expr = raw_expr.to_variant()

    if isinstance(expr, BinaryExpr):
        return _find_window_ref_in_expr(
            expr.left, fn_name, col_name, partitions, out_name, window_info
        ) or _find_window_ref_in_expr(
            expr.right, fn_name, col_name, partitions, out_name, window_info
        )

    if isinstance(expr, Alias):
        inner = expr.expr().to_variant()
        if isinstance(inner, Column):
            alias_name = expr.alias()
            col_fq = f"{col_name})" if "(" in inner.name else col_name
            if fn_name in alias_name and col_name in alias_name:
                window_info[out_name] = (fn_name.upper(), col_name, partitions)
                return True
        return _find_window_ref_in_expr(
            expr.expr(), fn_name, col_name, partitions, out_name, window_info
        )

    return False
```

- [ ] **Step 4: Run tests — 3 tests, verify all pass**

```bash
uv run pytest sql_transform/_state_test.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Run ruff, commit**

```bash
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/_state.py sql_transform/_state_test.py
git commit -m "feat: add state extraction from DataFusion logical plans"
```

---

### Task 3: Python code generation from logical plan

**Files:**
- Create: `sql_transform/_codegen.py`
- Create: `sql_transform/_codegen_test.py`

**Interfaces:**
- Consumes: `extract_state` from Task 2 (same signature)
- Produces: `generate_infer_fn(plan, state: dict) -> callable` — returns `(row: dict) -> dict`

- [ ] **Step 1: Write test file**

Create `sql_transform/_codegen_test.py`:

```python
"""Tests for Python code generation from DataFusion logical plans."""

import pyarrow as pa
import datafusion

from sql_transform._state import extract_state
from sql_transform._codegen import generate_infer_fn


def _setup(sql: str, data: dict) -> tuple:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    df = ctx.sql(sql)
    plan = df.logical_plan()
    state = extract_state(plan, ctx, "data")
    infer_fn = generate_infer_fn(plan, state)
    return infer_fn, state


def test_generate_constant_window_agg():
    infer_fn, state = _setup(
        "SELECT age / MEAN(age) OVER () AS age_norm FROM data",
        {"age": [25, 30, 35]},
    )
    result = infer_fn({"age": 40})
    assert result == {"age_norm": 40.0 / 30.0}


def test_generate_partitioned_window_agg():
    infer_fn, _ = _setup(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data",
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
    )
    result = infer_fn({"city": "a", "target": 10.0})
    assert result == {"city_enc": 2.0}


def test_generate_multiple_transforms():
    infer_fn, _ = _setup(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data",
        {"age": [25, 30, 35], "score": [10, 20, 30]},
    )
    result = infer_fn({"age": 40, "score": 5})
    assert result["age_norm"] == 40.0 / 30.0
    assert result["score_norm"] == 5.0 / 60.0


def test_generate_simple_column_pass_through():
    infer_fn, state = _setup(
        "SELECT age AS just_age FROM data",
        {"age": [1, 2, 3]},
    )
    result = infer_fn({"age": 42})
    assert result == {"just_age": 42}
```

- [ ] **Step 2: Run tests — fail (module not found)**

```bash
uv run pytest sql_transform/_codegen_test.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_codegen.py`**

Create `sql_transform/_codegen.py`:

```python
"""Generate pure-Python inference functions from DataFusion logical plans.

Walks the plan's projection expressions and converts them to Python code
that references a closure's `_state` dict for learned constants/lookups.
"""

from __future__ import annotations

import datafusion
from datafusion.expr import Alias, BinaryExpr, Column


def generate_infer_fn(
    plan: datafusion.plan.LogicalPlan,
    state: dict,
) -> callable:
    """Return a callable `(row: dict) -> dict` for single-row inference."""
    proj = plan.to_variant()
    expressions: list[tuple[str, str]] = []

    for raw_p in proj.projections():
        alias = raw_p.to_variant()
        out_name = alias.alias()
        code = _expr_to_code(alias.expr(), state)
        expressions.append((out_name, code))

    # Build the function body
    lines = ["def _infer(row):"]
    if not expressions:
        lines.append("    return {}")
    else:
        kv_lines = []
        for out_name, code in expressions:
            kv_lines.append(f'        "{out_name}": {code},')
        lines.append("    return {")
        lines.extend(kv_lines)
        lines.append("    }")

    body = "\n".join(lines)
    namespace: dict = {}
    exec(body, {"_state": state}, namespace)
    return namespace["_infer"]


def _expr_to_code(raw_expr, state: dict) -> str:
    """Convert a RawExpr to a Python code string."""
    expr = raw_expr.to_variant()

    if isinstance(expr, Column):
        return f'row["{expr.name}"]'

    if isinstance(expr, BinaryExpr):
        left = _expr_to_code(expr.left, state)
        right = _expr_to_code(expr.right, state)
        op_map = {
            "Divide": "/",
            "Plus": "+",
            "Minus": "-",
            "Multiply": "*",
        }
        op_name = type(expr.op).__name__.split(".")[-1]
        op_str = op_map.get(op_name, f" {op_name} ")
        return f"({left} {op_str} {right})"

    if isinstance(expr, Alias):
        inner = expr.expr().to_variant()
        alias_name = expr.alias()
        if isinstance(inner, Column):
            # This is a window aggregate alias. Look up in state using the
            # outer projection's alias as key (passed via state key matching).
            return f"_state[{alias_name!r}]"
        return _expr_to_code(expr.expr(), state)

    return f"repr({raw_expr!r})"
```
- [ ] **Step 3 (rewritten): Implement `_codegen.py`**

Create `sql_transform/_codegen.py`:

```python
"""Generate pure-Python inference functions from DataFusion logical plans."""

from __future__ import annotations

import datafusion
from datafusion.expr import Alias, BinaryExpr, Column


def generate_infer_fn(
    plan: datafusion.plan.LogicalPlan,
    state: dict,
) -> callable:
    """Return `(row: dict) -> dict` for single-row inference.

    Walks each top-level projection alias and converts its expression
    tree to Python code. Window aggregate references (Alias wrapping a
    Column) are replaced with state lookups. For constants, generates
    `_state["col"]`. For partitioned lookups, generates
    `_state["col"]["lookup"][row["partition_col"]]`.
    """
    proj = plan.to_variant()
    body_lines: list[str] = []

    for raw_p in proj.projections():
        alias = raw_p.to_variant()
        out_name = alias.alias()
        code = _expr_to_python(alias.expr(), state, out_alias=out_name)
        body_lines.append(f'        "{out_name}": {code},')

    source = (
        "def _infer(row, *, _state=None):\n"
        "    return {\n"
        + "\n".join(body_lines)
        + "\n    }"
    )
    namespace: dict = {}
    exec(source, {}, namespace)
    fn = namespace["_infer"]

    def bound(row: dict) -> dict:
        return fn(row, _state=state)

    return bound


def _expr_to_python(
    raw_expr, state: dict, out_alias: str = ""
) -> str:
    """Convert a RawExpr tree to a Python expression string."""
    expr = raw_expr.to_variant()

    if isinstance(expr, Column):
        return f'row["{expr.name}"]'

    if isinstance(expr, BinaryExpr):
        left = _expr_to_python(expr.left, state, out_alias)
        right = _expr_to_python(expr.right, state, out_alias)
        op_str = _op_to_str(expr.op)
        return f"({left} {op_str} {right})"

    if isinstance(expr, Alias):
        inner = expr.expr().to_variant()
        if isinstance(inner, Column):
            # Window aggregate reference. State may be:
            # - scalar (constant): just the value
            # - dict with "lookup" key (partitioned): need row[key] lookup
            val = state.get(out_alias)
            if isinstance(val, dict) and "lookup" in val:
                part_col = val["partition_col"]
                return f'_state[{out_alias!r}]["lookup"][row["{part_col}"]]'
            return f"_state[{out_alias!r}]"
        return _expr_to_python(expr.expr(), state, out_alias)

    return f"None  # unrecognized: {type(expr).__name__}"


def _op_to_str(op) -> str:
    name = str(op)
    op_map = {"Divide": "/", "Plus": "+", "Minus": "-", "Multiply": "*"}
    return op_map.get(name, name)
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
uv run pytest sql_transform/_codegen_test.py -v
```
Expected: 4 PASS.

- [ ] **Step 5: Run ruff, commit**

```bash
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/_codegen.py sql_transform/_codegen_test.py
git commit -m "feat: add Python code generation from DataFusion logical plans"
```

---

### Task 4: Wire SQLTransform class

**Files:**
- Modify: `sql_transform/__init__.py`
- Modify: `sql_transform/__init___test.py`

**Interfaces:**
- Consumes: `extract_state` from `_state.py`, `generate_infer_fn` from `_codegen.py`
- Produces: `SQLTransform(sql: str)` with methods:
  - `fit(table: pa.Table) -> SQLTransform`
  - `transform(table: pa.Table) -> pa.Table`
  - Private: `_infer(row: dict) -> dict`

- [ ] **Step 1: Update test file**

Overwrite `sql_transform/__init___test.py`:

```python
"""Tests for SQLTransform."""

import pyarrow as pa
import pytest


def test_fit_and_transform_batch_no_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM data")
    data = pa.table({"age": [1, 2, 3]})
    result = t.fit(data).transform(data)
    assert result.column("age").to_pylist() == [1, 2, 3]


def test_fit_and_transform_constant_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    data = pa.table({"age": [25, 30, 35]})
    result = t.fit(data).transform(data)

    expected = [25 / 30, 30 / 30, 35 / 30]
    actual = result.column("age_norm").to_pylist()
    for a, e in zip(actual, expected):
        assert abs(a - e) < 0.001


def test_transform_on_unseen_data():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    train = pa.table({"age": [25, 30, 35]})
    test_data = pa.table({"age": [40, 50]})
    result = t.fit(train).transform(test_data)

    expected = [40 / 30, 50 / 30]
    actual = result.column("age_norm").to_pylist()
    for a, e in zip(actual, expected):
        assert abs(a - e) < 0.001


def test_single_row_inference():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    t.fit(pa.table({"age": [25, 30, 35]}))

    result = t._infer({"age": 40})
    assert abs(result["age_norm"] - 40 / 30) < 0.001


def test_partitioned_agg_transform():
    from sql_transform import SQLTransform

    sql = (
        "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    )
    t = SQLTransform(sql)
    data = pa.table(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]}
    )
    result = t.fit(data).transform(data)
    actual = result.column("city_enc").to_pylist()
    expected = [2.0, 3.0, 2.0, 3.0]
    for a, e in zip(actual, expected):
        assert abs(a - e) < 0.001


def test_partitioned_single_row_inference():
    from sql_transform import SQLTransform

    sql = (
        "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    )
    t = SQLTransform(sql)
    t.fit(
        pa.table(
            {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]}
        )
    )

    result = t._infer({"city": "a", "target": 10.0})
    assert abs(result["city_enc"] - 2.0) < 0.001


def test_multiple_columns():
    from sql_transform import SQLTransform

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data"
    )
    t = SQLTransform(sql)
    data = pa.table({"age": [25, 30, 35], "score": [10, 20, 30]})
    result = t.fit(data).transform(data)

    assert "age_norm" in result.schema.names
    assert "score_norm" in result.schema.names


def test_fit_returns_self():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM data")
    result = t.fit(pa.table({"age": [1]}))
    assert result is t


def test_from_file(tmp_path):
    from sql_transform import SQLTransform

    sql_file = tmp_path / "features.sql"
    sql_file.write_text("SELECT age / MEAN(age) OVER () AS x FROM data")

    t = SQLTransform.from_file(str(sql_file))
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 10})
    assert "x" in result
```

- [ ] **Step 2: Run tests — fail (stub doesn't have fit)**

```bash
uv run pytest sql_transform/__init___test.py -v
```
Expected: multiple FAILs — `SQLTransform` has no `fit` method.

- [ ] **Step 3: Implement SQLTransform**

Overwrite `sql_transform/__init__.py`:

```python
"""SQLTransform — sklearn-compatible SQL-based feature transforms."""

from __future__ import annotations

from typing import Any

import datafusion
import pyarrow as pa

from sql_transform._codegen import generate_infer_fn
from sql_transform._state import extract_state


class SQLTransform:
    """A transformer that applies SQL window-aggregate transforms.

    fit() runs the SQL on training data via DataFusion, extracts
    learned state (aggregate constants and partition lookups), and
    generates a Python function for single-row inference.

    transform() applies the transforms to batch data via DataFusion.

    Usage:
        t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
        t.fit(train_table)
        out = t.transform(test_table)        # batch
        out_row = t._infer({"age": 42})      # single row
    """

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: dict[str, Any] = {}
        self._infer_fn: callable | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLTransform:
        with open(path) as f:
            return cls(f.read())

    def fit(self, table: pa.Table, /) -> SQLTransform:
        ctx = datafusion.SessionContext()
        # Use a fixed table name "data" — the SQL must reference "data"
        ctx.from_pyarrow(table, name="data")
        df = ctx.sql(self._sql)
        plan = df.logical_plan()

        self._state = extract_state(plan, ctx, "data")
        self._infer_fn = generate_infer_fn(plan, self._state)
        return self

    def transform(self, table: pa.Table, /) -> pa.Table:
        ctx = datafusion.SessionContext()
        ctx.from_pyarrow(table, name="data")
        result = ctx.sql(self._sql).collect()
        return pa.Table.from_batches(result)

    def _infer(self, row: dict) -> dict:
        """Single-row inference using generated Python function."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        return self._infer_fn(row)
```

- [ ] **Step 4: Run tests — verify all pass**

```bash
uv run pytest sql_transform/__init___test.py -v
```
Expected: 8 PASS.

- [ ] **Step 5: Run full test suite, ruff, commit**

```bash
uv run pytest -v
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/__init__.py sql_transform/__init___test.py
git commit -m "feat: wire SQLTransform class with fit/transform/infer"
```

---

### Task 5: End-to-end smoke test

**Files:**
- Create: `examples/smoke_test.py` — run and discard after, or add as a pytest test

**Interfaces:**
- Consumes: `SQLTransform` from `__init__.py`
- Produces: nothing — verification only

- [ ] **Step 1: Add integration test to existing test file**

Append to `sql_transform/__init___test.py`:

```python
def test_e2e_three_transforms():
    """End-to-end: fit on training, transform test batch, infer single row."""
    from sql_transform import SQLTransform

    sql = """
    SELECT
        age / MEAN(age) OVER () AS age_norm,
        income / SUM(income) OVER () AS income_share,
        MEAN(target) OVER (PARTITION BY city) AS city_enc
    FROM data
    """

    t = SQLTransform(sql)
    train = pa.table({
        "age": [25, 30, 35, 40],
        "income": [50_000, 60_000, 70_000, 80_000],
        "city": ["paris", "paris", "tehran", "tehran"],
        "target": [1.0, 2.0, 3.0, 4.0],
    })

    t.fit(train)

    # Batch transform
    out = t.transform(train)
    assert out.schema.names == ["age_norm", "income_share", "city_enc"]
    assert len(out) == 4

    # Single-row inference
    row = {"age": 50, "income": 100_000, "city": "tehran", "target": 5.0}
    result = t._infer(row)

    mean_age = 32.5  # (25+30+35+40)/4
    assert abs(result["age_norm"] - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result["income_share"] - 100_000 / total_income) < 0.001

    # city_enc for tehran = mean of [3.0, 4.0] = 3.5
    assert abs(result["city_enc"] - 3.5) < 0.001
```

- [ ] **Step 2: Run test**

```bash
uv run pytest sql_transform/__init___test.py::test_e2e_three_transforms -v
```
Expected: PASS.

- [ ] **Step 3: Run full suite, ruff, commit**

```bash
uv run pytest -v
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/__init___test.py
git commit -m "test: add end-to-end integration test"
```

---

### Task 6: Clean up pyproject.toml optional deps

**Files:**
- Modify: `pyproject.toml`

Remove stale `[tool.pytest.ini_options]` if present (ruff config is fine). Verify final state.

- [ ] **Step 1: Final pyproject.toml review**

Final `pyproject.toml` should be:

```toml
[project]
name = "sql-transform"
version = "0.1.0"
description = "SQL-based data transforms on DataFusion"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "datafusion>=46.0.0",
    "pyarrow>=19.0",
]

[dependency-groups]
dev = [
    "ipdb>=0.13.13",
    "ipython>=9.2.0",
    "pytest>=8.3.5",
    "ruff>=0.11.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
exclude = ["**/*_test.py"]

[tool.hatch.build.targets.wheel]
packages = ["./sql_transform"]

[tool.ruff]
fix = true
lint.select = ["E", "F", "B", "I", "UP", "C4", "S"]
lint.ignore = ["S101"]

[tool.ruff.lint.per-file-ignores]
"*test*.py" = ["S608"]

[tool.pytest.ini_options]
python_files = ["*_test.py"]
```

- [ ] **Step 2: Re-sync and verify**

```bash
uv sync
uv run pytest -v
uv run ruff check --fix .
uv run ruff format .
```

Expected: all tests pass, no ruff errors.

- [ ] **Step 3: Final commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: finalize pyproject.toml, drop stale deps"
```