# SQL Transform v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend SQLTransform with two-phase architecture — Phase 1 learns state and produces reduced DataFusion plan, Phase 2 interprets plan for single-row inference. Add sklearn transformer support and JOIN→lookup support.

**Architecture:** Phase 1 (fit): parse SQL via sqlglot, fit transformers, extract window agg values + JOIN lookups via typed AST walk, produce reduced SQL + DataFusion logical plan. Phase 2 (inference): row-by-row plan interpreter walks expressions, dispatches built-in functions and UDFs, returns result dict. No regex, no exec(), no code generation.

**Tech Stack:** DataFusion >=46.0.0, PyArrow >=19.0, sqlglot (re-added), Python >=3.13

## Global Constraints

- datafusion>=46.0.0, pyarrow>=19.0, sqlglot>=26.16.2
- scikit-learn added to dev dependencies for transformer tests
- No regex for plan inspection — typed AST via `RawExpr` methods
- No exec() — Phase 2 is an interpreter, not a code generator
- Public API: `SQLTransform(sql).register_transformer(name, instance).add_table(name, table).fit(table).transform(table)`
- Tests use pyarrow tables
- Ruff lint + format passes before each commit
- Pytest: `python_files = ["*_test.py"]`
- Ruff: select E,F,B,I,UP,C4,S; ignore S101; per-file-ignore `"*test*.py" = ["S608"]`, `"_state.py" = ["S608"]`

---

### Task 1: Add sqlglot dependency

**Files:**
- Modify: `pyproject.toml:1-22`

**Interfaces:**
- Consumes: nothing
- Produces: sqlglot available as import

- [ ] **Step 1: Add sqlglot to pyproject.toml**

Read `pyproject.toml`, find `dependencies` list. Add `"sqlglot>=26.16.2",`:

```
dependencies = [
    "datafusion>=46.0.0",
    "pyarrow>=19.0",
    "sqlglot>=26.16.2",
]
```

- [ ] **Step 2: Add sklearn to dev dependencies**

Add `"scikit-learn>=1.6.0"` to `[dependency-groups].dev` in `pyproject.toml`.

```toml
dev = [
    "ipdb>=0.13.13",
    "ipython>=9.2.0",
    "pytest>=8.3.5",
    "ruff>=0.11.7",
    "scikit-learn>=1.6.0",
]
```

- [ ] **Step 3: Sync dependencies**

```
uv sync
```

Expected: sqlglot + sklearn installed without errors.

- [ ] **Step 4: Verify import**

```
uv run python -c "import sqlglot; print(sqlglot.__version__)"
```

Expected: prints version string.

- [ ] **Step 5: Ruff + commit**

```
uv run ruff check --fix .
uv run ruff format .
git add pyproject.toml uv.lock
git commit -m "chore: add sqlglot for transformer call parsing"
```

---

### Task 2: Phase 2 plan interpreter (`_interpreter.py`)

**Files:**
- Create: `sql_transform/_interpreter.py`
- Create: `sql_transform/_interpreter_test.py`
- Modify: `sql_transform/_codegen.py` (will be kept until Task 5, but not used by interpreter)

**Interfaces:**
- Consumes: DataFusion logical plan (`plan: datafusion.plan.LogicalPlan`), row dict (`{"col": val, ...}`), state dict
- Produces: `interpret(plan, row, state) -> dict` — returns `{"alias": value, ...}` per projection

**Architecture:** The interpreter walks DataFusion's `RawExpr` tree. All operations (arithmetic, built-in functions, UDFs) go through `rex_call_operator()` + `rex_call_operands()`. Column references and literals have `to_variant()` → Column/Literal. Expression types that can't be variant-converted (ScalarFunction, UDF) are handled via the raw API.

- [ ] **Step 1: Write interpreter test file**

Create `sql_transform/_interpreter_test.py`:

```python
"""Tests for DataFusion plan interpreter."""

import pyarrow as pa
import datafusion

from sql_transform._interpreter import interpret


def _setup(sql: str, data: dict, state: dict | None = None):
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    plan = ctx.sql(sql).logical_plan()
    return plan, state or {}


def _expected(sql: str, data: dict):
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    batch = ctx.sql(sql).collect()[0]
    return batch.to_pylist()[0]


def test_column_pass_through():
    sql = "SELECT age FROM data"
    data = {"age": [30]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"age": 30}, state)
    assert result == expected


def test_literal():
    sql = "SELECT 42 AS x FROM data"
    data = {"age": [1]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"age": 1}, state)
    assert result == expected


def test_arithmetic():
    sql = "SELECT age / 2 AS half FROM data"
    data = {"age": [30]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"age": 30}, state)
    assert result == expected


def test_builtin_upper():
    sql = "SELECT UPPER(name) AS up FROM data"
    data = {"name": ["hello"]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"name": "hello"}, state)
    assert result == expected


def test_builtin_concat():
    sql = "SELECT CONCAT(a, '-', b) AS combo FROM data"
    data = {"a": ["x"], "b": ["y"]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"a": "x", "b": "y"}, state)
    assert result == expected


def test_multiple_columns():
    sql = "SELECT age, name FROM data"
    data = {"age": [30], "name": ["hello"]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"age": 30, "name": "hello"}, state)
    assert result == expected


def test_cast():
    sql = "SELECT CAST(age AS VARCHAR) AS s FROM data"
    data = {"age": [42]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"age": 42}, state)
    assert result == expected


def test_arithmetic_precedence():
    sql = "SELECT (a + b) * c AS x FROM data"
    data = {"a": [2], "b": [3], "c": [4]}
    plan, state = _setup(sql, data)
    expected = _expected(sql, data)
    result = interpret(plan, {"a": 2, "b": 3, "c": 4}, state)
    assert result == expected
```

- [ ] **Step 2: Run tests — verify all fail**

```
uv run pytest sql_transform/_interpreter_test.py -v
```

Expected: `ModuleNotFoundError: No module named 'sql_transform._interpreter'`

- [ ] **Step 3: Write minimal `_interpreter.py` placeholder**

Create `sql_transform/_interpreter.py`:

```python
def interpret(plan, row, state):
    return {}
```

- [ ] **Step 4: Run tests — verify import works, tests fail on assertion**

```
uv run pytest sql_transform/_interpreter_test.py -v
```

Expected: all FAIL with assertion errors (empty dict vs expected values).

- [ ] **Step 5: Implement interpreter — core expression walker**

Replace `sql_transform/_interpreter.py`:

```python
"""Row-by-row DataFusion logical plan interpreter.

Walks the logical plan's projection expressions and evaluates each node
for a single input row. All operations (arithmetic, built-in functions,
UDFs) flow through RawExpr.rex_call_operator() and rex_call_operands().
Column and Literal nodes can be variant-converted for value extraction.
"""

from __future__ import annotations

import datafusion
from datafusion.expr import Alias, Column, Literal

_BUILTIN_DISPATCH = {
    "upper": lambda args: str(args[0]).upper(),
    "lower": lambda args: str(args[0]).lower(),
    "concat": lambda args: "".join(str(a) for a in args),
    "substr": lambda args: _substr(*args),
    "trim": lambda args: str(args[0]).strip(),
    "abs": lambda args: abs(args[0]),
    "round": lambda args: round(args[0]),
    "cast": lambda args: args[0],
    "nullif": lambda args: None if args[0] == args[1] else args[0],
    "coalesce": lambda args: next((a for a in args if a is not None), None),
}

_ARITHMETIC = {"+", "-", "*", "/"}


def interpret(
    plan: datafusion.plan.LogicalPlan,
    row: dict,
    state: dict,
) -> dict:
    """Evaluate a logical plan for a single input row.

    Returns dict of {alias: value} for each projection.
    """
    proj = plan.to_variant()
    result: dict = {}

    for p in proj.projections():
        try:
            v = p.to_variant()
        except ValueError:
            v = None

        if v is not None and isinstance(v, Alias):
            out_name = v.alias()
            result[out_name] = _interpret_expr(v.expr(), row, state)
        elif v is not None and isinstance(v, Column):
            out_name = v.name()
            result[out_name] = row[out_name]
        else:
            # Expression without alias — use column name as key
            col_name = p.column_name()
            result[col_name] = row[col_name]

    return result


def _interpret_expr(raw_expr, row: dict, state: dict):
    """Recursively interpret a RawExpr node."""
    try:
        v = raw_expr.to_variant()
    except ValueError:
        v = None

    if v is not None and isinstance(v, Column):
        return row[v.name()]

    if v is not None and isinstance(v, Literal):
        return _literal_value(v)

    if v is not None and isinstance(v, Alias):
        return _interpret_expr(v.expr(), row, state)

    # Fall through: function call or arithmetic (ScalarFunction, UDF, etc.)
    op = raw_expr.rex_call_operator()
    operands = raw_expr.rex_call_operands()
    args = [_interpret_expr(o, row, state) for o in operands]

    return _dispatch(op, args, state, row)


def _dispatch(op: str, args: list, state: dict, row: dict):
    """Route operator to handler."""
    if op in _ARITHMETIC:
        return _arithmetic(op, args)
    if handler := _BUILTIN_DISPATCH.get(op.lower()):
        return handler(args)
    if op in state:
        return state[op](*args)
    if handler := state.get("_udf_registry", {}).get(op):
        return handler(*args)
    raise ValueError(
        f"Unknown operator: {op}. Available builtins: "
        f"{sorted(_BUILTIN_DISPATCH)}, state keys: {sorted(state)}"
    )


def _arithmetic(op: str, args: list):
    a, b = args
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    raise ValueError(f"Unknown arithmetic operator: {op}")


def _substr(s: str, start: int, length: int | None = None):
    """SUBSTR(s, start) or SUBSTR(s, start, length). 1-indexed like SQL."""
    idx = start - 1 if start > 0 else start
    if length is not None:
        return s[idx:idx + length]
    return s[idx:]


def _literal_value(lit: Literal):
    """Extract Python value from a DataFusion Literal variant."""
    dtype = lit.data_type()
    try:
        return lit.value_i64()
    except Exception:
        pass
    try:
        return lit.value_f64()
    except Exception:
        pass
    try:
        return lit.value_f32()
    except Exception:
        pass
    try:
        return lit.value_i32()
    except Exception:
        pass
    try:
        return lit.value_string()
    except Exception:
        pass
    raise ValueError(f"Cannot extract value from literal: {dtype}")
```

- [ ] **Step 6: Run tests — verify all pass**

```
uv run pytest sql_transform/_interpreter_test.py -v
```

Expected: 8 PASS.

- [ ] **Step 7: Ruff + commit**

```
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/_interpreter.py sql_transform/_interpreter_test.py
git commit -m "feat: add DataFusion plan interpreter for single-row inference"
```

---

### Task 3: Rewrite `_state.py` with typed AST (no regex)

**Files:**
- Modify: `sql_transform/_state.py` (full rewrite)
- Modify: `sql_transform/_state_test.py` (update for new API, add JOIN tests)

**Interfaces:**
- Consumes: DataFusion logical plan, session context, table name
- Produces: `extract_state(plan, ctx, table_name) -> dict` — window agg values + JOIN lookups (NO regex, typed AST walk via `plan.to_variant()`)

**Key change:** Replace `_WINDOW_AGG_RE` regex on `display_indent()` with walking `Projection.input()` → `Window.input()` → iterate `WindowExpr`. No change to state shape or external API.

- [ ] **Step 1: Update `_state_test.py` — keep existing tests, add JOIN tests**

Read current `sql_transform/_state_test.py`. Keep all three existing tests (constant window agg, partitioned window agg, multi-window-agg). Append JOIN tests:

```python
def test_single_key_join_lookup():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"id": [1, 2], "x": [10, 20]},
        name="data",
    )
    ctx.from_pydict(
        {"id": [1, 2], "temp": [22.5, 18.0]},
        name="ref",
    )

    sql = "SELECT data.x, ref.temp FROM data JOIN ref ON data.id = ref.id"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "ref" in state
    lookup = state["ref"]["lookup"]
    assert lookup[(1,)] == {"temp": 22.5}
    assert lookup[(2,)] == {"temp": 18.0}
    assert state["ref"]["keys"] == ["id"]


def test_multi_key_join_lookup():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"city": ["a", "b"], "country": ["X", "Y"], "x": [1, 2]},
        name="data",
    )
    ctx.from_pydict(
        {"city": ["a", "b"], "country": ["X", "Y"], "temp": [30.0, 20.0]},
        name="ref",
    )

    sql = (
        "SELECT data.x, ref.temp FROM data "
        "JOIN ref ON data.city = ref.city AND data.country = ref.country"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "ref" in state
    lookup = state["ref"]["lookup"]
    assert lookup[("a", "X")] == {"temp": 30.0}
    assert lookup[("b", "Y")] == {"temp": 20.0}
    assert state["ref"]["keys"] == ["city", "country"]


def test_mixed_window_and_join():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"id": [1, 2], "val": [10.0, 20.0]},
        name="data",
    )
    ctx.from_pydict(
        {"id": [1, 2], "label": ["low", "high"]},
        name="ref",
    )

    sql = (
        "SELECT data.id, ref.label, "
        "data.val / MEAN(data.val) OVER () AS norm "
        "FROM data JOIN ref ON data.id = ref.id"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "norm" in state
    assert state["norm"] == 15.0
    assert "ref" in state
    assert state["ref"]["lookup"][(1,)]["label"] == "low"
```

- [ ] **Step 2: Run tests — verify new JOIN tests fail**

```
uv run pytest sql_transform/_state_test.py -v -k "join or mixed"
```

Expected: FAIL — `extract_state` doesn't return "ref" key yet.

- [ ] **Step 3: Rewrite `_state.py` — typed AST walk**

Replace `sql_transform/_state.py` entirely:

```python
"""Extract learned state from DataFusion logical plans via typed AST walking.

Walks Projection.input() for Window nodes to find window aggregates.
Walks plan for Join nodes to build right-side table lookup dicts.
No regex — pure typed AST access.
"""

from __future__ import annotations

import datafusion
from datafusion.expr import Projection, SubqueryAlias, TableScan, Window


_INPUT_TABLE_ALIAS = "data"


def extract_state(
    plan: datafusion.plan.LogicalPlan,
    ctx: datafusion.SessionContext,
    table_name: str,
) -> dict:
    state: dict = {}

    proj = plan.to_variant()

    _extract_window_aggs(proj, ctx, table_name, state)
    _extract_join_lookups(proj, ctx, state)

    return state


def _extract_window_aggs(
    proj: Projection, ctx: datafusion.SessionContext, table_name: str, state: dict
) -> None:
    """Walk Projection → Window → WindowExpr to find window aggregates."""
    inp = proj.input()
    inv = inp.to_variant()

    if not isinstance(inv, Window):
        return

    window_exprs = inv.window_expr()
    for we_raw in window_exprs:
        we = we_raw.to_variant()
        fn_name = we.fun.name.upper()

        params = we.params
        args = params.args()

        if not args:
            continue

        col_name = args[0].to_variant().name

        partitions_raw = params.partition_by()
        partitions = [p.to_variant().name for p in partitions_raw]

        # Find which projection alias references this window agg
        for raw_p in proj.projections():
            alias_v = raw_p.to_variant()
            out_name = alias_v.alias()

            if _expr_refs_window(alias_v.expr(), fn_name, col_name):
                if not partitions:
                    sql = (
                        f"SELECT {fn_name}({col_name}) "
                        f"FROM {table_name}"  # noqa: S608
                    )
                    result = ctx.sql(sql).collect()
                    value = result[0].column(0)[0].as_py()
                    state[out_name] = float(value)
                else:
                    part_col = partitions[0]
                    sql = (
                        f"SELECT {part_col}, {fn_name}({col_name}) "
                        f"FROM {table_name} "
                        f"GROUP BY {part_col}"  # noqa: S608
                    )
                    result = ctx.sql(sql).collect()
                    keys: list = []
                    vals: list = []
                    for batch in result:
                        keys.extend(
                            batch.column(0).to_pylist()
                        )
                        vals.extend(
                            batch.column(1).to_pylist()
                        )
                    state[out_name] = {
                        "lookup": dict(
                            zip(keys, vals, strict=True)
                        ),
                        "partition_col": part_col,
                    }
                break


def _expr_refs_window(raw_expr, fn_name: str, col_name: str) -> bool:
    """Check if expression tree references a window aggregate by name."""
    try:
        v = raw_expr.to_variant()
    except ValueError:
        return False

    from datafusion.expr import Alias, BinaryExpr

    if isinstance(v, BinaryExpr):
        return _expr_refs_window(v.left, fn_name, col_name) or _expr_refs_window(
            v.right, fn_name, col_name
        )

    if isinstance(v, Alias):
        inner = v.expr()
        try:
            iv = inner.to_variant()
            from datafusion.expr import Column

            if isinstance(iv, Column):
                alias_name = v.alias()
                if fn_name in alias_name and col_name in alias_name:
                    return True
        except ValueError:
            pass
        return _expr_refs_window(inner, fn_name, col_name)

    return False


def _extract_join_lookups(
    proj: Projection, ctx: datafusion.SessionContext, state: dict
) -> None:
    """Walk plan tree for Join nodes, build lookup dicts for right-side tables."""
    inp = proj.input()
    inv = inp.to_variant()

    if isinstance(inv, datafusion.expr.Join):
        _process_join(inv, ctx, state)

    # Walk upward: handle nested inputs (SubqueryAlias, etc.)
    _walk_for_joins(inv, ctx, state)


def _walk_for_joins(node, ctx, state):
    """Recursively search for Join nodes in plan tree."""
    if isinstance(node, datafusion.expr.Join):
        _process_join(node, ctx, state)
    if hasattr(node, "input"):
        child = node.input()
        cv = child.to_variant()
        _walk_for_joins(cv, ctx, state)


def _process_join(join, ctx, state):
    """Process a Join node: extract right-side table as lookup dict."""
    right = join.right().to_variant()

    right_table_name = _table_name(right)

    if not right_table_name:
        return

    # Find the registered table data from the context
    left_keys, right_keys = _extract_join_keys(join)

    # Materialize right table from DataFusion query
    sql = f"SELECT * FROM {right_table_name}"  # noqa: S608
    result = ctx.sql(sql).collect()

    columns = result[0].schema.names
    lookup: dict = {}
    for batch in result:
        for i in range(batch.num_rows):
            key_vals = tuple(
                batch.column(
                    columns.index(rk)
                )[i].as_py()
                for rk in right_keys
            )
            row_data = {}
            for col in columns:
                if col not in right_keys:
                    row_data[col] = (
                        batch.column(columns.index(col))[i].as_py()
                    )
            lookup[key_vals] = row_data

    state[right_table_name] = {
        "lookup": lookup,
        "keys": left_keys,
    }


def _table_name(node) -> str | None:
    """Extract table name from a plan node (handles SubqueryAlias wrapping)."""
    if isinstance(node, SubqueryAlias):
        return node.alias()
    if isinstance(node, TableScan):
        return node.table_name()
    if hasattr(node, "input"):
        return _table_name(node.input().to_variant())
    return None


def _extract_join_keys(join) -> tuple[list[str], list[str]]:
    """Extract equality keys from JOIN ON condition."""
    left_keys: list[str] = []
    right_keys: list[str] = []

    # Walk join condition — iterate AND-connected equalities
    condition = join.on()
    cond_v = condition.to_variant()

    _walk_condition(cond_v, left_keys, right_keys)

    return left_keys, right_keys


def _walk_condition(node, left_keys: list, right_keys: list):
    """Walk JOIN ON condition tree extracting equality column pairs."""
    from datafusion.expr import BinaryExpr

    if isinstance(node, BinaryExpr):
        op_str = str(type(node.op).__name__).split(".")[-1]
        left = node.left.to_variant()
        right = node.right.to_variant()
        from datafusion.expr import Column

        if isinstance(left, Column) and isinstance(right, Column):
            left_name = left.name()
            right_name = right.name()
            left_keys.append(left_name)
            right_keys.append(right_name)
```

- [ ] **Step 4: Run tests — verify all pass**

```
uv run pytest sql_transform/_state_test.py -v
```

Expected: 6 PASS (3 existing + 3 new JOIN tests).

- [ ] **Step 5: Ruff + commit**

```
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/_state.py sql_transform/_state_test.py
git commit -m "refactor: rewrite state extraction with typed AST, add JOIN lookups"
```

---

### Task 4: Transformer call parsing + fitting (`_transformers.py`)

**Files:**
- Create: `sql_transform/_transformers.py`
- Create: `sql_transform/_transformers_test.py`

**Interfaces:**
- Consumes: SQL string + set of registered transformer names
- Produces:
  - `extract_transformer_calls(sql, registered_names) -> dict[str, TransformNode]` — parse SQL, find transformer calls, build DAG
  - `fit_transformers(nodes, table, state) -> dict` — fit in topological order, returns fitted instances indexed by name
  - `strip_transformer_calls(sql, registered_names) -> str` — remove calls, produce clean SQL for DataFusion

- [ ] **Step 1: Write test file**

Create `sql_transform/_transformers_test.py`:

```python
"""Tests for transformer call parsing and fitting."""

import pyarrow as pa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sql_transform._transformers import (
    TransformNode,
    extract_transformer_calls,
    fit_transformers,
    strip_transformer_calls,
)


def test_parse_simple_call():
    sql = "SELECT tfidf(text) AS bow FROM data"
    nodes = extract_transformer_calls(sql, {"tfidf"})

    assert len(nodes) == 1
    node = nodes["bow"]
    assert node.name == "tfidf"
    assert node.args == ["text"]


def test_parse_nested_call():
    sql = "SELECT svd(tfidf(text)) AS emb FROM data"
    nodes = extract_transformer_calls(sql, {"tfidf", "svd"})

    assert len(nodes) == 1
    node = nodes["emb"]
    assert node.name == "svd"
    assert len(node.args) == 1
    inner = node.args[0]
    assert isinstance(inner, TransformNode)
    assert inner.name == "tfidf"
    assert inner.args == ["text"]


def test_parse_multiple_calls():
    sql = (
        "SELECT svd(tfidf(text)) AS emb, "
        "countvec(title) AS bow FROM data"
    )
    nodes = extract_transformer_calls(sql, {"tfidf", "svd", "countvec"})

    assert len(nodes) == 2
    assert nodes["emb"].name == "svd"
    assert nodes["bow"].name == "countvec"


def test_topological_order():
    sql = "SELECT svd(tfidf(text)) AS emb FROM data"
    nodes = extract_transformer_calls(sql, {"tfidf", "svd"})

    order = TransformNode.topological_order(nodes)
    names = [n.name for n in order]
    # tfidf must come before svd
    assert names.index("tfidf") < names.index("svd")


def test_strip_transformer_calls():
    sql = (
        "SELECT svd(tfidf(text)) AS emb, "
        "age / MEAN(age) OVER () AS norm FROM data"
    )
    cleaned = strip_transformer_calls(sql, {"tfidf", "svd"})

    # Transformer calls replaced with NULL placeholders
    assert "tfidf" not in cleaned
    assert "svd" not in cleaned
    assert "NULL AS emb" in cleaned or "1 AS emb" in cleaned
    # Window agg preserved
    assert "MEAN(age) OVER ()" in cleaned


def test_fit_single_transformer():
    table = pa.table({"text": ["hello world", "foo bar"]})
    nodes = extract_transformer_calls(
        "SELECT vec(text) AS bow FROM data", {"vec"}
    )
    state = fit_transformers(nodes, table, {})

    assert "vec" in state
    # Fitted transformer should work
    result = state["vec"].transform(["hello world"])
    assert result.shape[1] > 0


def test_fit_nested_chain():
    import numpy as np
    table = pa.table({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    sql = "SELECT pca(scaler(x)) AS reduced FROM data"
    nodes = extract_transformer_calls(sql, {"scaler", "pca"})
    state = fit_transformers(
        nodes, table, {"scaler": StandardScaler(), "pca": PCA(n_components=1)}
    )

    assert "scaler" in state
    assert "pca" in state
    result = state["pca"].transform(state["scaler"].transform([[6.0]]))
    assert result.shape == (1, 1)
```

- [ ] **Step 2: Run tests — verify fail**

```
uv run pytest sql_transform/_transformers_test.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_transformers.py`**

Create `sql_transform/_transformers.py`:

```python
"""Parse SQL for sklearn transformer calls, fit in topological order.

Uses sqlglot to extract custom function calls like tfidf(text) or
svd(tfidf(text)) from SQL before passing to DataFusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa
import sqlglot
from sqlglot import exp


@dataclass
class TransformNode:
    """A single transformer call in the SQL expression tree."""

    name: str
    args: list = field(default_factory=list)  # str | TransformNode

    @staticmethod
    def topological_order(
        nodes: dict[str, TransformNode],
    ) -> list[TransformNode]:
        """Return nodes in dependency order (leaf-first)."""
        order: list[TransformNode] = []
        seen: set[int] = set()

        def visit(node: TransformNode):
            nid = id(node)
            if nid in seen:
                return
            seen.add(nid)
            for arg in node.args:
                if isinstance(arg, TransformNode):
                    visit(arg)
            order.append(node)

        for node in nodes.values():
            visit(node)

        return order


def extract_transformer_calls(
    sql: str, registered: set[str],
) -> dict[str, TransformNode]:
    """Parse SQL with sqlglot, find registered transformer function calls.

    Returns: {alias: TransformNode} for each transformer call.
    """
    parsed = sqlglot.parse_one(sql)
    nodes: dict[str, TransformNode] = {}

    for node in parsed.find_all(exp.Select):
        for expr in node.expressions:
            _extract_from_expression(expr, registered, nodes)

    return nodes


def _extract_from_expression(
    expr: exp.Expression, registered: set[str], nodes: dict
):
    """Walk a SELECT expression, find transformer calls, map to aliases."""
    alias_name = expr.alias
    if alias_name is None:
        return

    # Find function calls in the expression tree
    for func in expr.find_all(exp.Anonymous):
        func_name = func.name.lower()
        if func_name in registered:
            # Build argument list
            args = []
            for arg_expr in func.expressions:
                # Check if arg is another transformer call
                inner_funcs = list(arg_expr.find_all(exp.Anonymous))
                if inner_funcs:
                    inner = inner_funcs[0]
                    inner_name = inner.name.lower()
                    if inner_name in registered:
                        inner_args = [
                            e.name if isinstance(e, exp.Column) else e.sql()
                            for e in inner.expressions
                        ]
                        inner_node = nodes.get(
                            inner_name,
                            TransformNode(name=inner_name, args=inner_args),
                        )
                        nodes[inner_name] = inner_node
                        args.append(inner_node)
                        continue

                if isinstance(arg_expr, exp.Column):
                    args.append(arg_expr.name)
                else:
                    args.append(arg_expr.sql())

            nodes[str(alias_name)] = TransformNode(
                name=func_name, args=args
            )


def strip_transformer_calls(sql: str, registered: set[str]) -> str:
    """Remove transformer function calls from SQL, replace with placeholders.

    Produces valid DataFusion SQL that can be parsed for plan extraction.
    """
    parsed = sqlglot.parse_one(sql)

    # Build a mapping of alias → replacement
    replacements: dict[str, str] = {}

    for node in parsed.find_all(exp.Select):
        for expr in node.expressions:
            alias = expr.alias
            if alias is None:
                continue
            for func in expr.find_all(exp.Anonymous):
                if func.name.lower() in registered:
                    replacements[str(alias)] = f"NULL AS {alias}"

    if not replacements:
        return sql

    result = sql
    for alias, replacement in replacements.items():
        # Find and replace the original expression
        import re
        # Match expression ending with AS alias
        pattern = rf"\S+\s+AS\s+{re.escape(alias)}"
        result = re.sub(pattern, replacement, result)

    return result


def fit_transformers(
    nodes: dict[str, TransformNode],
    table: pa.Table,
    state: dict,
) -> dict:
    """Fit transformers in topological order. Returns fitted instances dict.

    state dict maps transformer name → fitted sklearn instance.
    Modifies state in-place and returns it.
    """
    if not nodes:
        return state

    order = TransformNode.topological_order(nodes)

    for node in order:
        if node.name in state:
            continue  # Already fitted (shared sub-node)

        # Collect input data for this transformer
        input_data = _collect_input(node, table, state)

        # Fit the transformer
        transformer = state.get(node.name)  # Pre-registered instance
        if transformer is None:
            raise ValueError(
                f"Transformer '{node.name}' not registered. "
                f"Registered: {list(state.keys())}"
            )

        if hasattr(transformer, "fit"):
            transformer.fit(input_data)

    return state


def _collect_input(
    node: TransformNode, table: pa.Table, state: dict,
):
    """Collect the input values for a transformer node."""
    if len(node.args) == 1 and isinstance(node.args[0], str):
        col_name = node.args[0]
        col = table.column(col_name)
        if hasattr(col, "to_pylist"):
            return col.to_pylist()
        return [col[i].as_py() for i in range(len(col))]

    if len(node.args) == 1 and isinstance(node.args[0], TransformNode):
        child = state[node.args[0].name]
        raw = _collect_input(node.args[0], table, state)
        return child.transform([[v] for v in raw])

    raise NotImplementedError(
        f"Multi-arg transformers not yet supported: {node.args}"
    )
```

- [ ] **Step 4: Run tests — iterate until all pass**

```
uv run pytest sql_transform/_transformers_test.py -v
```

Expected: 7 PASS.

Fix any issues (sqlglot parsing edge cases, regex in strip, topological sort).

- [ ] **Step 5: Ruff + commit**

```
uv run ruff check --fix .
uv run ruff format .
git add sql_transform/_transformers.py sql_transform/_transformers_test.py
git commit -m "feat: add transformer call parsing and fitting via sqlglot"
```

---

### Task 5: Wire SQLTransform — two-phase fit/transform

**Files:**
- Modify: `sql_transform/__init__.py` (full rewrite)
- Modify: `sql_transform/__init___test.py` (update all tests)
- Modify: `pyproject.toml` (per-file-ignores)

**Interfaces:**
- Consumes: `extract_state` from `_state.py`, `interpret` from `_interpreter.py`, `extract_transformer_calls` + `fit_transformers` + `strip_transformer_calls` from `_transformers.py`
- Produces: `SQLTransform` with builder API + two-phase fit/transform

- [ ] **Step 1: Update test file**

Overwrite `sql_transform/__init___test.py`:

```python
"""Tests for SQLTransform."""

import pyarrow as pa
import pytest

from sql_transform import SQLTransform


def test_fit_and_transform_batch_no_agg():
    t = SQLTransform("SELECT age FROM data")
    data = pa.table({"age": [1, 2, 3]})
    result = t.fit(data).transform(data)
    assert result.column("age").to_pylist() == [1, 2, 3]


def test_fit_and_transform_constant_agg():
    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    data = pa.table({"age": [25, 30, 35]})
    result = t.fit(data).transform(data)

    expected = [25 / 30, 30 / 30, 35 / 30]
    actual = result.column("age_norm").to_pylist()
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) < 0.001


def test_transform_on_unseen_data():
    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    train = pa.table({"age": [25, 30, 35]})
    test_data = pa.table({"age": [40, 50]})
    result = t.fit(train).transform(test_data)

    expected = [40 / 30, 50 / 30]
    actual = result.column("age_norm").to_pylist()
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) < 0.001


def test_single_row_inference():
    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM data")
    t.fit(pa.table({"age": [25, 30, 35]}))

    result = t._infer({"age": 40})
    assert abs(result["age_norm"] - 40 / 30) < 0.001


def test_partitioned_agg_transform():
    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    t = SQLTransform(sql)
    data = pa.table(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]}
    )
    result = t.fit(data).transform(data)
    actual = result.column("city_enc").to_pylist()
    expected = [2.0, 3.0, 2.0, 3.0]
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) < 0.001


def test_partitioned_single_row_inference():
    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    t = SQLTransform(sql)
    t.fit(
        pa.table(
            {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]}
        )
    )

    result = t._infer({"city": "a", "target": 10.0})
    assert abs(result["city_enc"] - 2.0) < 0.001


def test_multiple_columns():
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
    t = SQLTransform("SELECT age FROM data")
    result = t.fit(pa.table({"age": [1]}))
    assert result is t


def test_from_file(tmp_path):
    sql_file = tmp_path / "features.sql"
    sql_file.write_text("SELECT age / MEAN(age) OVER () AS x FROM data")

    t = SQLTransform.from_file(str(sql_file))
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 10})
    assert "x" in result


def test_e2e_three_transforms():
    """End-to-end: window agg + constant + partitioned."""
    sql = """
    SELECT
        age / MEAN(age) OVER () AS age_norm,
        income / SUM(income) OVER () AS income_share,
        MEAN(target) OVER (PARTITION BY city) AS city_enc
    FROM data
    """
    t = SQLTransform(sql)
    train = pa.table(
        {
            "age": [25, 30, 35, 40],
            "income": [50_000, 60_000, 70_000, 80_000],
            "city": ["paris", "paris", "tehran", "tehran"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    t.fit(train)

    out = t.transform(train)
    assert out.schema.names == ["age_norm", "income_share", "city_enc"]
    assert len(out) == 4

    row = {"age": 50, "income": 100_000, "city": "tehran", "target": 5.0}
    result = t._infer(row)

    mean_age = 32.5
    assert abs(result["age_norm"] - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result["income_share"] - 100_000 / total_income) < 0.001

    assert abs(result["city_enc"] - 3.5) < 0.001
```

- [ ] **Step 2: Run tests — verify fail**

```
uv run pytest sql_transform/__init___test.py -v
```

Expected: FAIL — current `SQLTransform` doesn't have `register_transformer`, `add_table`, or expect interpreter-based implementation (though current tests may pass if implementation not yet changed).

- [ ] **Step 3: Rewrite `__init__.py`**

```python
"""SQLTransform — sklearn-compatible SQL-based feature transforms.

Two-phase architecture:
  Phase 1 (fit): Learns state from training data via DataFusion,
    produces a reduced logical plan with constants and UDF references.
  Phase 2 (inference): Row-by-row plan interpreter, no DataFusion at runtime.
"""

from __future__ import annotations

from typing import Any

import datafusion
import pyarrow as pa

from sql_transform._interpreter import interpret
from sql_transform._state import extract_state
from sql_transform._transformers import (
    extract_transformer_calls,
    fit_transformers,
    strip_transformer_calls,
)


class SQLTransform:
    """A transformer that applies SQL-based feature transforms.

    Supports window aggregates, sklearn transformer calls, and JOIN
    lookups — all expressed in SQL, learned at fit time, executed row-by-row
    at inference time.
    """

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: dict[str, Any] = {}
        self._registered_transformers: dict[str, Any] = {}
        self._registered_tables: dict[str, pa.Table] = {}
        self._reduced_plan: datafusion.plan.LogicalPlan | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLTransform:
        with open(path) as f:
            return cls(f.read())

    def register_transformer(self, name: str, transformer) -> SQLTransform:
        self._registered_transformers[name] = transformer
        return self

    def add_table(self, name: str, table: pa.Table) -> SQLTransform:
        self._registered_tables[name] = table
        return self

    def fit(self, table: pa.Table, /) -> SQLTransform:
        # Step 1: Parse SQL, extract transformer calls
        registered = set(self._registered_transformers.keys())
        t_nodes = extract_transformer_calls(self._sql, registered)

        # Step 2: Strip transformer calls, get clean SQL
        clean_sql = strip_transformer_calls(self._sql, registered)

        # Step 3: Create DataFusion session with tables
        ctx = datafusion.SessionContext()
        ctx.from_pyarrow(table, name="data")
        for t_name, t in self._registered_tables.items():
            ctx.from_pyarrow(t, name=t_name)

        # Step 4: Run clean SQL to get logical plan + extract state
        df = ctx.sql(clean_sql)
        plan = df.logical_plan()
        self._state = extract_state(plan, ctx, "data")

        # Step 5: Fit transformers in topological order
        self._state.update(self._registered_transformers)
        fit_transformers(t_nodes, table, self._state)

        # Step 6: Build reduced SQL
        reduced_sql = _build_reduced_sql(
            self._sql, self._state, registered,
        )

        # Step 7: Parse reduced SQL through DataFusion for Phase 2 plan
        ctx2 = datafusion.SessionContext()
        ctx2.from_pydict({}, name="data")  # Dummy registration for parsing
        for t_name in self._registered_tables:
            ctx2.from_pydict({}, name=t_name)
        self._reduced_plan = ctx2.sql(reduced_sql).logical_plan()

        return self

    def transform(self, table: pa.Table, /) -> pa.Table:
        if self._reduced_plan is None:
            raise RuntimeError("Must call fit() before transform()")
        cols = table.column_names
        rows = table.to_pylist()
        out_rows = [
            self._infer_fn({c: row[c] for c in cols}) for row in rows
        ]
        if out_rows:
            return pa.table(
                {k: [r[k] for r in out_rows] for k in out_rows[0]}
            )
        return pa.table({})

    def _infer(self, row: dict) -> dict:
        if self._reduced_plan is None:
            raise RuntimeError("Must call fit() before transform()")
        return self._infer_fn(row)

    def _infer_fn(self, row: dict) -> dict:
        return interpret(self._reduced_plan, row, self._state)


def _build_reduced_sql(sql: str, state: dict, registered: set[str]) -> str:
    """Build reduced SQL with window agg sub-expressions replaced by constants.

    Original: SELECT svd(tfidf(text)) AS emb, age / MEAN(age) OVER() AS norm, ref.temp
              FROM data JOIN ref ON data.id = ref.id

    Reduced:  SELECT NULL AS emb, age / 30.0 AS norm, NULL AS temp FROM data
    """
    import sqlglot
    from sqlglot import exp

    parsed = sqlglot.parse_one(sql)

    # Build window agg info from state: alias → scalar value
    window_scalars = {
        k: v
        for k, v in state.items()
        if isinstance(v, (int, float))
    }

    # Build alias-to-state-value mapping for window aggs
    # (determined by walking the alias tree in state extraction)
    alias_to_scalar = {}
    for alias, val in window_scalars.items():
        alias_to_scalar[alias] = val

    expr_strs: list[str] = []

    for node in parsed.find_all(sqlglot.exp.Select):
        for expr in node.expressions:
            alias = str(expr.alias) if expr.alias else ""

            has_tfm = any(
                f.name.lower() in registered
                for f in expr.find_all(sqlglot.exp.Anonymous)
            )
            if has_tfm:
                expr_strs.append(f"NULL AS {alias}")
                continue

            has_join = any(
                (
                    isinstance(c, sqlglot.exp.Column)
                    and c.table
                    and str(c.table) != "data"
                )
                for c in expr.find_all(sqlglot.exp.Column)
            )
            if has_join:
                expr_strs.append(f"NULL AS {alias}")
                continue

            # Replace window function expressions with their constants
            expr_sql = expr.sql()
            for wf in expr.find_all(sqlglot.exp.Window):
                # Find the parent aggregate function
                agg_parent = wf.parent
                if isinstance(agg_parent, sqlglot.exp.Anonymous):
                    # Replace entire Anonymous(arg1, arg2, OVER (...)) with constant
                    if alias in alias_to_scalar:
                        old = agg_parent.parent.sql() if agg_parent.parent else agg_parent.sql()
                        # Replace just the aggregate+window part
                        agg_sql = agg_parent.sql()
                        val_sql = repr(alias_to_scalar[alias])
                        expr_sql = expr_sql.replace(agg_sql, val_sql)

            expr_strs.append(f"{expr_sql} AS {alias}")

    return "SELECT " + ", ".join(expr_strs) + " FROM data"
```

- [ ] **Step 4: Run tests — iterate until all pass**

```
uv run pytest sql_transform/__init___test.py -v
```

Expected: 10 PASS.

Debug: The `_build_reduced_sql` function needs to produce valid DataFusion SQL with no window aggs and no transformer calls. The key insight: since Phase 2's `interpret()` gets a reduced plan, and the reduced plan is parsed by DataFusion, the reduced SQL must be valid. For window aggs that become constants, the plan's expressions must reference the constant values.

Iterate on `_build_reduced_sql` until all tests pass. May need to:
- Replace window aggregate expressions with their constant values from state
- Keep non-window, non-transformer, non-join expressions as-is
- Remove JOIN clauses entirely (replace with single FROM data)

- [ ] **Step 5: Add S102 + S608 ignores to pyproject.toml**

Read `pyproject.toml`, update per-file-ignores:

```
[tool.ruff.lint.per-file-ignores]
"*test*.py" = ["S608"]
"_state.py" = ["S608"]
"_interpreter.py" = ["S608"]
```

- [ ] **Step 6: Full test suite + ruff**

```
uv run pytest -v
uv run ruff check --fix .
uv run ruff format .
```

Expected: all tests pass, no ruff errors.

- [ ] **Step 7: Commit**

```
git add sql_transform/__init__.py sql_transform/__init___test.py pyproject.toml
git commit -m "refactor: two-phase SQLTransform with plan interpreter"
```

---

### Task 6: Integration test + cleanup

**Files:**
- Modify: `sql_transform/__init___test.py` (add transformer + JOIN tests)
- Delete: `sql_transform/_codegen.py`, `sql_transform/_codegen_test.py`

- [ ] **Step 1: Add transformer integration test**

Append to `sql_transform/__init___test.py`:

```python
def test_simple_transformer_with_window_agg():
    from sklearn.preprocessing import StandardScaler

    sql = "SELECT scale(x) AS x_norm FROM data"
    t = SQLTransform(sql).register_transformer(
        "scale", StandardScaler()
    )

    data = pa.table({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    t.fit(data)

    result = t._infer({"x": 3.0})
    assert "x_norm" in result
    # StandardScaler with mean=3.0, std~1.58 → x=3.0 → ~0.0
    assert abs(result["x_norm"]) < 0.1

    batch = t.transform(pa.table({"x": [6.0, 7.0]}))
    assert len(batch) == 2
    assert "x_norm" in batch.schema.names
```

Actually, sklearn transformer test needs a simpler transformer. StandardScaler with single-feature text won't work (text is not numeric).

- [ ] **Step 1 (revised): Add simple transformer + JOIN test**

```python
def test_simple_transformer_with_window_agg():
    from sklearn.preprocessing import StandardScaler

    sql = "SELECT scale(x) AS x_norm, MEAN(x) OVER () AS x_mean FROM data"
    t = SQLTransform(sql).register_transformer(
        "scale", StandardScaler()
    )

    data = pa.table({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    t.fit(data)

    result = t._infer({"x": 3.0})
    assert "x_norm" in result
    assert abs(result["x_mean"] - 3.0) < 0.001

    batch = t.transform(pa.table({"x": [6.0, 7.0]}))
    assert len(batch) == 2


def test_join_lookup_single_key():
    ref = pa.table({"id": [1, 2], "label": ["low", "high"]})
    data = pa.table({"id": [1, 2], "val": [10.0, 20.0]})

    sql = (
        "SELECT data.val, ref.label FROM data "
        "JOIN ref ON data.id = ref.id"
    )
    t = SQLTransform(sql).add_table("ref", ref)
    t.fit(data)

    result = t._infer({"id": 1, "val": 10.0})
    assert result["label"] == "low"
    assert result["val"] == 10.0


def test_join_lookup_multi_key():
    ref = pa.table({
        "city": ["a", "b"], "country": ["X", "Y"], "label": ["one", "two"]
    })
    data = pa.table({
        "city": ["a", "b"], "country": ["X", "Y"], "val": [1.0, 2.0]
    })

    sql = (
        "SELECT data.val, ref.label FROM data "
        "JOIN ref ON data.city = ref.city AND data.country = ref.country"
    )
    t = SQLTransform(sql).add_table("ref", ref)
    t.fit(data)

    result = t._infer({"city": "b", "country": "Y", "val": 2.0})
    assert result["label"] == "two"


def test_error_before_fit():
    t = SQLTransform("SELECT age FROM data")
    with pytest.raises(RuntimeError):
        t.transform(pa.table({"age": [1]}))

    with pytest.raises(RuntimeError):
        t._infer({"age": 1})
```

- [ ] **Step 2: Run integration tests**

```
uv run pytest sql_transform/__init___test.py -v -k "transformer or join"
```

Expected: PASS for the new tests.

- [ ] **Step 3: Remove old codegen files**

```
git rm sql_transform/_codegen.py sql_transform/_codegen_test.py
```

- [ ] **Step 4: Full test suite**

```
uv run pytest -v
uv run ruff check --fix .
uv run ruff format .
```

Expected: all tests pass, no ruff errors.

- [ ] **Step 5: Final commit**

```
git add -A
git commit -m "feat: add transformer and JOIN support, remove codegen"
```