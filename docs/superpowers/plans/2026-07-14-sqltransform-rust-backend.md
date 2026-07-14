# SQLTransform on the Rust Interpreter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SQLTransform`'s Python-codegen (`exec()`-based) inference path with the Rust `InferFn` interpreter, by rewriting the user's SQL at `fit()` time into a form the Rust engine runs directly.

**Architecture:** `fit()` runs the user's SQL through DataFusion once to (a) extract window-aggregate constants into a synthesized Pydantic `StateModel` instance, and (b) rewrite the SQL's projection expressions into plain-column-reference SQL text (`__STATE__.avg_age` instead of `AVG(age) OVER ()`). That rewritten SQL, plus a synthesized-or-supplied `__THIS__` row model and the `StateModel` class, construct an `InferFn`. `transform()`/`_infer()` then just call `InferFn.infer()` with `__THIS__` rows and a 1-element `__STATE__` list (the constant state instance), relying on `InferFn`'s existing `CrossJoin` to repeat it per row.

**Tech Stack:** Python 3.13, DataFusion (`datafusion` package) for `fit()`-time SQL parsing/aggregation, Pydantic v2 (`create_model`) for schema synthesis, the existing Rust `InferFn` (`sql_transform._interpreter`, already built) for inference. No Rust changes in this plan — see spec Non-Goals.

## Global Constraints

- Source SQL must reference the input table as `__THIS__` (e.g. `FROM __THIS__`) — required convention, no rewriting of arbitrary table names.
- Only non-partitioned window aggregates are supported: `AGG(col) OVER ()` (empty window). `OVER (PARTITION BY ...)` raises `NotImplementedError` from `extract_state`, before any Rust call.
- State keys are `{fn}_{col}`, lowercase, table-qualifier stripped, **no leading underscore** (Pydantic v2 treats leading-underscore field names as private attributes — excluded from `model_fields`, unsettable via `model_validate`). Referenced in rewritten SQL as `__STATE__.{fn}_{col}`.
- `__STATE__` is a **row table** (not a static table) passed as `{"__STATE__": [state_instance]}` on every `.infer()` call — no `LookupJoin`/Arrow serialization involved.
- Duplicate `(fn, col)` pairs across projections are computed once (deduped), not once per occurrence.
- This is a breaking change to `SQLTransform`'s public SQL convention (`FROM __THIS__` replaces arbitrary table names like `FROM data`) — no backward-compat shim, per project convention (pre-1.0, breaking changes are made directly).
- Pydantic v2 only (`create_model`, `model_fields`, `model_validate`) — matches the rest of the codebase.

---

## File Structure

```
sql_transform/_schema.py        (new)      — pydantic model synthesis for __THIS__/__STATE__
sql_transform/_schema_test.py   (new)      — tests for _schema.py
sql_transform/_state.py         (rewrite)  — extract_state() + state_key(), dedup, partition rejection
sql_transform/_state_test.py    (rewrite)  — tests for the rewritten _state.py
sql_transform/_codegen.py       (rewrite)  — rewrite_sql(): plan projections -> SQL text
sql_transform/_codegen_test.py  (rewrite)  — tests for the rewritten _codegen.py
sql_transform/__init__.py       (rewrite)  — SQLTransform using InferFn
sql_transform/__init___test.py  (rewrite)  — tests for the rewritten SQLTransform
README.md                       (modify)   — update Quick Start example to `FROM __THIS__`
```

`_state.py` and `_codegen.py` keep their existing responsibilities (state extraction vs. expression-tree walking) — only their *output* changes.

---

### Task 1: `_schema.py` — Pydantic model synthesis

**Files:**
- Create: `sql_transform/_schema.py`
- Test: `sql_transform/_schema_test.py`

**Interfaces:**
- Consumes: nothing from other tasks (leaf module).
- Produces:
  - `synthesize_this_model(schema: pyarrow.Schema) -> type[pydantic.BaseModel]`
  - `synthesize_state_model(state: dict[str, float]) -> type[pydantic.BaseModel]`
  - Used by: Task 2 (`synthesize_state_model`), Task 4 (`synthesize_this_model`).

- [ ] **Step 1: Write the failing tests**

Create `sql_transform/_schema_test.py`:

```python
"""Tests for pydantic model synthesis (__THIS__ and __STATE__ schemas)."""

import pyarrow as pa

from sql_transform._schema import synthesize_state_model, synthesize_this_model


def test_synthesize_this_model_basic_types():
    schema = pa.schema(
        [
            pa.field("age", pa.int64(), nullable=False),
            pa.field("score", pa.float64(), nullable=False),
            pa.field("name", pa.string(), nullable=True),
            pa.field("active", pa.bool_(), nullable=False),
        ]
    )
    model = synthesize_this_model(schema)

    assert model.model_fields["age"].annotation is int
    assert model.model_fields["score"].annotation is float
    assert model.model_fields["name"].annotation == (str | None)
    assert model.model_fields["active"].annotation is bool


def test_synthesize_this_model_instantiates_from_values():
    schema = pa.schema([pa.field("age", pa.int64(), nullable=False)])
    model = synthesize_this_model(schema)
    instance = model(age=30)
    assert instance.age == 30


def test_synthesize_this_model_nullable_field_accepts_none():
    schema = pa.schema([pa.field("name", pa.string(), nullable=True)])
    model = synthesize_this_model(schema)
    instance = model(name=None)
    assert instance.name is None


def test_synthesize_state_model_all_float_fields():
    model = synthesize_state_model({"avg_age": 30.0, "sum_score": 60.0})
    assert model.model_fields["avg_age"].annotation is float
    assert model.model_fields["sum_score"].annotation is float


def test_synthesize_state_model_instantiates_from_values():
    model = synthesize_state_model({"avg_age": 30.0})
    instance = model(avg_age=30.0)
    assert instance.avg_age == 30.0


def test_synthesize_state_model_empty_state_is_valid():
    model = synthesize_state_model({})
    instance = model()
    assert instance.model_dump() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_schema_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sql_transform._schema'`

- [ ] **Step 3: Write the implementation**

Create `sql_transform/_schema.py`:

```python
"""Synthesize Pydantic models for SQLTransform's __THIS__ and __STATE__ tables."""

from __future__ import annotations

import typing

import pyarrow as pa
from pydantic import BaseModel, create_model


def synthesize_this_model(schema: pa.Schema) -> type[BaseModel]:
    """Build a Pydantic model matching an Arrow schema's columns, types, and
    nullability — used as __THIS__'s row schema when the caller doesn't
    supply their own `this_model`."""
    fields: dict[str, tuple[object, object]] = {}
    for field in schema:
        base = _arrow_type_to_python(field.type)
        py_type: object = base
        if base is not typing.Any and field.nullable:
            py_type = base | None
        fields[field.name] = (py_type, ...)
    return create_model("ThisRow", **fields)


def synthesize_state_model(state: dict[str, float]) -> type[BaseModel]:
    """Build a Pydantic model with one float field per state key — used as
    __STATE__'s row schema. Field names must have no leading underscore:
    Pydantic v2 treats those as private attributes, excluded from
    model_fields and unsettable via the constructor."""
    fields = {key: (float, ...) for key in state}
    return create_model("StateModel", **fields)


def _arrow_type_to_python(arrow_type: pa.DataType) -> object:
    if pa.types.is_integer(arrow_type):
        return int
    if pa.types.is_floating(arrow_type):
        return float
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return str
    if pa.types.is_boolean(arrow_type):
        return bool
    return typing.Any
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_schema_test.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_schema.py sql_transform/_schema_test.py
git commit -m "feat: synthesize pydantic models for __THIS__/__STATE__ schemas"
```

---

### Task 2: `_state.py` — rewrite `extract_state` for dedup, partition rejection, and typed output

**Files:**
- Modify (full rewrite): `sql_transform/_state.py`
- Test (full rewrite): `sql_transform/_state_test.py`

**Interfaces:**
- Consumes: `synthesize_state_model(state: dict[str, float]) -> type[BaseModel]` (Task 1).
- Produces:
  - `state_key(fn_name: str, col_name: str) -> str` — used by Task 3 (`_codegen.py`) to compute matching keys.
  - `extract_state(plan: datafusion.plan.LogicalPlan, ctx: datafusion.SessionContext, table_name: str) -> pydantic.BaseModel` — an *instance* (not a class, not a dict). Used by Task 4.
  - Raises `NotImplementedError` if any window aggregate in the plan has `PARTITION BY`.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/_state_test.py`:

```python
"""Tests for state extraction from DataFusion logical plans."""

import datafusion
import pytest

from sql_transform._state import extract_state, state_key


def test_state_key_lowercases_and_strips_qualifier():
    assert state_key("AVG", "age") == "avg_age"
    assert state_key("avg", "AGE") == "avg_age"


def test_extract_constant_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = "SELECT age / MEAN(age) OVER () AS age_norm FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.avg_age == 30.0


def test_extract_dedups_repeated_aggregate():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "MEAN(age) OVER () AS age_avg FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    # Both projections reference the same (fn, col) pair -> one field.
    # DataFusion normalizes MEAN to avg internally, so the key is avg_age.
    assert state.model_dump() == {"avg_age": 30.0}


def test_extract_multiple_distinct_aggregates():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35], "score": [10, 20, 30]}, name="data")

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.avg_age == 30.0
    assert state.sum_score == 60.0


def test_extract_no_aggregates_returns_empty_state():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [1, 2, 3]}, name="data")

    sql = "SELECT age FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.model_dump() == {}


def test_extract_partitioned_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
        name="data",
    )

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    with pytest.raises(NotImplementedError):
        extract_state(plan, ctx, "data")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_state_test.py -v`
Expected: FAIL — `state_key` doesn't exist yet, and `extract_state`'s current alias-keyed dict return won't have `.avg_age`/`.model_dump()`.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/_state.py`:

```python
"""Extract learned state from DataFusion logical plans.

Parses DataFusion plan display text to find window aggregate columns in
projection expressions, then executes one DataFusion query per DISTINCT
(function, column) pair to compute its scalar value. The result is a
synthesized Pydantic model instance ("StateModel") keyed by `{fn}_{col}`
(no leading underscore -- see state_key), suitable for use as InferFn's
__STATE__ row table.
"""

from __future__ import annotations

import re

import datafusion
from pydantic import BaseModel

from sql_transform._schema import synthesize_state_model


def state_key(fn_name: str, col_name: str) -> str:
    """The __STATE__ field name for a given aggregate function + column,
    e.g. state_key("AVG", "age") == "avg_age". No leading underscore --
    Pydantic v2 would treat that as a private attribute."""
    return f"{fn_name.lower()}_{col_name.lower()}"


def extract_state(
    plan: datafusion.plan.LogicalPlan,
    ctx: datafusion.SessionContext,
    table_name: str,
) -> BaseModel:
    """Return a synthesized StateModel instance with one float field per
    distinct (fn, col) window aggregate referenced in the plan.

    Raises NotImplementedError if any window aggregate uses PARTITION BY --
    not yet supported by the Rust-backed pipeline.
    """
    display = plan.display_indent()

    pairs: dict[tuple[str, str], None] = {}
    for m in _WINDOW_AGG_RE.finditer(display):
        if m.group("partition"):
            raise NotImplementedError(
                "PARTITION BY window aggregates are not yet supported by "
                "the Rust-backed SQLTransform pipeline"
            )
        pairs[(m.group("fn").lower(), m.group("col").lower())] = None

    values: dict[str, float] = {}
    for fn_name, col_name in pairs:
        sql = f"SELECT {fn_name}({col_name}) FROM {table_name}"
        result = ctx.sql(sql).collect()
        value = result[0].column(0)[0].as_py()
        values[state_key(fn_name, col_name)] = float(value)

    state_model = synthesize_state_model(values)
    return state_model(**values)


_WINDOW_AGG_RE = re.compile(
    r"(?P<fn>\w+)"
    r"\((?:\w+)\.(?P<col>\w+)\)"
    r"(?P<partition>\s+PARTITION\s+BY\s+\[(?:\w+)\.\w+\])?"
    r"\s+ROWS\s+BETWEEN[^,\n]+"
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_state_test.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_state.py sql_transform/_state_test.py
git commit -m "feat: dedup state extraction, typed StateModel, reject PARTITION BY"
```

---

### Task 3: `_codegen.py` — rewrite `rewrite_sql` to emit SQL text instead of Python source

**Files:**
- Modify (full rewrite): `sql_transform/_codegen.py`
- Test (full rewrite): `sql_transform/_codegen_test.py`

**Interfaces:**
- Consumes: `state_key(fn_name: str, col_name: str) -> str` (Task 2).
- Produces: `rewrite_sql(plan: datafusion.plan.LogicalPlan) -> str` — a full `SELECT ... FROM __THIS__, __STATE__` SQL string. Used by Task 4.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/_codegen_test.py`:

```python
"""Tests for SQL rewriting from DataFusion logical plans."""

import datafusion

from sql_transform._codegen import rewrite_sql


def _plan(sql: str, data: dict) -> datafusion.plan.LogicalPlan:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).logical_plan()


def test_rewrite_simple_column_pass_through():
    plan = _plan("SELECT age AS just_age FROM data", {"age": [1, 2, 3]})
    sql = rewrite_sql(plan)
    assert sql == "SELECT __THIS__.age AS just_age FROM __THIS__, __STATE__"


def test_rewrite_constant_window_agg():
    plan = _plan(
        "SELECT age / MEAN(age) OVER () AS age_norm FROM data",
        {"age": [25, 30, 35]},
    )
    sql = rewrite_sql(plan)
    # DataFusion normalizes MEAN to avg internally, so the key is avg_age.
    assert sql == (
        "SELECT (__THIS__.age / __STATE__.avg_age) AS age_norm "
        "FROM __THIS__, __STATE__"
    )


def test_rewrite_bare_window_agg_alias():
    plan = _plan(
        "SELECT MEAN(age) OVER () AS age_avg FROM data",
        {"age": [25, 30, 35]},
    )
    sql = rewrite_sql(plan)
    assert sql == "SELECT __STATE__.avg_age AS age_avg FROM __THIS__, __STATE__"


def test_rewrite_multiple_projections():
    plan = _plan(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data",
        {"age": [25, 30, 35], "score": [10, 20, 30]},
    )
    sql = rewrite_sql(plan)
    assert sql == (
        "SELECT (__THIS__.age / __STATE__.avg_age) AS age_norm, "
        "(__THIS__.score / __STATE__.sum_score) AS score_norm "
        "FROM __THIS__, __STATE__"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/_codegen_test.py -v`
Expected: FAIL — `rewrite_sql` doesn't exist yet (`_codegen.py` currently only exports `generate_infer_fn`).

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/_codegen.py`:

```python
"""Rewrite DataFusion logical plans into SQL runnable by the Rust InferFn.

Walks each top-level projection alias and converts its expression tree
into SQL text. Window aggregate references (Alias wrapping a Column, or a
bare Column with a DataFusion-generated window-agg name) are rewritten
into `__STATE__.<fn>_<col>` references; plain columns become
`__THIS__.<col>` references. The result is `SELECT ... FROM __THIS__,
__STATE__` -- a cross join, since __STATE__ is always exactly one row.
"""

from __future__ import annotations

import re

import datafusion
from datafusion.expr import Alias, BinaryExpr, Column

from sql_transform._state import state_key


def rewrite_sql(plan: datafusion.plan.LogicalPlan) -> str:
    """Return a SQL string equivalent to the plan's projection, with every
    window-aggregate reference replaced by a __STATE__ column reference."""
    proj = plan.to_variant()
    parts: list[str] = []

    for raw_p in proj.projections():
        alias = raw_p.to_variant()
        if isinstance(alias, Column):
            out_name = alias.name()
            expr_sql = _expr_to_sql(raw_p)
        else:
            out_name = alias.alias()
            expr_sql = _expr_to_sql(alias.expr())
        parts.append(f"{expr_sql} AS {out_name}")

    return "SELECT " + ", ".join(parts) + " FROM __THIS__, __STATE__"


def _expr_to_sql(raw_expr) -> str:
    """Convert a RawExpr tree to a SQL expression string."""
    expr = raw_expr.to_variant()

    if isinstance(expr, Column):
        return _column_to_sql(expr.name())

    if isinstance(expr, BinaryExpr):
        left = _expr_to_sql(expr.left())
        right = _expr_to_sql(expr.right())
        return f"({left} {expr.op()} {right})"

    if isinstance(expr, Alias):
        return _expr_to_sql(expr.expr())

    raise ValueError(f"Unrecognized expression node: {type(expr).__name__}")


def _column_to_sql(col_name: str) -> str:
    m = _WINDOW_COL_RE.match(col_name)
    if m:
        key = state_key(m.group("fn"), m.group("col"))
        return f"__STATE__.{key}"
    return f"__THIS__.{col_name}"


# DataFusion generates window aggregate column names like:
#   avg(data.age) ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
_WINDOW_COL_RE = re.compile(r"^(?P<fn>\w+)\((?:\w+\.)?(?P<col>\w+)\)\s")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/_codegen_test.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sql_transform/_codegen.py sql_transform/_codegen_test.py
git commit -m "feat: rewrite window-agg SQL to __STATE__/__THIS__ references"
```

---

### Task 4: `__init__.py` — `SQLTransform` on top of `InferFn`

**Files:**
- Modify (full rewrite): `sql_transform/__init__.py`
- Test (full rewrite): `sql_transform/__init___test.py`

**Interfaces:**
- Consumes:
  - `synthesize_this_model(schema: pa.Schema) -> type[BaseModel]` (Task 1)
  - `extract_state(plan, ctx, table_name: str) -> BaseModel` (Task 2)
  - `rewrite_sql(plan) -> str` (Task 3)
  - `InferFn(sql, row_tables, static_tables, output_model=None)` and `InferFn.infer(tables: dict[str, list]) -> list[BaseModel]` (existing Rust module, `sql_transform._interpreter`)
- Produces: `SQLTransform` public class (see Public API in the spec) — this is the top-level deliverable, nothing downstream in this plan consumes it further.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `sql_transform/__init___test.py`:

```python
"""Tests for SQLTransform."""

import pyarrow as pa
import pytest
from pydantic import BaseModel


def assert_approx_equal(actual: list, expected: list) -> None:
    for a, e in zip(actual, expected, strict=True):
        assert abs(a - e) < 0.001


def test_fit_and_transform_batch_no_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    data = pa.table({"age": [1, 2, 3]})
    result = t.fit(data).transform(data)
    assert result.column("age").to_pylist() == [1, 2, 3]


def test_fit_and_transform_constant_agg():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    data = pa.table({"age": [25, 30, 35]})
    result = t.fit(data).transform(data)
    assert_approx_equal(
        result.column("age_norm").to_pylist(),
        [25 / 30, 30 / 30, 35 / 30],
    )


def test_transform_on_unseen_data():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    train = pa.table({"age": [25, 30, 35]})
    test_data = pa.table({"age": [40, 50]})
    result = t.fit(train).transform(test_data)
    assert_approx_equal(
        result.column("age_norm").to_pylist(),
        [40 / 30, 50 / 30],
    )


def test_single_row_inference():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    result = t._infer({"age": 40})
    assert abs(result["age_norm"] - 40 / 30) < 0.001


def test_multiple_columns():
    from sql_transform import SQLTransform

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM __THIS__"
    )
    t = SQLTransform(sql)
    data = pa.table({"age": [25, 30, 35], "score": [10, 20, 30]})
    result = t.fit(data).transform(data)
    assert "age_norm" in result.schema.names
    assert "score_norm" in result.schema.names


def test_fit_returns_self():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    result = t.fit(pa.table({"age": [1]}))
    assert result is t


def test_from_file(tmp_path):
    from sql_transform import SQLTransform

    sql_file = tmp_path / "features.sql"
    sql_file.write_text("SELECT age / MEAN(age) OVER () AS x FROM __THIS__")

    t = SQLTransform.from_file(str(sql_file))
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 10})
    assert "x" in result


def test_e2e_two_transforms_and_dedup():
    """End-to-end: fit on training, transform batch, infer single row,
    with a repeated aggregate deduped across two projections."""
    from sql_transform import SQLTransform

    sql = """
    SELECT
        age / MEAN(age) OVER () AS age_norm,
        income / SUM(income) OVER () AS income_share
    FROM __THIS__
    """

    t = SQLTransform(sql)
    train = pa.table(
        {
            "age": [25, 30, 35, 40],
            "income": [50_000, 60_000, 70_000, 80_000],
        }
    )

    t.fit(train)

    out = t.transform(train)
    assert out.schema.names == ["age_norm", "income_share"]
    assert len(out) == 4

    row = {"age": 50, "income": 100_000}
    result = t._infer(row)

    mean_age = 32.5
    assert abs(result["age_norm"] - 50 / mean_age) < 0.001

    total_income = 260_000.0
    assert abs(result["income_share"] - 100_000 / total_income) < 0.001


def test_partitioned_agg_raises_not_implemented():
    from sql_transform import SQLTransform

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM __THIS__"
    t = SQLTransform(sql)
    data = pa.table({"city": ["a", "b"], "target": [1.0, 2.0]})
    with pytest.raises(NotImplementedError):
        t.fit(data)


def test_this_model_omitted_synthesizes_from_table_schema():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age FROM __THIS__")
    t.fit(pa.table({"age": [1, 2, 3]}))
    result = t._infer({"age": 5})
    assert result["age"] == 5


def test_this_model_supplied_compatible():
    from sql_transform import SQLTransform

    class Row(BaseModel):
        age: int

    t = SQLTransform("SELECT age FROM __THIS__")
    t.fit(pa.table({"age": [1, 2, 3]}), this_model=Row)
    result = t._infer({"age": 7})
    assert result["age"] == 7


def test_this_model_supplied_missing_referenced_column_raises():
    from sql_transform import SQLTransform

    class IncompleteRow(BaseModel):
        other: int  # doesn't declare "age", which the query references

    t = SQLTransform("SELECT age FROM __THIS__")
    with pytest.raises(ValueError):
        t.fit(pa.table({"age": [1, 2, 3]}), this_model=IncompleteRow)


def test_state_is_typed_pydantic_instance():
    from sql_transform import SQLTransform

    t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    t.fit(pa.table({"age": [25, 30, 35]}))
    # DataFusion normalizes MEAN to avg internally, so the field is avg_age.
    assert isinstance(t._state.avg_age, float)
    assert t._state.avg_age == 30.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: FAIL — current `SQLTransform` requires `FROM data`-style SQL (not `FROM __THIS__`), has no `this_model` parameter, and `_infer_fn`/`_state` aren't wired to `InferFn`.

- [ ] **Step 3: Write the implementation**

Replace the full contents of `sql_transform/__init__.py`:

```python
"""SQLTransform — sklearn-compatible SQL-based feature transforms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import datafusion
import pyarrow as pa
from pydantic import BaseModel

from sql_transform._codegen import rewrite_sql
from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model
from sql_transform._state import extract_state

__all__ = ["InferFn", "SQLTransform"]


class SQLTransform:
    """A transformer that applies SQL window-aggregate transforms.

    fit() runs the SQL on training data via DataFusion to extract window
    aggregate state, rewrites the SQL into plain-column-reference form,
    and builds a Rust InferFn for evaluation.

    transform() applies the transforms to batch data via InferFn.

    Usage:
        t = SQLTransform("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
        t.fit(train_table)
        out = t.transform(test_table)        # batch
        out_row = t._infer({"age": 42})      # single row
    """

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self._state: BaseModel | None = None
        self._infer_fn: InferFn | None = None

    @classmethod
    def from_file(cls, path: str) -> SQLTransform:
        with open(path) as f:
            return cls(f.read())

    def fit(
        self,
        table: pa.Table,
        /,
        this_model: type[BaseModel] | None = None,
    ) -> SQLTransform:
        this_model = this_model or synthesize_this_model(table.schema)

        ctx = datafusion.SessionContext()
        ctx.from_arrow(table, name="__THIS__")
        df = ctx.sql(self._sql)
        plan = df.logical_plan()

        self._state = extract_state(plan, ctx, "__THIS__")
        rewritten_sql = rewrite_sql(plan)
        self._infer_fn = InferFn(
            rewritten_sql,
            row_tables={"__THIS__": this_model, "__STATE__": type(self._state)},
            static_tables={},
        )
        return self

    def transform(self, table: pa.Table, /) -> pa.Table:
        """Apply transforms to batch data using learned state, via InferFn."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before transform()")
        rows = table.to_pylist()
        out_rows = self._infer_fn.infer(
            {
                "__THIS__": [SimpleNamespace(**row) for row in rows],
                "__STATE__": [self._state],
            }
        )
        out_dicts = [r.model_dump() for r in out_rows]
        return (
            pa.table({k: [r[k] for r in out_dicts] for k in out_dicts[0]})
            if out_dicts
            else pa.table({})
        )

    def _infer(self, row: dict[str, Any]) -> dict[str, Any]:
        """Single-row inference via InferFn."""
        if self._infer_fn is None:
            raise RuntimeError("Must call fit() before inference")
        out_rows = self._infer_fn.infer(
            {"__THIS__": [SimpleNamespace(**row)], "__STATE__": [self._state]}
        )
        return out_rows[0].model_dump()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest sql_transform/__init___test.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest`
Expected: All tests pass, including `tests/test_interpreter.py` (untouched, exercises `InferFn` directly).

- [ ] **Step 6: Commit**

```bash
git add sql_transform/__init__.py sql_transform/__init___test.py
git commit -m "feat: SQLTransform runs inference through the Rust InferFn"
```

---

### Task 5: Update README's Quick Start example

**Files:**
- Modify: `README.md:26-73`

**Interfaces:**
- Consumes: nothing (docs only).
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Update the Quick Start SQL examples**

In `README.md`, replace the "Basic Usage" example's SQL block (currently `FROM data` with `avg(feature1)`/`avg(feature2) over (partition by class)`, README.md:42-47) with:

```markdown
```python
from sql_transform import SQLTransform
import pyarrow as pa

# Create sample data
data = pa.table({
    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "feature2": [10, 20, 30, 40, 50],
})

# Define SQL transformation -- input table is always referenced as __THIS__
sql = """
SELECT
    feature1 / MEAN(feature1) OVER () AS feature1_norm,
    feature2 / SUM(feature2) OVER () AS feature2_share
FROM __THIS__
"""

# Fit and transform
transformer = SQLTransform(sql)
transformer.fit(data)
result = transformer.transform(data)
print(result)
```
```

(The partitioned `class`/`avg(feature2) over (partition by class)` example is dropped from this section — `PARTITION BY` isn't supported yet; see the sklearn section below, which is unaffected since it doesn't use window aggregates.)

Also update the "Current" bullet list under Features (README.md:100-107): change `- Basic SQL parsing (SELECT with aggregations)` to note the `__THIS__` convention, e.g. `- Rust-backed inference via a rewritten-SQL pipeline (window aggregates against __THIS__/__STATE__)`.

- [ ] **Step 2: Verify the example actually runs**

Run:
```bash
uv run python -c "
from sql_transform import SQLTransform
import pyarrow as pa

data = pa.table({'feature1': [1.0, 2.0, 3.0, 4.0, 5.0], 'feature2': [10, 20, 30, 40, 50]})
sql = '''
SELECT
    feature1 / MEAN(feature1) OVER () AS feature1_norm,
    feature2 / SUM(feature2) OVER () AS feature2_share
FROM __THIS__
'''
transformer = SQLTransform(sql)
transformer.fit(data)
print(transformer.transform(data))
"
```
Expected: prints a `pyarrow.Table` with `feature1_norm`/`feature2_share` columns, no errors.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update Quick Start example for the __THIS__/__STATE__ convention"
```

---

## Post-Plan Verification

- [ ] Run `mise run check` (fmt + full test suite) and confirm it's clean.
- [ ] Confirm `sql_transform/_codegen.py` no longer contains any `exec()` call (`grep -n exec sql_transform/_codegen.py` should return nothing) — this was the whole point of the migration.
