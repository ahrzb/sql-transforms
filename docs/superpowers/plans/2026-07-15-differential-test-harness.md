# Differential Test Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Rust `InferFn` interpreter a differential test harness — a `check(query, tables, expect=)` helper that runs a query through both DataFusion (oracle) and `InferFn` over the same typed input and asserts the values match — plus native `@pytest.mark.parametrize` test suites covering the Layer-1 surface.

**Architecture:** One helper module `tests/differential.py` (the `check`/`check_both_raise` runners, `row`/`rows`/`static` table builders, and an order-insensitive/float-tolerant/NULL-aware comparator). Tests are ordinary parametrized functions calling `check`; known divergences use `pytest.mark.xfail(strict=True)`, both-reject cases use `check_both_raise`. No YAML.

**Tech Stack:** Python 3.13, pytest, DataFusion (`datafusion`), the Rust `InferFn`, pyarrow, Pydantic v2 (via `synthesize_this_model`).

## Global Constraints

- Tests-only change — no engine (`src/`, `sql_transform/`) behavior is modified.
- Differential: DataFusion is the oracle; `check` asserts `InferFn` output equals it.
- **Order-insensitive comparison is required** (verified: DataFusion reorders join output; Layer-1 has no `ORDER BY`). Float tolerance `1e-9`; `None`-aware; output column set must match exactly.
- `schema` is a type-spec dict: base `int`/`float`/`str`/`bool`, trailing `?` = nullable; Python builtins `int/float/str/bool` accepted as values. No Pydantic-model schemas.
- The helper module is `tests/differential.py` — the name has no `test_` prefix / `_test` suffix, so pytest (`python_files = ["*_test.py", "test_*.py"]`) does NOT collect it as tests. Test modules import it as `from differential import ...` (verified: pytest prepends the test dir to `sys.path`).
- Known engine divergences are tracked as `xfail(strict=True)`, never silently skipped.

---

## File Structure

```
tests/differential.py                 (new) — check(), check_both_raise(), row/rows/static, comparator
tests/test_differential_selftest.py   (new) — tests OF the harness itself
tests/test_diff_expressions.py        (new) — arithmetic, comparisons, logic, CAST, string fns, NULL
tests/test_diff_relational.py         (new) — WHERE, INNER/CROSS/LookupJoin/LEFT-lookup, multi-row
tests/test_diff_errors.py             (new) — both-reject errors; known divergences (xfail)
tests/test_interpreter.py             (trim) — migrate differential cases to check(); keep the rest
```

---

### Task 1: `tests/differential.py` — the harness (+ self-tests)

**Files:**
- Create: `tests/differential.py`
- Test: `tests/test_differential_selftest.py`

**Interfaces:**
- Consumes: `InferFn` (`sql_transform._interpreter`), `synthesize_this_model` (`sql_transform._schema`), `datafusion`, `pyarrow`.
- Produces (used by Tasks 2-4):
  - `check(query: str, tables: dict[str, Table], expect: list[dict] | None = None) -> None`
  - `check_both_raise(query: str, tables: dict[str, Table], match: str | None = None) -> None`
  - `row(**cols) -> Table`, `rows(schema: dict, data: list[dict]) -> Table`, `static(schema: dict, data: list[dict]) -> Table`

- [ ] **Step 1: Write the failing self-tests**

Create `tests/test_differential_selftest.py`:

```python
"""Tests OF the differential harness itself."""

import pytest

from differential import (
    _rows_equal,
    check,
    check_both_raise,
    row,
    rows,
    static,
)


def test_check_passes_on_agreement():
    check("SELECT a + b AS s FROM t", {"t": row(a=6, b=4)}, expect=[{"s": 10}])


def test_check_expect_mismatch_raises():
    with pytest.raises(AssertionError):
        check("SELECT a + b AS s FROM t", {"t": row(a=6, b=4)}, expect=[{"s": 999}])


def test_rows_equal_order_insensitive():
    assert _rows_equal([{"x": 1}, {"x": 2}], [{"x": 2}, {"x": 1}])
    assert not _rows_equal([{"x": 1}], [{"x": 1}, {"x": 2}])


def test_rows_equal_float_tolerance_and_null():
    assert _rows_equal([{"x": 1.0}], [{"x": 1.0 + 1e-12}])
    assert _rows_equal([{"x": None}], [{"x": None}])
    assert not _rows_equal([{"x": None}], [{"x": 1}])


def test_row_infers_types_and_check_multirow():
    check(
        "SELECT age FROM t WHERE age > 18",
        {"t": rows({"age": "int"}, [{"age": 15}, {"age": 25}, {"age": 40}])},
        expect=[{"age": 25}, {"age": 40}],
    )


def test_static_join_agrees():
    check(
        "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id",
        {
            "data": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 10}]),
            "ref": static({"id": "int", "y": "str"}, [{"id": 1, "y": "a"}]),
        },
    )


def test_nullable_column_omitted_value():
    check(
        "SELECT UPPER(name) AS n FROM t",
        {"t": rows({"name": "str?"}, [{"name": "alice"}, {}])},  # 2nd row: name NULL
        expect=[{"n": "ALICE"}, {"n": None}],
    )


def test_check_both_raise_on_unknown_column():
    check_both_raise("SELECT nope FROM t", {"t": row(a=1)})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_differential_selftest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'differential'`.

- [ ] **Step 3: Write the harness**

Create `tests/differential.py`:

```python
"""Differential test harness for the Rust InferFn interpreter.

`check(query, tables)` runs a query through DataFusion (the oracle) AND the Rust
InferFn over the same typed input, and asserts their output values match. Tests
are native pytest parametrized decision tables (see test_diff_*.py). This module
is NOT collected by pytest (no test_ prefix / _test suffix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import datafusion
import pyarrow as pa

from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model

_ARROW = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}
_BUILTIN = {int: "int", float: "float", str: "str", bool: "bool"}


@dataclass
class Table:
    kind: str  # "row" or "static"
    schema: pa.Schema
    rows: list[dict[str, Any]]


def _arrow_field(name: str, spec: Any) -> pa.Field:
    if spec in _BUILTIN:  # python builtin type value
        spec = _BUILTIN[spec]
    if not isinstance(spec, str):
        raise ValueError(f"Unsupported column type {spec!r} for column {name!r}")
    nullable = spec.endswith("?")
    base = spec[:-1] if nullable else spec
    if base not in _ARROW:
        raise ValueError(f"Unknown column type {spec!r} for column {name!r}")
    return pa.field(name, _ARROW[base], nullable=nullable)


def _make(kind: str, schema: dict[str, Any], data: list[dict]) -> Table:
    pa_schema = pa.schema([_arrow_field(n, spec) for n, spec in schema.items()])
    cols = [f.name for f in pa_schema]
    # Fill omitted columns with None so both engines see identical rows.
    norm = [{c: r.get(c) for c in cols} for r in data]
    return Table(kind=kind, schema=pa_schema, rows=norm)


def row(**cols: Any) -> Table:
    """A single-row `row` table with column types inferred from the values.
    Use rows() for explicit types or nullable columns (a None value here is an
    error -- its type can't be inferred)."""
    schema: dict[str, Any] = {}
    for k, v in cols.items():
        if type(v) not in _BUILTIN:
            raise ValueError(
                f"row({k}={v!r}): can't infer a type; use rows() with an "
                "explicit schema"
            )
        schema[k] = _BUILTIN[type(v)]
    return _make("row", schema, [cols])


def rows(schema: dict[str, Any], data: list[dict]) -> Table:
    """A multi-row `row` table with an explicit type-spec schema."""
    return _make("row", schema, data)


def static(schema: dict[str, Any], data: list[dict]) -> Table:
    """A preloaded static table (goes to InferFn.static_tables)."""
    return _make("static", schema, data)


def _run_datafusion(query: str, tables: dict[str, Table]) -> list[dict]:
    ctx = datafusion.SessionContext()
    for name, tbl in tables.items():
        ctx.from_arrow(pa.Table.from_pylist(tbl.rows, schema=tbl.schema), name=name)
    df = ctx.sql(query)
    return pa.Table.from_batches(df.collect(), schema=df.schema()).to_pylist()


def _run_infer(query: str, tables: dict[str, Table]) -> list[dict]:
    row_models: dict[str, Any] = {}
    infer_rows: dict[str, list] = {}
    static_tables: dict[str, pa.Table] = {}
    for name, tbl in tables.items():
        if tbl.kind == "row":
            model = synthesize_this_model(tbl.schema)
            row_models[name] = model
            infer_rows[name] = [model(**r) for r in tbl.rows]
        else:
            static_tables[name] = pa.Table.from_pylist(tbl.rows, schema=tbl.schema)
    fn = InferFn(query, row_tables=row_models, static_tables=static_tables)
    return [r.model_dump() for r in fn.infer(infer_rows)]


def _canon(r: dict) -> tuple:
    # Sortable key for order-insensitive comparison. None sorts before values.
    return tuple(sorted((k, v is None, str(v)) for k, v in r.items()))


def _val_equal(a: Any, b: Any, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def _rows_equal(a: list[dict], b: list[dict], tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(sorted(a, key=_canon), sorted(b, key=_canon), strict=True):
        if set(ra) != set(rb):
            return False
        if any(not _val_equal(ra[k], rb[k], tol) for k in ra):
            return False
    return True


def check(
    query: str,
    tables: dict[str, Table],
    expect: list[dict] | None = None,
) -> None:
    """Run `query` through DataFusion (oracle) AND the Rust InferFn over the same
    typed tables; assert their output rows match (order-insensitive, float-
    tolerant, NULL-aware). If `expect` is given, also assert output == expect."""
    oracle = _run_datafusion(query, tables)
    actual = _run_infer(query, tables)
    assert _rows_equal(actual, oracle), (
        f"Rust InferFn disagrees with DataFusion.\n  query: {query}\n"
        f"  rust:       {actual}\n  datafusion: {oracle}"
    )
    if expect is not None:
        assert _rows_equal(actual, expect), (
            f"Output does not match expected.\n  query: {query}\n"
            f"  actual:   {actual}\n  expected: {expect}"
        )


def check_both_raise(
    query: str,
    tables: dict[str, Table],
    match: str | None = None,
) -> None:
    """Assert BOTH engines reject `query` (at build or execution). If `match` is
    given, each engine's error message must contain that regex."""
    for runner in (_run_datafusion, _run_infer):
        try:
            runner(query, tables)
        except Exception as e:  # noqa: BLE001 -- differential harness, any error counts
            if match is not None and not re.search(match, str(e)):
                raise AssertionError(
                    f"{runner.__name__} raised {e!r}, expected match {match!r}"
                ) from e
            continue
        raise AssertionError(f"{runner.__name__} did not raise for query: {query}")
```

- [ ] **Step 4: Run self-tests to verify they pass**

Run: `uv run pytest tests/test_differential_selftest.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Ruff**

Run: `uv run ruff check tests/differential.py tests/test_differential_selftest.py && uv run ruff format tests/differential.py tests/test_differential_selftest.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tests/differential.py tests/test_differential_selftest.py
git commit -m "test: differential harness (check/row/rows/static) for the Rust engine"
```

---

### Task 2: `tests/test_diff_expressions.py` — expression coverage

**Files:**
- Create: `tests/test_diff_expressions.py`

**Interfaces:**
- Consumes: `check`, `rows`, `row` (Task 1).
- Produces: nothing consumed downstream.

- [ ] **Step 1: Write the tests**

Create `tests/test_diff_expressions.py`:

```python
"""Differential coverage of the Rust engine's scalar expression surface."""

import pytest

from differential import check, rows


@pytest.mark.parametrize(
    "a, b, q",
    [
        (6, 2, 3),
        (7, 2, 3),  # int/int truncates toward zero
        (-7, 2, -3),
    ],
)
def test_int_division_truncates(a, b, q):
    check(
        "SELECT a / b AS q FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": a, "b": b}])},
        expect=[{"q": q}],
    )


@pytest.mark.parametrize(
    "a, b",
    [(6, 2), (7, 2), (-7, 2), (5, 3)],
)
def test_int_mod(a, b):
    check(
        "SELECT a % b AS r FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": a, "b": b}])},
    )


@pytest.mark.parametrize(
    "a, b",
    [(3, 2.0), (7, 4.0)],  # int / float promotes to float
)
def test_mixed_int_float_promotes(a, b):
    check(
        "SELECT a / b AS q FROM t",
        {"t": rows({"a": "int", "b": "float"}, [{"a": a, "b": b}])},
    )


@pytest.mark.parametrize(
    "x, y, expected",
    [(1, 2, True), (2, 1, False), (2, 2, False)],
)
def test_comparison(x, y, expected):
    check(
        "SELECT x < y AS lt FROM t",
        {"t": rows({"x": "int", "y": "int"}, [{"x": x, "y": y}])},
        expect=[{"lt": expected}],
    )


@pytest.mark.parametrize(
    "p, q, expected",
    [
        (True, False, False),
        (True, None, None),  # three-valued: TRUE AND NULL = NULL
        (False, None, False),  # FALSE AND NULL = FALSE
    ],
)
def test_and_three_valued(p, q, expected):
    check(
        "SELECT p AND q AS r FROM t",
        {"t": rows({"p": "bool?", "q": "bool?"}, [{"p": p, "q": q}])},
        expect=[{"r": expected}],
    )


@pytest.mark.parametrize(
    "val, expected",
    [(3.7, 3), (-3.7, -3)],  # float->int truncates toward zero
)
def test_cast_float_to_int(val, expected):
    check(
        "SELECT CAST(x AS BIGINT) AS c FROM t",
        {"t": rows({"x": "float"}, [{"x": val}])},
        expect=[{"c": expected}],
    )


def test_string_builtins():
    check(
        "SELECT UPPER(s) AS u, LOWER(s) AS l, TRIM(s) AS t2 FROM t",
        {"t": rows({"s": "str"}, [{"s": " AbC "}])},
    )


@pytest.mark.parametrize(
    "a, b",
    [(1, 2), (None, 2), (1, None)],  # NULL propagates through arithmetic
)
def test_null_propagation(a, b):
    check(
        "SELECT a + b AS s FROM t",
        {"t": rows({"a": "int?", "b": "int?"}, [{"a": a, "b": b}])},
    )


def test_coalesce_and_nullif():
    check(
        "SELECT COALESCE(a, b) AS c, NULLIF(a, b) AS n FROM t",
        {
            "t": rows(
                {"a": "int?", "b": "int?"},
                [{"a": None, "b": 5}, {"a": 3, "b": 3}, {"a": 7, "b": 2}],
            )
        },
    )


def test_abs_and_round():
    check(
        "SELECT ABS(x) AS a, ROUND(y) AS r FROM t",
        {"t": rows({"x": "int", "y": "float"}, [{"x": -4, "y": 2.6}])},
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_diff_expressions.py -v`
Expected: PASS. (If a specific parametrized case reveals a genuine Rust/DataFusion divergence, do NOT weaken the assertion — stop and report it; it may need an `xfail(strict=True, reason=...)` on that `pytest.param` and a note for the final review.)

- [ ] **Step 3: Ruff + commit**

```bash
uv run ruff check tests/test_diff_expressions.py && uv run ruff format tests/test_diff_expressions.py
git add tests/test_diff_expressions.py
git commit -m "test: differential coverage of the Rust engine's expression surface"
```

---

### Task 3: `tests/test_diff_relational.py` — WHERE, joins, multi-row

**Files:**
- Create: `tests/test_diff_relational.py`

**Interfaces:**
- Consumes: `check`, `rows`, `static` (Task 1).

- [ ] **Step 1: Write the tests**

Create `tests/test_diff_relational.py`:

```python
"""Differential coverage of the Rust engine's relational surface."""

import pytest

from differential import check, rows, static


def test_where_filters_rows():
    check(
        "SELECT age FROM t WHERE age >= 18",
        {"t": rows({"age": "int"}, [{"age": 10}, {"age": 18}, {"age": 40}])},
        expect=[{"age": 18}, {"age": 40}],
    )


def test_cross_join():
    check(
        "SELECT a.x, b.y FROM a, b",
        {
            "a": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 10}]),
            "b": rows({"id": "int", "y": "int"}, [{"id": 1, "y": 20}]),
        },
    )


def test_inner_join_multiple_rows():
    check(
        "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id",
        {
            "a": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 10}, {"id": 2, "x": 20}]),
            "b": rows({"id": "int", "y": "int"}, [{"id": 1, "y": 100}, {"id": 2, "y": 200}]),
        },
    )


def test_row_static_lookup_join():
    check(
        "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id",
        {
            "data": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 5}, {"id": 2, "x": 6}]),
            "ref": static({"id": "int", "y": "int"}, [{"id": 1, "y": 10}, {"id": 2, "y": 20}]),
        },
    )


def test_left_lookup_join_hit_and_miss():
    # id=99 has no match -> LEFT JOIN yields NULL for ref.y on BOTH engines.
    check(
        "SELECT data.x, ref.y FROM data LEFT JOIN ref ON data.id = ref.id",
        {
            "data": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 5}, {"id": 99, "x": 6}]),
            "ref": static({"id": "int", "y": "int"}, [{"id": 1, "y": 10}]),
        },
        expect=[{"x": 5, "y": 10}, {"x": 6, "y": None}],
    )


def test_multi_row_projection_all_rows_present():
    check(
        "SELECT age, age * 2 AS d FROM t",
        {"t": rows({"age": "int"}, [{"age": 1}, {"age": 2}, {"age": 3}])},
    )


def test_composite_key_lookup():
    check(
        "SELECT d.v, r.z FROM d JOIN r ON d.a = r.a AND d.b = r.b",
        {
            "d": rows({"a": "int", "b": "int", "v": "int"}, [{"a": 1, "b": 2, "v": 7}]),
            "r": static({"a": "int", "b": "int", "z": "int"}, [{"a": 1, "b": 2, "z": 9}]),
        },
    )
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_diff_relational.py -v`
Expected: PASS. (Same rule as Task 2 — a real divergence is reported, not asserted away.)

- [ ] **Step 3: Ruff + commit**

```bash
uv run ruff check tests/test_diff_relational.py && uv run ruff format tests/test_diff_relational.py
git add tests/test_diff_relational.py
git commit -m "test: differential coverage of WHERE, joins, and lookup joins"
```

---

### Task 4: `tests/test_diff_errors.py` + trim `test_interpreter.py`

**Files:**
- Create: `tests/test_diff_errors.py`
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `check_both_raise`, `rows` (Task 1).

- [ ] **Step 1: Write the error/divergence tests**

Create `tests/test_diff_errors.py`:

```python
"""Cases where both engines reject a query, and tracked engine divergences."""

import pytest

from differential import check, check_both_raise, rows


def test_unknown_column_rejected_by_both():
    check_both_raise("SELECT nope FROM t", {"t": rows({"age": "int"}, [{"age": 1}])})


def test_self_join_rejected():
    check_both_raise(
        "SELECT a.x FROM a JOIN a ON a.id = a.id",
        {"a": rows({"id": "int", "x": "int"}, [{"id": 1, "x": 1}])},
    )


@pytest.mark.xfail(
    strict=True,
    reason="int div-by-zero: Rust raises InterpError->KeyError, DataFusion raises "
    "Arrow DivideByZero -- error types diverge; see docs/BACKLOG.md",
)
def test_int_div_by_zero_same_error_type():
    # Both raise, but not the SAME error type -> we can't assert a shared match.
    # Documented divergence: if this ever passes (both agree), revisit the xfail.
    check_both_raise(
        "SELECT a / b AS q FROM t",
        {"t": rows({"a": "int", "b": "int"}, [{"a": 1, "b": 0}])},
        match="[Dd]ivide by zero",
    )
```

Note: `test_int_div_by_zero_same_error_type` uses `check_both_raise(..., match="Divide by zero")`, which requires BOTH engines' messages to contain that phrase. DataFusion's does; the Rust engine's does not (it raises a `KeyError`/`InterpError` with a different message), so the `match` assertion fails → the test `xfail`s. This documents the divergence as a strict-xfail that flips loud if the messages ever converge.

- [ ] **Step 2: Trim `tests/test_interpreter.py`**

The value-comparison tests in `test_interpreter.py` are now redundant with the differential suites. Remove the ones whose entire assertion is `_as_dicts(fn.infer(...)) == _expected(...)` (or the inline-oracle equivalents): `test_column_pass_through`, `test_arithmetic_and_where`, `test_builtin_function_and_cast`, `test_cross_join`, `test_inner_join`, `test_aliased_row_table`, `test_join_row_and_static_table`, and the three Task-4 lookup tests (`test_left_lookup_join_hit_returns_value`, `test_left_lookup_join_miss_returns_null`, `test_inner_lookup_join_miss_still_errors` — now covered by `test_diff_relational.py` and `test_diff_errors.py`).

KEEP the tests that are not plain differential value checks: `test_module_imports_and_constructs`, `test_error_unknown_row_column`, `test_error_unknown_static_column`, `test_error_self_join_still_rejected`, and any construction/validation test. Remove the now-unused `_expected`/`_as_dicts` helpers and the `Data`/`A`/`B` model classes only if nothing kept references them (check with grep before deleting).

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass, exactly one `xfail` from the pre-existing `test_transform_raises_clean_valueerror_on_div_by_zero` PLUS the new `test_int_div_by_zero_same_error_type` xfail = **2 xfailed**; no `XPASS`, no errors, no drop in total meaningful coverage.

- [ ] **Step 4: Ruff + commit**

```bash
uv run ruff check . && uv run ruff format .
git add tests/test_diff_errors.py tests/test_interpreter.py
git commit -m "test: both-reject + divergence cases; migrate test_interpreter differential cases"
```

---

## Post-Plan Verification

- [ ] `mise run check` — clean; total xfail count is 2 (the two documented divergences), no `XPASS`.
- [ ] `uv run pytest tests/ -v` shows one line per parametrized case (decision-table rows individually named).
- [ ] Adding a new engine test is a one-line `check(...)` inside a parametrized function — confirm by reading `test_diff_expressions.py`.
- [ ] `tests/differential.py` is NOT collected as a test module (it has no `test_`/`_test` affix); confirm it does not appear in `uv run pytest --collect-only -q`.
- [ ] No `src/` or `sql_transform/` files changed (tests-only): `git diff --stat <base> HEAD` touches only `tests/` and `docs/`.
