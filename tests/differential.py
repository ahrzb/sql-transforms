"""Differential test harness for the serving engines.

`check(query, tables)` runs a query through DataFusion (the oracle) AND the
serving engine selected by the active backend, and asserts their output values
match. The backend is set per-test by the `_backend` fixture in conftest.py, so
every case here runs once per engine:

  * "native"  — the native InferFn interpreter (sql_transform._interpreter)
  * "codegen" — the codegen engine (sql_transform._codegen)

Holding both to the same oracle is what makes them provably equivalent. Cases
touching surface a backend explicitly defers raise UnsupportedInCodegen and are
skipped loudly rather than passing silently.

Tests are native pytest parametrized decision tables (see test_diff_*.py). This
module is NOT collected by pytest (no test_ prefix / _test suffix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import datafusion
import pyarrow as pa
import pytest

from sql_transform._codegen import CodegenFn, UnsupportedInCodegen
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


def _split_struct_fields(inner: str) -> list[str]:
    # Split "x:int,y:list[int]" on top-level commas only (not commas nested
    # inside a struct{...}/list[...] sub-spec).
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])
    return parts


def _arrow_type(spec: str) -> tuple[pa.DataType, bool]:
    """Parse a type-spec string into (arrow type, nullable)."""
    nullable = spec.endswith("?")
    base = spec[:-1] if nullable else spec
    if base.startswith("struct{") and base.endswith("}"):
        inner = base[len("struct{") : -1]
        fields = [_arrow_field(*part.split(":", 1)) for part in _split_struct_fields(inner)]
        return pa.struct(fields), nullable
    if base.startswith("list[") and base.endswith("]"):
        elem_type, elem_nullable = _arrow_type(base[len("list[") : -1])
        return pa.list_(pa.field("item", elem_type, nullable=elem_nullable)), nullable
    if base not in _ARROW:
        raise ValueError(f"Unknown column type {spec!r}")
    return _ARROW[base], nullable


def _arrow_field(name: str, spec: Any) -> pa.Field:
    if spec in _BUILTIN:  # python builtin type value
        spec = _BUILTIN[spec]
    if not isinstance(spec, str):
        raise ValueError(f"Unsupported column type {spec!r} for column {name!r}")
    arrow_type, nullable = _arrow_type(spec)
    return pa.field(name, arrow_type, nullable=nullable)


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
    batches = df.collect()
    if not batches:
        # No rows -> no data to be wrong about; df.schema() is fine here, and
        # pa.Table.from_batches requires a schema when there are zero batches.
        return pa.Table.from_batches(batches, schema=df.schema()).to_pylist()
    # Measured: for CASE with mixed-numeric branches (e.g. THEN 1 ELSE 2.5),
    # DataFusion's logical df.schema() reports the first branch's type (int64)
    # while the actual result batches are correctly coerced to the common
    # supertype (double) -- a mismatch between DataFusion's own logical and
    # physical schemas. Building the table from the batches' own (correct)
    # schema avoids an ArrowInvalid over a discrepancy that isn't ours.
    return pa.Table.from_batches(batches).to_pylist()


def _run_engine(engine: Any, query: str, tables: dict[str, Table]) -> list[dict]:
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
    fn = engine(query, row_tables=row_models, static_tables=static_tables)
    return [r.model_dump() for r in fn.infer(infer_rows)]


def _run_infer(query: str, tables: dict[str, Table]) -> list[dict]:
    return _run_engine(InferFn, query, tables)


def _run_codegen(query: str, tables: dict[str, Table]) -> list[dict]:
    return _run_engine(CodegenFn, query, tables)


BACKENDS = {"native": _run_infer, "codegen": _run_codegen}
_backend = "native"


def set_backend(name: str) -> None:
    """Select which serving engine `check` exercises against the oracle.
    Driven by the `_backend` fixture in conftest.py."""
    global _backend
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}; expected one of {sorted(BACKENDS)}"
        )
    _backend = name


def _run_backend(query: str, tables: dict[str, Table]) -> list[dict]:
    return BACKENDS[_backend](query, tables)


def _canon(r: dict) -> tuple:
    # Sortable canonical key for order-insensitive comparison; both sides use
    # the same key so the sort+zip pairing is consistent regardless of order.
    return tuple(sorted((k, v is None, str(v)) for k, v in r.items()))


def _val_equal(a: Any, b: Any, tol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        return set(a) == set(b) and all(_val_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)):
            return False
        return len(a) == len(b) and all(
            _val_equal(x, y, tol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, bool) != isinstance(b, bool):
        return False  # bool is an int subclass (True == 1); keep them distinct
    if isinstance(a, float) != isinstance(b, float):
        return False  # int vs float is a real type divergence for the oracle
    if isinstance(a, float):
        if a == b:
            return True  # handles equal +-inf, where abs(a - b) is nan
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
    codegen_only: bool = False,
) -> None:
    """Run `query` through DataFusion (oracle) AND the active backend engine over
    the same typed tables; assert their output rows match (order-insensitive,
    float-tolerant, NULL-aware). If `expect` is given, also assert
    output == expect. `codegen_only=True` skips the native backend for surface
    codegen supports but native does not yet (e.g. CASE, until TASK-27)."""
    if codegen_only and _backend == "native":
        pytest.skip("native does not implement this surface yet (codegen_only)")
    oracle = _run_datafusion(query, tables)
    try:
        actual = _run_backend(query, tables)
    except UnsupportedInCodegen as e:
        pytest.skip(f"{_backend} defers this surface: {e}")
    assert _rows_equal(actual, oracle), (
        f"{_backend} engine disagrees with DataFusion.\n  query: {query}\n"
        f"  {_backend}: {actual}\n  datafusion: {oracle}"
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
    """Assert BOTH DataFusion and the active backend reject `query` (at build or
    execution). If `match` is given, each engine's error message must contain
    that regex."""
    for runner in (_run_datafusion, _run_backend):
        try:
            runner(query, tables)
        except UnsupportedInCodegen as e:
            # A deferred surface is not a rejection -- don't let it pass as one.
            pytest.skip(f"{_backend} defers this surface: {e}")
        except Exception as e:  # noqa: BLE001 -- differential harness, any error counts
            if match is not None and not re.search(match, str(e)):
                raise AssertionError(
                    f"{runner.__name__} raised {e!r}, expected match {match!r}"
                ) from e
            continue
        raise AssertionError(f"{runner.__name__} did not raise for query: {query}")
