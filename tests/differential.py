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
