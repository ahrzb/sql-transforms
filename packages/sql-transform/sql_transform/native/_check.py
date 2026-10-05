"""Swap-the-entry parity: one query, served by confit twice, once with the
Python step and once with its native twin, compared lane by lane.

The native twin is also checked against its own definition: registered on
DuckDB, the same query must answer exactly what confit serves.
"""

from __future__ import annotations

import math
import struct
from typing import Any

import pyarrow as pa
from confit import DuckDBInferFn, compare
from confit import sql as S
from confit.oracle import Oracle

from sql_transform._udf import PythonTransform


class ParityError(AssertionError):
    """The native twin breached its bound; the message names the row."""


def _ordered(x: float) -> int:
    """The double's position on the line of all doubles: adjacent doubles
    are adjacent integers, and -0.0 and 0.0 share a position."""
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def ulp_distance(a: float, b: float) -> int:
    """Doubles between `a` and `b` (0 when equal); NaN is only near NaN."""
    if math.isnan(a) or math.isnan(b):
        return 0 if math.isnan(a) and math.isnan(b) else 1 << 64
    return abs(_ordered(a) - _ordered(b))


def _same(a: Any, b: Any, ulps: int) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        if ulps == 0:
            return repr(a) == repr(b)  # bit-exact: -0.0 is not 0.0
        return ulp_distance(a, b) <= ulps
    return a == b


def query(step: Any, id_col: str = "__iid") -> str:
    """The query both entries serve: every output lane of one call."""
    args = ", ".join([S.col(id_col).sql(), *(S.col(n).sql() for n in step.takes.names)])
    call = f"{step.name}({args})"
    r = step.returns
    if pa.types.is_struct(r):
        items = [
            f"{call}.{S.col(r.field(i).name).sql()} AS {S.col(r.field(i).name).sql()}"
            for i in range(r.num_fields)
        ]
    else:
        items = [f"{call} AS o"]
    return f"SELECT {', '.join(items)} FROM __THIS__"  # noqa: S608


def _serve(sql: str, rows: pa.Table, fn: Any) -> list[dict] | Exception:
    try:
        f = DuckDBInferFn(sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[fn])
        return f.infer_arrow(rows).to_pylist()
    except Exception as e:  # noqa: BLE001 — a trap is an answer here
        return e


def check(
    step: PythonTransform,
    native: Any,
    rows: pa.Table,
    *,
    ulps: int | None = None,
    id_col: str = "__iid",
) -> int:
    """Raise `ParityError` unless `native` answers every row of `rows` as
    `step` does, within `ulps` doubles per lane (0 = bit-exact; by default
    the catalog's declared bound for the step). `rows`
    holds `id_col` and one column per declared feature.

    Where the step itself raises on a row (sklearn rejecting an input it
    validates), the native answer is not compared: docs/native/goal.md,
    "Where the twin raises". Returns the number of rows compared."""
    if ulps is None:
        from sql_transform.native._registry import bound

        ulps = bound(step)
    sql = query(step, id_col)

    # The native twin against its own definition, exactly.
    got = _serve(sql, rows, native)
    if isinstance(got, Exception):
        raise ParityError(f"the native twin does not serve: {got}")
    with Oracle() as o:
        native.register(o)
        o.load("__THIS__", rows)
        want = o.try_answer(sql)
    if isinstance(want, Exception) or not isinstance(want, pa.Table):
        raise ParityError(f"DuckDB does not run the native definition: {want}")
    compare.assert_rows(got, want.to_pylist(), ctx=sql)

    # The swap: the Python step on the same query.
    twin = _serve(sql, rows, step)
    if isinstance(twin, Exception):
        # Row by row, so a row the step rejects leaves the others compared.
        twin = [_serve(sql, rows.slice(i, 1), step) for i in range(rows.num_rows)]
        twin = [t[0] if isinstance(t, list) else None for t in twin]
    compared = 0
    for i, (a, b) in enumerate(zip(twin, got, strict=True)):
        if a is None:
            continue  # the step raised on this row
        compared += 1
        for k in a:
            if not _same(a[k], b[k], ulps):
                d = (
                    ulp_distance(a[k], b[k])
                    if isinstance(a[k], float) and isinstance(b[k], float)
                    else "n/a"
                )
                raise ParityError(
                    f"row {i} lane {k!r}: step {a[k]!r}, native {b[k]!r}"
                    f" ({d} ulps, bound {ulps}); input {rows.slice(i, 1).to_pylist()[0]}"
                )
    return compared
