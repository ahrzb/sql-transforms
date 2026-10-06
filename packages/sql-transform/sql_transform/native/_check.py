"""Swap-the-entry parity: one query, served by confit twice, once with the
Python step and once with its native twin, compared lane by lane.

The native twin is also checked against its own definition: registered on
DuckDB, the same query (its call made once per row) must answer exactly what
confit serves.
"""

from __future__ import annotations

import math
import struct
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn, compare
from confit import sql as S
from confit.oracle import Oracle

from sql_transform._udf import PythonTransform
from sql_transform.native._registry import ErrorScale, catalog, query

EPS = np.longdouble(2.0**-52)
_DBL_MAX = np.longdouble(sys.float_info.max)


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
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _same(x, y, ulps) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float) and isinstance(b, float):
        if ulps == 0:
            return repr(a) == repr(b)  # bit-exact: -0.0 is not 0.0
        return ulp_distance(a, b) <= ulps
    return a == b


def near(a: float, b: float, bound: Any, g: Callable[[Any], Any] | None = None) -> bool:
    """`a` and `b` within `bound` of each other after the map `g` (an error
    scale's comparison, matvec-parity-bound.md, Recommendation 6): NaN is
    only near NaN; where one side is infinite, the other passes if it is
    finite, has the same sign and lies within `bound` of DBL_MAX, since
    overflow on one side can be rounding alone; else `|g(a) - g(b)| <=
    bound`. In long doubles, so that neither the map nor the difference
    overflows (on x86-64)."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    ga, gb = np.longdouble(a), np.longdouble(b)
    if g is not None:
        ga, gb = g(ga), g(gb)
    if ga == gb:
        return True
    if np.isinf(ga) and np.isinf(gb):
        return False
    if np.isinf(ga) or np.isinf(gb):
        big, fin = (ga, gb) if np.isinf(ga) else (gb, ga)
        return bool(np.sign(fin) == np.sign(big) and _DBL_MAX - abs(fin) <= bound)
    return bool(abs(ga - gb) <= bound)


def _features(rows: pa.Table, i: int, names: list[str]) -> np.ndarray:
    """Row `i`'s features as the twin's `transform` sees them, for an
    error scale: NaN for NULL, a boolean as 0 or 1, as long doubles."""
    vals = [rows.column(n)[i].as_py() for n in names]
    return np.array(
        [math.nan if v is None else float(v) for v in vals], dtype=np.longdouble
    )


def _bounds(scale: ErrorScale, est: Any, x: np.ndarray, width: int) -> np.ndarray:
    """Each output field's bound at one row, K*eps*S + tau."""
    k, tau = scale.k(est), scale.tau(est)
    s = scale.s(est, x)
    b = np.asarray(k, np.longdouble) * EPS * np.asarray(s, np.longdouble)
    return np.broadcast_to(b + np.asarray(tau, np.longdouble), (width,))


def _lanes(rec: dict) -> list[tuple[str, Any]]:
    """One answered row's lanes, each `(name, value)`: a struct's fields,
    a list's elements, or the one value."""
    if len(rec) == 1:
        ((key, v),) = rec.items()
        if isinstance(v, list):
            return [(f"{key}[{j}]", x) for j, x in enumerate(v)]
    return list(rec.items())


def _once(step: PythonTransform, id_col: str) -> str:
    """`query(step)` with the call made once per row, its struct then read
    by field: the same answers, in the time DuckDB takes for one call.
    DuckDB expands a macro at every field read, so it makes a wide struct's
    call once per lane (285 lanes: 18 s, against 0.1 s once)."""
    r = step.returns
    if not pa.types.is_struct(r):
        return query(step, id_col)
    args = ", ".join([S.col(id_col).sql(), *(S.col(n).sql() for n in step.takes.names)])
    s = S.col("__s")
    reads = ", ".join(
        f"{S.fn('struct_extract', s, S.lit(f.name)).sql()} AS {S.col(f.name).sql()}"
        for f in r
    )
    call = f"{step.name}({args}) AS {s.sql()}"
    return f"SELECT {reads} FROM (SELECT {call} FROM __THIS__)"  # noqa: S608


def _serve(sql: str, rows: pa.Table, fn: Any) -> list[dict] | Exception:
    try:
        f = DuckDBInferFn(
            sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[fn]
        )
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
    `step` does, lane by lane, within the parity bound of the row's
    instance: its ulp bound (0 = bit-exact) or its error scale
    (`ErrorScale`). With `ulps`, every row is held to that many doubles
    per lane instead. `rows` holds `id_col` and one column per declared
    feature.

    Where the step itself raises on a row (sklearn rejecting an input it
    validates), the native answer is not compared: loops/native/goal.md,
    "Where the twin raises". Where it raises on every row, nothing is
    compared and `check` raises: such rows prove nothing. Returns the number
    of rows compared, at least 1."""
    sql = query(step, id_col)

    # The native twin against its own definition, exactly.
    got = _serve(sql, rows, native)
    if isinstance(got, Exception):
        raise ParityError(f"the native twin does not serve: {got}")
    with Oracle() as o:
        native.register(o)
        o.load("__THIS__", rows)
        want = o.try_answer(_once(step, id_col))
    if isinstance(want, Exception) or not isinstance(want, pa.Table):
        raise ParityError(f"DuckDB does not run the native definition: {want}")
    compare.assert_rows(got, want.to_pylist(), ctx=sql)

    # The swap: the Python step on the same query.
    twin = _serve(sql, rows, step)
    if isinstance(twin, Exception):
        # Row by row, so a row the step rejects leaves the others compared.
        twin = [_serve(sql, rows.slice(i, 1), step) for i in range(rows.num_rows)]
        twin = [t[0] if isinstance(t, list) else None for t in twin]
    entries = catalog()
    ids = rows.column(id_col).to_pylist()
    compared = 0
    for i, (a, b) in enumerate(zip(twin, got, strict=True)):
        if a is None:
            continue  # the step raised on this row
        compared += 1
        # The row's instance, and its bound. A NULL id answers NULL on both
        # sides, compared exactly.
        est = None if ids[i] is None else step.instances.get(ids[i])
        entry = None if est is None else entries.get(type(est))
        scale = None if ulps is not None or entry is None else entry.scale
        if ulps is not None:
            u = ulps
        elif entry is None or scale is not None:
            u = 0  # what is not a double (NULL, say) compares exactly
        else:
            u = entry.bound(est)
        la, lb = _lanes(a), _lanes(b)
        if len(la) != len(lb):
            raise ParityError(
                f"row {i}: step {a!r}, native {b!r}; input {_input(rows, i)}"
            )
        bounds = None
        for j, ((name, x), (_, y)) in enumerate(zip(la, lb, strict=True)):
            if scale is None or not (isinstance(x, float) and isinstance(y, float)):
                if _same(x, y, u):
                    continue
                d = (
                    ulp_distance(x, y)
                    if isinstance(x, float) and isinstance(y, float)
                    else "n/a"
                )
                raise ParityError(
                    f"row {i} lane {name!r}: step {x!r}, native {y!r}"
                    f" ({d} ulps, bound {u}); input {_input(rows, i)}"
                )
            if repr(x) == repr(y):
                continue
            if bounds is None:
                x_in = _features(rows, i, list(step.takes.names))
                bounds = _bounds(scale, est, x_in, len(la))
            if not near(x, y, bounds[j], scale.g):
                raise ParityError(
                    f"row {i} lane {name!r}: step {x!r}, native {y!r}, past"
                    f" its bound K*eps*S + tau = {float(bounds[j]):.6g}"
                    f" ({_apart(x, y, scale.g)} apart); input {_input(rows, i)}"
                )
    if not compared:
        raise ParityError(
            f"check compares no row: the step raises on each of {rows.num_rows} rows"
        )
    return compared


def _input(rows: pa.Table, i: int) -> dict:
    return rows.slice(i, 1).to_pylist()[0]


def _apart(a: float, b: float, g: Callable[[Any], Any] | None) -> str:
    """How far apart `a` and `b` are, after `g`, as a message reads it."""
    ga, gb = np.longdouble(a), np.longdouble(b)
    if g is not None:
        ga, gb = g(ga), g(gb)
    return f"{float(abs(ga - gb)):.6g}"
