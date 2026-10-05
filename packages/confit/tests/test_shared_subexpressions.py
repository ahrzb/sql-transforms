"""A subexpression repeated across a projection is computed once per row.

The shape is a Normalizer's: a struct SQL function whose lane j is
`x_j / g(norm(x))`, every lane carrying the same row norm, read lane by lane.
Sharing the norm must not change an answer (NULL and NaN rows included) nor
which rows trap.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, SqlFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

F64 = pa.float64()


def _norm(kind: str, xs: list) -> S.Expr:
    if kind == "l2":
        acc = xs[0] * xs[0]
        for x in xs[1:]:
            acc = acc + x * x
        return S.fn("sqrt", acc)
    if kind == "l1":
        acc = S.fn("abs", xs[0])
        for x in xs[1:]:
            acc = acc + S.fn("abs", x)
        return acc
    if kind == "max":
        return S.fn("greatest", *[S.fn("abs", x) for x in xs])
    if kind == "tournament":
        # The native catalog's row maximum: a two-way CASE per pair, each
        # operand written twice.
        def rm(v: list) -> S.Expr:
            if len(v) == 1:
                return v[0]
            h = (len(v) + 1) // 2
            a, b = rm(v[:h]), rm(v[h:])
            return S.case(a >= b, a).otherwise(b)

        return rm([S.fn("abs", x) for x in xs])
    raise ValueError(kind)


def normalizer(
    kind: str, n: int, extra: dict | None = None, guarded: bool = False
) -> tuple:
    """A struct function of `n` DOUBLEs normalizing each by the row norm, and
    the query reading every lane by field. `extra` adds fields by name;
    `guarded` makes the struct NULL where `x0` is (the body a CASE, as the
    native catalog's is)."""
    names = [f"x{i}" for i in range(n)]
    schema = pa.schema([(c, F64) for c in names])
    extra = extra or {}
    fields = [(f"y{i}", F64) for i in range(n)] + [
        (k, t) for k, (t, _) in extra.items()
    ]

    def body(*xs):
        norm = _norm(kind, list(xs))
        d = S.case(norm < S.lit(1e-10), S.lit(1.0)).otherwise(norm)
        out = {f"y{i}": x / d for i, x in enumerate(xs)}
        out.update({k: f(list(xs)) for k, (_, f) in extra.items()})
        return out

    null_when = (lambda *xs: xs[0].isnull()) if guarded else None
    fn = SqlFunction(
        f"nrm_{kind}", schema, pa.struct(fields), body, null_when=null_when
    )
    call = f"nrm_{kind}({', '.join(names)})"
    sql = (
        "SELECT "
        + ", ".join(f"{call}.{f} AS {f}" for f, _ in fields)
        + " FROM __THIS__"
    )
    return fn, sql, schema


def _rows(schema: pa.Schema, rows: list[list[float | None]]) -> pa.Table:
    # Each row's values repeat across the width.
    cols = {c: [r[i % len(r)] for r in rows] for i, c in enumerate(schema.names)}
    return pa.table(cols, schema=schema)


ROWS = [
    [0.5, -1.25, 3.0, 2.0],
    [None],
    [0.0],
    [math.nan, 1.0],
    [-0.0, 1e-300],
    [1e300, -1e300, 2.0],
    [math.inf, 1.0],
    [-7.0, None, 4.0],
]


@pytest.mark.parametrize("kind", ["l2", "l1", "max", "tournament"])
@pytest.mark.parametrize("n", [3, 16])
def test_a_normalizer_agrees_with_the_oracle(kind, n):
    fn, sql, schema = normalizer(kind, n)
    assert_parity(sql, _rows(schema, ROWS), udfs=[fn])


@pytest.mark.parametrize("kind", ["l2", "max", "tournament"])
def test_a_guarded_normalizer_agrees_with_the_oracle(kind):
    fn, sql, schema = normalizer(kind, 8, guarded=True)
    assert_parity(sql, _rows(schema, ROWS), udfs=[fn])


def test_a_call_over_a_lateral_alias():
    # The call's argument is an earlier item's alias: its reads bind against
    # that alias, not against the cache of a read made before it existed.
    fn, _, _ = normalizer("l2", 2, guarded=True)
    rows = _rows(pa.schema([("x0", F64), ("x1", F64)]), ROWS)
    sql = (
        "SELECT x1 * 2 AS w, nrm_l2(x0, w).y0 AS a, nrm_l2(x0, w).y1 AS b, "
        "x1 AS w2, nrm_l2(x0, x1).y1 AS c FROM __THIS__"
    )
    assert_parity(sql, rows, udfs=[fn])


# The tournament's text is cubic in the row: past 32 lanes it passes the
# macro token cap.
@pytest.mark.parametrize("kind", ["l2", "l1"])
def test_a_wide_normalizer_agrees_with_the_oracle(kind):
    fn, sql, schema = normalizer(kind, 32)
    assert_parity(sql, _rows(schema, ROWS[:4]), udfs=[fn])


def test_a_shared_norm_beside_a_trapping_field():
    # Field `big` casts x1 to BIGINT, which fails past its range: reading
    # every field, the rows that trap are the ones that trap without the
    # sharing, with DuckDB's message.
    extra = {"big": (pa.int64(), lambda xs: xs[1].cast("BIGINT"))}
    fn, sql, schema = normalizer("l2", 4, extra)
    ok = _rows(schema, [[0.5, -1.25, 3.0, 2.0], [None], [3.0, 4.0]])
    assert_parity(sql, ok, udfs=[fn])
    bad = _rows(schema, [[0.5, -1.25, 3.0, 2.0], [1.0, 1e300, 2.0, 3.0]])
    assert_parity(sql, bad, udfs=[fn], trap="out of range|Overflow|Conversion")
    # Only the trapping field and one lane read: the same trap.
    call = f"nrm_l2({', '.join(schema.names)})"
    lone = f"SELECT {call}.y0 AS y0, {call}.big AS big FROM __THIS__"
    assert_parity(lone, bad, udfs=[fn], trap="out of range|Overflow|Conversion")


def test_a_wide_normalizer_builds():
    # Before sharing, 32 l2 lanes hit the code generator's size limit after
    # 19 s. A generous bound: debug builds run this too.
    fn, sql, schema = normalizer("l2", 128)
    t0 = time.perf_counter()
    f = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={}, udfs=[fn])
    took = time.perf_counter() - t0
    assert f.infer_arrow(_rows(schema, ROWS[:1])).num_columns == 128
    assert took < 60, took
