"""A narrow arm widened by unification keeps its own overflow width.

DuckDB unifies CASE, COALESCE, GREATEST and LEAST arms by casting each arm's
FINISHED result to the common type; it never re-types the arm's operands. So
`c0 * c0` over TINYINT overflows at 127 even when a BIGINT arm widens the
whole expression (fuzz seed 5008 and 12 more). Pinned on both backends.
"""

import os

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare
from confit.oracle import Oracle

SCHEMA = pa.schema(
    [
        pa.field("c0", pa.int8()),
        pa.field("c1", pa.int64()),
        pa.field("c2", pa.int16()),
    ]
)


def _table(rows):
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def _confit(sql, rows, force_interp):
    prev = os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
    try:
        if force_interp:
            os.environ["SPECIALIZER_FORCE_INTERP"] = "1"
        fn = DuckDBInferFn(sql, row_tables={"__THIS__": SCHEMA}, static_tables={})
    finally:
        os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
        if prev is not None:
            os.environ["SPECIALIZER_FORCE_INTERP"] = prev
    return fn.infer_arrow(_table(rows))


def _oracle(sql, rows):
    o = Oracle()
    o.load("__THIS__", _table(rows))
    return o.answer(sql)


ARMS = [
    "CASE WHEN FALSE THEN c1 ELSE c0 * c0 END",
    "CASE WHEN c1 > 100 THEN c1 ELSE c0 * c0 END",
    "coalesce(c0 * c0, c1)",
    "greatest(c0 * c0, c1)",
    "least(c1, c0 + c0)",
    "CASE WHEN c1 > 100 THEN c1 ELSE c2 * c2 END",
    "coalesce(-c0, c1)",
]


@pytest.mark.parametrize("expr", ARMS)
@pytest.mark.parametrize("force_interp", [False, True])
def test_a_narrow_arm_overflows_at_its_own_width(expr, force_interp):
    sql = f"SELECT {expr} AS o FROM __THIS__"
    rows = [{"c0": -128, "c1": 5, "c2": -32768}]
    with pytest.raises(Exception, match="Overflow"):
        _oracle(sql, rows)
    with pytest.raises(Exception, match="Overflow"):
        _confit(sql, rows, force_interp)


@pytest.mark.parametrize("expr", ARMS)
@pytest.mark.parametrize("force_interp", [False, True])
def test_in_range_rows_answer_at_the_unified_width(expr, force_interp):
    sql = f"SELECT {expr} AS o FROM __THIS__"
    rows = [{"c0": 3, "c1": 5, "c2": 7}, {"c0": None, "c1": 9, "c2": None}]
    want = _oracle(sql, rows)
    got = _confit(sql, rows, force_interp)
    assert got.schema.field("o").type == want.schema.field("o").type
    compare.assert_rows(got.to_pylist(), compare.rows(want), ctx=sql)
