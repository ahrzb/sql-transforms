"""Row-local derived tables, on both backends, against DuckDB.

The DuckDB behaviours the subquery design must reproduce
(docs/specs/2026-09-26-row-local-subqueries-design.md, "DuckDB semantics to
reproduce"): every subquery column is evaluated read or not, the inner SELECT
finishes before the outer WHERE, the inner WHERE guards the inner SELECT,
constants do not fold across a level, the inner scope is closed, and the
boundary names columns as DuckDB does.
"""

import os

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare
from confit.oracle import Oracle

SCHEMA = pa.schema([pa.field("a", pa.int64()), pa.field("s", pa.string())])
MAX = 2**63 - 1


def _table(rows):
    return pa.Table.from_pylist([{"a": a, "s": s} for a, s in rows], schema=SCHEMA)


def _build(sql, force_interp, **kw):
    prev = os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
    try:
        if force_interp:
            os.environ["SPECIALIZER_FORCE_INTERP"] = "1"
        return DuckDBInferFn(
            sql,
            row_tables={"__THIS__": SCHEMA},
            static_tables=kw.pop("statics", {}),
            **kw,
        )
    finally:
        os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
        if prev is not None:
            os.environ["SPECIALIZER_FORCE_INTERP"] = prev


def _oracle(sql, rows, statics=None):
    o = Oracle()
    for name, t in (statics or {}).items():
        o.load(name, t)
    o.load("__THIS__", _table(rows))
    return o.answer(sql)


SERVES = {
    "unread column, in range": (
        "SELECT a FROM (SELECT a, a + 1 AS b FROM __THIS__)",
        [(5, "x"), (None, "y")],
    ),
    "inner WHERE guards the inner SELECT": (
        "SELECT CAST(s AS INTEGER) AS i FROM (SELECT s FROM __THIS__ WHERE s <> 'x')",
        [(1, "x"), (2, "5")],
    ),
    "constants do not fold, zero rows": (
        "SELECT k + 9223372036854775807 AS o FROM (SELECT 1 AS k FROM __THIS__)",
        [],
    ),
    "bare NULL column projected directly": (
        "SELECT x FROM (SELECT NULL AS x FROM __THIS__)",
        [(1, "x")],
    ),
    "unaliased subquery": (
        "SELECT unnamed_subquery.x FROM (SELECT a AS x FROM __THIS__)",
        [(1, "x")],
    ),
    "partial column-alias list": (
        "SELECT * FROM (SELECT a, s FROM __THIS__) AS t(p)",
        [(1, "x")],
    ),
    "duplicate names": ("SELECT * FROM (SELECT a, a FROM __THIS__)", [(1, "x")]),
    "qualified through the alias": (
        "SELECT t.a, t.s FROM (SELECT a, s FROM __THIS__) t WHERE t.a > 0",
        [(1, "x"), (-1, "y")],
    ),
    "nesting": (
        "SELECT z FROM (SELECT y + 1 AS z FROM (SELECT a * 2 AS y FROM __THIS__))",
        [(3, "x"), (None, "y")],
    ),
    "outer WHERE over a slot": (
        "SELECT b FROM (SELECT a AS b FROM __THIS__) WHERE b > 1",
        [(1, "x"), (3, "y"), (None, "z")],
    ),
    "star with EXCLUDE": (
        "SELECT * EXCLUDE (s) FROM (SELECT a, s, a + 1 AS c FROM __THIS__)",
        [(1, "x")],
    ),
    "a slot read twice": (
        "SELECT b, b * b AS c FROM (SELECT a + 1 AS b FROM __THIS__)",
        [(4, "x")],
    ),
    "string functions across the level": (
        "SELECT upper(t) AS u, length(t) AS n "
        "FROM (SELECT s || '!' AS t FROM __THIS__)",
        [(1, "ab"), (2, None)],
    ),
    "CASE over slots": (
        "SELECT CASE WHEN b > 2 THEN b ELSE -b END AS c "
        "FROM (SELECT a AS b FROM __THIS__)",
        [(1, "x"), (5, "y"), (None, "z")],
    ),
}

TRAPS = {
    "an unread column still traps": (
        "SELECT a FROM (SELECT a, a + 9223372036854775807 AS b FROM __THIS__)",
        [(MAX, "x"), (1, "y")],
        "Overflow",
    ),
    "the inner SELECT finishes before the outer WHERE": (
        "SELECT i FROM (SELECT CAST(s AS INTEGER) AS i, s FROM __THIS__) "
        "WHERE s <> 'x'",
        [(1, "x"), (2, "5")],
        "onver",
    ),
    "constants do not fold across a level": (
        "SELECT k + 9223372036854775807 AS o FROM (SELECT 1 AS k FROM __THIS__)",
        [(1, "x")],
        "Overflow",
    ),
}


@pytest.mark.parametrize("sql, rows", SERVES.values(), ids=SERVES.keys())
@pytest.mark.parametrize("force_interp", [False, True])
def test_serves_as_duckdb(sql, rows, force_interp):
    want = _oracle(sql, rows)
    got = _build(sql, force_interp).infer_arrow(_table(rows))
    assert got.schema.names == want.schema.names
    assert [f.type for f in got.schema] == [f.type for f in want.schema]
    compare.assert_rows(got.to_pylist(), compare.rows(want), ctx=sql)


@pytest.mark.parametrize("sql, rows, match", TRAPS.values(), ids=TRAPS.keys())
@pytest.mark.parametrize("force_interp", [False, True])
def test_traps_as_duckdb(sql, rows, match, force_interp):
    with pytest.raises(Exception, match=match):
        _oracle(sql, rows)
    fn = _build(sql, force_interp)
    with pytest.raises(Exception, match=match):
        fn.infer_arrow(_table(rows))


@pytest.mark.parametrize(
    "sql, oracle_error",
    [
        ("SELECT s FROM (SELECT a FROM __THIS__)", "not found"),
        ("SELECT __THIS__.a FROM (SELECT a FROM __THIS__)", "not found"),
        ("SELECT * FROM (SELECT a FROM __THIS__) AS t(p, q)", "columns specified"),
    ],
)
def test_the_inner_scope_is_closed_and_binder_errors_are_bind_errors(sql, oracle_error):
    with pytest.raises(Exception, match=oracle_error):
        _oracle(sql, [(1, "x")])
    with pytest.raises(ValueError, match="^bind error: "):
        _build(sql, False)


@pytest.mark.parametrize(
    "sql, named",
    [
        (
            "SELECT x || 'a' AS z FROM (SELECT NULL AS x FROM __THIS__)",
            "bare NULL subquery column",
        ),
        (
            "SELECT o FROM (SELECT a AS o FROM __THIS__) AS s, __THIS__",
            "beside a derived table",
        ),
        (
            "SELECT o FROM (SELECT a AS o FROM __THIS__) s JOIN d ON s.o = d.id",
            "beside a derived table",
        ),
        ("SELECT * FROM (SELECT a, a + 1 FROM __THIS__ ORDER BY a)", "ORDER BY"),
        ("SELECT * FROM (SELECT DISTINCT a FROM __THIS__)", "DISTINCT"),
    ],
)
def test_what_stays_refused_is_refused_by_name(sql, named):
    d = pa.table({"id": [1], "v": ["x"]})
    with pytest.raises(ValueError, match=named):
        _build(sql, False, statics={"d": d})


def test_joins_inside_the_subquery_serve():
    d = pa.table({"id": pa.array([1, 2], pa.int64()), "v": ["one", "two"]})
    sql = (
        "SELECT v || '!' AS w, a FROM "
        "(SELECT a, v FROM __THIS__ LEFT JOIN d ON a = d.id) AS t WHERE a > 0"
    )
    rows = [(1, "x"), (3, "y"), (-1, "z")]
    want = _oracle(sql, rows, {"d": d})
    for force in (False, True):
        got = _build(sql, force, statics={"d": d}).infer_arrow(_table(rows))
        compare.assert_rows(got.to_pylist(), compare.rows(want), ctx=sql)


def test_the_one_row_proof_walks_every_stage():
    # shape='map' serves a derived table that keeps every row, and refuses
    # one whose inner or outer WHERE can drop a row.
    _build("SELECT b FROM (SELECT a + 1 AS b FROM __THIS__)", False, shape="map")
    for sql in (
        "SELECT b FROM (SELECT a AS b FROM __THIS__ WHERE a > 0)",
        "SELECT b FROM (SELECT a AS b FROM __THIS__) WHERE b > 0",
    ):
        with pytest.raises(ValueError, match="shape='map'.*WHERE"):
            _build(sql, False, shape="map")
