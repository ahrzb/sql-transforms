"""The correlated scalar subquery with one aggregate over one static table,
`(SELECT sum(s.v) FROM s WHERE s.k = t.k)` (src/specializer/frontend/subquery.rs).

Confit binds it as a hidden LEFT join to a table that it computes at build:
one row per group of static rows with equal keys, holding the group's
aggregate. A request that matches no group reads the aggregate of no rows: 0
for count, NULL for the others. The oracle evaluates the subquery's WHERE in
written order over (distinct outer values) x (static rows), and the
aggregate's argument over the matched rows only (the pins at the end). Nothing
served here can trap, so the values are what compares.

Every test but the pins asserts the campaign's own verdict (`fuzz.parity`).
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle, Trap

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

NAN = float("nan")
U64 = 2**64 - 1

T = table(
    {
        "id": "int64",
        "k": "int64?",
        "j": "int32",
        "name": "string?",
        "x": "double?",
        "u": "uint64?",
    },
    [
        {"id": 1, "k": 1, "j": 1, "name": "a", "x": 0.0, "u": 1},
        {"id": 2, "k": 1, "j": 2, "name": "b", "x": -0.0, "u": U64},
        {"id": 3, "k": 2, "j": 1, "name": None, "x": NAN, "u": 3},
        {"id": 4, "k": 3, "j": 2, "name": "z", "x": 1.5, "u": None},
        {"id": 5, "k": None, "j": 1, "name": "a", "x": None, "u": 0},
        {"id": 6, "k": 1, "j": 1, "name": "", "x": 2.0, "u": 1},
    ],
)
S = table(
    {
        "k": "int64?",
        "j": "int32",
        "v": "int64?",
        "d": "decimal(9,4)?",
        "f": "double?",
        "s": "string?",
        "b": "bool?",
        "u": "uint64",
        "i8": "int8?",
    },
    [
        {
            "k": 1,
            "j": 1,
            "v": 10,
            "d": Decimal("1.5"),
            "f": 1.5,
            "s": "b",
            "b": True,
            "u": U64,
            "i8": -128,
        },
        {
            "k": 1,
            "j": 2,
            "v": 20,
            "d": Decimal("2.25"),
            "f": -0.0,
            "s": "a",
            "b": False,
            "u": U64,
            "i8": 127,
        },
        {
            "k": 2,
            "j": 1,
            "v": 5,
            "d": None,
            "f": NAN,
            "s": None,
            "b": None,
            "u": 3,
            "i8": 1,
        },
        {
            "k": None,
            "j": 1,
            "v": 7,
            "d": Decimal("-3"),
            "f": None,
            "s": "z",
            "b": True,
            "u": 1,
            "i8": None,
        },
        {
            "k": 1,
            "j": 1,
            "v": None,
            "d": Decimal("0.1"),
            "f": 0.5,
            "s": "c",
            "b": True,
            "u": 0,
            "i8": 5,
        },
        {
            "k": 3,
            "j": 2,
            "v": -3,
            "d": Decimal("1"),
            "f": 2.0,
            "s": "",
            "b": False,
            "u": 1,
            "i8": -1,
        },
    ],
)


def _sub(agg: str, where: str = "s.k = t.k") -> str:
    return f"SELECT id, (SELECT {agg} FROM s WHERE {where}) AS c FROM __THIS__ AS t"


AGGREGATES = [
    "count(*)",
    "count(v)",
    "count(s)",
    "count(1)",
    "count('x')",
    "sum(v)",
    "sum(b)",
    "sum(d)",
    "sum(u)",
    "sum(i8)",
    "min(v)",
    "max(s)",
    "min(b)",
    "max(d)",
    "min(f)",
    "max(f)",
    "min(u)",
    "max(i8)",
]

KEYS = [
    "s.j = t.k",  # INTEGER column, BIGINT outer side
    "s.k = t.j",
    "s.d = t.k",  # DECIMAL against BIGINT
    "s.f = t.k",  # DOUBLE against BIGINT
    "s.f = t.x",  # -0.0 matches 0.0, NaN matches NaN
    "s.s = t.name",
    "s.u = t.u",
    "s.u = t.k",
    "s.i8 = t.k",
    "s.k = t.id",
    "s.k = 1",  # a constant outer side
    "t.k = s.k",  # the outer side first
    "s.k = t.k AND s.j = t.j",
    "s.k = t.k AND s.k = t.j",
    "s.k = t.k AND s.j = t.j AND s.u = t.u",
]

FORMS = [
    # Unqualified, aliased and schema-qualified names of the inner table.
    "SELECT id, (SELECT count(*) FROM s WHERE k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s AS x WHERE x.k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM main.s WHERE main.s.k = t.k) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.k = __THIS__.k) AS c FROM __THIS__",
    # Uncorrelated: one group, empty or not.
    "SELECT id, (SELECT count(*) FROM s) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT sum(v) FROM s) AS c FROM __THIS__ AS t",
    # Inside expressions, twice, and in WHERE.
    "SELECT id, coalesce((SELECT sum(v) FROM s WHERE s.k = t.k), -1) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT min(s) FROM s WHERE s.k = t.k) || 'x' AS c FROM __THIS__ AS t",
    "SELECT id, CASE WHEN t.k > 1 THEN (SELECT sum(v) FROM s WHERE s.k = t.k) END AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) "
    "+ (SELECT sum(v) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) "
    "= (SELECT count(*) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id FROM __THIS__ AS t WHERE (SELECT count(*) FROM s WHERE s.k = t.k) > 1",
    # A lateral alias as the outer side, and WHERE reading a subquery's alias.
    "SELECT id, k AS kk, (SELECT count(*) FROM s WHERE s.k = kk) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
    "FROM __THIS__ AS t WHERE c > 1",
    # DuckDB's names of unaliased items.
    "SELECT (SELECT count(*) FROM s WHERE s.k = t.k), "
    "(SELECT max(x.v) FROM s AS x WHERE x.k = t.k AND x.j = t.j), "
    "(SELECT count(*) AS n FROM s) FROM __THIS__ AS t",
    # Beside a FROM join (under the default shape its duplicate keys refuse).
    "SELECT t.id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
    "FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("agg", AGGREGATES)
def test_aggregate_over_the_matched_rows(agg, shape):
    assert_parity(_sub(agg), T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", KEYS)
def test_keys(where, shape):
    assert_parity(_sub("count(*)", where), T, statics={"s": S}, shape=shape)
    assert_parity(_sub("sum(v)", where), T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", FORMS)
def test_forms(sql, shape):
    if shape is None and "JOIN s AS s2" in sql:
        assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize(
    "sql",
    [
        _sub("count(*)", "s.k IS NOT DISTINCT FROM t.k"),
        _sub("count(*)", "s.f IS NOT DISTINCT FROM t.x"),
        _sub("sum(v)", "s.k = t.k AND s.j IS NOT DISTINCT FROM t.j"),
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
        "FROM (SELECT * FROM __THIS__) AS t",
        "WITH w AS (SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
        "FROM __THIS__ AS t) SELECT * FROM w",
    ],
)
def test_forms_of_the_default_shape(sql):
    assert_parity(sql, T, statics={"s": S})


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("agg", ["count(*)", "sum(v)", "min(s)", "max(d)"])
def test_empty_static_table_and_empty_batch(agg, shape):
    empty = S.slice(0, 0)
    assert_parity(_sub(agg), T, statics={"s": empty}, shape=shape)
    assert_parity(
        _sub(agg).replace(" WHERE s.k = t.k", ""), T, statics={"s": empty}, shape=shape
    )
    assert_parity(_sub(agg), T.slice(0, 0), statics={"s": S}, shape=shape)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c, c + 1 AS d "
        "FROM __THIS__ AS t",
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) + 0 AS c, c + 1 AS d "
        "FROM __THIS__ AS t",
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c, "
        "(SELECT count(*) FROM s WHERE s.k = c) AS e FROM __THIS__ AS t",
    ],
)
def test_select_list_alias_of_a_subquery_is_duckdbs_binder_error(sql):
    v = assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
    assert v.oracle == "rejects"
    assert (
        'Alias "c" referenced in a SELECT clause - but the expression has a subquery'
        in v.detail
    )


@pytest.mark.parametrize(
    "sql, why",
    [
        (_sub("avg(v)"), "the aggregate avg"),
        (_sub("sum(f)"), "sum over a DOUBLE column"),
        (_sub("count(DISTINCT v)"), "DISTINCT"),
        (_sub("max(v) + 1"), "one aggregate call"),
        (_sub("v"), "one aggregate call"),
        (_sub("count(NULL)"), "must be a column of its table"),
        (_sub("count(*)", "s.k = t.k AND s.v > 5"), "the WHERE term s.v > 5"),
        (_sub("count(*)", "s.k = k"), "the WHERE term s.k = k"),
        (_sub("sum(v)", "s.k = t.k + 1"), "can trap"),
        (_sub("count(*)", "s.s = t.k"), "compares VARCHAR with another type"),
        (
            _sub("count(*)", "s.k = (SELECT max(k) FROM s)"),
            "outer side of a scalar subquery",
        ),
        (
            "SELECT id, (SELECT count(*) FROM __THIS__ AS r WHERE r.k = t.k) AS c "
            "FROM __THIS__ AS t",
            "over the request table",
        ),
        (
            "SELECT t.id FROM __THIS__ AS t "
            "JOIN s AS s2 ON s2.k = (SELECT max(k) FROM s)",
            "in FROM or in a JOIN condition",
        ),
        (
            "SELECT t.id, (SELECT count(*) FROM s WHERE s.k = s2.j) AS c "
            "FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
            "reads a joined table",
        ),
        (
            "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k GROUP BY s.k) AS c "
            "FROM __THIS__ AS t",
            "GROUP BY or HAVING",
        ),
    ],
)
def test_refused_by_name(sql, why):
    v = assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
    assert why in v.detail, v.detail


def test_shape_many_names_the_join_of_a_subquery_in_a_query_of_two_levels():
    sql = (
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
        "FROM (SELECT * FROM __THIS__) AS t"
    )
    v = assert_parity(sql, T, statics={"s": S}, shape="many", expect="REFUSED")
    assert "confit runs a scalar subquery over a static table as a join" in v.detail


@pytest.mark.parametrize("agg", ["min(f)", "max(f)"])
def test_double_extreme_that_ties_other_bits_refuses_the_build(agg):
    # -0.0 and 0.0 compare equal: which one DuckDB answers depends on the
    # order of the rows.
    ties = table(
        {"k": "int64", "f": "double"}, [{"k": 1, "f": 0.0}, {"k": 1, "f": -0.0}]
    )
    v = assert_parity(_sub(agg), T, statics={"s": ties}, expect="REFUSED")
    assert "ties DOUBLE values with other bits" in v.detail


def test_sum_past_the_range_of_its_result_refuses_the_build():
    big = Decimal("9" * 38)
    wide = table(
        {"k": "int64", "w": "decimal(38,0)"}, [{"k": 1, "w": big}, {"k": 1, "w": big}]
    )
    v = assert_parity(_sub("sum(w)"), T, statics={"s": wide}, expect="REFUSED")
    assert "adds absolute values past the range of its result" in v.detail


def test_plan_facts_name_the_aggregate_join():
    fn = DuckDBInferFn(
        _sub("sum(v)"),
        row_tables={"__THIS__": T.schema},
        static_tables={"s": S},
    )
    (j,) = fn.plan_facts["joins"]
    assert (j["kind"], j["static"], j["keys"], j["rows"]) == (
        "aggregate",
        "s",
        1,
        S.num_rows,
    )


# Pins: what the oracle evaluates, measured 2026-10-14. The served forms
# cannot trap, so these facts never change a served answer. They are why a
# static WHERE term and an outer side that can trap refuse today.

BIG = 2**63 - 1
PIN_S = pa.table(
    {"k": pa.array([1, 1, 3], pa.int64()), "z": pa.array([0, 0, 1], pa.int64())}
)


def _pin(sql, keys):
    with Oracle() as o:
        o.table("t", "k BIGINT", [(k,) for k in keys])
        o.load("s", PIN_S)
        got = o.try_answer(sql)
    if isinstance(got, Trap):
        return got
    return [tuple(r.values()) for r in got.to_pylist()]


def test_pin_the_aggregate_argument_runs_on_matched_rows_only():
    # No request has k = 3, so its argument, which would overflow, never runs.
    sql = f"SELECT k, (SELECT sum(s.z + {BIG}) FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [1, 2]) == [(1, Decimal(2 * BIG)), (2, None)]


def test_pin_the_where_runs_in_written_order_over_every_static_row():
    # Before the key, a static term runs on the row k = 3 too, and traps.
    before = (
        f"SELECT k, (SELECT count(*) FROM s WHERE s.z + {BIG} > 0 AND s.k = t.k) FROM t"
    )
    got = _pin(before, [1, 2])
    assert isinstance(got, Trap), got
    assert "Overflow in addition of INT64" in got.message
    after = (
        f"SELECT k, (SELECT count(*) FROM s WHERE s.k = t.k AND s.z + {BIG} > 0) FROM t"
    )
    assert _pin(after, [1, 2]) == [(1, 2), (2, 0)]


def test_pin_an_empty_batch_runs_only_an_uncorrelated_subquery():
    correlated = f"SELECT k, (SELECT sum(s.z + {BIG}) FROM s WHERE s.k = t.k) FROM t"
    assert _pin(correlated, []) == []
    got = _pin(f"SELECT k, (SELECT max(s.z + {BIG}) FROM s) FROM t", [])
    assert isinstance(got, Trap), got
    assert "Overflow in addition of INT64" in got.message
