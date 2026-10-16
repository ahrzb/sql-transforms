"""The correlated scalar subquery with aggregates over one static table,
`(SELECT sum(s.v) FROM s WHERE s.k = t.k)` (src/specializer/frontend/subquery.rs).

Confit binds it as a hidden LEFT join to a table that it computes at build:
one row per group of static rows with equal keys, holding the group's
aggregate. The WHERE terms that read only the static table filter its rows
before the build groups them. A request that matches no group reads the
aggregate of no rows: 0 for count, NULL for the others. If the item is an
expression over aggregates, `max(s.v) + 1`, the row holds the expression
over the group's aggregates, and a request that matches no group reads the
expression over the aggregates of no rows. `EXISTS (SELECT * FROM s WHERE
...)` is the count of the same rows above 0, and `x IN (SELECT s.c FROM s
WHERE ...)` is a CASE over three counts: of the rows whose item equals x, of
all the rows, and of the rows whose item is not NULL. `x < ANY (SELECT ...)`
and the other comparisons but = test x against the extremes of the items in
place of the first count; `x op ALL` is NOT over ANY with the opposite
comparison. The oracle evaluates the subquery's WHERE in written order over
(distinct outer values) x (static rows), the aggregate's argument over the
matched rows only, and the item once for each distinct outer value (the pins
at the end). Only the outer side of a first key and the left side of IN, ANY
or ALL can trap, and confit evaluates them for the same rows as the oracle;
an item that traps before any request refuses; elsewhere the values are what
compares.

Every test but the pins asserts the campaign's own verdict (`fuzz.parity`).
"""

from __future__ import annotations

import math
import random
import struct
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, SqlFunction
from confit.oracle import Oracle, Trap
from confit.sql import lit

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
    "avg(v)",
    "avg(j)",
    "avg(d)",
    "avg(u)",
    "avg(i8)",
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
    "s.k = 1",  # reads only the table: a filter, not a key
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
    "SELECT (SELECT count(DISTINCT s.v) FROM s WHERE s.k = t.k), "
    "(SELECT sum(ALL s.v) FROM s WHERE s.k = t.k AND s.v NOT BETWEEN 1 AND 3), "
    "(SELECT max(DISTINCT x.v) + count(x.v) FROM s AS x WHERE x.k = t.k) "
    "FROM __THIS__ AS t",
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


# WHERE terms that read only the static table: the build keeps the rows on
# which every one is TRUE, then groups them.
STATIC_TERMS = [
    "s.k = t.k AND s.v > 5",
    "s.v > 5 AND s.k = t.k",  # before the key
    "s.k = t.k AND s.f < 1.0",  # NaN and -0.0
    "s.k = t.k AND s.b",
    "s.k = t.k AND NOT s.b",
    "s.k = t.k AND s.v",  # a BIGINT term: nonzero is TRUE
    "s.k = t.k AND s.f",  # a DOUBLE term: NaN is TRUE, -0.0 FALSE
    "s.k = t.k AND s.u",
    "s.k = t.k AND s.s IS NULL",
    "s.k = t.k AND s.v BETWEEN 0 AND 10",
    "s.k = t.k AND s.v IN (10, 20)",
    "s.k = t.k AND (s.j > 1 OR s.k IS NULL)",
    "s.k = t.k AND CASE WHEN s.b THEN s.v > 5 ELSE s.f > 0 END",
    "s.k = t.k AND coalesce(s.v, 0) > 5",
    "s.k = t.k AND TRY_CAST(s.s AS BIGINT) IS NULL",
    "s.k = t.k AND s.d > 1.5",
    "s.k = t.k AND s.s > 'a'",
    "s.k = t.k AND s.j = s.k",  # two columns of the table
    "s.k = t.k AND NULL",  # no row passes
    "s.k = t.k AND TRUE",
    "s.k = t.k AND FALSE",
    "s.k = k",  # `k` names s.k: the inner scope binds first
    "s.v > 5",  # uncorrelated
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", STATIC_TERMS)
def test_static_terms(where, shape):
    for agg in ["count(*)", "sum(v)", "max(s)", "avg(d)"]:
        assert_parity(_sub(agg, where), T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize("agg", AGGREGATES)
def test_aggregate_over_the_rows_a_static_term_keeps(agg):
    assert_parity(_sub(agg, "s.k = t.k AND s.j = 1"), T, statics={"s": S})


@pytest.mark.parametrize(
    "sql",
    [
        # The names of the table, aliased with its columns renamed, and
        # qualified by schema and by catalog.
        "SELECT id, (SELECT count(*) FROM s AS x(kk, jj) "
        "WHERE x.kk = t.k AND x.jj = 2) AS c FROM __THIS__ AS t",
        "SELECT id, (SELECT sum(v) FROM main.s "
        "WHERE main.s.v > 5 AND main.s.k = t.k) AS c FROM __THIS__ AS t",
        "SELECT id, (SELECT sum(v) FROM memory.main.s "
        "WHERE memory.main.s.v > 5 AND s.k = t.k) AS c FROM __THIS__ AS t",
        # Two subqueries over one table, each with terms of its own.
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k AND s.v > 5) "
        "+ (SELECT sum(v) FROM s WHERE s.k = t.k AND s.v < 15) AS c FROM __THIS__ AS t",
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k AND s.v > 5) AS c "
        "FROM __THIS__ AS t WHERE c > 0",
    ],
)
def test_static_term_forms(sql):
    assert_parity(sql, T, statics={"s": S})


def test_static_terms_over_a_table_with_columns_it_does_not_serve():
    # A struct and a timestamp column beside the plain ones: a term reads
    # the plain columns, and a term that reads the others refuses by name.
    w = pa.table(
        {
            "k": pa.array([1, 1, 2, None], pa.int64()),
            "ts": pa.array([datetime(2020, 1, 1)] * 4, pa.timestamp("us")),
            "st": pa.array(
                [{"a": 1}, None, {"a": 3}, {"a": 4}], pa.struct([("a", pa.int64())])
            ),
            "v": pa.array([5, 6, None, 8], pa.int64()),
        }
    )
    assert_parity(_sub("sum(v)", "s.k = t.k AND s.v > 5"), T, statics={"s": w})
    v = assert_parity(
        _sub("sum(v)", "s.k = t.k AND s.st.a > 1"),
        T,
        statics={"s": w},
        expect="REFUSED",
    )
    assert "a struct field" in v.detail, v.detail
    v = assert_parity(
        _sub("sum(v)", "s.k = t.k AND s.ts IS NULL"),
        T,
        statics={"s": w},
        expect="REFUSED",
    )
    assert "column 'ts' has type timestamp[us]" in v.detail, v.detail
    # A term that reads no column, over a table with no plain column.
    only = w.select(["ts", "st"])
    for where in ["TRUE", "FALSE", "NULL"]:
        assert_parity(_sub("count(*)", where), T, statics={"s": only})


@pytest.mark.parametrize("shape", [None, "many"])
def test_static_terms_over_an_empty_table_and_an_empty_batch(shape):
    where = "s.k = t.k AND s.v > 5"
    assert_parity(_sub("count(*)", where), T, statics={"s": S.slice(0, 0)}, shape=shape)
    assert_parity(_sub("sum(v)", where), T.slice(0, 0), statics={"s": S}, shape=shape)
    assert_parity(
        _sub("max(v)", "s.v > 5"), T.slice(0, 0), statics={"s": S}, shape=shape
    )


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
        _sub("count(*)", "s.v IS NULL AND s.k IS NOT DISTINCT FROM t.k"),
        "SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
        "FROM (SELECT * FROM __THIS__) AS t",
        "WITH w AS (SELECT id, (SELECT count(*) FROM s WHERE s.k = t.k) AS c "
        "FROM __THIS__ AS t) SELECT * FROM w",
    ],
)
def test_forms_of_the_default_shape(sql):
    assert_parity(sql, T, statics={"s": S})


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("agg", ["count(*)", "sum(v)", "avg(d)", "min(s)", "max(d)"])
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
        "SELECT id, EXISTS (SELECT 1 FROM s WHERE s.k = t.k) AS c, c AND TRUE AS d "
        "FROM __THIS__ AS t",
    ],
)
def test_select_list_alias_of_a_subquery_is_duckdbs_binder_error(sql):
    v = assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
    assert v.oracle == "rejects"
    assert (
        'Alias "c" referenced in a SELECT clause - but the expression has a subquery'
        in v.detail
    )


# A SQL function whose field `q` does not read its argument: a read of `q`
# binds no subquery, and still holds the one in the argument.
G = SqlFunction(
    "g",
    pa.schema([("v", pa.int64())]),
    pa.struct([("p", pa.int64()), ("q", pa.float64())]),
    lambda v: {"p": v * lit(2), "q": lit(1.5)},
)
SUB = "(SELECT count(*) FROM s WHERE s.k = t.k)"


@pytest.mark.parametrize(
    "sql, alias",
    [
        (f"SELECT g({SUB}).q AS a, a + 1 AS c FROM __THIS__ AS t", "a"),
        # Every read of one call holds the same marker, and the check for a
        # subquery answers each marker once: the later reads reuse it.
        (
            f"SELECT g({SUB}).q AS a, g({SUB}).q AS b, g({SUB}).q AS c, c + 1 AS d "
            "FROM __THIS__ AS t",
            "c",
        ),
        (
            f"SELECT g(g({SUB}).p).q AS a, g(g({SUB}).p).q AS b, b + 1 AS c "
            "FROM __THIS__ AS t",
            "b",
        ),
    ],
)
def test_select_list_alias_of_a_call_over_a_subquery_is_duckdbs_binder_error(
    sql, alias
):
    v = assert_parity(sql, T, statics={"s": S}, udfs=[G], expect="REFUSED")
    assert v.oracle == "rejects"
    assert (
        f'Alias "{alias}" referenced in a SELECT clause - but the expression has a '
        "subquery" in v.detail
    )


def test_select_list_alias_of_a_call_beside_one_over_a_subquery_serves():
    assert_parity(
        f"SELECT g({SUB}).q AS a, g(k).q AS b, g(k).p AS c, b + c AS d "
        "FROM __THIS__ AS t",
        T,
        statics={"s": S},
        udfs=[G],
    )


def test_static_term_that_calls_a_sql_function_refuses():
    v = assert_parity(
        _sub("count(*)", "s.k = t.k AND g(s.v).p > 0"),
        T,
        statics={"s": S},
        udfs=[G],
        expect="REFUSED",
    )
    assert "reads only its table and calls a SQL function" in v.detail, v.detail


@pytest.mark.parametrize(
    "sql, why",
    [
        (_sub("median(v)"), "function median"),
        (
            _sub("sum(f) + 1"),
            "computes over sum(f), a DOUBLE sum or avg",
        ),
        (_sub("count(NULL)"), "must be a column of its table"),
        (
            _sub("count(*)", "s.k = t.k AND s.v > t.k + 1"),
            "the WHERE term s.v > t.k + 1",
        ),
        (_sub("count(*)", "s.k + 1 = t.k"), "the WHERE term s.k + 1 = t.k"),
        (
            _sub("sum(v)", "s.j = t.j AND s.k = t.k + 1"),
            "can trap and follows another key",
        ),
        (
            _sub("count(*)", "s.k = t.k AND s.v + 9223372036854775807 > 0"),
            "reads only its table and can trap",
        ),
        (_sub("count(*)", "s.k = t.k AND s.s"), "reads only its table and can trap"),
        (_sub("count(*)", "s.s = t.k"), "compares VARCHAR with another type"),
        (_sub("count(*)", "s.k = (SELECT max(k) FROM s)"), "which holds a subquery"),
        (_sub("count(*)", "s.v > (SELECT min(v) FROM s)"), "which holds a subquery"),
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


@pytest.mark.parametrize("arg, ty", [("b", "BOOLEAN"), ("s", "VARCHAR")])
def test_avg_over_boolean_or_varchar_is_duckdbs_binder_error(arg, ty):
    v = assert_parity(_sub(f"avg({arg})"), T, statics={"s": S}, expect="REFUSED")
    assert v.oracle == "rejects"
    assert (
        f"No function matches the given name and argument types 'avg({ty})'" in v.detail
    )


# Groups (sum, count) of each storage type whose avg tells DuckDB's
# rounding apart from the others (src/duckdb/avg.rs). SMALLINT and
# DECIMAL(4,2) divide in double, which differs here from x87 long double.
# The other types divide in long double, which differs here from double and
# from one rounding of the exact quotient. Found by a search over the
# models; each group spreads its sum evenly over its rows.
AVG_GROUPS = {
    "int8": [(6406, 3075), (-33758, 2211)],
    "int16": [(57813847, 3923), (33050602, 3247)],
    "int32": [(-2827971762031, 3555), (-1175830622509, 2619)],
    "int64": [(239779659854332136443, 26), (129125398997375843331, 14)],
    "uint8": [(100498, 2679), (16350, 2311)],
    "uint16": [(511573, 3719), (113396039, 3599)],
    "uint32": [(2773561945545, 3479), (6420972326810, 3039)],
    "uint64": [(36890665975583623169, 2), (332037418797826136062, 18)],
    "decimal(4,2)": [(-7033692, 2280), (7176710, 2594)],
    "decimal(9,4)": [(-306314197041, 2361), (1305803900508, 3848)],
    "decimal(18,6)": [(-357814364270915024781, 2736), (-409514676728703535941, 3629)],
    "decimal(38,0)": [
        (41781269006313210299918023857145420718, 6),
        (-24465413588787404663834588390323333212, 3),
    ],
}


def _groups(spec, groups):
    """A static table (g, v) whose group g has sum and count groups[g]."""
    scale = int(spec[:-1].split(",")[1]) if spec.startswith("decimal") else None
    rows = []
    for g, (total, n) in enumerate(groups):
        q, r = divmod(total, n)
        for x in [q + 1] * r + [q] * (n - r):
            v = x if scale is None else Decimal(f"{x}E-{scale}")
            rows.append({"g": g, "v": v})
    return table({"g": "int64", "v": spec}, rows)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("spec", sorted(AVG_GROUPS))
def test_avg_rounds_as_duckdb_finalizes_it(spec, shape):
    s = _groups(spec, AVG_GROUPS[spec])
    t = table(
        {"id": "int64", "g": "int64?"},
        [{"id": i, "g": g} for i, g in enumerate([0, 1, 2, None, 1])],
    )
    sql = "SELECT id, (SELECT avg(s.v) FROM s WHERE s.g = t.g) AS a FROM __THIS__ AS t"
    assert_parity(sql, t, statics={"s": s}, shape=shape)


def test_avg_past_the_range_of_its_sum_refuses_the_build():
    # DuckDB keeps a HUGEINT sum, and an addition past its range traps: in
    # the order of these rows, the first one. The total itself fits.
    near = Decimal(8 * 10**37)
    ok = table({"k": "int64", "w": "decimal(38,0)"}, [{"k": 1, "w": near}] * 2)
    assert_parity(_sub("avg(w)"), T, statics={"s": ok})
    big = ["9" * 38, "9" * 38, "-" + "9" * 38]
    wide = table(
        {"k": "int64", "w": "decimal(38,0)"},
        [{"k": 1, "w": Decimal(b)} for b in big],
    )
    v = assert_parity(_sub("avg(w)"), T, statics={"s": wide}, expect="REFUSED")
    assert "past the range of the sum that DuckDB keeps for it" in v.detail


# A DOUBLE sum or avg. Confit adds the values of a group in table order, from
# +0.0. The oracle adds them in the same order over a small table. Over a
# large table it adds parts in parallel, and its answer can differ in the
# last bits from one run to the next. The comparison contract compares such
# an output column within the bound B = 2 (n - 1) 2^-52 sum|v| of its group
# (docs/oracle/05, claim: float-reduction-bound), and the verdict applies it:
# AGREE with the class `float-reduction-bound`. Group 1 of FS cancels: its
# sum is 0.1 in table order and 1.1 in another order.

DMAX = 1.7976931348623157e308

FS = table(
    {"k": "int64?", "f": "double?"},
    [
        {"k": 1, "f": 1e16},
        {"k": 1, "f": 1.0},
        {"k": 1, "f": -1e16},
        {"k": 1, "f": 0.1},
        {"k": 2, "f": -0.0},
        {"k": 2, "f": None},
        {"k": 3, "f": 0.1},
        {"k": 3, "f": 0.2},
        {"k": None, "f": 5e-324},
        {"k": 4, "f": -0.0},
        {"k": 4, "f": 0.0},
        {"k": 4, "f": 1.5},
    ],
)

DOUBLE_SUMS = ["sum(f)", "avg(f)", "sum(DISTINCT f)", "avg(DISTINCT f)", "sum(ALL f)"]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(
    "where", ["s.k = t.k", "s.k = t.k AND s.f > 0", "s.f < 1", None]
)
@pytest.mark.parametrize("agg", DOUBLE_SUMS)
def test_a_double_sum_or_avg(agg, where, shape):
    cond = f" WHERE {where}" if where else ""
    sql = f"SELECT id, (SELECT {agg} FROM s{cond}) AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": FS}, shape=shape, expect="AGREE")


@pytest.mark.parametrize(
    "sql",
    [
        # Through a derived table and a CTE, and beside another subquery.
        "SELECT * FROM (SELECT id, (SELECT sum(f) FROM s WHERE s.k = t.k) AS c "
        "FROM __THIS__ AS t) AS q",
        "WITH q AS (SELECT id, (SELECT avg(f) FROM s WHERE s.k = t.k) AS c "
        "FROM __THIS__ AS t) SELECT c, id FROM q",
        "SELECT (SELECT sum(f) FROM s WHERE s.k = t.k) AS a, "
        "(SELECT avg(f) FROM s) AS b, (SELECT count(*) FROM s WHERE s.k = t.k) AS n "
        "FROM __THIS__ AS t",
        # A lateral alias as the key, and the same subquery twice.
        "SELECT id, k AS kk, (SELECT sum(f) FROM s WHERE s.k = kk) AS c "
        "FROM __THIS__ AS t",
        "SELECT (SELECT sum(f) FROM s WHERE s.k = t.k) AS a, "
        "(SELECT sum(f) FROM s WHERE s.k = t.k) AS b FROM __THIS__ AS t",
        # A WHERE that reads another column.
        "SELECT id, (SELECT avg(f) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t "
        "WHERE t.j = 1",
    ],
)
def test_a_double_sum_or_avg_forms(sql):
    assert_parity(sql, T, statics={"s": FS}, expect="AGREE")


@pytest.mark.parametrize(
    "sql, why",
    [
        (
            "SELECT id, (SELECT sum(f) FROM s WHERE s.k = t.k) + 1 AS c "
            "FROM __THIS__ AS t",
            "the expression of the column 'c' reads a DOUBLE sum or avg",
        ),
        (
            "SELECT id, coalesce((SELECT sum(f) FROM s WHERE s.k = t.k), 0) AS c "
            "FROM __THIS__ AS t",
            "the expression of the column 'c' reads a DOUBLE sum or avg",
        ),
        (
            "SELECT id, (SELECT avg(f) FROM s WHERE s.k = t.k) IS NULL AS c "
            "FROM __THIS__ AS t",
            "the expression of the column 'c' reads a DOUBLE sum or avg",
        ),
        (
            "SELECT id, CAST((SELECT avg(f) FROM s WHERE s.k = t.k) AS VARCHAR) AS c "
            "FROM __THIS__ AS t",
            "the expression of the column 'c' reads a DOUBLE sum or avg",
        ),
        (
            "SELECT c * 2 AS d FROM (SELECT (SELECT sum(f) FROM s "
            "WHERE s.k = t.k) AS c FROM __THIS__ AS t) AS q",
            "the expression of the column 'd' reads a DOUBLE sum or avg",
        ),
        (
            "SELECT id, (SELECT sum(f) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t "
            "WHERE c > 0",
            "a WHERE clause reads a DOUBLE sum or avg",
        ),
        (
            "SELECT id FROM (SELECT id, (SELECT sum(f) FROM s WHERE s.k = t.k) AS c "
            "FROM __THIS__ AS t) AS q WHERE q.c IS NOT NULL",
            "a WHERE clause reads a DOUBLE sum or avg",
        ),
        (
            "SELECT q.id, s2.k FROM (SELECT id, (SELECT sum(f) FROM s WHERE s.k = t.k) "
            "AS c FROM __THIS__ AS t) AS q LEFT JOIN s AS s2 ON s2.f = q.c",
            "a join condition reads a DOUBLE sum or avg",
        ),
        (
            "SELECT id, struct_pack(a := (SELECT sum(f) FROM s WHERE s.k = t.k)) AS c "
            "FROM __THIS__ AS t",
            "the struct or list output 'c' reads a DOUBLE sum or avg",
        ),
        (
            _sub("coalesce(avg(f), 0)"),
            "computes over avg(f), a DOUBLE sum or avg",
        ),
        (
            _sub("sum(f) / count(*)"),
            "computes over sum(f), a DOUBLE sum or avg",
        ),
    ],
)
def test_a_double_sum_or_avg_that_is_not_a_whole_column_refuses_by_name(sql, why):
    v = assert_parity(sql, T, statics={"s": FS}, expect="REFUSED")
    assert why in v.detail, v.detail


@pytest.mark.parametrize("bad", [NAN, float("inf"), float("-inf")])
@pytest.mark.parametrize("agg", ["sum(f)", "avg(DISTINCT f)"])
def test_a_double_sum_over_a_value_that_is_not_finite_refuses_the_build(agg, bad):
    rows = [{"k": 1, "f": 1.0}, {"k": 2, "f": bad}]
    s = table({"k": "int64", "f": "double"}, rows)
    v = assert_parity(_sub(agg), T, statics={"s": s}, expect="REFUSED")
    assert "confit serves a DOUBLE sum or avg only over finite values" in v.detail


@pytest.mark.parametrize(
    "vals, served",
    [
        ([DMAX, DMAX], False),
        # The sum of the absolute values fits, but not with the bound added.
        ([DMAX / 3] * 3, False),
        ([DMAX / 3] * 2, True),
        # The rule reads the absolute values, so it also refuses this sum,
        # whose partial sums do not overflow in any order.
        ([DMAX, -DMAX, 1.0], False),
        ([DMAX / 4, -DMAX / 4, 1.0], True),
        # (n - 1) times the sum of the absolute values is past the range, but
        # this sum with the bound added is not.
        ([DMAX * 0.375, -DMAX * 0.375, 1.0], True),
        # The sum of the absolute values is DMAX, so B is finite. In table
        # order the last addition rounds to inf, and DuckDB answers inf. In
        # the reverse order each addition is exact, and the sum is DMAX.
        ([2.0**1023 - 2.0**970, 2.0**1022, 2.0**1022 - 2.0**970], False),
    ],
)
def test_a_double_sum_past_the_range_refuses_the_build(vals, served):
    s = table({"k": "int64", "f": "double"}, [{"k": 1, "f": v} for v in vals])
    if served:
        assert_parity(_sub("sum(f)"), T, statics={"s": s}, expect="AGREE")
        return
    v = assert_parity(_sub("avg(f)"), T, statics={"s": s}, expect="REFUSED")
    assert "adds absolute values past the range of a DOUBLE" in v.detail


def test_the_bound_build_answers_the_bound_of_each_group():
    # The bound that the build computes, against Python's `math.fsum` for the
    # sum of the absolute values: B = (n - 1) * fsum * 2^-51 for a sum, B / n
    # for an avg, 0.0 for two values or fewer, and NULL for none.
    rng = random.Random(20261015)
    rows = []
    for g in range(300):
        for _ in range(rng.choice([0, 1, 2, 3, 7, 40])):
            e = rng.randrange(-30, 30)
            f = None if rng.random() < 0.1 else rng.uniform(-1, 1) * 2.0**e
            rows.append({"k": g, "f": f})
    rng.shuffle(rows)
    s = table({"k": "int64", "f": "double?"}, rows)
    t = table({"id": "int64", "k": "int64?"}, [{"id": g, "k": g} for g in range(301)])
    for func in ("sum", "avg"):
        sql = (
            f"SELECT id, (SELECT {func}(f) FROM s WHERE s.k = t.k) AS c "
            "FROM __THIS__ AS t"
        )
        fn = DuckDBInferFn(
            sql,
            row_tables={"__THIS__": t.schema},
            static_tables={"s": s},
            float_bounds=True,
        )
        assert fn.plan_facts["bounds"] == [{"column": "c", "func": func}]
        got = {r["id"]: r["c"] for r in fn.infer_arrow(t).to_pylist()}
        for g in range(301):
            vals = [r["f"] for r in rows if r["k"] == g and r["f"] is not None]
            n = len(vals)
            b = (n - 1) * math.fsum(abs(v) for v in vals) * 2.0**-51
            want = None if n == 0 else 0.0 if n <= 2 else b if func == "sum" else b / n
            assert repr(got[g]) == repr(want), (func, g, vals)


def test_a_large_table_agrees_within_the_bound():
    # 300,000 rows in 300 groups: the oracle adds the parts of a group in
    # parallel, so its answer can differ from the table-order sum.
    rng = random.Random(7)
    n = 300_000
    s = pa.table(
        {
            "k": pa.array([rng.randrange(300) for _ in range(n)], pa.int64()),
            "f": pa.array(
                [
                    rng.uniform(-1e6, 1e6) * 10.0 ** rng.randrange(-3, 4)
                    for _ in range(n)
                ],
                pa.float64(),
            ),
        }
    )
    t = table({"id": "int64", "k": "int64?"}, [{"id": g, "k": g} for g in range(300)])
    sql = "SELECT id, (SELECT avg(f) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t"
    assert_parity(sql, t, statics={"s": s}, expect="AGREE")


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
    assert fn.plan_facts["bounds"] == []


# A first key whose outer side can trap. DuckDB evaluates that side over the
# static rows that the WHERE terms before it pass (the pins below), so it
# traps only if one such row exists. Confit counts those rows at build and
# evaluates the side only when the count is above 0. A subquery in the SELECT
# list runs over the rows that the outer WHERE passes, and one in the WHERE
# over the rows that reach that WHERE. The cast traps for rows 2 and 3 of
# KT, and row 1 matches the key 0 of KS.

KT = table(
    {"pid": "int", "x": "int?", "y": "int"},
    [
        {"pid": 1, "x": 0, "y": 0},
        {"pid": 2, "x": 1, "y": 1},
        {"pid": 3, "x": 5, "y": 1},
        {"pid": 4, "x": None, "y": 0},
    ],
)
KS = table(
    {"k": "int?", "v": "int?", "f": "int?"},
    [
        {"k": 10, "v": 1, "f": 0},
        {"k": 0, "v": 2, "f": 1},
        {"k": 2, "v": 3, "f": 1},
        {"k": None, "v": None, "f": None},
    ],
)
TRAP = "CAST(1000 * x AS TINYINT)"
# Optimizer-on DuckDB can order these terms another way, so where the
# optimizer-off oracle and confit agree, the verdict can read DIVERGE_OPT.
ROWS_OR_OPT = ("AGREE", "DIVERGE_OPT")
TRAP_OR_OPT = ("AGREE_TRAP", "DIVERGE_OPT")


def _ksub(where: str, agg: str = "count(*)", tail: str = "") -> str:
    return f"SELECT pid, (SELECT {agg} FROM s WHERE {where}) AS c FROM __THIS__{tail}"


@pytest.mark.parametrize(
    "sql, expect",
    [
        # The outer WHERE passes no row that traps, and the others match.
        (_ksub(f"s.k = {TRAP}", tail=" WHERE x = 0 OR x IS NULL"), "AGREE"),
        (_ksub(f"s.k = {TRAP}", "sum(v)", " WHERE x = 0 OR x IS NULL"), "AGREE"),
        (_ksub(f"s.k = {TRAP}", "max(v)", " WHERE x = 0 OR x IS NULL"), "AGREE"),
        (_ksub(f"s.k = {TRAP}", "s.v", " WHERE x = 0 OR x IS NULL"), "AGREE"),
        (
            _ksub(
                f"s.k = {TRAP} ORDER BY s.v DESC LIMIT 1",
                "s.v",
                " WHERE x = 0 OR x IS NULL",
            ),
            "AGREE",
        ),
        (_ksub(f"s.f = 1 AND s.k = {TRAP}", tail=" WHERE x = 0"), "AGREE"),
        (_ksub(f"s.k = {TRAP}"), "AGREE_TRAP"),
        (_ksub(f"s.k = {TRAP}", "s.v * 2"), "AGREE_TRAP"),
        (_ksub(f"{TRAP} = s.k"), "AGREE_TRAP"),
        (_ksub(f"s.k IS NOT DISTINCT FROM {TRAP}"), "AGREE_TRAP"),
        (_ksub(f"s.k = {TRAP} AND s.v = y"), "AGREE_TRAP"),
        # A term before the key that passes no row: the side never runs.
        (_ksub(f"s.f = 7 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.f = 7 AND s.k = {TRAP}", "s.v"), ROWS_OR_OPT),
        (_ksub(f"s.f = 7 AND s.k = {TRAP} ORDER BY s.v LIMIT 1", "s.v"), ROWS_OR_OPT),
        (_ksub(f"s.f = NULL AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.v NOT BETWEEN 0 AND 10 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"(s.v BETWEEN 0 AND -1 OR s.f = 9) AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.v >= 0 AND s.v <= -1 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.v BETWEEN 100 AND 200 AND s.k = {TRAP}"), ROWS_OR_OPT),
        # A term before the key that passes a row, and terms after the key.
        (_ksub(f"s.f = 0 AND s.k = {TRAP}"), "AGREE_TRAP"),
        (_ksub(f"s.k = {TRAP} AND s.f = 7"), "AGREE_TRAP"),
        (
            _ksub(f"s.f = 0 AND s.k = {TRAP} ORDER BY s.v LIMIT 1 OFFSET 1", "s.v"),
            "AGREE_TRAP",
        ),
        # The upper bound of BETWEEN moves after the key.
        (_ksub(f"s.v BETWEEN 0 AND -1 AND s.k = {TRAP}"), TRAP_OR_OPT),
        (_ksub(f"(s.v BETWEEN 0 AND -1 AND s.f = 1) AND s.k = {TRAP}"), TRAP_OR_OPT),
        # In the SELECT list the subquery runs for every row that the WHERE
        # passes, under a CASE too.
        (
            f"SELECT pid, CASE WHEN x = 0 THEN (SELECT count(*) FROM s "
            f"WHERE s.k = {TRAP}) END AS c FROM __THIS__",
            "AGREE_TRAP",
        ),
        (
            "SELECT pid, (SELECT count(*) FROM s "
            f"WHERE s.f = 7 AND s.k = {TRAP}) AS a, "
            f"(SELECT max(v) FROM s WHERE s.k = {TRAP}) AS b FROM __THIS__",
            "AGREE_TRAP",
        ),
        (
            f"SELECT pid, (SELECT count(*) FROM s WHERE s.k = {TRAP}) AS c "
            "FROM (SELECT * FROM __THIS__ WHERE x = 0) AS d",
            "AGREE",
        ),
        # In the WHERE it runs before the WHERE: for the rows that the terms
        # before it drop too, and again when the WHERE reads its alias.
        (
            f"SELECT pid FROM __THIS__ WHERE x = 0 AND (SELECT count(*) FROM s "
            f"WHERE s.k = {TRAP}) >= 0",
            TRAP_OR_OPT,
        ),
        (_ksub(f"s.k = {TRAP}", tail=" WHERE x = 0 AND c >= 0"), TRAP_OR_OPT),
        (
            _ksub(
                f"s.k = {TRAP}",
                tail=" WHERE x = 0 AND (SELECT count(*) FROM s "
                f"WHERE s.k = {TRAP}) >= 0",
            ),
            TRAP_OR_OPT,
        ),
        (
            f"SELECT pid FROM __THIS__ WHERE x = 0 OR (SELECT count(*) FROM s "
            f"WHERE s.k = {TRAP}) >= 0",
            TRAP_OR_OPT,
        ),
    ],
)
def test_a_first_key_whose_outer_side_can_trap(sql, expect):
    assert_parity(sql, KT, statics={"s": KS}, expect=expect)


@pytest.mark.parametrize(
    "sql",
    [
        _ksub(f"s.k = {TRAP}"),
        _ksub(f"s.k = {TRAP}", "sum(v)"),
        "SELECT pid FROM __THIS__ "
        f"WHERE (SELECT count(*) FROM s WHERE s.k = {TRAP}) >= 0",
    ],
)
def test_a_first_key_whose_outer_side_can_trap_over_an_empty_table(sql):
    empty = KS.slice(0, 0)
    assert_parity(sql, KT, statics={"s": empty}, expect="AGREE")


S2 = table({"id": "int", "w": "str"}, [{"id": 1, "w": "a"}])


@pytest.mark.parametrize(
    "sql, expect",
    [
        # After the WHERE, a comma join has matched its keys.
        (_ksub(f"s.k = {TRAP}", tail=", s2 WHERE pid = s2.id"), "AGREE"),
        (
            f"SELECT pid FROM __THIS__ JOIN s2 ON pid = s2.id "
            f"WHERE (SELECT count(*) FROM s WHERE s.k = {TRAP}) >= 0",
            "AGREE",
        ),
        (_ksub(f"s.k = {TRAP}", tail=" LEFT JOIN s2 ON pid = s2.id"), "AGREE_TRAP"),
    ],
)
def test_a_first_key_whose_outer_side_can_trap_beside_a_join(sql, expect):
    assert_parity(sql, KT, statics={"s": KS, "s2": S2}, expect=expect)


@pytest.mark.parametrize(
    "sql, shape, why",
    [
        (
            _ksub(f"s.v = y + 100 AND s.k = {TRAP}"),
            None,
            "can trap and follows another key",
        ),
        # DuckDB runs a subquery in the WHERE over every pair of a comma
        # join, before the WHERE matches the comma table's key.
        (
            f"SELECT pid FROM __THIS__, s2 WHERE pid = s2.id "
            f"AND (SELECT count(*) FROM s WHERE s.k = {TRAP}) >= 0",
            None,
            "beside a comma join",
        ),
        (_ksub(f"s.k = {TRAP}"), "many", "under shape='many'"),
    ],
)
def test_a_key_whose_outer_side_can_trap_refuses_by_name(sql, shape, why):
    v = assert_parity(
        sql, KT, statics={"s": KS, "s2": S2}, shape=shape, expect="REFUSED"
    )
    assert why in v.detail, v.detail


# An expression over the aggregates: the build computes it for each group,
# and a request that matches no group reads it over the aggregates of no
# rows (0 for a count, NULL for the others).

EXPRESSIONS = [
    "max(v) + 1",
    "count(*) + 1",
    "coalesce(max(v), 0)",
    "coalesce(sum(v), 0) + count(*)",
    "max(v) - min(v)",
    "sum(v) / count(*)",
    "count(*) > 1",
    "max(s) || 'x'",
    "CASE WHEN count(*) > 1 THEN 'many' ELSE 'few' END",
    "-max(v)",
    "abs(min(v))",
    "max(d) + 1",
    "sum(d) * 2",
    "avg(v) * 2",
    "count(v) * 2.5",
    "max(v) IS NULL",
    "least(max(v), 3)",
    "greatest(coalesce(max(v), 9), 3)",
    "max(v) - max(v)",
    "count(1) + count(*)",
    "max(u) + 0",
    "sum(u) + 1",
    "min(f) * 2",
    "max(b) OR count(*) = 0",
    "CAST(sum(v) AS INTEGER)",
    "max(s) LIKE 'b%'",
    "regexp_matches(max(s), 'b')",
    "min(i8) + 1",
    "coalesce(max(s), 'none')",
    "coalesce(sum(d), 1.5)",
    "coalesce(avg(v), -1.5)",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", ["s.k = t.k", None])
@pytest.mark.parametrize("item", EXPRESSIONS)
def test_an_expression_over_the_aggregates(item, where, shape):
    if where is None:
        sql = f"SELECT id, (SELECT {item} FROM s) AS c FROM __THIS__ AS t"
    else:
        sql = _sub(item, where)
    assert_parity(sql, T, statics={"s": S}, shape=shape)


AT = table(
    {"id": "int", "k": "int?"},
    [{"id": 1, "k": 1}, {"id": 2, "k": 2}, {"id": 3, "k": 3}, {"id": 4, "k": None}],
)
# The group k = 1 has only NULL values: a match whose aggregates differ from
# those of no rows.
AS = table(
    {"k": "int?", "v": "int?"},
    [{"k": 1, "v": None}, {"k": 1, "v": None}, {"k": 2, "v": 3}, {"k": None, "v": 4}],
)


@pytest.mark.parametrize(
    "sql",
    [
        _sub("coalesce(max(v), -1) + count(*)"),
        _sub("count(*) - count(v)"),
        _sub("coalesce(max(v), -1) + count(*)", "s.k IS NOT DISTINCT FROM t.k"),
        _sub("count(*) - count(v)", "s.k IS NOT DISTINCT FROM t.k"),
        # NULL on the match k = 1, and 0 over no rows: the match, not the
        # value, picks the group's value.
        _sub("CASE WHEN count(*) = 0 THEN 0 ELSE max(v) END"),
        _sub("max(v) + 1", "s.k = t.k + 1"),
        _sub("max(v) * 2", "s.k = t.k AND s.v > 2"),
        "SELECT id, CASE WHEN t.k > 1 THEN (SELECT count(*) + 1 FROM s "
        "WHERE s.k = t.k) END AS c FROM __THIS__ AS t",
        "SELECT id FROM __THIS__ AS t "
        "WHERE (SELECT count(*) + 1 FROM s WHERE s.k = t.k) > 2",
        "SELECT id, (SELECT max(v) - min(v) FROM s WHERE s.k = t.k) AS a, "
        "(SELECT max(v) - min(v) FROM s WHERE s.k = t.k) AS b FROM __THIS__ AS t",
        "SELECT id, (SELECT max(x.v) * 2 FROM s AS x WHERE x.k = t.k) AS c "
        "FROM __THIS__ AS t",
        # No static row passes the filter: every request reads the item over
        # the aggregates of no rows, with a key and without one.
        _sub("count(*) + 1", "s.k = t.k AND s.v > 100"),
        "SELECT id, (SELECT count(*) + 1 FROM s WHERE s.v > 100) AS c "
        "FROM __THIS__ AS t",
    ],
)
def test_an_expression_over_the_aggregates_of_a_match_and_of_no_rows(sql):
    assert_parity(sql, AT, statics={"s": AS})


class Halve:
    name = "halve"
    takes = pa.schema([("x", pa.int64())])
    returns = pa.float64()

    def __call__(self, x):
        return (None if x is None else x / 2.0,)


@pytest.mark.parametrize(
    "sql, udfs, why",
    [
        (_sub("v + max(v)"), [], "reads v outside an aggregate call"),
        (_sub("max(v) + t.k"), [], "reads t.k outside an aggregate call"),
        (_sub("max(max(v))"), [], "must be a column of its table"),
        (_sub("max(v) + (SELECT 1)"), [], "which holds a subquery"),
        (_sub("g(max(v)).p"), [G], "which calls a SQL function"),
        (_sub("halve(max(v))"), [Halve()], "which calls a UDF or a model"),
        # The oracle evaluates the item for each request that matches no
        # group (the one with k NULL here), where ln(0) traps.
        (_sub("ln(count(*))"), [], "traps over the aggregates of no rows"),
        # The group k = 1 has max(i8) = 127, and TINYINT + 1 overflows there.
        (_sub("max(i8) + 1"), [], "traps over the rows of one key"),
        (
            "SELECT id, (SELECT max(i8) + 1 FROM s) AS c FROM __THIS__ AS t",
            [],
            "traps over the table",
        ),
    ],
)
def test_an_expression_over_the_aggregates_refuses_by_name(sql, udfs, why):
    v = assert_parity(sql, T, statics={"s": S}, udfs=udfs, expect="REFUSED")
    assert why in v.detail, v.detail


# An aggregate over DISTINCT values reads each value of a class of equal
# values once. The oracle's equality decides the classes: -0.0 and 0.0 are
# one DOUBLE, as are all NaNs (the pin at the end). Each group of DS repeats
# values: k = 1 holds equal values that differ in their bits, and k = 3 only
# NULL values.

NAN2 = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]
NEG_NAN = struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000000))[0]
DS = table(
    {
        "k": "int64?",
        "v": "int64?",
        "f": "double?",
        "s": "string?",
        "b": "bool?",
        "d": "decimal(9,4)?",
        "u": "uint64",
        "i8": "int8?",
    },
    [
        {
            "k": 1,
            "v": 10,
            "f": 0.0,
            "s": "a",
            "b": True,
            "d": Decimal("1.5"),
            "u": U64,
            "i8": 127,
        },
        {
            "k": 1,
            "v": 10,
            "f": -0.0,
            "s": "a ",
            "b": True,
            "d": Decimal("1.50"),
            "u": U64,
            "i8": 127,
        },
        {
            "k": 1,
            "v": 20,
            "f": NAN,
            "s": "A",
            "b": False,
            "d": None,
            "u": 1,
            "i8": -128,
        },
        {
            "k": 1,
            "v": None,
            "f": NAN2,
            "s": "a",
            "b": None,
            "d": Decimal("2.25"),
            "u": U64,
            "i8": 127,
        },
        {
            "k": 2,
            "v": -5,
            "f": 1.5,
            "s": None,
            "b": None,
            "d": Decimal("-3"),
            "u": 3,
            "i8": 1,
        },
        {
            "k": 2,
            "v": -5,
            "f": NEG_NAN,
            "s": "",
            "b": False,
            "d": Decimal("-3"),
            "u": 3,
            "i8": 1,
        },
        {
            "k": None,
            "v": 7,
            "f": 2.0,
            "s": "z",
            "b": True,
            "d": Decimal("1"),
            "u": 0,
            "i8": None,
        },
        {
            "k": 3,
            "v": None,
            "f": None,
            "s": None,
            "b": None,
            "d": None,
            "u": 0,
            "i8": None,
        },
    ],
)

DISTINCT_AGGREGATES = [
    "count(DISTINCT v)",
    "count(DISTINCT f)",
    "count(DISTINCT s)",
    "count(DISTINCT b)",
    "count(DISTINCT d)",
    "count(DISTINCT u)",
    "count(DISTINCT i8)",
    "sum(DISTINCT v)",
    "sum(DISTINCT b)",
    "sum(DISTINCT d)",
    "sum(DISTINCT u)",
    "sum(DISTINCT i8)",
    "avg(DISTINCT v)",
    "avg(DISTINCT d)",
    "avg(DISTINCT u)",
    "avg(DISTINCT i8)",
    "min(DISTINCT v)",
    "max(DISTINCT s)",
    "min(DISTINCT b)",
    "max(DISTINCT d)",
    "min(DISTINCT u)",
    "max(DISTINCT i8)",
    "count(ALL v)",
    "sum(ALL d)",
    # Beside the same aggregate over all values.
    "count(DISTINCT v) + count(v)",
    "sum(v) - sum(DISTINCT v)",
    "coalesce(avg(DISTINCT d), -1.5)",
    "count(DISTINCT f) * 10 + count(f)",
    "max(DISTINCT v) - min(v)",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", ["s.k = t.k", "s.k = t.k AND s.v > 5", None])
@pytest.mark.parametrize("agg", DISTINCT_AGGREGATES)
def test_an_aggregate_over_distinct_values(agg, where, shape):
    if where is None:
        sql = f"SELECT id, (SELECT {agg} FROM s) AS c FROM __THIS__ AS t"
    else:
        sql = _sub(agg, where)
    assert_parity(sql, T, statics={"s": DS}, shape=shape)


@pytest.mark.parametrize(
    "sql, why",
    [
        (_sub("count(DISTINCT *)"), "the aggregate call count(DISTINCT *)"),
        (_sub("count(DISTINCT 1)"), "must be a column of its table"),
        # DS holds a NaN, and confit adds only finite DOUBLE values.
        (_sub("sum(DISTINCT f)"), "a DOUBLE sum or avg only over finite values"),
        (_sub("sum(v ORDER BY v)"), "ORDER BY or another clause"),
        # The class -0.0 = 0.0 of the group k = 1: which value DuckDB keeps
        # depends on the order of the rows.
        (_sub("min(DISTINCT f)"), "ties DOUBLE values with other bits"),
    ],
)
def test_an_aggregate_over_distinct_values_refuses_by_name(sql, why):
    v = assert_parity(sql, T, statics={"s": DS}, expect="REFUSED")
    assert why in v.detail, v.detail


def test_a_sum_over_distinct_values_adds_each_value_once():
    # The sum over all the values overflows its result, and the sum over the
    # distinct values does not.
    big = Decimal("9" * 38)
    wide = table(
        {"k": "int64", "w": "decimal(38,0)"}, [{"k": 1, "w": big}, {"k": 1, "w": big}]
    )
    assert_parity(_sub("sum(DISTINCT w)"), T, statics={"s": wide})
    assert_parity(_sub("avg(DISTINCT w)"), T, statics={"s": wide})
    v = assert_parity(_sub("sum(w)"), T, statics={"s": wide}, expect="REFUSED")
    assert "adds absolute values past the range of its result" in v.detail
    two = table(
        {"k": "int64", "w": "decimal(38,0)"},
        [
            {"k": 1, "w": big},
            {"k": 1, "w": Decimal("9" * 37 + "8")},
            {"k": 1, "w": big},
        ],
    )
    v = assert_parity(_sub("sum(DISTINCT w)"), T, statics={"s": two}, expect="REFUSED")
    assert "sum(DISTINCT w) over the rows of one key" in v.detail, v.detail


# EXISTS and NOT EXISTS: the count of the rows that the WHERE matches, above
# 0. Each item is `*`, a constant or a column of the table, none of which can
# trap where the oracle evaluates the items, over the matched rows.


def _exists(item="1", where="s.k = t.k", neg=False):
    w = "" if where is None else f" WHERE {where}"
    return f"{'NOT ' if neg else ''}EXISTS (SELECT {item} FROM s{w})"


EXISTS_WHERES = [
    "s.k = t.k",
    "s.k = t.k AND s.v > 5",
    "s.v > 5 AND s.k = t.k",
    "s.f = t.x",  # -0.0 matches 0.0, NaN matches NaN
    "s.s = t.name",
    "s.u = t.u",
    "s.j = t.k",
    "s.k = t.k AND s.j = t.j",
    "s.k = t.k AND s.v IS NULL",
    "s.k = t.k AND FALSE",
    "s.v > 5",  # uncorrelated
    "s.v > 100",  # uncorrelated, and no row passes
    None,
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("neg", [False, True])
@pytest.mark.parametrize("where", EXISTS_WHERES)
def test_exists(where, neg, shape):
    sql = f"SELECT id, {_exists('1', where, neg)} AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize(
    "item",
    ["*", "1", "NULL", "'x'", "-1", "TRUE", "s.v", "v", "(s.f)", "s.v, 2", "1 AS one"],
)
def test_exists_items(item):
    sql = f"SELECT id, {_exists(item)} AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": S})


EXISTS_FORMS = [
    # In the WHERE, alone, negated and beside other terms.
    f"SELECT id FROM __THIS__ AS t WHERE {_exists()}",
    f"SELECT id FROM __THIS__ AS t WHERE {_exists('*', neg=True)}",
    "SELECT id FROM __THIS__ AS t WHERE t.j = 1 "
    f"AND {_exists('1', 's.k = t.k AND s.v > 5')}",
    f"SELECT id FROM __THIS__ AS t WHERE t.j = 2 OR {_exists(neg=True)}",
    # Under CASE, inside expressions, and beside the count that it tests.
    f"SELECT id, CASE WHEN {_exists()} THEN 'y' ELSE 'n' END AS c FROM __THIS__ AS t",
    f"SELECT id, coalesce({_exists()}, FALSE) AS c FROM __THIS__ AS t",
    f"SELECT id, {_exists()} = ((SELECT count(*) FROM s WHERE s.k = t.k) > 0) AS c "
    "FROM __THIS__ AS t",
    f"SELECT id, {_exists()} AND {_exists('1', 's.k = t.k AND s.v > 5', True)} AS c "
    "FROM __THIS__ AS t",
    # NOT takes `EXISTS (...) IS NULL`, as the oracle's NOT does.
    f"SELECT id, {_exists(neg=True)} IS NULL AS c FROM __THIS__ AS t",
    f"SELECT id, ({_exists(neg=True)}) IS NULL AS c FROM __THIS__ AS t",
    f"SELECT id, {_exists(neg=True)} IS NOT DISTINCT FROM t.j > 1 AS c "
    "FROM __THIS__ AS t",
    # The table aliased, its columns renamed, and qualified by schema.
    "SELECT id, EXISTS (SELECT * FROM s AS x(kk, jj) WHERE x.kk = t.k AND x.jj = 2) "
    "AS c FROM __THIS__ AS t",
    "SELECT id, EXISTS (SELECT x.v FROM s AS x WHERE x.k = t.k) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, EXISTS (SELECT 1 FROM main.s WHERE main.s.k = t.k) AS c "
    "FROM __THIS__ AS t",
    # A lateral alias as the outer side, and WHERE reading an item's alias.
    "SELECT id, k AS kk, EXISTS (SELECT 1 FROM s WHERE s.k = kk) AS c "
    "FROM __THIS__ AS t",
    f"SELECT id, {_exists()} AS c FROM __THIS__ AS t WHERE c",
    # DuckDB's names of unaliased items.
    f"SELECT {_exists()}, NOT EXISTS (SELECT * FROM s AS x WHERE x.k = t.k AND "
    "x.j = t.j), EXISTS (SELECT s.v, 'a', TRUE AS b, -1 FROM s) FROM __THIS__ AS t",
    f"SELECT {_exists(neg=True)} IS NULL, ({_exists('NULL', neg=True)}) IS NULL "
    "FROM __THIS__ AS t",
    # Beside a FROM join (under the default shape its duplicate keys refuse).
    f"SELECT t.id, {_exists()} AS c FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", EXISTS_FORMS)
def test_exists_forms(sql, shape):
    if shape is None and "JOIN s AS s2" in sql:
        assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize(
    "where",
    ["s.k IS NOT DISTINCT FROM t.k", "s.f IS NOT DISTINCT FROM t.x"],
)
def test_exists_with_a_key_that_matches_null(where):
    for neg in (False, True):
        sql = f"SELECT id, {_exists('1', where, neg)} AS c FROM __THIS__ AS t"
        assert_parity(sql, T, statics={"s": S})


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", ["s.k = t.k", "s.v > 5", None])
def test_exists_over_an_empty_table_and_an_empty_batch(where, shape):
    for neg in (False, True):
        sql = f"SELECT id, {_exists('*', where, neg)} AS c FROM __THIS__ AS t"
        assert_parity(sql, T, statics={"s": S.slice(0, 0)}, shape=shape)
        assert_parity(sql, T.slice(0, 0), statics={"s": S}, shape=shape)


def _kexists(where: str, neg: bool = False, tail: str = "") -> str:
    not_ = "NOT " if neg else ""
    return (
        f"SELECT pid, {not_}EXISTS (SELECT 1 FROM s WHERE {where}) AS c "
        f"FROM __THIS__{tail}"
    )


@pytest.mark.parametrize(
    "sql, expect",
    [
        (_kexists(f"s.k = {TRAP}"), "AGREE_TRAP"),
        (_kexists(f"s.k = {TRAP}", True, " WHERE x = 0 OR x IS NULL"), "AGREE"),
        (_kexists(f"s.f = 7 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_kexists(f"s.f = 0 AND s.k = {TRAP}", True), "AGREE_TRAP"),
        (_kexists(f"s.k IS NOT DISTINCT FROM {TRAP}"), "AGREE_TRAP"),
        (
            f"SELECT pid FROM __THIS__ WHERE x = 0 AND EXISTS (SELECT 1 FROM s "
            f"WHERE s.k = {TRAP})",
            TRAP_OR_OPT,
        ),
        (
            f"SELECT pid FROM __THIS__ WHERE x = 0 AND NOT EXISTS (SELECT * FROM s "
            f"WHERE s.k = {TRAP})",
            TRAP_OR_OPT,
        ),
    ],
)
def test_exists_whose_first_key_can_trap(sql, expect):
    assert_parity(sql, KT, statics={"s": KS}, expect=expect)


@pytest.mark.parametrize(
    "sql, why",
    [
        (
            f"SELECT id, {_exists(item)} AS c FROM __THIS__ AS t",
            f"the SELECT item {item} under EXISTS",
        )
        # The first traps where the oracle evaluates it, over the rows k = 1.
        for item in [
            "s.v + 9223372036854775807",
            "count(*)",
            "t.k",
            "s.*",
            "* EXCLUDE (k)",
        ]
    ]
    + [
        (
            f"SELECT id, {_exists('1', 's.k = t.k LIMIT 1')} AS c FROM __THIS__ AS t",
            "ORDER BY or LIMIT",
        ),
        (
            f"SELECT id, {_exists('DISTINCT s.v')} AS c FROM __THIS__ AS t",
            "a subquery with DISTINCT",
        ),
        (
            f"SELECT id, {_exists('1', 's.k = t.k GROUP BY s.k')} AS c "
            "FROM __THIS__ AS t",
            "GROUP BY or HAVING",
        ),
        (
            f"SELECT id, {_exists('1', 's.k = t.k AND s.v + 9223372036854775807 > 0')} "
            "AS c FROM __THIS__ AS t",
            "reads only its table and can trap",
        ),
        (
            "SELECT id, EXISTS (SELECT 1 FROM __THIS__ AS r WHERE r.k = t.k) AS c "
            "FROM __THIS__ AS t",
            "over the request table",
        ),
        (
            f"SELECT t.id FROM __THIS__ AS t JOIN s AS s2 ON {_exists()}",
            "in FROM or in a JOIN condition",
        ),
    ],
)
def test_exists_refuses_by_name(sql, why):
    v = assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
    assert why in v.detail, v.detail


# x IN (SELECT ...) and NOT IN: three counts of the rows that the WHERE
# matches, those whose item equals x (the IN test, one more key), all of them,
# and those whose item is not NULL. The item is a column of the table. The
# oracle evaluates x once the subquery has an item that is not NULL: without
# a key in the WHERE the build knows whether it has one; with a key it
# depends on the other requests of the batch, so there an x that can trap
# refuses (the pins).


def _in(x="t.k", item="s.v", where="s.k = t.k", neg=False):
    w = "" if where is None else f" WHERE {where}"
    return f"{x} {'NOT ' if neg else ''}IN (SELECT {item} FROM s{w})"


IN_TESTS = [
    ("t.k", "s.v", "s.k = t.k"),  # the group k = 1 holds a NULL item
    ("t.j", "s.j", "s.k = t.k"),
    ("t.k", "s.v", "s.k = t.k AND s.j = t.j"),
    ("t.k", "s.k", "s.k = t.k AND s.v IS NULL"),
    ("t.j", "s.k", "s.k IS NOT DISTINCT FROM t.k"),
    ("t.k", "s.k", None),  # without a key, and a NULL item
    ("t.k", "s.v", "s.v > 5"),  # without a key, and no NULL item
    ("t.k", "s.v", "s.v > 100"),  # without a key, and no row
    ("t.x", "s.f", None),  # -0.0 matches 0.0, NaN matches NaN
    ("t.x", "s.f", "s.k = t.k"),
    ("t.name", "s.s", None),
    ("t.name", "s.s", "s.k = t.k"),
    ("t.u", "s.u", None),
    ("t.k", "s.u", None),  # BIGINT against UBIGINT
    ("t.k", "s.d", None),  # BIGINT against DECIMAL
    ("t.x", "s.d", None),  # DOUBLE against DECIMAL
    ("t.k", "s.i8", "s.j = t.j"),  # BIGINT against TINYINT
    ("t.k", "s.f", "s.k = t.k"),  # BIGINT against DOUBLE
    ("t.k + 1", "s.v", None),  # can trap: runs once the subquery has a row
    ("1", "s.k", "s.k = t.k"),
    ("NULL", "s.k", "s.k = t.k"),
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("neg", [False, True])
@pytest.mark.parametrize("x, item, where", IN_TESTS)
def test_in(x, item, where, neg, shape):
    sql = f"SELECT id, {_in(x, item, where, neg)} AS c FROM __THIS__ AS t"
    if shape == "many" and (x == "t.k + 1" or "DISTINCT" in (where or "")):
        # As in a scalar subquery: under shape='many' a key whose outer side
        # can trap, and IS NOT DISTINCT FROM, refuse.
        assert_parity(sql, T, statics={"s": S}, shape=shape, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape=shape)


IN_FORMS = [
    # In the WHERE, alone, negated and beside other terms.
    f"SELECT id FROM __THIS__ AS t WHERE {_in()}",
    f"SELECT id FROM __THIS__ AS t WHERE {_in('t.j', 's.j', neg=True)}",
    f"SELECT id FROM __THIS__ AS t WHERE t.j = 1 AND {_in('t.k', 's.k', None)}",
    f"SELECT id FROM __THIS__ AS t WHERE t.j = 2 OR {_in(neg=True)}",
    # Under CASE, inside expressions, and under NOT and IS NULL.
    f"SELECT id, CASE WHEN {_in()} THEN 'y' ELSE 'n' END AS c FROM __THIS__ AS t",
    f"SELECT id, coalesce({_in()}, FALSE) AS c FROM __THIS__ AS t",
    f"SELECT id, NOT {_in()} AS c FROM __THIS__ AS t",
    f"SELECT id, {_in()} IS NULL AS c FROM __THIS__ AS t",
    f"SELECT id, {_in(neg=True)} IS NOT DISTINCT FROM t.j > 1 AS c FROM __THIS__ AS t",
    # Two tests of one subquery, and the subquery's other forms beside them.
    f"SELECT id, {_in()} AND {_in('t.j')} AS c FROM __THIS__ AS t",
    f"SELECT id, {_in()} = {_exists()} AS c, (SELECT count(*) FROM s WHERE s.k = t.k) "
    "AS n FROM __THIS__ AS t",
    # The table aliased, its columns renamed, and qualified by schema.
    "SELECT id, t.k IN (SELECT v FROM s AS z(kk, jj, v) WHERE z.kk = t.k) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, t.k IN (SELECT x.v FROM s AS x WHERE x.k = t.k) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, t.k IN (SELECT main.s.v FROM main.s WHERE main.s.k = t.k) AS c "
    "FROM __THIS__ AS t",
    # x names the outer column; a bare name in the subquery names its own.
    "SELECT id, k IN (SELECT k FROM s) AS c FROM __THIS__ AS t",
    "SELECT id, k IN (SELECT v FROM s WHERE s.j = j) AS c FROM __THIS__ AS t",
    # A lateral alias as x, and WHERE reading an item's alias.
    "SELECT id, j AS jj, jj IN (SELECT s.j FROM s) AS c FROM __THIS__ AS t",
    f"SELECT id, {_in('t.k', 's.k', None)} AS c FROM __THIS__ AS t WHERE c",
    # DuckDB's names of unaliased items.
    f"SELECT {_in()}, t.j NOT IN (SELECT x.j FROM s AS x), NOT {_in('t.j', 's.j')} "
    "FROM __THIS__ AS t",
    # Beside a FROM join (under the default shape its duplicate keys refuse).
    f"SELECT t.id, {_in()} AS c FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", IN_FORMS)
def test_in_forms(sql, shape):
    if shape is None and "JOIN s AS s2" in sql:
        assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", ["s.k = t.k", "s.v > 5", None])
def test_in_over_an_empty_table_and_an_empty_batch(where, shape):
    for neg in (False, True):
        sql = f"SELECT id, {_in('t.k', 's.v', where, neg)} AS c FROM __THIS__ AS t"
        assert_parity(sql, T, statics={"s": S.slice(0, 0)}, shape=shape)
        assert_parity(sql, T.slice(0, 0), statics={"s": S}, shape=shape)


def _kin(x: str, where: str | None = None, neg: bool = False, tail: str = "") -> str:
    w = "" if where is None else f" WHERE {where}"
    not_ = "NOT " if neg else ""
    return f"SELECT pid, {x} {not_}IN (SELECT s.k FROM s{w}) AS c FROM __THIS__{tail}"


@pytest.mark.parametrize(
    "sql, expect",
    [
        # Without a key, x runs once the subquery has a row.
        (_kin(TRAP), "AGREE_TRAP"),
        (_kin(TRAP, neg=True), "AGREE_TRAP"),
        (_kin(TRAP, "s.f = 7"), ROWS_OR_OPT),
        (_kin(TRAP, "s.f = 0"), "AGREE_TRAP"),
        # The IN test runs after the whole WHERE, both bounds of BETWEEN too.
        (_kin(TRAP, "s.v BETWEEN 2 AND 1"), ROWS_OR_OPT),
        (_kin(TRAP, "s.v BETWEEN 1 AND 3", True), "AGREE_TRAP"),
        # x runs only if an item that passes the WHERE is not NULL: here the
        # one row that passes has the item NULL.
        (_kin(TRAP, "s.f IS NULL"), ROWS_OR_OPT),
        (_kin(TRAP, "s.f IS NULL", True), ROWS_OR_OPT),
        (_kin(TRAP, "s.f IS NULL OR s.f = 1"), "AGREE_TRAP"),
        # In the SELECT list after the outer WHERE, in the WHERE before it,
        # and under CASE for every request.
        (_kin(TRAP, tail=" WHERE x = 0 OR x IS NULL"), "AGREE"),
        (
            f"SELECT pid FROM __THIS__ WHERE x = 0 AND {TRAP} IN (SELECT s.k FROM s)",
            TRAP_OR_OPT,
        ),
        (
            f"SELECT pid, CASE WHEN x = 0 THEN {TRAP} IN (SELECT s.k FROM s) END "
            "AS c FROM __THIS__",
            "AGREE_TRAP",
        ),
        (
            f"SELECT pid, CASE WHEN x = 0 THEN {TRAP} IN (SELECT s.k FROM s "
            "WHERE s.f = 7) END AS c FROM __THIS__",
            ROWS_OR_OPT,
        ),
        # The first key of the WHERE runs as in a scalar subquery: after the
        # lower bound of a BETWEEN before it.
        (_kin("x", f"s.k = {TRAP}"), "AGREE_TRAP"),
        (_kin("x", f"s.f = 7 AND s.k = {TRAP}", True), ROWS_OR_OPT),
        (_kin("x", f"s.v BETWEEN 2 AND 1 AND s.k = {TRAP}"), TRAP_OR_OPT),
    ],
)
def test_in_whose_left_side_can_trap(sql, expect):
    assert_parity(sql, KT, statics={"s": KS}, expect=expect)


@pytest.mark.parametrize(
    "sql, why",
    [
        (
            f"SELECT id, {_in('t.k', item)} AS c FROM __THIS__ AS t",
            f"the SELECT item {item} under IN",
        )
        # The first traps where the oracle evaluates it, over the rows k = 1.
        for item in ["s.v + 9223372036854775807", "count(*)", "t.j", "*", "1"]
    ]
    + [
        (
            f"SELECT id, {_in('t.k', 's.v, s.k')} AS c FROM __THIS__ AS t",
            "more than one SELECT item under IN",
        ),
        (
            f"SELECT id, {_in('t.name', 's.k')} AS c FROM __THIS__ AS t",
            "compares VARCHAR with another type",
        ),
        (
            f"SELECT id, {_in('(SELECT max(s.k) FROM s)', 's.v', None)} AS c "
            "FROM __THIS__ AS t",
            "which holds a subquery",
        ),
        (
            f"SELECT id, {_in(where='s.k = t.k LIMIT 1')} AS c FROM __THIS__ AS t",
            "ORDER BY or LIMIT",
        ),
        (
            f"SELECT id, {_in(item='DISTINCT s.v')} AS c FROM __THIS__ AS t",
            "a subquery with DISTINCT",
        ),
        (
            f"SELECT id, {_in(item='s.k', where='s.k = t.k GROUP BY s.k')} AS c "
            "FROM __THIS__ AS t",
            "GROUP BY or HAVING",
        ),
        (
            f"SELECT id, {_in(where='s.k = t.k AND s.v + 9223372036854775807 > 0')} "
            "AS c FROM __THIS__ AS t",
            "reads only its table and can trap",
        ),
        (
            "SELECT id, t.k IN (SELECT r.k FROM __THIS__ AS r) AS c FROM __THIS__ AS t",
            "over the request table",
        ),
        (
            f"SELECT t.id FROM __THIS__ AS t JOIN s AS s2 ON {_in()}",
            "in FROM or in a JOIN condition",
        ),
    ]
    + [
        (
            f"SELECT id, {_in('t.k + 9223372036854775807', where=w)} AS c "
            "FROM __THIS__ AS t",
            "which can trap, over a WHERE with a key",
        )
        # The oracle traps over the first WHERE, where a request matches a
        # static row, and not over the second, where none does.
        for w in ["s.k = t.k", "s.k = t.k AND s.v > 1000"]
    ],
)
def test_in_refuses_by_name(sql, why):
    v = assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
    assert why in v.detail, v.detail


def test_in_whose_left_side_reads_a_joined_table_refuses():
    sql = (
        "SELECT t.id, s2.v IN (SELECT s.v FROM s) AS c FROM __THIS__ AS t "
        "JOIN s AS s2 ON s2.k = t.k"
    )
    v = assert_parity(sql, T, statics={"s": S}, shape="many", expect="REFUSED")
    assert "which reads a joined table" in v.detail, v.detail


# x op ANY (SELECT ...), SOME alike, and x op ALL. The oracle answers ALL as
# NOT over ANY with the opposite comparison, and = ANY as IN (so <> ALL as NOT
# IN). For another comparison, one more hidden join tests x against the
# extremes of the item in the group, and the counts of IN tell FALSE from
# NULL. The oracle evaluates x once a row passes the whole WHERE, NULL items
# too, unlike IN (the pins).

QUANTIFIED = [
    "= ANY",
    "<> ANY",
    "< ANY",
    "<= SOME",
    "> ANY",
    ">= ANY",
    "= ALL",
    "<> ALL",
    "< ALL",
    "<= ALL",
    "> ALL",
    ">= ALL",
]


def _any(op="< ANY", x="t.k", item="s.v", where="s.k = t.k"):
    w = "" if where is None else f" WHERE {where}"
    return f"{x} {op} (SELECT {item} FROM s{w})"


ANY_TESTS = [
    *IN_TESTS,
    ("t.x", "s.d", "s.k = t.k"),  # DOUBLE against DECIMAL, with a key
    ("t.k", "s.f", None),  # NaN above every number
    ("t.j", "s.b", None),  # INTEGER against BOOLEAN
    ("(t.j = 1)", "s.b", "s.k = t.k"),  # BOOLEAN against BOOLEAN
]


@pytest.mark.parametrize("op", QUANTIFIED)
@pytest.mark.parametrize("x, item, where", ANY_TESTS)
def test_any(x, item, where, op):
    sql = f"SELECT id, {_any(op, x, item, where)} AS c FROM __THIS__ AS t"
    if (x, item) == ("t.j", "s.b") and op in ("= ANY", "<> ALL"):
        # As IN, a hash join: its key does not compare BOOLEAN with INTEGER.
        assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S})


@pytest.mark.parametrize("op", ["<> ANY", "< ALL", "= ALL", ">= SOME"])
@pytest.mark.parametrize("x, item, where", IN_TESTS)
def test_any_under_shape_many(x, item, where, op):
    sql = f"SELECT id, {_any(op, x, item, where)} AS c FROM __THIS__ AS t"
    if x == "t.k + 1" or "DISTINCT" in (where or ""):
        # As in a scalar subquery: under shape='many' an outer expression
        # that can trap, and IS NOT DISTINCT FROM, refuse.
        assert_parity(sql, T, statics={"s": S}, shape="many", expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape="many")


# Extremes that tie values with other bits: -0.0 with 0.0, and two NaNs.
# Only the comparison reads them, so the build answers either.
TIES = table(
    {"k": "int64", "f": "double"},
    [
        {"k": 1, "f": 0.0},
        {"k": 1, "f": -0.0},
        {"k": 2, "f": NAN},
        {"k": 2, "f": NAN2},
        {"k": 2, "f": 1.0},
    ],
)


@pytest.mark.parametrize("op", ["< ANY", "> ALL", "<> ANY", "= ALL", ">= ANY"])
@pytest.mark.parametrize("where", ["s.k = t.k", None])
def test_any_over_extremes_that_tie_with_other_bits(op, where):
    sql = f"SELECT id, {_any(op, 't.x', 's.f', where)} AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": TIES})


@pytest.mark.parametrize("op", ["< ANY", ">= ANY", "<> ANY", "= ALL", "> ALL"])
def test_any_compares_decimals_at_their_max_type(op):
    # DuckDB casts both sides of ANY to their max type: a DECIMAL(10,4)
    # against a DECIMAL(38,0) is a DECIMAL(38,0), so x = 1.5 rounds to 2.
    # The comparison operator would compare them at DECIMAL(38,4).
    s = table({"h": "decimal(38,0)?"}, [{"h": Decimal(2)}, {"h": Decimal(3)}])
    x = "CAST(t.j AS DECIMAL(9,4)) + 0.5"
    sql = f"SELECT id, {_any(op, x, 's.h', None)} AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": s})


ANY_FORMS = [
    # In the WHERE, alone, negated and beside other terms.
    f"SELECT id FROM __THIS__ AS t WHERE {_any()}",
    f"SELECT id FROM __THIS__ AS t WHERE NOT {_any('>= ALL', 't.j', 's.j')}",
    f"SELECT id FROM __THIS__ AS t WHERE t.j = 1 AND {_any('<> ANY', where=None)}",
    f"SELECT id FROM __THIS__ AS t WHERE t.j = 2 OR {_any('= ALL')}",
    # Under CASE, inside expressions, and under IS NULL.
    f"SELECT id, CASE WHEN {_any()} THEN 'y' ELSE 'n' END AS c FROM __THIS__ AS t",
    f"SELECT id, coalesce({_any('> ALL')}, FALSE) AS c FROM __THIS__ AS t",
    f"SELECT id, {_any('<= SOME')} IS NULL AS c FROM __THIS__ AS t",
    f"SELECT id, {_any('<> ALL')} IS NOT DISTINCT FROM t.j > 1 AS c FROM __THIS__ AS t",
    # Two tests of one subquery, and its other forms beside them.
    f"SELECT id, {_any()} AND {_any('> ANY')} AS c FROM __THIS__ AS t",
    f"SELECT id, {_any()} = {_in()} AS c, {_any('< ALL')} AS d, "
    "(SELECT max(s.v) FROM s WHERE s.k = t.k) AS n FROM __THIS__ AS t",
    # The table aliased, its columns renamed, and qualified by schema.
    "SELECT id, t.k < ANY (SELECT v FROM s AS z(kk, jj, v) WHERE z.kk = t.k) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, t.k >= ALL (SELECT main.s.v FROM main.s WHERE main.s.k = t.k) AS c "
    "FROM __THIS__ AS t",
    # x names the outer column; a bare name in the subquery names its own.
    "SELECT id, k > ANY (SELECT k FROM s) AS c FROM __THIS__ AS t",
    # A lateral alias as x, and WHERE reading an item's alias.
    "SELECT id, j AS jj, jj <> ALL (SELECT s.j FROM s) AS c FROM __THIS__ AS t",
    f"SELECT id, {_any('< ANY', where=None)} AS c FROM __THIS__ AS t WHERE c",
    # The subquery in parentheses of its own.
    "SELECT id, t.k < ANY ((SELECT s.v FROM s)) AS c FROM __THIS__ AS t",
    # DuckDB's names of unaliased items: SOME as ANY, ALL as NOT over ANY.
    f"SELECT {_any()}, t.j <= SOME (SELECT x.j FROM s AS x), "
    f"{_any('> ALL', 't.j', 's.j')}, {_any('<> ALL')}, NOT {_any('= ANY')} "
    "FROM __THIS__ AS t",
    # Beside a FROM join (under the default shape its duplicate keys refuse).
    f"SELECT t.id, {_any()} AS c FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", ANY_FORMS)
def test_any_forms(sql, shape):
    if shape is None and "JOIN s AS s2" in sql:
        assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize("op", ["<> ANY", "> ALL", "= ALL"])
@pytest.mark.parametrize("where", ["s.k = t.k", "s.v > 5", None])
def test_any_over_an_empty_table_and_an_empty_batch(op, where):
    sql = f"SELECT id, {_any(op, 't.k', 's.v', where)} AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": S.slice(0, 0)})
    assert_parity(sql, T.slice(0, 0), statics={"s": S})


def _kany(op: str, x: str, where: str | None = None, tail: str = "") -> str:
    w = "" if where is None else f" WHERE {where}"
    return f"SELECT pid, {x} {op} (SELECT s.k FROM s{w}) AS c FROM __THIS__{tail}"


@pytest.mark.parametrize("op", ["<> ANY", "< ANY", ">= ALL", "= ALL"])
@pytest.mark.parametrize(
    "case, expect",
    [
        # Without a key, x runs once a row passes the whole WHERE.
        ((None, ""), "AGREE_TRAP"),
        (("s.f = 7", ""), ROWS_OR_OPT),
        (("s.f = 0", ""), "AGREE_TRAP"),
        (("s.v BETWEEN 2 AND 1", ""), ROWS_OR_OPT),
        (("s.v BETWEEN 1 AND 3", ""), "AGREE_TRAP"),
        # A row whose item is NULL counts too, unlike under IN.
        (("s.f IS NULL", ""), "AGREE_TRAP"),
        (("s.k IS NULL", ""), "AGREE_TRAP"),
        # In the SELECT list after the outer WHERE.
        ((None, " WHERE x = 0 OR x IS NULL"), "AGREE"),
    ],
)
def test_any_whose_left_side_can_trap(op, case, expect):
    where, tail = case
    assert_parity(_kany(op, TRAP, where, tail), KT, statics={"s": KS}, expect=expect)


@pytest.mark.parametrize("op", ["<> ANY", "< ANY", ">= ALL", "= ALL"])
@pytest.mark.parametrize(
    "sql, expect",
    [
        # In the WHERE before the outer WHERE runs, and under CASE for every
        # request.
        (
            "SELECT pid FROM __THIS__ WHERE x = 0 AND {t} {op} (SELECT s.k FROM s)",
            TRAP_OR_OPT,
        ),
        (
            "SELECT pid, CASE WHEN x = 0 THEN {t} {op} (SELECT s.k FROM s) END AS c "
            "FROM __THIS__",
            "AGREE_TRAP",
        ),
        (
            "SELECT pid, CASE WHEN x = 0 THEN {t} {op} (SELECT s.k FROM s "
            "WHERE s.f = 7) END AS c FROM __THIS__",
            ROWS_OR_OPT,
        ),
        (
            "SELECT pid, CASE WHEN x = 0 THEN {t} {op} (SELECT s.k FROM s "
            "WHERE s.f IS NULL) END AS c FROM __THIS__",
            "AGREE_TRAP",
        ),
        # The first key of the WHERE runs as in a scalar subquery.
        (
            "SELECT pid, x {op} (SELECT s.k FROM s WHERE s.k = {t}) AS c FROM __THIS__",
            "AGREE_TRAP",
        ),
        (
            "SELECT pid, x {op} (SELECT s.k FROM s WHERE s.f = 7 AND s.k = {t}) AS c "
            "FROM __THIS__",
            ROWS_OR_OPT,
        ),
    ],
)
def test_any_runs_where_duckdb_plans_it(op, sql, expect):
    sql = sql.format(t=TRAP, op=op)
    assert_parity(sql, KT, statics={"s": KS}, expect=expect)


@pytest.mark.parametrize(
    "sql, why",
    [
        (
            f"SELECT id, {_any('< ANY', 't.k', item)} AS c FROM __THIS__ AS t",
            f"the SELECT item {item} under < ANY",
        )
        for item in ["s.v + 9223372036854775807", "count(*)", "t.j", "*", "1"]
    ]
    + [
        (
            f"SELECT id, {_any('> ALL', 't.k', 's.v, s.k')} AS c FROM __THIS__ AS t",
            "more than one SELECT item under > ALL",
        ),
        (
            f"SELECT id, {_any('<> ANY', 't.name', 's.k')} AS c FROM __THIS__ AS t",
            "compares VARCHAR with another type",
        ),
        (
            # The BIGINT item casts to DECIMAL(38,20), which holds 18 digits
            # before the point, and DuckDB casts every item.
            f"SELECT id, {_any('< ANY', 'CAST(t.j AS DECIMAL(38,20))', 's.v', None)} "
            "AS c FROM __THIS__ AS t",
            "whose cast to the type of the comparison can trap",
        ),
        (
            f"SELECT id, {_any('= ALL', 't.k', 's.s')} AS c FROM __THIS__ AS t",
            "compares VARCHAR with another type",
        ),
        (
            f"SELECT id, {_any('< ANY', '(SELECT max(s.k) FROM s)', 's.v', None)} AS c "
            "FROM __THIS__ AS t",
            "which holds a subquery",
        ),
        (
            f"SELECT id, {_any(where='s.k = t.k LIMIT 1')} AS c FROM __THIS__ AS t",
            "ORDER BY or LIMIT",
        ),
        (
            f"SELECT id, {_any(item='DISTINCT s.v')} AS c FROM __THIS__ AS t",
            "a subquery with DISTINCT",
        ),
        (
            f"SELECT id, {_any(where='s.k = t.k AND s.v + 9223372036854775807 > 0')} "
            "AS c FROM __THIS__ AS t",
            "reads only its table and can trap",
        ),
        (
            "SELECT id, t.k < ANY (SELECT r.k FROM __THIS__ AS r) AS c "
            "FROM __THIS__ AS t",
            "over the request table",
        ),
        (
            f"SELECT t.id FROM __THIS__ AS t JOIN s AS s2 ON {_any()}",
            "in FROM or in a JOIN condition",
        ),
        (
            "SELECT id, t.k = ANY ([1, 2]) AS c FROM __THIS__ AS t",
            "ANY/ALL over",
        ),
    ]
    + [
        (
            f"SELECT id, {_any(op, 't.k + 9223372036854775807', 's.v', w)} AS c "
            "FROM __THIS__ AS t",
            "which can trap, over a WHERE with a key",
        )
        for op in ["<> ANY", ">= ALL"]
        for w in ["s.k = t.k", "s.k = t.k AND s.v > 1000"]
    ],
)
def test_any_refuses_by_name(sql, why):
    v = assert_parity(sql, T, statics={"s": S}, expect="REFUSED")
    assert why in v.detail, v.detail


def test_any_whose_left_side_reads_a_joined_table_refuses():
    sql = (
        "SELECT t.id, s2.v < ANY (SELECT s.v FROM s) AS c FROM __THIS__ AS t "
        "JOIN s AS s2 ON s2.k = t.k"
    )
    v = assert_parity(sql, T, statics={"s": S}, shape="many", expect="REFUSED")
    assert "which reads a joined table" in v.detail, v.detail


def test_any_beside_a_comma_join_refuses_an_x_that_can_trap():
    # The oracle runs the subquery over every pair of the comma join, before
    # the WHERE matches the comma table's key.
    sql = (
        f"SELECT pid FROM __THIS__, s2 WHERE pid = s2.id AND {TRAP} < ANY "
        "(SELECT s.k FROM s)"
    )
    s2 = table({"id": "int"}, [{"id": 1}, {"id": 2}])
    v = assert_parity(sql, KT, statics={"s": KS, "s2": s2}, expect="REFUSED")
    assert "beside a comma join" in v.detail, v.detail


# An item with no aggregate reads the row that a request matches,
# `(SELECT s.v FROM s WHERE s.k = t.k)`. The build reads the one row of each
# group, and a request that matches no group reads NULL: the item does not
# run there (the pins at the end). A group of more than one row refuses the
# build, since the oracle traps when a request matches it. LS has one row for
# each key k, one for the NULL key, and DOUBLE keys f that differ under the
# oracle's equality (-0.0 matches 0.0, NaN matches NaN).

LS = table(
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
            "k": 2,
            "j": 1,
            "v": None,
            "d": Decimal("-2.25"),
            "f": -0.0,
            "s": "a",
            "b": False,
            "u": 3,
            "i8": 127,
        },
        {
            "k": 3,
            "j": 2,
            "v": -3,
            "d": None,
            "f": NAN,
            "s": None,
            "b": None,
            "u": 1,
            "i8": 1,
        },
        {
            "k": None,
            "j": 2,
            "v": 7,
            "d": Decimal("0.1"),
            "f": None,
            "s": "z",
            "b": True,
            "u": 0,
            "i8": None,
        },
        {
            "k": 5,
            "j": 1,
            "v": 20,
            "d": Decimal("3"),
            "f": 2.0,
            "s": "",
            "b": False,
            "u": 2,
            "i8": 5,
        },
    ],
)

ROW_ITEMS = [
    "v",
    "s.v",
    "s.k",
    "j",
    "d",
    "f",
    "s.s",
    "s",  # the column s, not the whole row of s
    "b",
    "u",
    "i8",
    "s.v * 2 + s.j",
    "coalesce(s.v, 7)",
    "upper(s.s) || 'x'",
    "s.f IS NULL",
    "CASE WHEN s.b THEN s.v ELSE -s.j END",
    "s.d + 1",
    "s.v + j",  # j binds in the subquery's table first
    "1",
    "'x'",
]

ROW_WHERES = [
    "s.k = t.k",
    "t.k = s.k",
    "s.k IS NOT DISTINCT FROM t.k",
    "s.k = t.k AND s.j = t.j",
    "s.j = t.j AND s.k = t.k",
    "s.k = t.k AND s.v > 0",  # the row k = 2 has v NULL: no match
    "s.f = t.x",
    "s.u = t.u",
    "s.s = t.name",
    "s.k = 1",  # no key: one row
    "s.k > 100",  # no key: no row
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("item", ROW_ITEMS)
def test_an_item_over_one_row(item, shape):
    assert_parity(_sub(item), T, statics={"s": LS}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("item", ["s.v", "coalesce(s.s, 'none')"])
@pytest.mark.parametrize("where", ROW_WHERES)
def test_an_item_over_one_row_keys(where, item, shape):
    if shape == "many" and "IS NOT DISTINCT FROM" in where:
        v = assert_parity(
            _sub(item, where), T, statics={"s": LS}, shape=shape, expect="REFUSED"
        )
        assert "IS NOT DISTINCT FROM" in v.detail, v.detail
        return
    assert_parity(_sub(item, where), T, statics={"s": LS}, shape=shape)


ROW_FORMS = [
    # DuckDB's names of unaliased items, over aliased, bare and qualified
    # names.
    "SELECT (SELECT s.v FROM s WHERE s.k = t.k), (SELECT x.s FROM s AS x "
    "WHERE x.k = t.k), (SELECT v * 2 FROM s WHERE k = t.k AND j = t.j), "
    "(SELECT 1 FROM s WHERE s.k = t.k), (SELECT main.s.d FROM main.s "
    "WHERE main.s.k = t.k) FROM __THIS__ AS t",
    # Inside expressions, twice, beside an aggregate of the same table, and
    # in WHERE.
    "SELECT id, coalesce((SELECT s.v FROM s WHERE s.k = t.k), -1) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k) "
    "+ (SELECT s.v FROM s WHERE s.k = t.k) "
    "+ (SELECT count(*) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id, CASE WHEN t.k > 1 THEN (SELECT s.s FROM s WHERE s.k = t.k) END AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k) IS NULL AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT s.s FROM s WHERE s.k = t.k) = t.name AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT s.d FROM s WHERE s.k = t.k) * 2 AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT s.f FROM s WHERE s.k = t.k) "
    "IS NOT DISTINCT FROM (SELECT s.f FROM s WHERE s.f = t.x) AS c FROM __THIS__ AS t",
    "SELECT id FROM __THIS__ AS t WHERE (SELECT s.v FROM s WHERE s.k = t.k) > 0",
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t "
    "WHERE c > 0",
    # A lateral alias as the outer side.
    "SELECT id, k AS kk, (SELECT s.s FROM s WHERE s.k = kk) AS c FROM __THIS__ AS t",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", ROW_FORMS)
def test_an_item_over_one_row_forms(sql, shape):
    assert_parity(sql, T, statics={"s": LS}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", ["s.k = t.k", "s.v > 5", None])
def test_an_item_over_one_row_over_an_empty_table_and_an_empty_batch(where, shape):
    w = "" if where is None else f" WHERE {where}"
    sql = f"SELECT id, (SELECT s.v FROM s{w}) AS c FROM __THIS__ AS t"
    assert_parity(sql, T, statics={"s": LS.slice(0, 0)}, shape=shape)
    if where == "s.k = t.k":
        assert_parity(sql, T.slice(0, 0), statics={"s": LS}, shape=shape)
        return
    # Without a key, three or five rows: the oracle traps on every call, an
    # empty batch too, and the build refuses.
    v = assert_parity(
        sql, T.slice(0, 0), statics={"s": LS}, shape=shape, expect="REFUSED"
    )
    assert v.oracle == "traps", v.oracle


DUP_F = table({"f": "double", "v": "int64"}, [{"f": 0.0, "v": 1}, {"f": -0.0, "v": 2}])
ST = pa.table(
    {
        "k": pa.array([1, 2, 3], pa.int64()),
        "st": pa.array([{"a": 1}, {"a": 2}, None], pa.struct([("a", pa.int64())])),
    }
)


@pytest.mark.parametrize(
    "sql, statics, udfs, why",
    [
        # The rows k = 1 of S: the request k = 1 matches three rows.
        (_sub("v"), S, [], "reads 3 rows of one key, where DuckDB traps if a request"),
        # -0.0 and 0.0 are one key.
        (_sub("s.v", "s.f = t.x"), DUP_F, [], "reads 2 rows of one key"),
        (
            _sub("s.v", "s.v > 5"),
            LS,
            [],
            "reads 3 rows of the table, where DuckDB traps on every call",
        ),
        # The item traps over the row k = 1, and over the one row v > 15.
        (_sub("s.v + 9223372036854775807"), LS, [], "traps over a row of one key"),
        (_sub("s.i8 - 1"), LS, [], "traps over a row of one key"),
        (
            _sub("s.v + 9223372036854775807", "s.v > 15"),
            LS,
            [],
            "traps over a row of the table",
        ),
        (_sub("s.v + t.j"), LS, [], "which reads t.j of the outer query"),
        (_sub("s.v + zz"), LS, [], "column 'zz' does not exist"),
        (_sub("s"), ST, [], "which reads the whole row of s"),
        (_sub("s.st['a']"), ST, [], "reads a field or an element of a column of s"),
        (_sub("st['a'] + 1"), ST, [], "reads a field or an element of a column of s"),
        (_sub("NULL"), LS, [], "which DuckDB types as NULL"),
        (_sub("median(v)"), LS, [], "function median"),
        (_sub("first(v)"), LS, [], "aggregate function first"),
        (_sub("s.v + (SELECT 1)"), LS, [], "which holds a subquery"),
        (_sub("g(s.v).p"), LS, [G], "which calls a SQL function"),
        (_sub("halve(s.v)"), LS, [Halve()], "which calls a UDF or a model"),
        (_sub("DISTINCT s.v"), LS, [], "a subquery with DISTINCT"),
        (_sub("s.v, s.k"), LS, [], "more than one SELECT item"),
        (_sub("*"), LS, [], "a star in its SELECT"),
    ],
)
def test_an_item_over_one_row_refuses_by_name(sql, statics, udfs, why):
    v = assert_parity(sql, T, statics={"s": statics}, udfs=udfs, expect="REFUSED")
    assert why in v.detail, v.detail


# ORDER BY, LIMIT and OFFSET over an item that reads one row: per key, the
# build sorts the rows of the group, keeps those from position OFFSET on, at
# most LIMIT of them, and reads the item over the one row kept. Per key of
# OS, each of the columns o, f, s and d, and the pair (b, o), orders the rows
# without a tie; j ties the rows j = 1 of k = 1. The requests k = 1 match
# four rows, k = 2 two and k = 3 one.

INF = float("inf")
OS = table(
    {
        "k": "int64?",
        "v": "int64?",
        "o": "int64?",
        "f": "double?",
        "s": "string?",
        "b": "bool?",
        "d": "decimal(9,4)?",
        "j": "int32",
        "i8": "int8?",
    },
    [
        {
            "k": 1,
            "v": 10,
            "o": 3,
            "f": 1.5,
            "s": "b",
            "b": True,
            "d": Decimal("1.5"),
            "j": 1,
            "i8": 1,
        },
        {
            "k": 1,
            "v": 20,
            "o": 1,
            "f": -0.0,
            "s": "B",
            "b": False,
            "d": Decimal("-2.25"),
            "j": 1,
            "i8": 2,
        },
        {
            "k": 1,
            "v": 30,
            "o": None,
            "f": NAN,
            "s": "é",
            "b": None,
            "d": None,
            "j": 2,
            "i8": 3,
        },
        {
            "k": 1,
            "v": 40,
            "o": 2,
            "f": -INF,
            "s": "a",
            "b": True,
            "d": Decimal("0.5"),
            "j": 1,
            "i8": 4,
        },
        {
            "k": 2,
            "v": 50,
            "o": 5,
            "f": INF,
            "s": "",
            "b": False,
            "d": Decimal("0"),
            "j": 1,
            "i8": 5,
        },
        {
            "k": 2,
            "v": None,
            "o": 4,
            "f": None,
            "s": None,
            "b": True,
            "d": Decimal("3"),
            "j": 1,
            "i8": None,
        },
        {
            "k": 3,
            "v": 70,
            "o": 7,
            "f": 2.0,
            "s": "z",
            "b": True,
            "d": Decimal("0.1"),
            "j": 2,
            "i8": 7,
        },
        {
            "k": None,
            "v": 80,
            "o": 0,
            "f": 0.0,
            "s": "y",
            "b": False,
            "d": Decimal("9"),
            "j": 1,
            "i8": 8,
        },
    ],
)

ORDER_BYS = [
    "s.o",
    "o",
    "s.o DESC",
    "s.o NULLS FIRST",
    "s.o DESC NULLS FIRST",
    "s.o ASC NULLS LAST",
    "s.f",  # -inf, -0.0, 1.5, NaN
    "s.f DESC",
    "s.f NULLS FIRST",
    "s.s",  # byte order: 'B', 'a', 'b', 'é'
    "s.s DESC",
    "s.d",
    "s.d DESC NULLS FIRST",
    "s.b, s.o",
    "s.b DESC, s.o DESC",
    "s.j DESC, s.o",
    "1",  # the item
    "1 DESC",
]

LIMITS = [
    "LIMIT 1",
    "LIMIT 1 OFFSET 1",
    "LIMIT 1 OFFSET 3",  # past the rows of k = 2: NULL
    "LIMIT 0",
    "LIMIT 1 OFFSET 9",
    "OFFSET 3",
    "LIMIT NULL OFFSET 3",
    "LIMIT 5 OFFSET 3",
    "LIMIT 1 OFFSET NULL",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("order", ORDER_BYS)
def test_an_item_over_one_row_ordered(order, limit, shape):
    sql = _sub("v", f"s.k = t.k ORDER BY {order} {limit}")
    assert_parity(sql, T, statics={"s": OS}, shape=shape)


ORDER_ITEMS = [
    "v",
    "s.o",
    "s.f",
    "s.s",
    "s.b",
    "s.d",
    "s.i8",
    "s.v * 2 + s.o",
    "coalesce(s.v, -1)",
    "upper(s.s) || 'x'",  # 'B' and 'b' give one item: a tie under 1
    "s.k",
    "1",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(
    "tail",
    [
        "ORDER BY s.o DESC LIMIT 1",
        "ORDER BY s.f LIMIT 1 OFFSET 2",
        "ORDER BY 1 OFFSET 3",
    ],
)
@pytest.mark.parametrize("item", ORDER_ITEMS)
def test_an_item_over_one_row_ordered_items(item, tail, shape):
    sql = _sub(item, f"s.k = t.k {tail}")
    assert_parity(sql, T, statics={"s": OS}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(
    "where",
    [
        "t.k = s.k",
        "s.k IS NOT DISTINCT FROM t.k",  # the row k = NULL
        "s.k = t.k AND s.j = t.j",
        "s.k = t.k AND s.v > 15",
        "s.f = t.x",  # -0.0 and 0.0 are one key: two rows
        "s.s = t.name",
        "s.v > 15",  # no key: the rows of the table
        "s.k > 100",  # no key: no row
    ],
)
def test_an_item_over_one_row_ordered_keys(where, shape):
    sql = _sub("s.v", f"{where} ORDER BY s.o DESC LIMIT 1")
    if shape == "many" and "IS NOT DISTINCT FROM" in where:
        v = assert_parity(sql, T, statics={"s": OS}, shape=shape, expect="REFUSED")
        assert "IS NOT DISTINCT FROM" in v.detail, v.detail
        return
    assert_parity(sql, T, statics={"s": OS}, shape=shape)


ORDER_FORMS = [
    # DuckDB's names of unaliased subqueries: ASC, DESC and NULLS as
    # written, LIMIT before OFFSET, and LIMIT ALL as LIMIT NULL.
    "SELECT (SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o LIMIT 1), "
    "(SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o DESC NULLS FIRST LIMIT 1 "
    "OFFSET 0), (SELECT x.v FROM s AS x WHERE x.k = t.k ORDER BY x.o ASC NULLS LAST "
    "LIMIT 1 OFFSET 1), "
    "(SELECT v FROM s WHERE k = t.k ORDER BY 1 DESC LIMIT NULL OFFSET 3), "
    "(SELECT s.v AS w FROM s WHERE s.k = t.k ORDER BY w LIMIT ALL OFFSET 3), "
    "(SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o OFFSET 1 ROWS LIMIT 1), "
    "(SELECT s.v FROM s WHERE s.k = t.k ORDER BY (s.o) LIMIT (1) OFFSET (2)) "
    "FROM __THIS__ AS t",
    # Two orders of one subquery, and the same subquery twice.
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o LIMIT 1) "
    "- (SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o DESC LIMIT 1) "
    "+ (SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o LIMIT 1) AS c "
    "FROM __THIS__ AS t",
    # Beside an aggregate of the same table.
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o LIMIT 1) "
    "* (SELECT count(*) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id, coalesce((SELECT s.s FROM s WHERE s.k = t.k ORDER BY s.f DESC "
    "LIMIT 1), "
    "'-') AS c FROM __THIS__ AS t",
    "SELECT id, CASE WHEN t.k > 1 THEN (SELECT s.v FROM s WHERE s.k = t.k "
    "ORDER BY s.o LIMIT 1) END AS c FROM __THIS__ AS t",
    "SELECT id FROM __THIS__ AS t WHERE (SELECT s.v FROM s WHERE s.k = t.k "
    "ORDER BY s.o LIMIT 1) > 15",
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o LIMIT 1) AS c "
    "FROM __THIS__ AS t WHERE c > 15",
    # Without a WHERE.
    "SELECT id, (SELECT s.v FROM s ORDER BY s.o DESC LIMIT 1) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT s.d FROM s WHERE s.k = t.k ORDER BY s.d DESC LIMIT 1) * 2 AS c "
    "FROM __THIS__ AS t",
    # A lateral alias as the outer side.
    "SELECT id, k AS kk, (SELECT s.s FROM s WHERE s.k = kk ORDER BY s.s LIMIT 1) AS c "
    "FROM __THIS__ AS t",
    # The item's alias names the item, before the column o, in any case; a
    # qualified name names the column.
    "SELECT id, (SELECT s.v AS o FROM s WHERE s.k = t.k ORDER BY o DESC LIMIT 1) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT s.v AS O FROM s WHERE s.k = t.k ORDER BY o LIMIT 1 "
    "OFFSET 1) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT s.v AS o FROM s WHERE s.k = t.k ORDER BY s.o LIMIT 1) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k ORDER BY (1) DESC LIMIT 1) AS c "
    "FROM __THIS__ AS t",
    # Without ORDER BY: a group whose rows give one item.
    "SELECT id, (SELECT s.k FROM s WHERE s.k = t.k LIMIT 1) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT s.v FROM s WHERE s.k = t.k AND s.j = 2 OFFSET 0) AS c "
    "FROM __THIS__ AS t",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", ORDER_FORMS)
def test_an_item_over_one_row_ordered_forms(sql, shape):
    assert_parity(sql, T, statics={"s": OS}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(
    "where, tail",
    [
        # Under a key, LIMIT NULL is the greatest BIGINT, and a LIMIT plus
        # the OFFSET past that BIGINT keeps no row.
        ("s.k = t.k", "LIMIT NULL OFFSET 1"),
        ("s.k = t.k", "LIMIT ALL OFFSET 3"),
        ("s.k = t.k", "OFFSET 3 LIMIT NULL"),
        ("s.k = t.k", "LIMIT 9223372036854775804 OFFSET 3"),
        ("s.k = t.k", "LIMIT 9223372036854775805 OFFSET 3"),
        ("s.k = t.k", "LIMIT 1 OFFSET 9223372036854775807"),
        ("s.k = t.k", "OFFSET 9223372036854775807"),
        # Without a key, no addition wraps.
        ("s.v > 0", "LIMIT NULL OFFSET 6"),
        ("s.v > 0", "LIMIT 9223372036854775807 OFFSET 6"),
        ("s.v > 0", "LIMIT 1 OFFSET 9223372036854775807"),
    ],
)
def test_an_item_over_one_row_ordered_at_the_greatest_bigint(where, tail, shape):
    sql = _sub("v", f"{where} ORDER BY s.o {tail}")
    assert_parity(sql, T, statics={"s": OS}, shape=shape)


ONE_PER_KEY = table(
    {"k": "int64?", "v": "int64?", "o": "int64?"},
    [
        {"k": 1, "v": 10, "o": 3},
        {"k": 2, "v": None, "o": None},
        {"k": 3, "v": 70, "o": 7},
    ],
)


@pytest.mark.parametrize(
    "tail",
    [
        "ORDER BY s.o LIMIT NULL",
        "ORDER BY s.o LIMIT ALL",
        "ORDER BY s.o DESC LIMIT 1000000",
        "ORDER BY 1 ASC NULLS LAST LIMIT NULL",
    ],
)
def test_a_limit_of_a_million_or_more_serves_where_duckdbs_optimizer_raises(tail):
    # With the optimizer on, DuckDB plans one ORDER BY term without NULLS
    # FIRST and no OFFSET as arg_min or arg_max with n = the LIMIT, and that
    # aggregate refuses an n of a million or more. The oracle, with the
    # optimizer off, serves (measured 2026-10-15). The divergence ledger
    # lists this as subquery-limit-ceiling.
    sql = _sub("v", f"s.k = t.k {tail}")
    v = assert_parity(sql, T, statics={"s": ONE_PER_KEY}, expect="DIVERGE_OPT")
    assert "n value must be < 1000000" in v.detail, v.detail


@pytest.mark.parametrize(
    "tail",
    [
        "ORDER BY s.o LIMIT 999999",
        "ORDER BY s.o NULLS FIRST LIMIT NULL",
        "ORDER BY s.o, s.v LIMIT NULL",
        "ORDER BY s.o LIMIT NULL OFFSET 0",
    ],
)
def test_other_forms_of_a_limit_of_a_million_or_more_agree(tail):
    assert_parity(_sub("v", f"s.k = t.k {tail}"), T, statics={"s": ONE_PER_KEY})


DUP_FV = table({"f": "double", "v": "int64"}, [{"f": 0.0, "v": 1}, {"f": -0.0, "v": 2}])


@pytest.mark.parametrize("shape", [None, "many"])
def test_an_item_over_one_row_ordered_ties_that_give_one_item(shape):
    # -0.0 and 0.0 tie under ORDER BY, and -0.0 + 0 is 0.0: one item.
    sql = _sub("s.f + 0", "s.f = t.x ORDER BY s.f LIMIT 1")
    assert_parity(sql, T, statics={"s": DUP_FV}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", ["s.k = t.k", "s.v > 5", None])
def test_an_item_over_one_row_ordered_over_an_empty_table_and_an_empty_batch(
    where, shape
):
    w = "" if where is None else f" WHERE {where}"
    sql = (
        f"SELECT id, (SELECT s.v FROM s{w} ORDER BY s.o LIMIT 1) AS c "
        "FROM __THIS__ AS t"
    )
    assert_parity(sql, T, statics={"s": OS.slice(0, 0)}, shape=shape)
    assert_parity(sql, T.slice(0, 0), statics={"s": OS}, shape=shape)


TS = pa.table(
    {
        "k": pa.array([1, 2], pa.int64()),
        "v": pa.array([1, 2], pa.int64()),
        "ts": pa.array([datetime(2026, 1, 1), None], pa.timestamp("us")),
        "st": pa.array([{"a": 1}, None], pa.struct([("a", pa.int64())])),
    }
)


@pytest.mark.parametrize(
    "sql, statics, why",
    [
        # Ties: j ties the rows j = 1 of k = 1, and without ORDER BY each
        # row of a group ties.
        (
            _sub("v", "s.k = t.k ORDER BY s.j LIMIT 1"),
            OS,
            "its ORDER BY ties that row with a row whose item differs (10 and 20)",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.b DESC LIMIT 1"),
            OS,
            "its ORDER BY ties that row with a row whose item differs (10 and 40)",
        ),
        (
            _sub("v", "s.k = t.k LIMIT 1"),
            OS,
            "keeps one row of one key without an ORDER BY, and the item differs",
        ),
        (
            _sub("v", "s.v > 0 LIMIT 1 OFFSET 2"),
            OS,
            "keeps one row of the table without an ORDER BY",
        ),
        # -0.0 and 0.0 tie, and as items they differ.
        (_sub("s.v", "s.f = t.x ORDER BY s.f LIMIT 1"), DUP_FV, "(1 and 2)"),
        (_sub("s.f", "s.f = t.x ORDER BY s.f LIMIT 1"), DUP_FV, "(0.0 and -0.0)"),
        # More than one row kept.
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT 2"),
            OS,
            "keeps 2 of the 4 rows of one key",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o OFFSET 1"),
            OS,
            "keeps 3 of the 4 rows of one key",
        ),
        (_sub("v", "s.k = t.k ORDER BY s.o"), OS, "reads 4 rows of one key"),
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT NULL"),
            OS,
            "keeps 4 of the 4 rows of one key",
        ),
        (
            _sub("v", "s.v > 0 ORDER BY s.o LIMIT 2"),
            OS,
            "keeps 2 of the 7 rows of the table, where DuckDB traps on every call",
        ),
        # The item runs over every row of a group, the rows that the
        # OFFSET skips too.
        (
            _sub("s.i8 * 100", "s.k = t.k ORDER BY s.o LIMIT 1 OFFSET 9"),
            OS,
            "traps over a row of one key",
        ),
        (
            _sub("s.v + 9223372036854775807", "s.v > 5 ORDER BY s.o LIMIT 0"),
            OS,
            "traps over a row of the table",
        ),
        # ORDER BY terms.
        (
            _sub("v", "s.k = t.k ORDER BY t.k LIMIT 1"),
            OS,
            "the ORDER BY term t.k, which reads the outer query",
        ),
        (_sub("v", "s.k = t.k ORDER BY zz LIMIT 1"), OS, "column 'zz' does not exist"),
        (
            _sub("v", "s.k = t.k ORDER BY s.o + 1 LIMIT 1"),
            OS,
            "the ORDER BY term s.o + 1 (each term",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY -s.o LIMIT 1"),
            OS,
            "the ORDER BY term -s.o (each term",
        ),
        (_sub("v", "s.k = t.k ORDER BY s.s COLLATE nocase LIMIT 1"), OS, "(each term"),
        (
            _sub("s.v AS x", "s.k = t.k ORDER BY x + 1 LIMIT 1"),
            OS,
            "the ORDER BY term x + 1",
        ),
        (_sub("v", "s.k = t.k ORDER BY 2 LIMIT 1"), OS, "'ORDER term out of range'"),
        (_sub("v", "s.k = t.k ORDER BY 0 LIMIT 1"), OS, "'ORDER term out of range'"),
        (
            _sub("v", "s.k = t.k ORDER BY 'a' LIMIT 1"),
            OS,
            "'ORDER BY non-integer literal",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY NULL LIMIT 1"),
            OS,
            "'ORDER BY non-integer literal",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY 1.5 LIMIT 1"),
            OS,
            "'ORDER BY non-integer literal",
        ),
        (
            _sub("s.k", "s.k = t.k ORDER BY s LIMIT 1"),
            ST,
            "the ORDER BY term s, which reads the whole row",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.st LIMIT 1"),
            TS,
            "is a struct where a scalar is needed",
        ),
        (_sub("v", "s.k = t.k ORDER BY s.ts LIMIT 1"), TS, "has type timestamp"),
        # LIMIT and OFFSET.
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT -1"),
            OS,
            "LIMIT -1, which is not a number",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT 1.0"),
            OS,
            "a subquery with LIMIT 1.0 --",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT '1'"),
            OS,
            "a subquery with LIMIT '1' --",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT 1 + 0"),
            OS,
            "LIMIT 1 + 0, which is not",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o OFFSET t.k"),
            OS,
            "OFFSET t.k, which is not",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT 9223372036854775808"),
            OS,
            "LIMIT 9223372036854775808 --",
        ),
        (_sub("v", "s.k = t.k ORDER BY s.o LIMIT 1, 1"), OS, "LIMIT <offset>, <count>"),
        (
            _sub("v", "s.k = t.k ORDER BY s.o LIMIT 1 BY s.k"),
            OS,
            "a subquery with LIMIT BY",
        ),
        (
            _sub("v", "s.k = t.k ORDER BY s.o FETCH FIRST 1 ROWS ONLY"),
            OS,
            "FETCH FIRST",
        ),
        # An aggregate call in the item.
        (
            _sub("max(v)", "s.k = t.k LIMIT 1"),
            OS,
            "and an aggregate call in its SELECT item",
        ),
        (
            _sub("count(*)", "s.k = t.k ORDER BY s.o"),
            OS,
            "and an aggregate call in its SELECT",
        ),
        (
            _sub("max(v) + 1", "s.k = t.k LIMIT 1"),
            OS,
            "and an aggregate call in its SELECT",
        ),
    ],
)
def test_an_item_over_one_row_ordered_refuses_by_name(sql, statics, why):
    v = assert_parity(sql, T, statics={"s": statics}, expect="REFUSED")
    assert why in v.detail, v.detail


# A WHERE term that compares the table with the outer query other than by a
# key, `s.v > t.id`: confit scans, for each request, the rows of the table
# that share the request's keys and pass the terms that read only the table,
# at most 100 of them (the build checks), and folds the aggregate over the
# rows on which every other term is TRUE. The oracle evaluates such a term
# over each pair of a static row and a distinct outer value. No such term can
# trap, and the aggregate does not depend on the order of the rows: count,
# sum over an integer or BOOLEAN column, min and max over an integer, DECIMAL
# or VARCHAR column.

SCAN_AGGREGATES = [
    "count(*)",
    "count(v)",
    "count(s)",
    "count(1)",
    "sum(v)",
    "sum(b)",
    "sum(d)",
    "sum(u)",
    "sum(i8)",
    "min(v)",
    "max(v)",
    "min(s)",
    "max(s)",
    "min(d)",
    "max(u)",
    "max(i8)",
]

SCAN_WHERES = [
    "s.v > t.id",
    "s.v > t.id AND s.k = t.k",  # the key after the term
    "s.k = t.k AND s.j <> t.j",
    "s.v BETWEEN t.j AND t.id",
    "s.f < t.x",  # -0.0 and 0.0 compare equal, NaN is the greatest
    "s.f * t.x > 0",  # -0.0 and 0.0 give equal products
    "s.d <= t.x",  # DECIMAL against DOUBLE
    "s.s < t.name",
    "s.u > t.u",
    "s.v > t.id OR s.s = t.name",
    "coalesce(s.v, 0) < t.id",
    "t.k > 1 AND s.v > t.id",  # a term that reads only the request
    "t.k IS NULL",
    "s.k IS NOT DISTINCT FROM t.k AND s.v > t.id",
    "s.v > 5 AND s.j < t.j",  # a term that reads only the table filters
    "s.f = t.x AND s.v < t.id",  # a DOUBLE key
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("where", SCAN_WHERES)
@pytest.mark.parametrize("agg", SCAN_AGGREGATES)
def test_a_scan(agg, where, shape):
    assert_parity(_sub(agg, where), T, statics={"s": S}, shape=shape)


SCAN_FORMS = [
    # EXISTS and NOT EXISTS, in the WHERE, and under CASE.
    "SELECT id, EXISTS (SELECT 1 FROM s WHERE s.v > t.id) AS c FROM __THIS__ AS t",
    "SELECT id, NOT EXISTS (SELECT * FROM s WHERE s.f < t.x) AS c FROM __THIS__ AS t",
    "SELECT id FROM __THIS__ AS t WHERE (SELECT count(*) FROM s WHERE s.v > t.id) > 2",
    "SELECT id FROM __THIS__ AS t WHERE EXISTS (SELECT 1 FROM s WHERE s.s < t.name)",
    "SELECT id, CASE WHEN t.k > 1 THEN (SELECT sum(v) FROM s WHERE s.v > t.id) END "
    "AS c FROM __THIS__ AS t",
    # Two scans, the same scan twice, and a scan beside a key.
    "SELECT id, (SELECT count(*) FROM s WHERE s.v > t.id) AS a, "
    "(SELECT max(s) FROM s WHERE s.v < t.id) AS b, "
    "(SELECT sum(v) FROM s WHERE s.k = t.k) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.v > t.id) "
    "+ (SELECT count(*) FROM s WHERE s.v > t.id) AS c FROM __THIS__ AS t",
    # Inside expressions, a lateral alias as the outer side, and WHERE
    # reading the subquery's alias.
    "SELECT id, coalesce((SELECT max(v) FROM s WHERE s.v < t.id), -1) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT min(s) FROM s WHERE s.v > t.j) || 'x' AS c FROM __THIS__ AS t",
    "SELECT id, j AS jj, (SELECT count(*) FROM s WHERE s.j < jj) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.v > t.id) AS c "
    "FROM __THIS__ AS t WHERE c > 1",
    # The table aliased, its columns renamed, qualified by schema, and its
    # columns unqualified.
    "SELECT id, (SELECT count(*) FROM s AS x WHERE x.v > t.id) AS c FROM __THIS__ AS t",
    "SELECT id, (SELECT sum(vv) FROM s AS x(kk, jj, vv) WHERE x.vv > t.id) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM main.s WHERE main.s.v > t.id) AS c "
    "FROM __THIS__ AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE v > t.id) AS c FROM __THIS__ AS t",
    # A query of two levels (under shape='many' the subquery refuses), where
    # DuckDB correlates on the column of the derived table: -inf and inf.
    "SELECT id, (SELECT count(*) FROM s WHERE s.v > t.id) AS c "
    "FROM (SELECT * FROM __THIS__) AS t",
    "SELECT id, (SELECT count(*) FROM s WHERE s.f < t.y) AS c "
    "FROM (SELECT id, 1.0 / x AS y FROM __THIS__) AS t",
    # Beside a FROM join (under the default shape its duplicate keys refuse).
    "SELECT t.id, (SELECT count(*) FROM s WHERE s.v > t.id) AS c "
    "FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("sql", SCAN_FORMS)
def test_a_scan_forms(sql, shape):
    if (shape is None and "JOIN s AS s2" in sql) or (
        shape == "many" and "FROM (SELECT " in sql
    ):
        assert_parity(sql, T, statics={"s": S}, shape=shape, expect="REFUSED")
        return
    assert_parity(sql, T, statics={"s": S}, shape=shape)


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("agg", ["count(*)", "sum(v)", "max(s)"])
def test_a_scan_over_an_empty_table_and_an_empty_batch(agg, shape):
    sql = _sub(agg, "s.v > t.id")
    assert_parity(sql, T, statics={"s": S.slice(0, 0)}, shape=shape)
    assert_parity(sql, T.slice(0, 0), statics={"s": S}, shape=shape)


def _rows(n, key=lambda i: 1):
    return table(
        {"k": "int64?", "v": "int64"}, [{"k": key(i), "v": i} for i in range(n)]
    )


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(
    "where, statics",
    [
        ("s.v > t.id", _rows(100)),
        # Two keys of 100 rows each.
        ("s.k = t.k AND s.v > t.id", _rows(200, lambda i: i % 2)),
        # The term that reads only the table keeps 100 rows of 150.
        ("s.v >= 50 AND s.v > t.id", _rows(150)),
        # A row whose key is NULL matches no request.
        ("s.k = t.k AND s.v > t.id", _rows(300, lambda i: None if i < 200 else 1)),
    ],
)
def test_a_scan_of_at_most_100_rows(where, statics, shape):
    assert_parity(_sub("sum(v)", where), T, statics={"s": statics}, shape=shape)


@pytest.mark.parametrize(
    "where, statics",
    [
        ("s.v > t.id", _rows(101)),
        ("s.k = t.k AND s.v > t.id", _rows(201, lambda i: i % 2)),
        ("s.v >= 49 AND s.v > t.id", _rows(150)),
    ],
)
def test_a_scan_of_more_than_100_rows_refuses_the_build(where, statics):
    v = assert_parity(
        _sub("sum(v)", where), T, statics={"s": statics}, expect="REFUSED"
    )
    assert "101 rows do here, more than the 100 that confit scans" in v.detail, v.detail


@pytest.mark.parametrize(
    "spec, top",
    [
        ("decimal(18,6)", Decimal("999999999999.999999")),
        ("decimal(38,0)", Decimal("9" * 38)),
    ],
)
def test_a_scan_sums_a_decimal_of_at_most_36_digits(spec, top):
    # 100 values below 10^36 add up below 10^38, the bound of the result.
    wide = table({"v": "int64", "w": spec}, [{"v": i, "w": top} for i in range(100)])
    sql = _sub("sum(w)", "s.v > t.id")
    if spec == "decimal(18,6)":
        assert_parity(sql, T, statics={"s": wide})
        return
    v = assert_parity(sql, T, statics={"s": wide}, expect="REFUSED")
    assert "its SELECT item is sum(w)" in v.detail, v.detail


def test_a_scan_over_a_table_with_columns_it_does_not_serve():
    # A struct and a timestamp column beside the plain ones: a term reads
    # the plain columns, and a term that reads the others refuses by name.
    w = pa.table(
        {
            "k": pa.array([1, 1, 2, None], pa.int64()),
            "ts": pa.array([datetime(2020, 1, 1)] * 4, pa.timestamp("us")),
            "st": pa.array(
                [{"a": 1}, None, {"a": 3}, {"a": 4}], pa.struct([("a", pa.int64())])
            ),
            "v": pa.array([5, 6, None, 8], pa.int64()),
        }
    )
    assert_parity(_sub("sum(v)", "s.k = t.k AND s.v > t.id"), T, statics={"s": w})
    v = assert_parity(
        _sub("sum(v)", "s.st.a > t.id"), T, statics={"s": w}, expect="REFUSED"
    )
    assert "a struct field" in v.detail, v.detail
    v = assert_parity(
        _sub("sum(v)", "s.ts IS NULL AND s.v > t.id"),
        T,
        statics={"s": w},
        expect="REFUSED",
    )
    assert "column 'ts' has type timestamp[us]" in v.detail, v.detail


def test_plan_facts_name_the_scan_and_its_fold():
    fn = DuckDBInferFn(
        _sub("sum(v)", "s.k = t.k AND s.v > t.id"),
        row_tables={"__THIS__": T.schema},
        static_tables={"s": S},
    )
    facts = [(j["kind"], j["static"], j["keys"]) for j in fn.plan_facts["joins"]]
    assert facts == [("scan", "s", 1), ("fold", "s", 0)]
    # The scan holds the rows whose key is not NULL.
    assert fn.plan_facts["joins"][0]["built"] == 5


@pytest.mark.parametrize(
    "sql, udfs, why",
    [
        # Forms that the scan does not serve.
        (
            f"SELECT id, {_in('t.k', 's.v', 's.v > t.id')} AS c FROM __THIS__ AS t",
            [],
            "the subquery is under IN, ANY or ALL",
        ),
        (
            "SELECT id, t.k < ANY (SELECT s.v FROM s WHERE s.v > t.id) AS c "
            "FROM __THIS__ AS t",
            [],
            "the subquery is under IN, ANY or ALL",
        ),
        (
            "SELECT id, (SELECT s.v FROM s WHERE s.v > t.id ORDER BY s.v LIMIT 1) AS c "
            "FROM __THIS__ AS t",
            [],
            "the subquery has ORDER BY or LIMIT",
        ),
        (_sub("max(v) + 1", "s.v > t.id"), [], "max(v) + 1 is not one aggregate call"),
        (_sub("s.v", "s.v > t.id"), [], "s.v is not one aggregate call"),
        (_sub("avg(v)", "s.v > t.id"), [], "its SELECT item is avg(v)"),
        (_sub("count(DISTINCT v)", "s.v > t.id"), [], "is count(DISTINCT v)"),
        (_sub("sum(f)", "s.v > t.id"), [], "its SELECT item is sum(f)"),
        (_sub("min(f)", "s.v > t.id"), [], "its SELECT item is min(f)"),
        (_sub("max(b)", "s.v > t.id"), [], "its SELECT item is max(b)"),
        # Terms that can trap.
        (_sub("count(*)", "s.v > t.id + 1"), [], "s.v > t.id + 1"),
        (_sub("count(*)", "s.k = t.k + 1 AND s.v > t.id"), [], "a key can trap"),
        (_sub("count(*)", "s.v + 1 > t.id"), [], "where the term can trap"),
        # Terms that tell -0.0 from 0.0, which the oracle evaluates once for
        # both (the pin at the end).
        (_sub("count(*)", "s.f < 1.0 / t.x"), [], "can tell -0.0 from 0.0"),
        (_sub("count(*)", "s.s = CAST(t.x AS VARCHAR) AND s.v > t.id"), [], "-0.0"),
        # A lateral alias reads x, on which DuckDB correlates.
        (
            "SELECT id, 1.0 / x AS ix, (SELECT count(*) FROM s WHERE s.f < ix) AS c "
            "FROM __THIS__ AS t",
            [],
            "can tell -0.0 from 0.0",
        ),
        # Terms that read what the scan does not.
        (_sub("count(*)", "g(s.v).p > t.id"), [G], "the term calls a SQL function"),
        (_sub("count(*)", "halve(s.v) > t.id"), [Halve()], "calls a UDF or a model"),
        (
            "SELECT t.id, (SELECT count(*) FROM s WHERE s.v > s2.j) AS c "
            "FROM __THIS__ AS t JOIN s AS s2 ON s2.k = t.k",
            [],
            "reads a joined table",
        ),
    ],
)
def test_a_scan_refuses_by_name(sql, udfs, why):
    v = assert_parity(sql, T, statics={"s": S}, udfs=udfs, expect="REFUSED")
    assert why in v.detail, v.detail


# Pins: what the oracle evaluates, measured 2026-10-14. The served forms
# cannot trap but in the outer side of a first key and in the left side of
# IN, ANY or ALL, so these facts never change a served answer. They are why a WHERE term
# that reads only the table and can trap refuses today.

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


# The outer side of a key, measured 2026-10-15 over KT (as t) and KS: rows 2
# and 3 overflow the cast.
KT_ROWS = [(1, 0, 0), (2, 1, 1), (3, 5, 1), (4, None, 0)]
KTRAP = TRAP.replace("x", "t.x")


@pytest.mark.parametrize(
    "where, traps",
    [
        # It runs over the static rows that the terms before it pass.
        (f"s.k = {KTRAP}", True),
        (f"s.f = 7 AND s.k = {KTRAP}", False),
        (f"s.f = 0 AND s.k = {KTRAP}", True),
        (f"s.k = {KTRAP} AND s.f = 7", True),
        # DuckDB's planner moves the upper bound of BETWEEN after it.
        (f"s.v BETWEEN 0 AND -1 AND s.k = {KTRAP}", True),
        (f"s.v >= 0 AND s.v <= -1 AND s.k = {KTRAP}", False),
        # A second key's runs over the static rows that the first one matches.
        (f"s.v = t.y + 100 AND s.k = {KTRAP}", False),
        (f"s.v = t.y + 1 AND s.k = {KTRAP}", True),
    ],
)
@pytest.mark.parametrize(
    "form",
    [
        "(SELECT count(*) FROM s WHERE {})",
        "EXISTS (SELECT 1 FROM s WHERE {})",
        "(SELECT s.v FROM s WHERE {} ORDER BY s.v LIMIT 1)",
    ],
)
def test_pin_the_outer_side_of_a_key_runs_over_the_rows_the_terms_before_it_pass(
    where, traps, form
):
    with Oracle() as o:
        o.table("t", "pid BIGINT, x BIGINT, y BIGINT", KT_ROWS)
        o.load("s", KS)
        got = o.try_answer(f"SELECT pid, {form.format(where)} FROM t")
    assert isinstance(got, Trap) == traps, got


@pytest.mark.parametrize(
    "sql, traps",
    [
        # An empty static table: no pair to evaluate.
        (f"SELECT pid, (SELECT count(*) FROM e WHERE e.k = {KTRAP}) FROM t", False),
        # In the SELECT list after the WHERE, in the WHERE before it.
        (
            f"SELECT pid, (SELECT count(*) FROM s WHERE s.k = {KTRAP}) FROM t "
            "WHERE t.x = 0",
            False,
        ),
        (
            "SELECT pid FROM t WHERE t.x = 0 "
            f"AND (SELECT count(*) FROM s WHERE s.k = {KTRAP}) >= 0",
            True,
        ),
        # Under a comma join, in the WHERE over every pair before the WHERE
        # matches the key; in the SELECT list over the matched pairs.
        (
            "SELECT pid FROM t, s2 WHERE t.pid = s2.id "
            f"AND (SELECT count(*) FROM s WHERE s.k = {KTRAP}) >= 0",
            True,
        ),
        (
            f"SELECT pid, (SELECT count(*) FROM s WHERE s.k = {KTRAP}) FROM t, s2 "
            "WHERE t.pid = s2.id",
            False,
        ),
        # EXISTS runs where the count does.
        (
            "SELECT pid FROM t WHERE t.x = 0 "
            f"AND EXISTS (SELECT 1 FROM s WHERE s.k = {KTRAP})",
            True,
        ),
        (
            f"SELECT pid, EXISTS (SELECT 1 FROM s WHERE s.k = {KTRAP}) FROM t "
            "WHERE t.x = 0",
            False,
        ),
        (
            f"SELECT pid, CASE WHEN t.x = 0 THEN EXISTS (SELECT 1 FROM s "
            f"WHERE s.k = {KTRAP}) END FROM t",
            True,
        ),
        # The left side of IN without a key runs where the key does.
        (f"SELECT pid, {KTRAP} IN (SELECT s.k FROM s) FROM t WHERE t.x = 0", False),
        (
            f"SELECT pid, CASE WHEN t.x = 0 THEN {KTRAP} IN (SELECT s.k FROM s) "
            "END FROM t",
            True,
        ),
    ],
)
def test_pin_the_subquery_runs_where_duckdb_plans_it(sql, traps):
    with Oracle() as o:
        o.table("t", "pid BIGINT, x BIGINT, y BIGINT", KT_ROWS)
        o.load("s", KS)
        o.load("e", KS.slice(0, 0))
        o.load("s2", S2)
        got = o.try_answer(sql)
    assert isinstance(got, Trap) == traps, got


def test_pin_the_subquery_runs_once_for_minus_zero_and_zero():
    # The oracle evaluates the subquery once for each distinct outer value,
    # and -0.0 and 0.0 are one value: here the value of the first row. So a
    # term that tells them apart answers by the order of the batch, and a
    # scan refuses such a term (measured 2026-10-15).
    sql = (
        "SELECT id, (SELECT count(*) FROM s "
        "WHERE CAST(t.x AS VARCHAR) = '-0.0') FROM t ORDER BY id"
    )
    for rows, n in [([(1, -0.0), (2, 0.0)], 3), ([(1, 0.0), (2, -0.0)], 0)]:
        with Oracle() as o:
            o.table("t", "id BIGINT, x DOUBLE", rows)
            o.load("s", PIN_S)
            got = o.try_answer(sql)
        assert [tuple(r.values()) for r in got.to_pylist()] == [(1, n), (2, n)]


# The SELECT item over the aggregates, measured 2026-10-15 over PIN_S: the
# group k = 1 has z = 0 twice, the group k = 3 has z = 1. The build computes
# the item for every group and refuses one that traps, and the bind refuses
# an item that traps over no rows, so these facts never change a served
# answer.


def test_pin_the_item_runs_over_the_groups_that_requests_match():
    sql = f"SELECT k, (SELECT max(s.z) + {BIG} FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [1, 2]) == [(1, BIG), (2, None)]
    got = _pin(sql, [1, 3])
    assert isinstance(got, Trap), got
    assert "Overflow in addition of INT64" in got.message


def test_pin_the_item_runs_over_no_rows_for_a_request_that_matches_no_group():
    sql = f"SELECT k, (SELECT count(*) - 2 - {BIG} FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [1, 3]) == [(1, -BIG), (3, -BIG - 1)]
    for keys in ([1, 2], [1, None]):
        got = _pin(sql, keys)
        assert isinstance(got, Trap), (keys, got)
        assert "Overflow in subtraction of INT64" in got.message


def test_pin_an_uncorrelated_item_runs_on_every_call():
    for sql in (
        f"SELECT k, (SELECT max(s.z) + {BIG} FROM s) FROM t",
        "SELECT k, (SELECT ln(count(*)) FROM s WHERE s.z > 5) FROM t",
    ):
        got = _pin(sql, [])
        assert isinstance(got, Trap), (sql, got)


def test_pin_a_column_outside_an_aggregate_call_is_a_binder_error():
    got = _pin("SELECT k, (SELECT s.z + max(s.z) FROM s WHERE s.k = t.k) FROM t", [1])
    assert isinstance(got, Trap), got
    assert "must appear in the GROUP BY clause" in got.message


def test_pin_distinct_reads_one_value_for_each_class_of_equal_values():
    # -0.0 and 0.0 are one value, as are NaNs with other bits; NULL is none.
    d = pa.table(
        {"f": pa.array([0.0, -0.0, NAN, NAN2, NEG_NAN, 1.5, None], pa.float64())}
    )
    with Oracle() as o:
        o.table("t", "k BIGINT", [(1,)])
        o.load("d", d)
        got = o.try_answer("SELECT (SELECT count(DISTINCT d.f) FROM d) FROM t")
    assert [tuple(r.values()) for r in got.to_pylist()] == [(3,)]


# EXISTS, measured 2026-10-15 over PIN_S: the group k = 1 has z = 0 twice,
# the group k = 3 has z = 1. Its value and its traps are those of the count
# of its rows above 0, and its items run over the matched rows, so an item
# that can trap refuses.


@pytest.mark.parametrize(
    "where",
    [
        "s.k = t.k",
        f"s.z + {BIG} > 0 AND s.k = t.k",
        f"s.k = t.k AND s.z + {BIG} > 0",
        "s.z > 5",
    ],
)
def test_pin_exists_answers_and_traps_as_the_count_of_its_rows_above_0(where):
    exists = _pin(f"SELECT k, EXISTS (SELECT 1 FROM s WHERE {where}) FROM t", [1, 2])
    count = _pin(f"SELECT k, (SELECT count(*) FROM s WHERE {where}) > 0 FROM t", [1, 2])
    if isinstance(count, Trap):
        assert isinstance(exists, Trap), exists
        assert exists.message == count.message
    else:
        assert exists == count


def test_pin_the_items_under_exists_run_over_the_matched_rows():
    sql = f"SELECT k, EXISTS (SELECT s.z + {BIG} FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [1, 2]) == [(1, True), (2, False)]
    got = _pin(sql, [1, 3])
    assert isinstance(got, Trap), got
    assert "Overflow in addition of INT64" in got.message


def test_pin_exists_over_an_aggregate_is_true():
    # An aggregate without GROUP BY gives one row, over no rows too.
    sql = "SELECT k, EXISTS (SELECT count(*) FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [1, 2]) == [(1, True), (2, True)]


# IN, measured 2026-10-15 over PIN_S: the group k = 1 has z = 0 twice, the
# group k = 3 has z = 1. Its value is a CASE over three counts. The oracle
# evaluates x once the subquery has an item that is not NULL: without a key,
# after the whole WHERE; with a key, for every request of the batch once one
# request matches a static row. Its item runs over the matched rows, so an
# item that can trap refuses.
def _counts(hit: str) -> str:
    return (
        f"CASE WHEN {hit} THEN TRUE "
        "WHEN (SELECT count(*) FROM v) = 0 THEN FALSE "
        "WHEN t.k IS NULL OR (SELECT count(v.c) FROM v) < (SELECT count(*) FROM v) "
        "THEN NULL ELSE FALSE END"
    )


IN_COUNTS = _counts("(SELECT count(*) FROM v WHERE v.c = t.k) > 0")


@pytest.mark.parametrize("neg", [False, True])
@pytest.mark.parametrize("vals", [[1, 2], [1, None], [None], []])
def test_pin_in_answers_as_three_counts(vals, neg):
    # TRUE if an item equals x, else FALSE over no rows, else NULL if x or an
    # item is NULL, else FALSE.
    not_ = "NOT " if neg else ""
    with Oracle() as o:
        o.table("t", "k BIGINT", [(1,), (3,), (None,)])
        o.load("v", pa.table({"c": pa.array(vals, pa.int64())}))
        got = o.try_answer(f"SELECT k {not_}IN (SELECT v.c FROM v) FROM t")
        want = o.try_answer(f"SELECT {not_}({IN_COUNTS}) FROM t")
    assert got.column(0).to_pylist() == want.column(0).to_pylist()


def test_pin_in_runs_its_left_side_once_its_subquery_has_an_item_not_null():
    sql = f"SELECT k, (k + {BIG}) {{}}IN (SELECT {{}} FROM s{{}}) FROM t"
    # The row z = 1 passes the lower bound of BETWEEN 1 AND 0, and x does not
    # run: it runs after the whole WHERE.
    for where in [" WHERE s.z > 5", " WHERE s.z BETWEEN 1 AND 0"]:
        assert _pin(sql.format("", "s.z", where), [1]) == [(1, False)], where
    # The rows k = 1 pass, and NULLIF makes their items NULL: the hash table
    # of the oracle holds no NULL, so x does not run. IN is NULL.
    for not_ in ["", "NOT "]:
        got = _pin(sql.format(not_, "NULLIF(s.z, 0)", " WHERE s.k = 1"), [1])
        assert got == [(1, None)], (not_, got)
    for item, where in [
        ("s.z", ""),
        ("s.z", " WHERE s.z BETWEEN 0 AND 0"),
        ("NULLIF(s.z, 0)", ""),
    ]:
        got = _pin(sql.format("", item, where), [1])
        assert isinstance(got, Trap), (item, where, got)
        assert "Overflow in addition of INT64" in got.message


def test_pin_in_with_a_key_runs_its_left_side_over_the_whole_batch():
    # Once the request k = 1 matches a static row, x runs for k = 2 too,
    # which matches none. This holds also if the items of the matched rows
    # are NULL (NULLIF makes them NULL).
    sql = (
        "SELECT k, k * 4611686018427387904 IN (SELECT {} FROM s WHERE s.k = t.k) FROM t"
    )
    for item in ["s.z", "NULLIF(s.z, 0)"]:
        assert _pin(sql.format(item), [2]) == [(2, False)], item
        got = _pin(sql.format(item), [1, 2])
        assert isinstance(got, Trap), (item, got)
        assert "(2 * 4611686018427387904)" in got.message


@pytest.mark.parametrize(
    "form, want",
    [
        ("IN", [(1, False), (2, False)]),
        ("< ANY", [(1, True), (2, False)]),
        ("= ALL", [(1, False), (2, True)]),
    ],
)
def test_pin_the_item_under_in_or_any_runs_over_the_matched_rows(form, want):
    sql = f"SELECT k, k {form} (SELECT s.z + {BIG} FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [1, 2]) == want
    # An uncorrelated item runs on every call.
    for q, keys in [(sql, [3]), (sql.replace(" WHERE s.k = t.k", ""), [])]:
        got = _pin(q, keys)
        assert isinstance(got, Trap), (q, got)
        assert "Overflow in addition of INT64" in got.message


# ANY and ALL, measured 2026-10-15 over PIN_S. The oracle answers x op ALL as
# NOT over x op' ANY, where op' is the opposite comparison, and = ANY as IN.
# For another comparison, the value is that of IN with a test of x against
# the extremes of the items in place of the count of the items equal to x.
# The oracle evaluates x once a row passes the whole WHERE, NULL items too.
ANY_HIT = {
    "<": "t.k < (SELECT max(v.c) FROM v)",
    "<=": "t.k <= (SELECT max(v.c) FROM v)",
    ">": "t.k > (SELECT min(v.c) FROM v)",
    ">=": "t.k >= (SELECT min(v.c) FROM v)",
    "<>": "(t.k <> (SELECT min(v.c) FROM v) OR t.k <> (SELECT max(v.c) FROM v))",
}
OPPOSITE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "=": "<>"}
ANY_PIN_VALUES = [
    *(("BIGINT", [1, 2, 3, None], v) for v in [[1, 2], [2, 2], [1, None], [None], []]),
    *(
        ("DOUBLE", [0.0, -0.0, 1.0, NAN, None], v)
        for v in [[1.0, NAN], [-0.0, 0.0], [NAN, None], [-1.0, 1.0]]
    ),
]


@pytest.mark.parametrize(
    "form",
    ["< ANY", "<= ANY", "> ANY", ">= ANY", "<> ANY"]
    + ["< ALL", "<= ALL", "> ALL", ">= ALL", "= ALL"],
)
@pytest.mark.parametrize("ty, keys, vals", ANY_PIN_VALUES)
def test_pin_any_answers_as_its_extremes_under_the_counts_of_in(form, ty, keys, vals):
    # NaN is above every number, and -0.0 equals 0.0, in the comparison and
    # in min and max alike.
    op, quant = form.split()
    want = _counts(ANY_HIT[op if quant == "ANY" else OPPOSITE[op]])
    want = want if quant == "ANY" else f"NOT ({want})"
    arrow = pa.int64() if ty == "BIGINT" else pa.float64()
    with Oracle() as o:
        o.table("t", f"k {ty}", [(k,) for k in keys])
        o.load("v", pa.table({"c": pa.array(vals, arrow)}))
        got = o.try_answer(f"SELECT t.k {form} (SELECT v.c FROM v) FROM t")
        want = o.try_answer(f"SELECT {want} FROM t")
    assert got.column(0).to_pylist() == want.column(0).to_pylist()


@pytest.mark.parametrize(
    "form", ["< ANY", "<> ANY", "> ALL", "= ALL", "= ANY", "<> ALL"]
)
def test_pin_any_runs_its_left_side_once_a_row_passes_the_whole_where(form):
    sql = f"SELECT k, (k + {BIG}) {form} (SELECT {{}} FROM s{{}}) FROM t"
    empty = form.endswith("ALL")  # the value over no rows
    # The row z = 1 passes the lower bound of BETWEEN 1 AND 0, and x does not
    # run: it runs after the whole WHERE.
    for where in [" WHERE s.z > 5", " WHERE s.z BETWEEN 1 AND 0"]:
        assert _pin(sql.format("s.z", where), [1]) == [(1, empty)], where
    # The rows k = 1 pass, and NULLIF makes their items NULL. x runs, but
    # = ANY runs as IN, and <> ALL as NOT IN: then x does not run.
    got = _pin(sql.format("NULLIF(s.z, 0)", " WHERE s.k = 1"), [1])
    if form in ("= ANY", "<> ALL"):
        assert got == [(1, None)], got
    else:
        assert isinstance(got, Trap), got
        assert "Overflow in addition of INT64" in got.message
    for where in ["", " WHERE s.z BETWEEN 0 AND 0"]:
        got = _pin(sql.format("s.z", where), [1])
        assert isinstance(got, Trap), (where, got)
        assert "Overflow in addition of INT64" in got.message


@pytest.mark.parametrize(
    "form", ["< ANY", "<> ANY", "> ALL", "= ALL", "= ANY", "<> ALL"]
)
def test_pin_any_with_a_key_runs_its_left_side_over_the_whole_batch(form):
    # Once the request k = 1 matches a static row, x runs for k = 2 too,
    # which matches none. This holds also if the items of the matched rows
    # are NULL (NULLIF makes them NULL).
    sql = (
        f"SELECT k, k * 4611686018427387904 {form} "
        "(SELECT {} FROM s WHERE s.k = t.k) FROM t"
    )
    empty = form.endswith("ALL")  # the value over no rows
    for item in ["s.z", "NULLIF(s.z, 0)"]:
        assert _pin(sql.format(item), [2]) == [(2, empty)], item
        got = _pin(sql.format(item), [1, 2])
        assert isinstance(got, Trap), (item, got)
        assert "(2 * 4611686018427387904)" in got.message


# The SELECT item over one row, measured 2026-10-15 over PIN_S: the group
# k = 1 has two rows, the group k = 3 one. The build refuses a group of more
# than one row, and an item that traps over a row, so these facts never
# change a served answer.


def test_pin_a_request_that_matches_more_than_one_row_traps():
    sql = "SELECT k, (SELECT s.z FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [2, 3]) == [(2, None), (3, 1)]
    for sql in (
        "SELECT k, (SELECT s.z FROM s WHERE s.k = t.k) FROM t",
        # Under a CASE that the request does not take, too.
        "SELECT k, CASE WHEN k > 5 THEN (SELECT s.z FROM s WHERE s.k = t.k) END FROM t",
    ):
        got = _pin(sql, [1, 3])
        assert isinstance(got, Trap), (sql, got)
        assert "More than one row returned by a subquery" in got.message


def test_pin_the_item_over_one_row_runs_over_the_matched_row_only():
    sql = f"SELECT k, (SELECT s.z + {BIG} FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [2]) == [(2, None)]
    got = _pin(sql, [3])
    assert isinstance(got, Trap), got
    assert "Overflow in addition of INT64" in got.message
    # No row matches: NULL, not the item over no rows.
    sql = "SELECT k, (SELECT coalesce(s.z, 7) FROM s WHERE s.k = t.k) FROM t"
    assert _pin(sql, [2]) == [(2, None)]


def test_pin_an_uncorrelated_item_over_more_than_one_row_traps_on_every_call():
    got = _pin("SELECT k, (SELECT s.z FROM s) FROM t", [])
    assert isinstance(got, Trap), got
    assert "More than one row returned by a subquery" in got.message
    assert _pin("SELECT k, (SELECT s.z FROM s WHERE s.k = 3) FROM t", [1]) == [(1, 1)]
    assert _pin("SELECT k, (SELECT s.z FROM s WHERE s.k = 2) FROM t", [1]) == [
        (1, None)
    ]


# ORDER BY, LIMIT and OFFSET over an item that reads one row, measured
# 2026-10-15 over PIN_O: the group k = 1 has three rows, the group k = 3
# one. The build sorts the rows of a group as DuckDB does, and refuses a tie
# that changes the item, a group that keeps more than one row, and an item
# that traps over a row, so these facts never change a served answer.

PIN_O = pa.table(
    {
        "k": pa.array([1, 1, 1, 3], pa.int64()),
        "z": pa.array([0, 1, 2, 9], pa.int64()),
        "o": pa.array([2, None, 1, 0], pa.int64()),
    }
)


def _pin_o(sql, keys, s=PIN_O):
    with Oracle() as o:
        o.table("t", "k BIGINT", [(k,) for k in keys])
        o.load("s", s)
        got = o.try_answer(sql)
    if isinstance(got, Trap):
        return got
    return [tuple(r.values()) for r in got.to_pylist()]


@pytest.mark.parametrize(
    "order, z",
    [
        # o: 1 (z = 2), 2 (z = 0) and NULL (z = 1).
        ("s.o", 1),
        ("s.o ASC", 1),
        ("s.o DESC", 1),
        ("s.o NULLS FIRST", 0),
        ("s.o DESC NULLS FIRST", 2),
    ],
)
def test_pin_order_by_sorts_nulls_last_under_asc_and_desc(order, z):
    sql = (
        "SELECT k, (SELECT s.z FROM s WHERE s.k = t.k "
        f"ORDER BY {order} LIMIT 1 OFFSET 2) FROM t"
    )
    assert _pin_o(sql, [1]) == [(1, z)]


@pytest.mark.parametrize("tail", ["ORDER BY s.o LIMIT 1 OFFSET 1", "LIMIT 0"])
def test_pin_the_item_runs_over_every_matched_row_before_limit_and_offset(tail):
    # The one row of k = 3, which the OFFSET skips and LIMIT 0 drops.
    sql = f"SELECT k, (SELECT s.z + {BIG} FROM s WHERE s.k = t.k {tail}) FROM t"
    got = _pin_o(sql, [3])
    assert isinstance(got, Trap), got
    assert "Overflow in addition of INT64" in got.message
    # A group that no request matches: the item does not run.
    assert _pin_o(sql, [2]) == [(2, None)]


def test_pin_a_limit_that_keeps_more_than_one_row_traps():
    for tail in [
        "ORDER BY s.o LIMIT 2",
        "ORDER BY s.o LIMIT NULL",
        "ORDER BY s.o OFFSET 1",
    ]:
        sql = f"SELECT k, (SELECT s.z FROM s WHERE s.k = t.k {tail}) FROM t"
        assert _pin_o(sql, [3, 2]) == [
            (3, 9 if "OFFSET" not in tail else None),
            (2, None),
        ]
        got = _pin_o(sql, [1])
        assert isinstance(got, Trap), (tail, got)
        assert "More than one row returned by a subquery" in got.message
    # OFFSET NULL skips no row.
    sql = (
        "SELECT k, (SELECT s.z FROM s WHERE s.k = t.k "
        "ORDER BY s.o LIMIT 1 OFFSET NULL) FROM t"
    )
    assert _pin_o(sql, [1]) == [(1, 2)]


def test_pin_a_limit_under_a_key_keeps_rows_up_to_a_sum_that_wraps():
    # Under a key, DuckDB keeps the rows numbered offset < r <= offset +
    # limit, LIMIT NULL is the greatest BIGINT, and the sum wraps.
    sql = "SELECT k, (SELECT s.z FROM s WHERE s.k = t.k ORDER BY s.o {}) FROM t"
    assert _pin_o(sql.format("LIMIT NULL OFFSET 2"), [1]) == [(1, None)]
    assert _pin_o(sql.format("OFFSET 2"), [1]) == [(1, 1)]
    assert _pin_o(sql.format(f"LIMIT {BIG - 2} OFFSET 2"), [1]) == [(1, 1)]
    assert _pin_o(sql.format(f"LIMIT {BIG - 1} OFFSET 2"), [1]) == [(1, None)]
    # Without a key, the same LIMIT keeps the row.
    sql = "SELECT k, (SELECT s.z FROM s WHERE s.k < 3 ORDER BY s.o {}) FROM t"
    assert _pin_o(sql.format("LIMIT NULL OFFSET 2"), [1]) == [(1, 1)]
    assert _pin_o(sql.format(f"LIMIT {BIG - 1} OFFSET 2"), [1]) == [(1, 1)]


def test_pin_order_by_names_the_alias_before_a_column():
    sql = (
        "SELECT k, (SELECT s.z AS o FROM s WHERE s.k = t.k "
        "ORDER BY {} DESC LIMIT 1) FROM t"
    )
    assert _pin_o(sql.format("o"), [1]) == [(1, 2)]
    assert _pin_o(sql.format("O"), [1]) == [(1, 2)]
    assert _pin_o(sql.format("s.o"), [1]) == [(1, 0)]
    # Not the name of an item without an alias.
    sql = (
        "SELECT k, (SELECT s.z + 0 FROM s WHERE s.k = t.k "
        'ORDER BY "(s.z + 0)" LIMIT 1) FROM t'
    )
    got = _pin_o(sql, [1])
    assert isinstance(got, Trap), got
    assert "not found in FROM clause" in got.message


def test_pin_the_order_of_doubles_strings_and_booleans():
    s = pa.table(
        {
            "k": pa.array([1] * 5, pa.int64()),
            "v": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "f": pa.array([0.0, -0.0, NAN, INF, -INF], pa.float64()),
            "s": pa.array(["b", "B", "é", "a", ""], pa.string()),
            "b": pa.array([True, False, True, None, False], pa.bool_()),
        }
    )
    sql = (
        "SELECT k, (SELECT s.v FROM s WHERE s.k = t.k "
        "ORDER BY {} LIMIT 1 OFFSET {}) FROM t"
    )

    def order(by):
        return [_pin_o(sql.format(by, i), [1], s)[0][1] for i in range(5)]

    # -0.0 and 0.0 tie, so v orders them both ways; NaN sorts above inf.
    assert order("s.f, s.v") == [5, 1, 2, 4, 3]
    assert order("s.f DESC, s.v") == [3, 4, 1, 2, 5]
    # Byte order.
    assert order("s.s") == [5, 2, 4, 1, 3]
    assert order("s.b, s.v") == [2, 5, 1, 3, 4]
