"""The correlated scalar subquery with one aggregate over one static table,
`(SELECT sum(s.v) FROM s WHERE s.k = t.k)` (src/specializer/frontend/subquery.rs).

Confit binds it as a hidden LEFT join to a table that it computes at build:
one row per group of static rows with equal keys, holding the group's
aggregate. The WHERE terms that read only the static table filter its rows
before the build groups them. A request that matches no group reads the
aggregate of no rows: 0 for count, NULL for the others. The oracle evaluates
the subquery's WHERE in written order over (distinct outer values) x (static
rows), and the aggregate's argument over the matched rows only (the pins at
the end). Only the outer side of a first key can trap, and confit evaluates it
for the same rows as the oracle; elsewhere the values are what compares.

Every test but the pins asserts the campaign's own verdict (`fuzz.parity`).
"""

from __future__ import annotations

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
        (_sub("median(v)"), "the aggregate median"),
        (_sub("sum(f)"), "sum over a DOUBLE column"),
        (_sub("avg(f)"), "avg over a DOUBLE column"),
        (_sub("count(DISTINCT v)"), "DISTINCT"),
        (_sub("max(v) + 1"), "one aggregate call"),
        (_sub("v"), "one aggregate call"),
        (_sub("count(NULL)"), "must be a column of its table"),
        (_sub("count(*)", "s.k = t.k AND s.v > t.k"), "the WHERE term s.v > t.k"),
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
        (_ksub(f"s.f = 1 AND s.k = {TRAP}", tail=" WHERE x = 0"), "AGREE"),
        (_ksub(f"s.k = {TRAP}"), "AGREE_TRAP"),
        (_ksub(f"{TRAP} = s.k"), "AGREE_TRAP"),
        (_ksub(f"s.k IS NOT DISTINCT FROM {TRAP}"), "AGREE_TRAP"),
        (_ksub(f"s.k = {TRAP} AND s.v = y"), "AGREE_TRAP"),
        # A term before the key that passes no row: the side never runs.
        (_ksub(f"s.f = 7 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.f = NULL AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.v NOT BETWEEN 0 AND 10 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"(s.v BETWEEN 0 AND -1 OR s.f = 9) AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.v >= 0 AND s.v <= -1 AND s.k = {TRAP}"), ROWS_OR_OPT),
        (_ksub(f"s.v BETWEEN 100 AND 200 AND s.k = {TRAP}"), ROWS_OR_OPT),
        # A term before the key that passes a row, and terms after the key.
        (_ksub(f"s.f = 0 AND s.k = {TRAP}"), "AGREE_TRAP"),
        (_ksub(f"s.k = {TRAP} AND s.f = 7"), "AGREE_TRAP"),
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


# Pins: what the oracle evaluates, measured 2026-10-14. The served forms
# cannot trap but in the outer side of a first key, so these facts never
# change a served answer. They are why a WHERE term that reads only the
# table and can trap refuses today.

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
def test_pin_the_outer_side_of_a_key_runs_over_the_rows_the_terms_before_it_pass(
    where, traps
):
    with Oracle() as o:
        o.table("t", "pid BIGINT, x BIGINT, y BIGINT", KT_ROWS)
        o.load("s", KS)
        got = o.try_answer(f"SELECT pid, (SELECT count(*) FROM s WHERE {where}) FROM t")
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
