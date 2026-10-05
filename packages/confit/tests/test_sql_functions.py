"""SQL functions (`confit.SqlFunction`): a call means its body.

DuckDB registers the body as a macro and substitutes the arguments; confit
substitutes the same arguments into the same text before parsing. Each case
runs through the campaign verdict, which registers the macro by its own
recipe, and the scalar ones also through the function's own `register`.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, ExternFunction, FunctionError, SqlFunction, compare
from confit import sql as S
from confit.oracle import Oracle

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from fuzz.parity import assert_parity, table  # noqa: E402
from test_sql_builder import _tree  # noqa: E402

ROWS = table(
    {"a": "int?", "b": "int?", "x": "float?", "y": "float?", "s": "str?"},
    [
        {"a": 3, "b": -2, "x": 0.5, "y": -1.25, "s": "it's"},
        {"a": None, "b": 7, "x": None, "y": 2.0, "s": None},
        {"a": 0, "b": 0, "x": -0.0, "y": 1e300, "s": ""},
        {"a": 9223372036854775807, "b": 4, "x": 3.5, "y": None, "s": "q"},
    ],
)

F64 = pa.float64()


def _schema(**cols) -> pa.Schema:
    return pa.schema(list(cols.items()))


SCALE = SqlFunction(
    "scale", _schema(v=F64), F64, lambda v: (v - S.lit(3.5)) * S.lit(2.0)
)
AFFINE = SqlFunction(
    "affine",
    _schema(v=F64, w=F64, c=F64),
    F64,
    lambda v, w, c: v * w + c,
)
STATS = SqlFunction(
    "stats",
    _schema(p=pa.int64(), q=pa.int64()),
    pa.struct([("lo", pa.int64()), ("mean", F64), ("tag", pa.string())]),
    lambda p, q: {
        "lo": S.fn("least", p, q),
        "mean": (p + q) / S.lit(2),
        "tag": S.case(p.isnull(), S.lit("none")).otherwise(S.lit("some")),
    },
)
# Its body calls SCALE: a call in a body expands too.
TWICE = SqlFunction(
    "twice", _schema(v=F64), F64, lambda v: S.fn("scale", S.fn("scale", v))
)
# A parameter named after CASE's END, read twice.
KEYWORD = SqlFunction(
    "kw", _schema(end=pa.int64()), pa.int64(), lambda e: S.case(e > 0, e).otherwise(-e)
)
INCR = SqlFunction("incr", _schema(n=pa.int64()), pa.int64(), lambda n: n + 1)
# A struct body reading another struct function by field.
WRAP = SqlFunction(
    "wrap",
    _schema(p=pa.int64(), q=pa.int64()),
    pa.struct([("lo", pa.int64()), ("m2", F64)]),
    lambda p, q: {
        "lo": S.fn("stats", p, q).field("lo"),
        "m2": S.fn("stats", q, p).field("mean") * S.lit(2.0),
    },
)
# A field that traps: reading its sibling still traps.
TRAPS = SqlFunction(
    "traps",
    _schema(n=pa.int64()),
    pa.struct([("v", pa.int64()), ("big", pa.int64())]),
    lambda n: {"v": n, "big": n * S.lit(9223372036854775807)},
)
# A NULL struct where the id is NULL: its body is a CASE.
MAYBE = SqlFunction(
    "maybe",
    _schema(n=pa.int64()),
    pa.struct([("v", pa.int64()), ("w", F64)]),
    lambda n: {"v": n * 2, "w": n / S.lit(4)},
    null_when=lambda n: n.isnull(),
)


@pytest.mark.parametrize(
    "sql, fns",
    [
        ("SELECT scale(x) AS o FROM __THIS__", [SCALE]),
        ("SELECT SCALE(a) + scale(y) AS o FROM __THIS__", [SCALE]),
        ("SELECT affine(x, least(y, 1.0), (a + 1) * 2) AS o FROM __THIS__", [AFFINE]),
        (
            "SELECT o FROM (SELECT scale(x) AS o FROM __THIS__) AS d WHERE o > 0",
            [SCALE],
        ),
        ("SELECT a FROM __THIS__ WHERE scale(x) BETWEEN -10 AND 10", [SCALE]),
        (
            "SELECT CASE WHEN a > 0 THEN scale(x) ELSE scale(y) END AS o FROM __THIS__",
            [SCALE],
        ),
        ("SELECT twice(x) AS o FROM __THIS__", [SCALE, TWICE]),
        ("SELECT scale(scale(scale(x))) AS o FROM __THIS__", [SCALE]),
        ("SELECT kw(a) AS o, kw(b) AS p FROM __THIS__", [KEYWORD]),
        ("SELECT stats(a, b).lo AS lo, stats(a, b).mean AS m FROM __THIS__", [STATS]),
        ("SELECT stats(a, b) AS st FROM __THIS__", [STATS]),
        ("SELECT stats(b, b).tag AS t FROM __THIS__", [STATS]),
        # Reads of one call share its expansion; unaliased ones are named
        # after the call, as DuckDB names them.
        (
            "SELECT stats(a, b).lo, stats(a, b).mean, stats(b, a).tag FROM __THIS__",
            [STATS],
        ),
        (
            "SELECT stats(a, b).lo AS o FROM __THIS__ WHERE stats(a, b).mean > 0",
            [STATS],
        ),
        (
            "SELECT CASE WHEN a > 0 THEN stats(a, b).mean ELSE stats(b, a).mean END "
            "AS o FROM __THIS__",
            [STATS],
        ),
        (
            "SELECT stats(incr(b), b).lo AS o, stats(b, b).lo AS p FROM __THIS__",
            [INCR, STATS],
        ),
        ("SELECT wrap(a, b).lo AS o, wrap(a, b).m2 AS p FROM __THIS__", [STATS, WRAP]),
        (
            "SELECT o FROM (SELECT stats(b, b).mean AS o FROM __THIS__) AS d "
            "WHERE o > stats(1, 2).mean",
            [STATS],
        ),
        ("SELECT traps(b).v AS o FROM __THIS__", [TRAPS]),
        ("SELECT maybe(a).v AS o, maybe(b).w AS p FROM __THIS__", [MAYBE]),
        # An argument that traps (a = BIGINT max): both engines trap.
        ("SELECT incr(a) AS o FROM __THIS__", [INCR]),
        ("SELECT incr(incr(b)) AS o FROM __THIS__", [INCR]),
    ],
)
def test_a_sql_function_agrees_with_the_oracle(sql, fns):
    assert_parity(sql, ROWS, udfs=fns)


def test_a_sql_function_over_a_join_key_and_a_static():
    d = table({"k": "int", "w": "float"}, [{"k": 3, "w": 0.5}, {"k": 0, "w": -2.0}])
    sql = (
        "SELECT affine(t.x, d.w, 1.0) AS o FROM __THIS__ AS t "
        "LEFT JOIN d ON incr(t.b) = d.k"
    )
    assert_parity(sql, ROWS, statics={"d": d}, udfs=[AFFINE, INCR])


def test_a_sql_function_calling_an_extern():
    dbl = ExternFunction(
        "dbl", _schema(v=F64), F64, lambda v: None if v is None else (2 * v,)
    )
    f = SqlFunction("f", _schema(v=F64), F64, lambda v: S.fn("dbl", v) + S.lit(1.0))
    assert_parity("SELECT f(x) AS o FROM __THIS__", ROWS, udfs=[dbl, f])


@pytest.mark.parametrize(
    "sql, fns",
    [
        ("SELECT scale(x) AS o FROM __THIS__", [SCALE]),
        ("SELECT stats(a, b).mean AS m, stats(a, b).tag AS t FROM __THIS__", [STATS]),
        ("SELECT twice(y) AS o FROM __THIS__", [SCALE, TWICE]),
    ],
)
def test_register_is_the_definition_duckdb_runs(sql, fns):
    rows = ROWS.slice(0, 3)  # no trapping row
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=fns
    )
    got = fn.infer_arrow(rows)
    with Oracle() as o:
        for f in fns:
            f.register(o)
        o.load("__THIS__", rows)
        want = o.answer(sql)
    compare.assert_rows(got.to_pylist(), want.to_pylist(), ctx=sql)


@pytest.mark.parametrize("seed", range(40))
def test_a_random_body_agrees_with_the_oracle(seed):
    # The builder fuzz's trees over columns a, b, x, y, here the parameters.
    rng = random.Random(seed)  # noqa: S311
    tree = _tree(rng, 4, "num")
    f = SqlFunction(
        "f",
        _schema(a=pa.int64(), b=pa.int64(), x=F64, y=F64),
        F64,
        lambda *_: tree,
    )
    args = rng.choice(
        ["a, b, x, y", "b, a, y, x", "a + 1, b * 2, x - y, 0.5", "b, b, x, x"]
    )
    assert_parity(f"SELECT f({args}) AS o FROM __THIS__", ROWS, udfs=[f])


# ------------------------------------------------------------------ refusals


def test_a_body_reading_another_column_refuses():
    with pytest.raises(FunctionError, match='reads "z"'):
        SqlFunction("f", _schema(v=F64), F64, lambda v: v + S.col("z"))


def test_a_struct_body_names_every_field_in_order():
    ret = pa.struct([("p", F64), ("q", F64)])
    with pytest.raises(FunctionError, match="dict of"):
        SqlFunction("f", _schema(v=F64), ret, lambda v: {"q": v, "p": v})
    with pytest.raises(FunctionError, match="one expression"):
        SqlFunction("f", _schema(v=F64), F64, lambda v: {"p": v})


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: SqlFunction("f(", _schema(v=F64), F64, lambda v: v), "identifier"),
        (
            lambda: SqlFunction("f", _schema(v=F64, V=F64), F64, lambda v, w: v),
            "collide",
        ),
        (
            lambda: SqlFunction("f", _schema(v=F64), pa.list_(F64, 2), lambda v: v),
            "list",
        ),
    ],
)
def test_a_bad_declaration_refuses_at_construction(build, match):
    with pytest.raises(FunctionError, match=match):
        build()


def _build(sql, fns):
    return DuckDBInferFn(
        sql, row_tables={"__THIS__": ROWS.schema}, static_tables={}, udfs=fns
    )


def test_the_wrong_argument_count_refuses():
    with pytest.raises(ValueError, match="takes 1 arguments, got 2"):
        _build("SELECT scale(x, y) AS o FROM __THIS__", [SCALE])


def test_a_recursive_body_refuses():
    loop = SqlFunction("loop", _schema(v=F64), F64, lambda v: S.fn("loop", v))
    with pytest.raises(ValueError, match="expands more than"):
        _build("SELECT loop(x) AS o FROM __THIS__", [loop])


def test_a_builtin_name_refuses():
    abs_ = SqlFunction("abs", _schema(v=F64), F64, lambda v: v)
    with pytest.raises(ValueError, match="collides with the builtin"):
        _build("SELECT abs(x) AS o FROM __THIS__", [abs_])


def test_a_wide_struct_function_builds_once_per_call():
    # The native catalog's shape: n lanes, each a CASE over fitted groups
    # ending in error(), every lane read. Each call expands once, so 128
    # lanes over 3 groups build well inside the expansion budget.
    n, groups = 128, 3

    def body(iid, *xs):
        out = {}
        for i, x in enumerate(xs):
            lane = None
            for g in range(groups):
                v = (S.coalesce(x, S.lit(float("nan"))) - S.lit(1.0 + g)) / S.lit(
                    2.0 + i
                )
                lane = (
                    S.case(iid == S.lit(g), v)
                    if lane is None
                    else lane.when(iid == S.lit(g), v)
                )
            out[f"f{i}"] = lane.when(iid.isnull(), S.lit(None, F64)).otherwise(
                S.fn("error", S.lit("unknown id"))
            )
        return out

    takes = pa.schema([("iid", pa.int64())] + [(f"x{i}", F64) for i in range(n)])
    fn = SqlFunction("wide", takes, pa.struct([(f"f{i}", F64) for i in range(n)]), body)
    args = ", ".join(["iid"] + [f"x{i}" for i in range(n)])
    sql = "SELECT " + ", ".join(f"wide({args}).f{i} AS o{i}" for i in range(n))
    sql += " FROM __THIS__"
    rows = pa.table(
        {"iid": [0, 2, None], **{f"x{i}": [float(i), None, -1.5] for i in range(n)}}
    )
    infer = DuckDBInferFn(
        sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[fn]
    )
    got = infer.infer_arrow(rows).to_pylist()
    assert got[0]["o5"] == (5.0 - 1.0) / 7.0
    assert got[2]["o5"] is None
    with pytest.raises(Exception, match="unknown id"):
        infer.infer_arrow(rows.slice(0, 1).set_column(0, "iid", pa.array([7])))
