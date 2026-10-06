"""A SQL function's lets: a value its body uses in several places is
declared once, bound once and, where it cannot trap, computed once per row.

`SqlFunction` declares every `confit.sql` node its body reads more than once
(`sql_lets`); DuckDB registers the body with each spelled out at every read
(`sql_body`). So each test here is also a test that the two forms mean the
same: confit serves the declared form, the oracle runs the spelled-out one.
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

from fuzz.parity import assert_parity, verdict  # noqa: E402

F64 = pa.float64()
X = pa.schema([("x", F64)])
ROWS = pa.table(
    {
        "x": [
            0.5,
            None,
            0.0,
            -0.0,
            -3.25,
            1e300,
            -1e300,
            math.inf,
            -math.inf,
            math.nan,
            7.0,
        ]
    },
    schema=X,
)


def doubling(steps: int):
    """`h_{i+1} = h_i * 0.5 + h_i * 0.25`, each step reading the one before
    twice: spelled out, 2^steps reads of `x`."""

    def body(x):
        h = x
        for _ in range(steps):
            h = h * S.lit(0.5) + h * S.lit(0.25)
        return h

    return SqlFunction("dbl", X, F64, body)


def test_a_value_read_twice_is_declared_once():
    fn = doubling(3)
    # `h1` and `h2`: `h0` is the parameter (a leaf costs less to repeat),
    # and the root reads `h3` once.
    assert len(fn.sql_lets) == 2
    assert fn.sql_let_body == (
        "CAST(((__cf_let(1) * CAST('0.5' AS DOUBLE)) + "
        "(__cf_let(1) * CAST('0.25' AS DOUBLE))) AS DOUBLE)"
    )
    # DuckDB's definition spells every read out.
    assert fn.sql_body.count('"x"') == 2**3
    assert "__cf_let" not in fn.sql_body


def test_a_body_without_a_value_read_twice_has_no_lets():
    fn = SqlFunction("inc", X, F64, lambda x: x + S.lit(1.0))
    assert not hasattr(fn, "sql_lets")
    assert (
        fn.sql_body
        == """CAST((CAST("x" AS DOUBLE) + CAST('1.0' AS DOUBLE)) AS DOUBLE)"""
    )


@pytest.mark.parametrize("steps", [1, 4, 10])
def test_a_doubling_recurrence_agrees_with_the_oracle(steps):
    fn = doubling(steps)
    assert_parity("SELECT dbl(x) AS o FROM __THIS__", ROWS, udfs=[fn])


def test_a_long_doubling_recurrence_builds_in_linear_time():
    # Spelled out, 60 steps are 2^60 reads: past the expansion cap from 17.
    fn = doubling(60)
    t0 = time.perf_counter()
    f = DuckDBInferFn(
        "SELECT dbl(x) AS o FROM __THIS__",
        row_tables={"__THIS__": X},
        static_tables={},
        udfs=[fn],
    )
    took = time.perf_counter() - t0
    got = f.infer_arrow(ROWS).column("o").to_pylist()
    for x, o in zip(ROWS.column("x").to_pylist(), got, strict=True):
        want = x
        if want is not None:
            for _ in range(60):
                want = want * 0.5 + want * 0.25
        assert repr(o) == repr(want), (x, o, want)
    assert took < 10, took


def trapping(steps: int):
    """`doubling` under a cast that can trap at every step: a value that
    can trap binds again at every read, where it stands."""

    def body(x):
        h = x
        for _ in range(steps):
            h = (h * S.lit(0.5) + h * S.lit(0.25)).cast("BIGINT")
        return h

    return SqlFunction("tdb", X, pa.int64(), body)


def test_a_recurrence_that_can_trap_agrees_with_the_oracle():
    fn = trapping(3)
    assert fn.sql_lets
    sql = "SELECT tdb(x) AS o FROM __THIS__"
    rows = pa.table({"x": [0.5, None, 7.0, -3.25, 1e6, -0.0]}, schema=X)
    assert_parity(sql, rows, udfs=[fn])
    big = pa.table({"x": [1.0, 1e300]}, schema=X)
    assert_parity(sql, big, udfs=[fn], trap="out of range|Conversion|Overflow")


def _refuses(sql, fn, match, **tables):
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=match):
        DuckDBInferFn(
            sql,
            row_tables={"__THIS__": tables.get("rows", X)},
            static_tables=tables.get("statics", {}),
            udfs=[fn],
        )
    assert time.perf_counter() - t0 < 10


def test_a_long_recurrence_that_can_trap_refuses_at_the_expansion_cap():
    # 2^40 binds spelled out: the expansion cap refuses it, as it refuses
    # the text.
    _refuses("SELECT tdb(x) AS o FROM __THIS__", trapping(40), "expands past")


def test_a_long_recurrence_read_outside_the_projection_refuses_at_the_cap():
    # A WHERE and a join read the value itself, 2^40 nodes spelled out.
    fn = doubling(40)
    _refuses("SELECT x AS o FROM __THIS__ WHERE dbl(x) > 0", fn, "spells out past")
    _refuses("SELECT dbl(x) AS o, x FROM __THIS__ WHERE o > 0", fn, "spells out past")
    s = pa.table({"k": [1.0], "v": [1]})
    _refuses(
        "SELECT v FROM __THIS__ JOIN s ON dbl(x) = s.k",
        fn,
        "spells out past",
        statics={"s": s},
    )


def test_an_unaliased_read_is_named_after_its_text_spelled_out():
    fn = doubling(3)
    sql = "SELECT dbl(x), dbl(x) + 1 FROM __THIS__"
    names = [
        DuckDBInferFn(sql, row_tables={"__THIS__": X}, static_tables={}, udfs=[f])
        .infer_arrow(ROWS)
        .column_names
        for f in (fn, _SpelledOut(fn))
    ]
    assert names[0] == names[1]
    assert "__cf_let" not in names[0][0]


def test_an_unaliased_read_named_past_the_cap_asks_for_an_alias():
    _refuses("SELECT dbl(x) FROM __THIS__", doubling(40), "give it an alias")


def test_reads_outside_the_projection_count_toward_one_cap():
    # Each read is 3.1M nodes spelled out, under the cap alone; the text
    # spelled out passes it at the second, and so does the query here.
    _refuses(
        "SELECT x AS o FROM __THIS__ WHERE dbl(x) > 0 AND dbl(x) < 1",
        doubling(19),
        "spells out past",
    )


def test_unaliased_reads_count_toward_one_cap():
    # One name spelled out is 2.6M tokens, under the cap; two pass it, as
    # their text does in the expansion of the spelled-out body.
    fn = doubling(16)
    DuckDBInferFn(
        "SELECT dbl(x) FROM __THIS__",
        row_tables={"__THIS__": X},
        static_tables={},
        udfs=[fn],
    )
    _refuses("SELECT dbl(x), dbl(x) + 1 FROM __THIS__", fn, "give it an alias")


def test_a_let_read_in_a_join_residual():
    fn = doubling(4)
    s = pa.table({"k": [1, 2], "w": [0.1, 5.0]})
    rows = pa.table({"x": [0.5, -2.0, None, 8.0], "k": [1, 1, 2, 3]})
    assert_parity(
        "SELECT w, dbl(x) AS o FROM __THIS__ JOIN s "
        "ON __THIS__.k = s.k AND dbl(x) < s.w",
        rows,
        statics={"s": s},
        udfs=[fn],
    )
    # On one side only, the residual refuses as the text spelled out does:
    # the check sees the value, not a read of it.
    sql = "SELECT w FROM __THIS__ JOIN s ON __THIS__.k = s.k AND dbl(x) > 0"
    kinds = [
        verdict(sql, rows, statics={"s": s}, udfs=[f]).klass
        for f in (fn, _SpelledOut(fn))
    ]
    assert kinds[0] == kinds[1], kinds


def _struct_fn(name, lanes, takes=X, null_when=None):
    fields = pa.struct(
        [(f"y{i}", F64) for i in range(len(lanes(*[S.col(n) for n in takes.names])))]
    )
    return SqlFunction(
        name,
        takes,
        fields,
        lambda *xs: {f"y{i}": e for i, e in enumerate(lanes(*xs))},
        null_when=null_when,
    )


def _reads(fn, args="x"):
    call = f"{fn.name}({args})"
    names = [fn.returns.field(i).name for i in range(fn.returns.num_fields)]
    return "SELECT " + ", ".join(f"{call}.{n} AS {n}" for n in names) + " FROM __THIS__"


def test_lanes_that_share_a_value_agree_with_the_oracle():
    def lanes(x):
        m = x % S.lit(2.5)
        p = m + S.case(m < S.lit(0.0), S.lit(2.5)).otherwise(S.lit(0.0))
        return [p * p, p - S.lit(1.0), S.case(p > S.lit(1.0), p).otherwise(-p)]

    for null_when in (None, lambda x: x.isnull()):
        fn = _struct_fn("per", lanes, null_when=null_when)
        assert fn.sql_lets
        assert_parity(_reads(fn), ROWS, udfs=[fn])


def test_a_let_that_can_trap_traps_where_it_is_read():
    # `big` fails past BIGINT's range; read only under `x < 0`, it traps
    # only on a negative row out of range, and on every read there.
    def body(x):
        big = (x * S.lit(1e10)).cast("BIGINT")
        return S.case(x < S.lit(0.0), big + big).otherwise(S.lit(0, "BIGINT"))

    fn = SqlFunction("trp", X, pa.int64(), body)
    assert fn.sql_lets
    sql = "SELECT trp(x) AS o FROM __THIS__"
    ok = pa.table({"x": [0.5, None, 1e300, math.inf, -3.0]}, schema=X)
    assert_parity(sql, ok, udfs=[fn])
    bad = pa.table({"x": [0.5, -1e300]}, schema=X)
    assert_parity(sql, bad, udfs=[fn], trap="out of range|Conversion|Overflow")


def test_a_guarded_let_keeps_its_guard():
    # `ln(x)` cannot trap under `x > 0`; read beside it unguarded too, it
    # traps on the rows that reach that read.
    def body(x):
        lx = S.fn("ln", x)
        return S.case(x > S.lit(0.0), lx * lx).otherwise(S.lit(0.0))

    fn = SqlFunction("lng", X, F64, body)
    assert_parity("SELECT lng(x) AS o FROM __THIS__", ROWS, udfs=[fn])

    def unguarded(x):
        lx = S.fn("ln", x)
        return S.case(x > S.lit(0.0), lx * lx).otherwise(lx)

    fn = SqlFunction("lnu", X, F64, unguarded)
    sql = "SELECT lnu(x) AS o FROM __THIS__"
    assert_parity(sql, pa.table({"x": [0.5, None, 2.0]}, schema=X), udfs=[fn])
    assert_parity(
        sql, pa.table({"x": [0.5, -1.0]}, schema=X), udfs=[fn], trap="logarithm"
    )


def test_a_nested_function_read_by_field_beside_a_let():
    inner = _struct_fn("inr", lambda x: [x * S.lit(2.0), x * x])

    def lanes(x):
        call = S.fn("inr", x)
        a = call.field("y0") + call.field("y1")
        return [a * a, a + S.lit(1.0)]

    fn = _struct_fn("otr", lanes)
    assert fn.sql_lets
    assert_parity(_reads(fn), ROWS, udfs=[inner, fn])


def test_a_lateral_alias_in_the_arguments():
    def lanes(x, w):
        s = x * x + w * w
        return [x / s, w / s]

    takes = pa.schema([("x", F64), ("w", F64)])
    fn = _struct_fn("lat", lanes, takes=takes)
    rows = pa.table({"x": [0.5, 2.0, None], "v": [1.0, -3.0, 4.0]})
    sql = (
        "SELECT v * 2 AS w, lat(x, w).y0 AS a, lat(x, w).y1 AS b, "
        "v AS w2, lat(x, v).y1 AS c FROM __THIS__"
    )
    assert_parity(sql, rows, udfs=[fn])


def test_a_let_read_in_a_where_and_a_join_key():
    fn = doubling(4)
    rows = pa.table({"x": [0.5, -2.0, None, 8.0]}, schema=X)
    assert_parity("SELECT dbl(x) AS o FROM __THIS__ WHERE dbl(x) > 0", rows, udfs=[fn])
    s = pa.table({"k": [0.5 * 0.75**4, 8.0 * 0.75**4], "v": [1, 2]})
    assert_parity(
        "SELECT v, dbl(x) AS o FROM __THIS__ LEFT JOIN s ON dbl(x) = s.k",
        rows,
        statics={"s": s},
        udfs=[fn],
    )


def test_a_let_under_a_derived_table():
    fn = doubling(5)
    assert_parity(
        "SELECT o * 2 AS p FROM (SELECT dbl(x) AS o, dbl(x) + 1 AS q FROM __THIS__)",
        ROWS,
        udfs=[fn],
    )


def test_a_let_in_a_many_shaped_query():
    fn = doubling(4)
    s = pa.table({"k": [1, 1, 2], "v": [10.0, 20.0, 30.0]})
    rows = pa.table({"x": [0.5, -2.0, None], "k": [1, 2, 3]})
    assert_parity(
        "SELECT v, dbl(x) + v AS o FROM __THIS__ JOIN s ON __THIS__.k = s.k",
        rows,
        statics={"s": s},
        udfs=[fn],
        shape="many",
    )


class _SpelledOut:
    """`fn` as a protocol object with only its spelled-out body."""

    def __init__(self, fn):
        self.name, self.takes, self.returns = fn.name, fn.takes, fn.returns
        self.sql_body = fn.sql_body

    def register(self, con):
        con.execute(f'CREATE MACRO "{self.name}"(x) AS {self.sql_body}')


@pytest.mark.parametrize("deep", [150, 330, 340, 480])
def test_a_let_read_deeper_refuses_as_the_spelled_out_text_does(deep):
    # A let 300 levels deep, read once near the top and once `deep` levels
    # down: the second read is where the text spelled out passes the depth
    # limit, or does not. Spelled out, the parser's own nesting limit can
    # refuse first; declared, the let parses apart, and the binder refuses.
    def body(x):
        v = x
        for _ in range(300):
            v = v + S.lit(1.0)
        e = v
        for _ in range(deep):
            e = -e
        return v + e

    fn = SqlFunction("dep", X, F64, body)
    assert fn.sql_lets
    sql = "SELECT dep(x) AS o FROM __THIS__"

    def builds(f) -> bool:
        try:
            DuckDBInferFn(sql, row_tables={"__THIS__": X}, static_tables={}, udfs=[f])
        except ValueError as e:
            assert "expression depth" in str(e) or "recursion limit" in str(e), e
            return False
        return True

    assert builds(fn) == builds(_SpelledOut(fn))
