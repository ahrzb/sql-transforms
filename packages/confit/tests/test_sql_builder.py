"""The SQL builder (`confit.sql`): its text means what was built.

Three readings of one tree must agree: the rendered text run by DuckDB, the
`to_duckdb()` expression object evaluated by DuckDB, and the rendered text
served by confit (the campaign verdict).
"""

from __future__ import annotations

import decimal
import math
import random
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
from confit import ExternFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

a, b, x, y, s = (S.col(n) for n in ("a", "b", "x", "y", "s"))

ROWS = table(
    {"a": "int?", "b": "int?", "x": "float?", "y": "float?", "s": "str?"},
    [
        {"a": 3, "b": -2, "x": 0.5, "y": -1.25, "s": "it's"},
        {"a": None, "b": 7, "x": None, "y": 2.0, "s": None},
        {"a": 0, "b": 0, "x": -0.0, "y": 1e300, "s": ""},
        {"a": -9, "b": 4, "x": 3.5, "y": None, "s": 'q"x'},
    ],
)


def _same(u, v) -> bool:
    if isinstance(u, float) and isinstance(v, float):
        return repr(u) == repr(v)  # NaN equals NaN, -0.0 differs from 0.0
    return u == v


def _duck_text(e: S.Expr):
    return duckdb.sql(f"SELECT {e.sql()}").fetchone()[0]


def _duck_object(e: S.Expr):
    return duckdb.sql("SELECT 1").select(e.to_duckdb()).fetchone()[0]


CONSTANTS = [
    0.1,
    0.30000000000000004,
    -0.0,
    5e-324,
    1.7976931348623157e308,
    math.inf,
    -math.inf,
    1 / 3,
    0,
    -(2**63),
    2**63 - 1,
    True,
    False,
    "",
    "it's",
    'q"x',
    "ünï",
    decimal.Decimal("1.50"),
    decimal.Decimal("-12345678901234567.89"),
]


@pytest.mark.parametrize("v", CONSTANTS, ids=repr)
def test_a_constant_renders_its_own_value(v):
    e = S.lit(v)
    assert _same(_duck_text(e), v)
    assert _same(_duck_object(e), v)


def test_nan_renders_as_nan():
    assert math.isnan(_duck_text(S.lit(math.nan)))


@pytest.mark.parametrize(
    "v, t",
    [
        (None, pa.int64()),
        (None, "VARCHAR"),
        (7, pa.int8()),
        (7, pa.float64()),
        (1.5, "FLOAT"),
    ],
)
def test_a_typed_constant_has_its_type(v, t):
    e = S.lit(v, t)
    got = duckdb.sql(f"SELECT typeof({e.sql()})").fetchone()[0]
    assert got == S.type_name(t)


def test_text_keeps_double_arithmetic_where_duckdbs_str_does_not():
    e = S.lit(0.1) + S.lit(0.2)
    assert _duck_text(e) == 0.30000000000000004
    # The motivating loss: DuckDB's own rendering reads back as DECIMAL.
    lossy = str(duckdb.ConstantExpression(0.1) + duckdb.ConstantExpression(0.2))
    assert duckdb.sql(f"SELECT {lossy}").fetchone()[0] != 0.30000000000000004


HAND = [
    (a + b) * S.lit(2),
    a // S.lit(2) + a % S.lit(3),
    x / y,
    -x + S.lit(1.5),
    x**2,
    S.case(a.isnull(), S.lit(0)).when(a > 0, a).otherwise(-a),
    S.case(x > y, S.lit("gt")),
    S.coalesce(a, b, S.lit(0)),
    S.fn("least", a, b, S.lit(1)),
    S.fn("abs", x),
    a.isin(S.lit(3), S.lit(0), S.lit(None, pa.int64())),
    a.isnotin(S.lit(3)),
    x.between(S.lit(0.0), S.lit(1.0)),
    (a > 0) & (b < 0) | ~(x >= y),
    a.is_not_distinct_from(S.lit(None, pa.int64())),
    a.is_distinct_from(b),
    s.concat(S.lit("!")),
    s.isnull(),
    s.isnotnull(),
    a.cast(pa.float64()) + x,
    x.cast("VARCHAR"),
    a == b,
    a != b,
]


@pytest.mark.parametrize("e", HAND, ids=lambda e: e.sql())
def test_text_and_object_agree_in_duckdb(e):
    rel = duckdb.arrow(ROWS)
    by_text = [r[0] for r in duckdb.sql(f"SELECT {e.sql()} FROM rel").fetchall()]
    by_object = [r[0] for r in rel.select(e.to_duckdb()).fetchall()]
    assert all(_same(u, v) for u, v in zip(by_text, by_object, strict=True))


@pytest.mark.parametrize("e", HAND, ids=lambda e: e.sql())
def test_confit_serves_what_was_built(e):
    assert_parity(S.select(e.alias("o")).from_("__THIS__").sql(), ROWS)


def test_a_query_with_a_join_and_a_where():
    d = table({"k": "int", "v": "str"}, [{"k": 3, "v": "three"}, {"k": 0, "v": "zero"}])
    q = (
        S.select(S.col("t", "a").alias("a"), S.col("d", "v").alias("v"))
        .from_("__THIS__", "t")
        .join("d", S.col("t", "a") == S.col("d", "k"), how="left", alias="d")
        .where(S.col("t", "b") < 5)
        .where(S.col("t", "x").isnotnull())
    )
    assert_parity(q.sql(), ROWS, statics={"d": d})


def test_a_function_passed_in_udfs_is_called_by_name():
    double = ExternFunction(
        "dbl",
        pa.schema([("x", pa.float64())]),
        pa.float64(),
        lambda v: None if v is None else (2 * v,),
    )
    q = S.select(S.fn("dbl", x + 1.0).alias("o")).from_("__THIS__")
    assert_parity(q.sql(), ROWS, udfs=[double])


# ------------------------------------------------------------------ random


def _tree(rng: random.Random, depth: int, kind: str) -> S.Expr:
    """A random well-typed tree: kind is "num" (DOUBLE or BIGINT, mixed
    freely, as DuckDB unifies them) or "bool"."""
    leaf = depth <= 0 or rng.random() < 0.3
    if kind == "bool":
        if leaf:
            return rng.choice([a.isnull(), x > y, a <= b, S.lit(rng.random() < 0.5)])
        pick = rng.randrange(6)
        n = lambda: _tree(rng, depth - 1, "num")  # noqa: E731
        q = lambda: _tree(rng, depth - 1, "bool")  # noqa: E731
        if pick == 0:
            return q() & q()
        if pick == 1:
            return q() | q()
        if pick == 2:
            return ~q()
        if pick == 3:
            return rng.choice([n() < n(), n() == n(), n() >= n()])
        if pick == 4:
            return n().between(n(), n())
        return n().isin(n(), n())
    if leaf:
        return rng.choice(
            [
                a,
                b,
                x,
                y,
                S.lit(rng.choice([0, 1, -3, 7])),
                S.lit(rng.choice([0.5, -2.0, 1e-3])),
            ]
        )
    n = lambda: _tree(rng, depth - 1, "num")  # noqa: E731
    pick = rng.randrange(7)
    if pick == 0:
        return n() + n()
    if pick == 1:
        return n() - n()
    if pick == 2:
        return n() * n()
    if pick == 3:
        return -n()
    if pick == 4:
        return S.case(_tree(rng, depth - 1, "bool"), n()).otherwise(n())
    if pick == 5:
        return S.coalesce(n(), n())
    return n().cast(pa.float64())


@pytest.mark.parametrize("seed", range(60))
def test_a_random_tree_agrees_everywhere(seed):
    rng = random.Random(seed)  # noqa: S311
    e = _tree(rng, 4, rng.choice(["num", "bool"]))
    test_text_and_object_agree_in_duckdb(e)
    assert_parity(S.select(e.alias("o")).from_("__THIS__").sql(), ROWS)


# ------------------------------------------------------------------ shape


def test_walk_is_preorder_over_the_tree():
    e = S.case(a.isnull(), S.lit(0)).otherwise(a + 1)
    kinds = [type(n).__name__ for n in e.walk()]
    assert kinds == ["Case", "Postfix", "Column", "Const", "BinOp", "Column", "Const"]


def test_identifiers_are_quoted():
    assert S.col("t", 'we"ird').sql() == '"t"."we""ird"'


@pytest.mark.parametrize(
    "build, err",
    [
        (lambda: bool(a > 1), TypeError),
        (lambda: S.lit(None), ValueError),
        (lambda: S.lit(object()), TypeError),
        (lambda: S.fn("drop table", a), ValueError),
        (lambda: S.case(a > 1, a).otherwise(b).when(a > 2, b), ValueError),
        (lambda: S.select(a).from_("t").join("d", how="left"), ValueError),
        (lambda: S.select(a).from_("t").join("d", a == b, how="cross"), ValueError),
        (lambda: S.select(a).sql(), ValueError),
        (lambda: hash(a), TypeError),
    ],
)
def test_a_malformed_build_refuses(build, err):
    with pytest.raises(err):
        build()
