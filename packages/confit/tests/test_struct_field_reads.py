"""Field reads over built structs: `struct_pack` and CASE-valued structs.

DuckDB builds every field of a struct_pack, so reading one field still
fires a sibling's trap: `(struct_pack(p := a, q := <overflow>)).p` traps
(measured on 1.5.5, optimizer off). A field read over a CASE reads the field
of the arm taken, and a NULL arm is a NULL struct whose field is NULL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import ExternFunction, SqlFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

ROWS = table(
    {"a": "int?", "x": "float?", "s": "str?"},
    [
        {"a": 2, "x": 1.5, "s": "p"},
        {"a": None, "x": None, "s": None},
        {"a": 0, "x": -1.0, "s": ""},
    ],
)
SAFE = ROWS.slice(1, 2)
BIG = "a * 9223372036854775807"


def q(expr: str) -> str:
    return f"SELECT {expr} AS o FROM __THIS__"


@pytest.mark.parametrize(
    "expr",
    [
        f"(struct_pack(p := a, q := {BIG})).p",
        f"(struct_pack(q := {BIG}, p := a)).p",
        f"struct_extract(struct_pack(p := a, q := {BIG}), 'p')",
        f"(struct_pack(p := a, q := struct_pack(r := x, z := {BIG}))).q.r",
        f"(CASE WHEN a > 0 THEN struct_pack(p := a, q := {BIG}) END).p",
    ],
)
def test_a_sibling_field_still_traps(expr):
    assert_parity(q(expr), ROWS, trap="Overflow")
    assert_parity(q(expr), SAFE)


def _lane(value: str, msg: str = "bad id", cond: str = "a = 0") -> str:
    return f"CASE WHEN {cond} THEN {value} ELSE error('{msg}') END"


@pytest.mark.parametrize(
    "expr, trap",
    [
        # Siblings kept only for their traps, reduced to their conditions:
        # equal ones are kept once, different messages keep field order.
        (
            f"(struct_pack(p := x, q := {_lane('x')}, r := {_lane('x * 2')})).p",
            "bad id",
        ),
        (
            f"(struct_pack(p := x, q := {_lane('x', 'm1')}, "
            f"r := {_lane('x', 'm2')})).p",
            "m1",
        ),
        (
            f"(struct_pack(q := {_lane('x', 'm2')}, p := x, "
            f"r := {_lane('x', 'm1')})).p",
            "m2",
        ),
        # A condition that traps still traps, before its error() arm.
        (f"(struct_pack(p := x, q := {_lane('x', cond=f'{BIG} > 0')})).p", "Overflow"),
        # A trapping value in a taken arm is kept as it is.
        (f"(struct_pack(p := x, q := CASE WHEN a > 0 THEN {BIG} END)).p", "Overflow"),
    ],
)
def test_a_sibling_kept_for_its_traps(expr, trap):
    assert_parity(q(expr), ROWS, trap=trap)
    assert_parity(q(expr), ROWS.filter(pa.compute.equal(ROWS["a"], 0)))


@pytest.mark.parametrize(
    "expr",
    [
        "(struct_pack(p := a, q := x)).p",
        "(struct_pack(p := a, q := s)).q",
        "(struct_pack(q := struct_pack(r := x))).q.r",
        "(CASE WHEN a IS NULL THEN NULL ELSE struct_pack(p := a, q := x) END).p",
        "(CASE WHEN a IS NULL THEN NULL ELSE struct_pack(p := a, q := x) END).q + 1",
        "(CASE WHEN a = 0 THEN struct_pack(p := a) ELSE struct_pack(p := x) END).p",
        "(CASE WHEN a = 0 THEN struct_pack(p := 1) ELSE struct_pack(p := 2.5) END).p",
        "(CASE WHEN a = 0 THEN struct_pack(p := 300::SMALLINT) "
        "ELSE struct_pack(p := a::TINYINT) END).p",
        "(CASE a WHEN 2 THEN struct_pack(p := s) END).p",
        "struct_extract(CASE WHEN a = 0 THEN struct_pack(p := a) END, 'p')",
        "(CASE WHEN a = 0 THEN struct_pack(p := a, q := struct_pack(r := x)) END).q.r",
        "(CASE WHEN a = 0 THEN struct_pack(p := a) "
        "ELSE CASE WHEN a = 2 THEN struct_pack(p := a + 1) END END).p",
    ],
)
def test_a_field_read_agrees_with_the_oracle(expr):
    assert_parity(q(expr), ROWS)


@pytest.mark.parametrize(
    "expr",
    [
        # DuckDB evaluates a struct that names no column when it binds the
        # read. A NULL one makes the read a bare NULL: INTEGER alone, of
        # the type beside it in a context that unifies.
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x') END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x') END).p || 'y'",
        "(CASE WHEN 1 = 1 THEN NULL ELSE struct_pack(p := 2.5) END).p + 1",
        "coalesce((CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 1.5) END).p, 2.5)",
        "struct_extract(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x') END, 'p')",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(q := struct_pack(r := 'x')) END)"
        ".q.r",
        "(CASE WHEN NULL THEN struct_pack(p := 'x') END).p",
        "(CASE WHEN 'ab' LIKE 'a_' THEN NULL ELSE struct_pack(p := 'x') END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := x + NULL) END).p",
        # The arm not taken does not run, so it does not trap.
        "(CASE WHEN TRUE THEN NULL"
        " ELSE struct_pack(p := 9223372036854775807 + 1) END).p",
        # Not evaluated: a column anywhere in the struct, also one that
        # cannot change its value, or a struct that is there.
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x', q := x) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := coalesce(2.0, x)) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := TRUE OR x > 0) END).p",
        "(CASE WHEN 1 = 1 THEN NULL WHEN a > 0 THEN struct_pack(p := 'x') END).p",
        "(CASE WHEN FALSE THEN NULL ELSE struct_pack(p := 'x') END).p",
        # A call is a NULL constant to DuckDB's binder when an argument has
        # type SQLNULL, or names no column and is NULL; then its columns do
        # not count. A cast to VARCHAR is no longer SQLNULL, so the column
        # under it counts (nightly seed 4848122).
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x',"
        " q := levenshtein('a', CASE WHEN a > 0 THEN NULL ELSE NULL END)) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x',"
        " q := levenshtein(s, CAST(NULL AS VARCHAR))) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x',"
        " q := ((a + NULL) * a) - a) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x',"
        " q := nullif(NULL, a)) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x', q := nullif(a + NULL, a))"
        " END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x', q := levenshtein('a',"
        " CAST(CASE WHEN a > 0 THEN NULL ELSE NULL END AS VARCHAR))) END).p",
        "(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := 'x',"
        " q := 1 >= levenshtein('a', CAST(CASE WHEN a > 0 THEN NULL END AS VARCHAR)))"
        " END).p",
    ],
)
def test_a_read_of_a_struct_that_folds_to_null(expr):
    assert_parity(q(expr), ROWS)


def test_a_lateral_alias_over_constants_folds_with_the_struct():
    sql = (
        "SELECT 1.5 AS z, (CASE WHEN TRUE THEN NULL ELSE struct_pack(p := z) END).p"
        " AS o FROM __THIS__"
    )
    assert_parity(sql, ROWS)


def test_a_pure_extern_over_constants_folds_with_the_struct():
    # DuckDB evaluates a call of a function without side effects at bind
    # time when its arguments are constants.
    one = ExternFunction(
        "one",
        pa.schema([("v", pa.float64())]),
        pa.float64(),
        lambda v: None if v is None else (v,),
    )
    for p in ("one(1.5)", "one(x)"):
        sql = q(f"(CASE WHEN TRUE THEN NULL ELSE struct_pack(p := {p}) END).p")
        assert_parity(sql, ROWS, udfs=[one])


def test_a_pure_extern_in_the_condition_folds_with_the_struct():
    # DuckDB evaluates the condition at bind time, the call included: a
    # call that raises leaves the read to run, and trap, per row. Ours
    # panicked running the call (nightly seed 4704064).
    one = ExternFunction(
        "one",
        pa.schema([("v", pa.float64())]),
        pa.float64(),
        lambda v: None if v is None else (v,),
    )

    def raises(v):
        raise ValueError("boom")

    boom = ExternFunction(
        "boom", pa.schema([("v", pa.float64())]), pa.float64(), raises
    )
    g = SqlFunction(
        "g",
        pa.schema([("v", pa.float64())]),
        pa.struct([("p", pa.float64())]),
        lambda v: {"p": v},
        null_when=lambda v: v > S.lit(1.0),
    )
    for cond in ("one(1.5) > 1", "one(1.5) < 1", "one(NULL) IS NULL", "one(x) > 1"):
        sql = q(f"(CASE WHEN {cond} THEN NULL ELSE struct_pack(p := 'x') END).p")
        assert_parity(sql, ROWS, udfs=[one])
    for expr in (
        "(g(one(1.5))).p",
        "(g(one(0.5))).p",
        "g(one(1.5)).p",
        "(g(one(1.5))).p || 'y'",
    ):
        assert_parity(q(expr), ROWS, udfs=[one, g])
    sql = q("(CASE WHEN boom(1.5) > 1 THEN NULL ELSE struct_pack(p := 'x') END).p")
    assert_parity(sql, ROWS, udfs=[boom], trap="boom")


def test_a_null_struct_from_a_sql_function_over_constants():
    f = SqlFunction(
        "g",
        pa.schema([("iid", pa.int64()), ("v", pa.float64())]),
        pa.struct([("p", pa.float64()), ("q", pa.float64())]),
        lambda iid, v: {"p": v, "q": v * S.lit(2.0)},
        null_when=lambda iid, v: iid.isnull(),
    )
    for sql in (
        "SELECT g(NULL, 1.5).p AS o FROM __THIS__",
        "SELECT g(NULL, 1.5).q + 1 AS o FROM __THIS__",
        "SELECT g(NULL, a + NULL).p AS o FROM __THIS__",
        "SELECT g(1, 1.5).p AS o FROM __THIS__",
        "SELECT g(NULL, x).p AS o FROM __THIS__",
    ):
        assert_parity(sql, ROWS, udfs=[f])


def test_a_struct_sql_function_keeps_its_sibling_traps():
    f = SqlFunction(
        "pair",
        pa.schema([("n", pa.int64())]),
        pa.struct([("lo", pa.int64()), ("hi", pa.int64())]),
        lambda n: {"lo": n - S.lit(1), "hi": n * S.lit(9223372036854775807)},
    )
    assert_parity(
        "SELECT pair(a).lo AS o FROM __THIS__", ROWS, udfs=[f], trap="Overflow"
    )
    assert_parity("SELECT pair(a).lo AS o FROM __THIS__", SAFE, udfs=[f])


def test_a_null_struct_from_a_sql_function():
    # The native catalog's shape: a NULL id answers a NULL struct, read
    # whole or by field.
    f = SqlFunction(
        "g",
        pa.schema([("iid", pa.int64()), ("v", pa.float64())]),
        pa.struct([("p", pa.float64()), ("q", pa.float64())]),
        lambda iid, v: {"p": v, "q": v * S.lit(2.0)},
        null_when=lambda iid, v: iid.isnull(),
    )
    for sql in (
        "SELECT g(a, x).p AS p, g(a, x).q AS q FROM __THIS__",
        "SELECT g(a, x) AS st FROM __THIS__",
    ):
        assert_parity(sql, ROWS, udfs=[f])


EDGES = table(
    {"x": "float?"},
    [
        {"x": v}
        for v in (2.0, -1.0, 0.0, -0.0, float("nan"), float("inf"), float("-inf"), None)
    ],
)
POSITIVE = table({"x": "float?"}, [{"x": 2.0}, {"x": float("nan")}, {"x": None}])


@pytest.mark.parametrize(
    "p",
    [
        "-x * 2.0",
        "exp(x * 1000.0)",
        "CASE WHEN x <= 0 THEN NULL ELSE ln(x) END",
        "CASE WHEN x > 0 THEN ln(x) END",
        "CASE WHEN 0 >= x THEN NULL ELSE exp(0.5 * ln(x)) / 0.5 END",
        "CASE WHEN x < 0 THEN 0.0 ELSE sqrt(x) END",
        # Guards that leave the domain error reachable: the read still traps.
        "ln(x)",
        "CASE WHEN x < 0 THEN NULL ELSE ln(x) END",
        "CASE WHEN x >= 0 THEN ln(x) END",
        "CASE WHEN x < -1 THEN NULL ELSE sqrt(x) END",
        # sin/cos/tan raise exactly on +-inf; round(DOUBLE) never.
        "CASE WHEN abs(x) = CAST('inf' AS DOUBLE) THEN NULL ELSE sin(x) END",
        "CASE WHEN abs(x) < CAST('inf' AS DOUBLE) THEN cos(x) END",
        "CASE WHEN abs(x) <> CAST('inf' AS DOUBLE) THEN tan(x) END",
        "round(x) * 2.0",
        "sin(x)",
        "CASE WHEN x = CAST('inf' AS DOUBLE) THEN NULL ELSE cos(x) END",
    ],
)
def test_a_sibling_raises_exactly_where_duckdb_does(p):
    # Siblings proven unable to raise (plan.rs `can_trap`) are not
    # evaluated by a field read; the rest still raise.
    sql = f"SELECT (struct_pack(p := {p}, q := x)).q AS o FROM __THIS__"
    assert_parity(sql, EDGES)
    assert_parity(sql, POSITIVE)


def test_a_null_struct_keeps_its_sibling_traps():
    # null_when wraps the body in a CASE whose struct_pack arm reads as a
    # call of its own (calls.rs `split_case_arms`): the read field still
    # carries its siblings' traps, only on rows that build the struct.
    f = SqlFunction(
        "h",
        pa.schema([("iid", pa.int64()), ("n", pa.int64())]),
        pa.struct([("lo", pa.int64()), ("hi", pa.int64())]),
        lambda iid, n: {"lo": n - S.lit(1), "hi": n * S.lit(9223372036854775807)},
        null_when=lambda iid, n: iid.isnull(),
    )
    for sql in (
        "SELECT h(a, a).lo AS o FROM __THIS__",
        "SELECT h(a, a).lo AS o, h(a, a).hi AS p FROM __THIS__",
        "SELECT h(a, a) AS st FROM __THIS__",
        # A lateral alias named by the arguments rebinds the call.
        "SELECT a + 1 AS a, h(a, a).lo AS o, h(a, a).hi AS p FROM __THIS__",
        "SELECT a + 1 AS b, h(b, a).lo AS o, h(a, b).lo AS p FROM __THIS__",
    ):
        assert_parity(sql, ROWS, udfs=[f], trap="Overflow")
        assert_parity(sql, SAFE, udfs=[f])


@pytest.mark.parametrize(
    "expr, why",
    [
        (f"(struct_pack(p := NULL, q := {BIG})).p", "bare-NULL struct field"),
        (
            "(CASE WHEN a = 0 THEN struct_pack(p := a) ELSE struct_pack(q := a) END).p",
            "key",
        ),
        ("__cf_seq(0, a)", "reserved identifier"),
    ],
)
def test_other_forms_refuse_by_name(expr, why):
    v = assert_parity(q(expr), SAFE, expect="REFUSED")
    assert why in v.detail
