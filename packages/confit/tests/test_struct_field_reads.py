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
from confit import SqlFunction
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
