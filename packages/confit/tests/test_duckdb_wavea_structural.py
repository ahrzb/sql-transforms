"""Wave-A structural tails vs the duckdb oracle.

Pins: packages/confit/docs/specs/2026-07-28-waveA-structural-tails.md +
pins-waveA/*.json — structs-as-lanes, FROM colon alias, reverse()
graphemes, COLUMNS(* REPLACE), paren-less * REPLACE, NULL regex
patterns, lazy non-scalar rejection.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from test_duckdb_interpreter import duck_check

Inner = pa.struct([pa.field("i", pa.int64()), pa.field("j", pa.int64())])
S = pa.schema([pa.field("x", pa.int64(), nullable=False), pa.field("a", Inner)])
S_ROWS = [
    {"x": 9, "a": {"i": 1, "j": 2}},
    {"x": 8, "a": None},
    {"x": 7, "a": {"i": 3, "j": None}},
]


def _struct_check(sql: str, want_cols: list[str], want_rows: list[tuple]):
    """Engine output vs hand-pinned DuckDB values (the oracle rows come
    from the pins run; duck_check's schema DSL has no struct spelling)."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": S}, static_tables={})
    got = fn.infer_rows(S_ROWS)
    assert [list(r.keys()) for r in got] == [want_cols] * len(want_rows)
    assert [tuple(r.values()) for r in got] == want_rows


def test_struct_star_and_field_access():
    # pins-waveA/struct-star.json: bare field names, declaration order,
    # NULL struct -> NULL per field.
    _struct_check(
        "SELECT a.* FROM __THIS__",
        ["i", "j"],
        [(1, 2), (None, None), (3, None)],
    )
    _struct_check("SELECT a.* EXCLUDE(J) FROM __THIS__", ["i"], [(1,), (None,), (3,)])
    _struct_check(
        "SELECT a.* REPLACE(a.i + 3 AS i) FROM __THIS__",
        ["i", "j"],
        [(4, 2), (None, None), (6, None)],
    )
    # REPLACE expr sees other table columns; alias case wins the name.
    _struct_check(
        "SELECT a.* REPLACE(x + a.i AS I) FROM __THIS__",
        ["I", "j"],
        [(10, 2), (None, None), (10, None)],
    )
    _struct_check(
        "SELECT x, a.i, __THIS__.a.j FROM __THIS__",
        ["x", "i", "j"],
        [(9, 1, 2), (8, None, None), (7, 3, None)],
    )


def test_struct_rejections_are_named():
    def rejects(sql, needle):
        with pytest.raises(ValueError, match=needle):
            DuckDBInferFn(sql, row_tables={"__THIS__": S}, static_tables={})

    rejects("SELECT a FROM __THIS__", "whole value")
    rejects("SELECT * FROM __THIS__", "non-scalar")
    rejects("SELECT a.nope FROM __THIS__", "Could not find key")
    rejects("SELECT a.i.j FROM __THIS__", "not a struct")
    # Excluding the struct serves the rest.
    fn = DuckDBInferFn(
        "SELECT * EXCLUDE (a) FROM __THIS__",
        row_tables={"__THIS__": S},
        static_tables={},
    )
    assert [r["x"] for r in fn.infer_rows(S_ROWS)] == [9, 8, 7]


T = {"a": "int", "s": "str?"}
T_ROWS = [
    {"a": 3, "s": "hello"},
    {"a": -1, "s": None},
    {"a": 0, "s": "MotörHead"},
    {"a": 7, "s": "a\r\nb"},
    {"a": 5, "s": "\U0001f1fa\U0001f1f8\U0001f1eb"},
]


def test_from_colon_alias_vs_oracle():
    duck_check("SELECT * FROM b : __THIS__", T, T_ROWS)
    duck_check('SELECT * FROM "b" : __THIS__', T, T_ROWS)
    duck_check("SELECT b.a FROM b:__THIS__ WHERE b.a > 0", T, T_ROWS)


def test_reverse_vs_oracle():
    # The ASCII byte path (CRLF splits!) and the grapheme path, oracle-run.
    duck_check("SELECT reverse(s) AS r FROM __THIS__", T, T_ROWS)
    duck_check("SELECT reverse(s || s) AS r, reverse('') AS e FROM __THIS__", T, T_ROWS)


def test_columns_star_replace_and_parenless_vs_oracle():
    duck_check("SELECT COLUMNS(* REPLACE (a + 10 AS a)) FROM __THIS__", T, T_ROWS)
    duck_check("SELECT COLUMNS(* EXCLUDE (s)) FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * REPLACE a+100 AS a FROM __THIS__", T, T_ROWS)
    # The comma ends the paren-less item (measured): a second select item.
    duck_check("SELECT * REPLACE a+100 AS a, a+1 AS a2 FROM __THIS__", T, T_ROWS)


def test_null_regex_pattern_vs_oracle():
    duck_check(
        "SELECT regexp_matches(s, CAST(NULL AS VARCHAR)) AS m, "
        "s SIMILAR TO CAST(NULL AS VARCHAR) AS f, "
        "regexp_replace(s, CAST(NULL AS VARCHAR), 'x') AS r FROM __THIS__",
        T,
        T_ROWS,
    )


def test_unreferenced_nonscalar_column_serves():
    # Lazy rejection: a non-vocabulary field blocks nothing unless it is
    # referenced (including via *).
    D = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("d", pa.timestamp("us")),
            pa.field("s", pa.string()),
        ]
    )
    rows = [
        {"a": 1, "d": datetime.datetime(2020, 1, 1), "s": "x"},
        {"a": 2, "d": None, "s": None},
    ]
    fn = DuckDBInferFn(
        "SELECT a + 1 AS b, s FROM __THIS__",
        row_tables={"__THIS__": D},
        static_tables={},
    )
    assert [r["b"] for r in fn.infer_rows(rows)] == [2, 3]
    fn = DuckDBInferFn(
        "SELECT * EXCLUDE (d) FROM __THIS__",
        row_tables={"__THIS__": D},
        static_tables={},
    )
    assert [list(r.keys()) for r in fn.infer_rows(rows)] == [["a", "s"]] * 2
    for sql in ["SELECT d FROM __THIS__", "SELECT * FROM __THIS__"]:
        with pytest.raises(ValueError, match="non-scalar"):
            DuckDBInferFn(sql, row_tables={"__THIS__": D}, static_tables={})
