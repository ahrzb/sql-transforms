"""String operations that cannot trap, and where confit relies on it.

`can_trap` (`src/specializer/plan.rs`) tells confit where it may evaluate a
term in another order than DuckDB does, or not at all: a WHERE term written
before a key of a comma join, a WHERE term of a scalar subquery that reads
only its static table, a struct field that no read needs. It names these
string operations as ones that cannot trap: LIKE and ILIKE without ESCAPE,
GLOB, `||` and concat, trim, the searches (instr, contains, starts_with,
ends_with, `^@`), replace and translate, a VARCHAR slice, upper and lower,
the lengths, unicode, ord and ascii, strip_accents and reverse. Their kernels
return no error, and the pin below measures that the oracle raises none over
inputs at the edges of each one. LIKE with ESCAPE, substr, lpad and rpad,
repeat, a VARCHAR subscript and the edit distances can trap: they stay
unnamed, so a term that holds one still refuses where its order matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

# The edges of each operation: the LIKE and GLOB wildcards, a pattern that
# ends in a backslash (LIKE's escape character when it has an ESCAPE), a GLOB
# class left open, combining and zero-width-joined characters, case maps that
# change the length, and a long string.
STRS = [
    "",
    "a",
    "abc",
    "%",
    "_",
    "\\",
    "a\\",
    "%_\\",
    "[a-",
    "[!a]",
    "*",
    "?",
    "é",
    "é",
    "☃",
    "😀",
    "👨‍👩‍👧",
    "ß",
    "İ",
    "ǅ",
    "  pad  ",
    "x" * 10_000,
    None,
]
# Slice bounds past both ends of a string, and past substr's range guard.
BOUNDS = [
    -(2**63),
    -(2**32) - 1,
    -(2**32),
    -5,
    -1,
    0,
    1,
    2,
    5,
    2**32 - 1,
    2**32,
    2**63 - 1,
    None,
]

# Each spelling with the relation of hostile inputs it runs over.
OPS = [
    *[
        (e, "v")
        for e in [
            "lower(x)",
            "upper(x)",
            "lcase(x)",
            "ucase(x)",
            "strip_accents(x)",
            "reverse(x)",
            "trim(x)",
            "ltrim(x)",
            "rtrim(x)",
            "length(x)",
            "len(x)",
            "char_length(x)",
            "character_length(x)",
            "strlen(x)",
            "unicode(x)",
            "ord(x)",
            "ascii(x)",
        ]
    ],
    *[
        (e, "v2")
        for e in [
            "x LIKE y",
            "x NOT LIKE y",
            "x ILIKE y",
            "x NOT ILIKE y",
            "x GLOB y",
            "x || y",
            "concat(x, y)",
            "trim(x, y)",
            "ltrim(x, y)",
            "rtrim(x, y)",
            "instr(x, y)",
            "strpos(x, y)",
            "position(y IN x)",
            "starts_with(x, y)",
            "prefix(x, y)",
            "ends_with(x, y)",
            "suffix(x, y)",
            "contains(x, y)",
            "x ^@ y",
        ]
    ],
    ("replace(x, y, z)", "v3"),
    ("translate(x, y, z)", "v3"),
    ("x[lo:hi]", "vs"),
    ("array_slice(x, lo, hi)", "vs"),
    ("list_slice(x, lo, hi)", "vs"),
]


def test_pin_the_oracle_raises_no_error_for_the_string_operations(oracle):
    # Measured: DuckDB 1.5.5, optimizer off, answers each spelling on every
    # input (count evaluates it on each row, and the sum reads each value).
    oracle.table("v", "x VARCHAR", [(s,) for s in STRS])
    oracle.table("n", "i BIGINT", [(i,) for i in BOUNDS])
    oracle.execute("CREATE TABLE v2 AS SELECT a.x AS x, b.x AS y FROM v a, v b")
    oracle.execute(
        "CREATE TABLE v3 AS SELECT a.x AS x, b.x AS y, c.x AS z FROM v a, v b, v c"
    )
    oracle.execute(
        "CREATE TABLE vs AS SELECT v.x AS x, a.i AS lo, b.i AS hi FROM v, n a, n b"
    )
    for expr, rel in OPS:
        oracle.execute(
            f"SELECT count({expr}), sum(length(CAST({expr} AS VARCHAR))) FROM {rel}"
        ).fetchall()


ROWS = table(
    {"pid": "int", "x": "str?"},
    [
        {"pid": 1, "x": "a%"},
        {"pid": 2, "x": "%☃"},
        {"pid": 3, "x": None},
        {"pid": 4, "x": "_"},
        {"pid": 9, "x": "b"},
    ],
)
S = table(
    {"id": "int", "s": "str?"},
    [
        {"id": 1, "s": "abc"},
        {"id": 2, "s": "é☃"},
        {"id": 3, "s": ""},
        {"id": 4, "s": None},
        {"id": 5, "s": "a\\"},
    ],
)

# Terms that read the static table, and some the request row too, each with
# one of the operations above.
TERMS = [
    "s.s LIKE x",
    "s.s NOT LIKE 'a%'",
    "s.s ILIKE upper(x)",
    "s.s GLOB '*☃'",
    "(s.s || x) <> 'abc'",
    "concat(s.s, x) IS NOT NULL",
    "trim(s.s, x) = ''",
    "ltrim(s.s) = s.s",
    "instr(s.s, x) > 0",
    "contains(s.s, 'b')",
    "starts_with(s.s, x)",
    "ends_with(s.s, '☃')",
    "s.s ^@ 'a'",
    "replace(s.s, 'b', x) <> ''",
    "translate(s.s, 'abc', 'x') = 'x'",
    "array_slice(s.s, 2, 3) = 'bc'",
    "lower(s.s) = upper(s.s)",
    "length(s.s) > 1",
    "strlen(s.s) > 2",
    "ascii(s.s) > 0",
    "unicode(s.s) > 0",
    "strip_accents(s.s) = 'e☃'",
    "reverse(s.s) > x",
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("term", TERMS)
def test_a_comma_join_serves_a_string_term_before_the_key(term, shape):
    # DuckDB evaluates the term over every pair of a request and a static
    # row, and confit only where the key matched: the same rows pass.
    sql = f"SELECT pid, s.s FROM __THIS__, s WHERE {term} AND pid = s.id"
    assert_parity(sql, ROWS, statics={"s": S}, shape=shape)


@pytest.mark.parametrize(
    "term",
    [
        "s.s LIKE 'a%'",
        "s.s ILIKE '%B%'",
        "s.s GLOB 'a*'",
        "lower(s.s) = 'abc'",
        "length(s.s) > 1",
        "NOT contains(s.s, '☃')",
    ],
)
def test_a_scalar_subquery_serves_a_string_filter_term(term):
    sql = (
        f"SELECT pid, (SELECT count(*) FROM s WHERE {term} AND s.id = pid) AS c "
        "FROM __THIS__"
    )
    assert_parity(sql, ROWS, statics={"s": S})


@pytest.mark.parametrize(
    "term",
    [
        "s.s LIKE x ESCAPE '\\'",
        "substr(s.s, 2) = 'bc'",
        "lpad(s.s, 4, '') = 'abc'",
        "repeat(s.s, 2) = 'abcabc'",
        "array_extract(s.s, 2) = 'b'",
        "levenshtein(s.s, x) > 1",
    ],
)
def test_a_comma_join_refuses_a_string_term_that_can_trap_before_the_key(term):
    sql = f"SELECT pid, s.s FROM __THIS__, s WHERE {term} AND pid = s.id"
    v = assert_parity(sql, ROWS, statics={"s": S}, expect="REFUSED")
    assert "comma join" in v.detail, v.detail
