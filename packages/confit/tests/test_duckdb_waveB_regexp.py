"""Regexp family vs the duckdb oracle.

Pins: packages/confit/docs/specs/2026-07-27-waveB-regexp-pins.md — the rust
`regex` crate behind the bind-time RE2 translation layer; ~ / SIMILAR TO
are FULL match; extract returns '' on no-match; replace backrefs are
backslash-style with the invalid-rewrite quirks.
"""

from __future__ import annotations

import pytest
from fuzz.parity import assert_parity, table
from test_duckdb_interpreter import duck_check

T = {"a": "int", "s": "str?"}
T_ROWS = [
    {"a": 1, "s": "hello world"},
    {"a": 2, "s": "abc123def"},
    {"a": 3, "s": None},
    {"a": 4, "s": ""},
    {"a": 5, "s": "héllo ٣٤"},
    {"a": 6, "s": "HELLO"},
]


def test_regexp_match_forms():
    duck_check(
        "SELECT regexp_matches(s, 'ell') AS m, regexp_full_match(s, 'ell') AS f, "
        "s ~ 'h.*' AS t, s !~ 'h.*' AS nt, s SIMILAR TO 'h%o' AS pct, "
        "s SIMILAR TO 'h.llo.*' AS re, s NOT SIMILAR TO 'h.*' AS ns, "
        "regexp_matches(s, '') AS emp, regexp_matches(s, 'HELLO', 'i') AS ci "
        "FROM __THIS__",
        T,
        T_ROWS,
    )
    # ASCII perl classes vs Unicode property classes (the differential pin).
    duck_check(
        "SELECT regexp_matches(s, '\\d') AS d, regexp_matches(s, '\\p{Nd}') AS nd, "
        "regexp_matches(s, '\\w+') AS w FROM __THIS__",
        T,
        T_ROWS,
    )


def test_regexp_extract():
    duck_check(
        "SELECT regexp_extract(s, '[0-9]+') AS num, "
        "regexp_extract(s, '(\\w+) (\\w+)', 2) AS g2, "
        "regexp_extract(s, '(h)(x)?', 2) AS npart, "
        "regexp_extract(s, 'zzz') AS miss, "
        "regexp_extract(s, '(a)|(b)', 1) AS alt, "
        "regexp_extract(s, '.', 0, 'i') AS opt FROM __THIS__",
        T,
        T_ROWS,
    )


def test_regexp_replace():
    duck_check(
        "SELECT regexp_replace(s, 'l', 'L') AS one, "
        "regexp_replace(s, 'l', 'L', 'g') AS all_, "
        "regexp_replace(s, '(h)ello', '[\\1]') AS bref, "
        "regexp_replace(s, '(h)ello', '[$1]') AS dollar_lit, "
        "regexp_replace(s, '(h)', '\\2') AS oor, "
        "regexp_replace(s, 'o', 'O', 'gi') AS gi FROM __THIS__",
        T,
        T_ROWS,
    )


def test_star_similar_and_columns():
    duck_check("SELECT * SIMILAR TO 'a' FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * NOT SIMILAR TO 'a.*' FROM __THIS__", T, T_ROWS)
    duck_check("SELECT COLUMNS('s') FROM __THIS__", T, T_ROWS)
    duck_check("SELECT COLUMNS(*) FROM __THIS__", T, T_ROWS)


# A subject and a NULL one; `a + MAX` overflows on both rows.
NULLS = table({"s": "str?", "a": "int"}, [{"s": "abc", "a": 1}, {"s": None, "a": 2}])
TRAP = "(a + 9223372036854775807)"


@pytest.mark.parametrize(
    "node",
    [
        "CAST(({x} ~ NULL) AS BIGINT)",
        "CAST(({x} !~ NULL) AS BIGINT)",
        "CAST(({x} SIMILAR TO NULL) AS BIGINT)",
        "CAST(({x} NOT SIMILAR TO NULL) AS BIGINT)",
        "CAST(regexp_full_match({x}, NULL) AS BIGINT)",
        "CAST(regexp_full_match({x} || 'x', CAST(NULL AS VARCHAR)) AS BIGINT)",
        "CAST(regexp_full_match({x}, CASE WHEN FALSE THEN 'a' END) AS BIGINT)",
        "length(regexp_extract({x}, NULL))",
        "length(regexp_extract({x}, NULL, 1))",
        "length(regexp_extract({x}, NULL, NULL))",
    ],
)
def test_a_regexp_call_over_a_null_pattern_reads_its_subject(node):
    # DuckDB's regexp functions handle a NULL themselves, so its binder keeps
    # the call: it reads its subject for each row and answers NULL, and the
    # `+` over it still evaluates the trap (measured, 1.5.5). Over a subject
    # that reads no column the call is foldable, and the `+` folds to NULL.
    column = node.format(x="s")
    assert_parity(
        f"SELECT {column} + {TRAP} AS o FROM __THIS__", NULLS, trap="Overflow"
    )
    closed = node.format(x="'abc'")
    assert_parity(f"SELECT {closed} + {TRAP} AS o FROM __THIS__", NULLS, expect="AGREE")


def test_regexp_matches_over_a_null_pattern_is_the_optimizer_off_reading():
    # DuckDB's optimizer makes `regexp_matches(s, NULL)` a NULL constant (its
    # regex rule), so only the optimizer-off reading traps, as confit does.
    for node in [
        "CAST(regexp_matches(s, NULL) AS BIGINT) + " + TRAP,
        "CAST(regexp_matches(CAST(a + 9223372036854775807 AS VARCHAR), NULL) "
        "AS BIGINT)",
    ]:
        assert_parity(f"SELECT {node} AS o FROM __THIS__", NULLS, expect="DIVERGE_OPT")


def test_a_null_pattern_over_a_subject_that_drops_its_column_refuses():
    # `coalesce('x', s)` binds as 'x' here, but DuckDB's call reads `s` for
    # each row and does not fold.
    v = assert_parity(
        f"SELECT CAST(regexp_matches(coalesce('x', s), NULL) AS BIGINT) + {TRAP} AS o "
        "FROM __THIS__",
        NULLS,
        expect="REFUSED",
    )
    assert "NULL pattern" in v.detail, v


@pytest.mark.parametrize(
    "call",
    [
        "regexp_matches(NULL, 'a(')",
        "NULL ~ 'a('",
        "regexp_extract(NULL, 'a(')",
        "regexp_extract(s, 'a(', NULL)",
        "regexp_extract(NULL, 'a', 99)",
        "regexp_extract(NULL, NULL, 99)",
        "regexp_extract(s, 'a', CASE WHEN a > 0 THEN NULL END)",
    ],
)
def test_a_null_argument_skips_no_bind_check(call):
    # DuckDB checks the options, the group and the pattern at bind, whatever
    # the subject and the group are: a NULL subject or group answers NULL or
    # '' only at run time (measured, 1.5.5).
    v = assert_parity(f"SELECT {call} AS o FROM __THIS__", NULLS, expect="REFUSED")
    assert v.oracle != "serves", v


@pytest.mark.parametrize(
    "call",
    [
        "regexp_replace(s, NULL, 'x', 'zz')",
        "regexp_replace(CASE WHEN FALSE THEN 'q' END, 'a(', 'x')",
        "regexp_replace(s, 'a', 'x', CASE WHEN FALSE THEN 'g' END)",
    ],
)
def test_regexp_replace_over_a_null_argument_is_null_before_its_options(call):
    # regexp_replace has the default NULL handling: DuckDB's binder makes the
    # call a NULL constant before it reads the options or compiles the
    # pattern (measured, 1.5.5).
    assert_parity(f"SELECT {call} AS o FROM __THIS__", NULLS, expect="AGREE")


@pytest.mark.parametrize(
    "call",
    [
        "regexp_matches(s, 'a', coalesce('i', s))",
        "regexp_matches(s, NULL, coalesce('i', s))",
        "regexp_replace(s, 'a', 'b', coalesce('g', s))",
        "regexp_extract(s, 'a', coalesce(1, CAST(a AS INTEGER)))",
    ],
)
def test_options_or_a_group_that_name_a_column_are_no_constant(call):
    # Our binder binds `coalesce('i', s)` as 'i', but DuckDB keys on the text:
    # the options or the group read a column, so they are no constant there
    # (measured, 1.5.5: "must be a constant").
    v = assert_parity(f"SELECT {call} AS o FROM __THIS__", NULLS, expect="REFUSED")
    assert v.oracle != "serves", v
