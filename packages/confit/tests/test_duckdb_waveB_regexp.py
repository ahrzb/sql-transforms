"""Wave-B regexp family vs the duckdb oracle.

Pins: packages/confit/docs/specs/2026-07-27-waveB-regexp-pins.md — the rust
`regex` crate behind the bind-time RE2 translation layer; ~ / SIMILAR TO
are FULL match; extract returns '' on no-match; replace backrefs are
backslash-style with the invalid-rewrite quirks.
"""

from __future__ import annotations

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
