"""Wave-5 structural + dialect forms vs the duckdb oracle.

Pins: docs/superpowers/specs/2026-07-26-wave5-structural-pins.md — colon
prefix aliases (token pre-rewrite), slices, extended subscripts, bitwise
ops, ^@/GLOB, star forms, duplicate-name contract, binder tail. The file
grows stage by stage with TASK-52.
"""

from __future__ import annotations

from test_duckdb_interpreter import duck_check

T = {"a": "int", "s": "str?"}
T_ROWS = [
    {"a": 3, "s": "hello"},
    {"a": -1, "s": None},
    {"a": 0, "s": ""},
    {"a": 7, "s": "héllo"},
]


def test_colon_prefix_alias():
    duck_check("SELECT k: a + 1, j: a * 2 FROM __THIS__", T, T_ROWS)
    duck_check('SELECT "K": a - 1, a AS plain FROM __THIS__ WHERE a > 0', T, T_ROWS)


S_ROWS = [
    {"a": 2, "s": "hello"},
    {"a": -1, "s": ""},
    {"a": 100, "s": None},
    {"a": 3, "s": "héllo"},
    {"a": 0, "s": "a😀b€c"},
]


def test_bracket_subscripts_vs_oracle():
    # Codepoint extract: negative from-end, 0/out-of-range -> '', dynamic
    # index from a column, NULL string/index -> NULL.
    duck_check(
        "SELECT s[2], s[-1], s[0], s[100], s[a], s[NULL] FROM __THIS__",
        T,
        S_ROWS,
    )
    # DuckDB derives 's[(a + 1)]' for the unaliased compound form (it
    # parenthesizes sub-expressions) — alias to keep the check on values.
    duck_check("SELECT s[a + 1] AS x FROM __THIS__", T, S_ROWS)


def test_bracket_slices_vs_oracle():
    # Both-inclusive codepoint slices, open bounds, negatives, clamps,
    # reversed -> '', NULL bound -> NULL (NOT an open bound).
    duck_check(
        "SELECT s[2:4], s[:2], s[2:], s[:], s[4:2], s[-3:-1], s[-99:2], "
        "s[2:-1], s[50:60], s[NULL:2], s[2:NULL], s[a:a+2] AS dyn FROM __THIS__",
        T,
        S_ROWS,
    )
    duck_check("SELECT s[2:4][1] FROM __THIS__", T, S_ROWS)
