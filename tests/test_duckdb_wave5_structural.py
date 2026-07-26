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
