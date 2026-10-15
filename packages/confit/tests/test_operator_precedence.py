"""Operators parse with DuckDB's precedence (src/specializer/precedence.rs).

sqlparser ranks several operators unlike DuckDB's grammar, so each query here
parsed to another tree before: it served another answer (`a IS DISTINCT FROM b
AND c` read as `a IS DISTINCT FROM (b AND c)`) or refused. The parser now
ranks them as DuckDB does; the Rust tests hold it to DuckDB's own parses in
`docs/specs/pins-wave5/operator-precedence.json`. Here each query agrees with
DuckDB on the rows, the types and the output names, and the forms DuckDB's
grammar refuses are refused.

Every test asserts the campaign's own verdict (`fuzz.parity`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

T = table(
    {
        "id": "int64",
        "a": "int64?",
        "b": "int64?",
        "c": "bool?",
        "k": "int64",
        "s": "string?",
        "x": "int64",
        "y": "int64",
        "z": "int64",
    },
    [
        {
            "id": 1,
            "a": 1,
            "b": 1,
            "c": False,
            "k": 1,
            "s": "ab",
            "x": 4,
            "y": 1,
            "z": 1,
        },
        {"id": 2, "a": 1, "b": 2, "c": True, "k": 2, "s": "ba", "x": 6, "y": 3, "z": 2},
        {
            "id": 3,
            "a": None,
            "b": None,
            "c": None,
            "k": 3,
            "s": None,
            "x": 1,
            "y": 2,
            "z": 1,
        },
        {
            "id": 4,
            "a": None,
            "b": 5,
            "c": True,
            "k": 1,
            "s": "a",
            "x": 0,
            "y": 7,
            "z": 3,
        },
    ],
)

# Each item reads one way under DuckDB's ranks and another under sqlparser's.
ITEMS = [
    # IS [NOT] DISTINCT FROM is below the comparisons and above NOT.
    "a IS DISTINCT FROM b AND c",
    "a IS NOT DISTINCT FROM b OR c",
    "c OR a IS DISTINCT FROM b AND c",
    "c AND a IS DISTINCT FROM b OR c",
    "NOT a IS DISTINCT FROM b AND c",
    "a = b IS DISTINCT FROM c OR c",
    # LIKE, IN and BETWEEN are above the comparisons.
    "c = s LIKE 'a%'",
    "s LIKE 'a%' = c",
    "c = k IN (1, 2)",
    "c = k BETWEEN 1 AND 2",
    "c <> k NOT IN (1)",
    # || ^@ ~ & | << >> are one level, left-associative, below + and *.
    "s || x + y",
    "s || x * y",
    "x + y || s",
    "s ^@ 'a' || 'z'",
    "s || 'b' ^@ 'a'",
    "s ~ 'a.*' || 'z'",
    "c = s ~ 'a.*'",
    "x | y & z",
    "x << y & z",
    "x | y << z",
]


@pytest.mark.parametrize("item", ITEMS)
def test_an_item_reads_as_duckdb_reads_it(item):
    assert_parity(f"SELECT id, {item} AS r FROM __THIS__", T)


@pytest.mark.parametrize("item", ITEMS)
def test_an_unaliased_item_is_named_after_duckdbs_tree(item):
    # DuckDB names the item after its parse, every operator in parentheses:
    # `((a IS DISTINCT FROM b) AND c)`.
    assert_parity(f"SELECT id, {item} FROM __THIS__", T)


@pytest.mark.parametrize(
    "item", ["s ^@ 'a'", "s ~ 'a.*'", "s !~ 'a.*'", "s NOT SIMILAR TO 'a.*'"]
)
def test_a_text_operator_is_named_as_duckdb_names_it(item):
    # `(s ^@ 'a')`, `regexp_full_match(s, 'a.*')`,
    # `(NOT regexp_full_match(s, 'a.*'))`: confit kept the SQL text before.
    assert_parity(f"SELECT id, {item} FROM __THIS__", T)


@pytest.mark.parametrize(
    "where",
    ["a IS DISTINCT FROM b AND c", "a IS NOT DISTINCT FROM b OR c", "c = s LIKE 'a%'"],
)
def test_a_where_reads_as_duckdb_reads_it(where):
    assert_parity(f"SELECT id FROM __THIS__ WHERE {where}", T)


def test_a_join_key_before_another_term():
    s = table(
        {"k": "int64?", "v": "int64"},
        [{"k": 1, "v": 2}, {"k": None, "v": 3}, {"k": 2, "v": 1}],
    )
    assert_parity(
        "SELECT t.id, s.v FROM __THIS__ AS t LEFT JOIN s"
        " ON t.a IS NOT DISTINCT FROM s.k AND s.v > 1",
        T,
        statics={"s": s},
    )


@pytest.mark.parametrize(
    ("item", "why"),
    [
        ("a = b = c", "does not chain comparisons"),
        ("a < b <> c", "does not chain comparisons"),
        ("s LIKE 'a%' IN (TRUE)", "does not chain LIKE"),
        ("k BETWEEN 1 AND 2 LIKE 'a'", "does not chain LIKE"),
        ("a IS DISTINCT FROM b IS NULL", "takes no IS test right after IS"),
        ("s || NOT c", "takes no NOT right after this operator"),
        ("k BETWEEN NOT c AND 2", "takes no NOT as the lower bound of BETWEEN"),
    ],
)
def test_a_form_duckdbs_grammar_refuses_is_refused(item, why):
    # DuckDB refuses each with "syntax error at or near"; confit served them.
    v = assert_parity(f"SELECT id, {item} AS r FROM __THIS__", T, expect="REFUSED")
    assert why in v.detail, v.detail
