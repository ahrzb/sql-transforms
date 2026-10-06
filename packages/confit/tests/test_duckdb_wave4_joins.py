"""Join forms vs the duckdb oracle.

Pins: packages/confit/docs/specs/2026-07-26-wave4-join-pins.md — USING
desugar (merged column = left value at the left position), residual ON
predicates (LEFT keeps residual-failing key-matches with a NULL side),
all-key/semi joins, star-over-join key reconstruction, comma-join
rewrite, cross join to a 1-row static.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from test_duckdb_interpreter import duck_check, static

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

R = static(
    {"id": "int", "category": "int", "budget": "int"},
    [
        {"id": 1, "category": 1, "budget": 100},
        {"id": 2, "category": 2, "budget": 1000},
        {"id": 3, "category": 1, "budget": 50},
    ],
)
L_ROWS = [
    {"lid": 1, "val": 0, "amount": 200},
    {"lid": 1, "val": 5, "amount": 200},
    {"lid": 2, "val": 9, "amount": 500},
    {"lid": 4, "val": 9, "amount": 500},
    {"lid": None, "val": 1, "amount": 1},
]
L = {"lid": "int?", "val": "int", "amount": "int"}


def test_inner_residual_matches_where_placement():
    duck_check(
        "SELECT lid, val, r.budget AS b FROM __THIS__ JOIN r ON lid = r.id AND val > 1",
        L,
        L_ROWS,
        {"r": R},
    )


def test_left_residual_keeps_row_with_null_side():
    # THE pin: key matches but residual fails -> row SURVIVES, right side
    # NULL (val=0 row); WHERE placement would drop it.
    duck_check(
        "SELECT lid, val, r.budget AS b, r.category AS c FROM __THIS__ "
        "LEFT JOIN r ON lid = r.id AND val > 1",
        L,
        L_ROWS,
        {"r": R},
    )


def test_left_right_side_residual():
    duck_check(
        "SELECT lid, r.budget AS b FROM __THIS__ "
        "LEFT JOIN r ON lid = r.id AND r.category = 1",
        L,
        L_ROWS,
        {"r": R},
    )


def test_constant_residuals_true_false():
    for c in ["true", "false"]:
        duck_check(
            f"SELECT lid, r.budget AS b FROM __THIS__ "
            f"LEFT JOIN r ON lid = r.id AND {c}",
            L,
            L_ROWS,
            {"r": R},
        )
        duck_check(
            f"SELECT lid, r.budget AS b FROM __THIS__ JOIN r ON lid = r.id AND {c}",
            L,
            L_ROWS,
            {"r": R},
        )


def test_both_sides_residual():
    duck_check(
        "SELECT lid, r.budget AS b FROM __THIS__ "
        "LEFT JOIN r ON lid = r.id AND amount + r.budget > 1100",
        L,
        L_ROWS,
        {"r": R},
    )


def test_left_only_constant_equality_residual():
    duck_check(
        "SELECT lid, val, r.budget AS b FROM __THIS__ JOIN r ON r.id = lid AND val = 9",
        L,
        L_ROWS,
        {"r": R},
    )


def test_all_key_semi_join():
    keys = static({"id": "int"}, [{"id": 1}, {"id": 3}])
    duck_check(
        "SELECT lid, val FROM __THIS__ JOIN k ON lid = k.id",
        L,
        L_ROWS,
        {"k": keys},
    )


def test_star_over_join_reconstructs_key():
    # Star includes r's key column (named distinctly to keep the typed
    # output model happy); LEFT misses show it NULL.
    r2 = static(
        {"rid": "int", "budget": "int"},
        [{"rid": 1, "budget": 100}, {"rid": 2, "budget": 1000}],
    )
    duck_check(
        "SELECT * FROM __THIS__ LEFT JOIN r2 ON lid = r2.rid",
        L,
        L_ROWS,
        {"r2": r2},
    )
    duck_check(
        "SELECT r2.rid AS k FROM __THIS__ JOIN r2 ON lid = r2.rid",
        L,
        L_ROWS,
        {"r2": r2},
    )


def test_using_merges_key_and_binds_both_quals():
    r3 = static(
        {"lid": "int", "budget": "int"},
        [{"lid": 1, "budget": 100}, {"lid": 2, "budget": 1000}],
    )
    # Merged star column once, left value; r3.lid NULL on LEFT miss.
    duck_check(
        "SELECT * FROM __THIS__ LEFT JOIN r3 USING (lid)",
        L,
        L_ROWS,
        {"r3": r3},
    )
    duck_check(
        "SELECT lid, r3.lid AS rk, budget FROM __THIS__ LEFT JOIN r3 USING (lid)",
        L,
        L_ROWS,
        {"r3": r3},
    )
    duck_check(
        "SELECT lid, budget FROM __THIS__ JOIN r3 USING (lid)",
        L,
        L_ROWS,
        {"r3": r3},
    )


def test_comma_join_equi_rewrite():
    duck_check(
        "SELECT lid, val, budget FROM __THIS__, r WHERE lid = r.id AND val > 1",
        L,
        L_ROWS,
        {"r": R},
    )


def test_comma_join_three_way():
    dim = static(
        {"category": "int", "label": "str"},
        [
            {"category": 1, "label": "low"},
            {"category": 2, "label": "high"},
        ],
    )
    duck_check(
        "SELECT lid, budget, label FROM __THIS__, r, dim "
        "WHERE lid = r.id AND r.category = dim.category",
        L,
        L_ROWS,
        {"r": R, "dim": dim},
    )


def test_cross_join_to_one_row_static():
    one = static({"base": "int"}, [{"base": 7}])
    duck_check(
        "SELECT lid, val + base AS vb FROM __THIS__, one",
        L,
        L_ROWS,
        {"one": one},
    )


def test_cross_join_to_empty_static_annihilates():
    empty = static({"base": "int"}, [])
    duck_check(
        "SELECT lid, base FROM __THIS__, empty",
        L,
        L_ROWS,
        {"empty": empty},
    )


@pytest.mark.parametrize(
    "on",
    [
        "s = e.k",
        "lid = e.id AND CAST(s AS DOUBLE) = e.k",
        "lid = e.id AND lid / 0 > e.k",
    ],
)
def test_a_left_join_to_an_empty_static_never_reads_its_on(on):
    # DuckDB answers every row unmatched and never evaluates the ON clause,
    # so the cast that fails on 'One' never runs (nightly seed 4712724).
    empty = static({"id": "int", "k": "float"}, [])
    duck_check(
        f"SELECT lid, e.k FROM __THIS__ LEFT JOIN empty e ON {on}",
        {"lid": "int?", "s": "str?"},
        [{"lid": 1, "s": "One"}, {"lid": None, "s": "2.5"}],
        {"empty": empty},
    )


def test_cross_join_to_multirow_static_rejects_cleanly():
    two = static({"base": "int"}, [{"base": 1}, {"base": 2}])
    with pytest.raises(ValueError, match="has no equality key"):
        duck_check(
            "SELECT lid, base FROM __THIS__, two",
            L,
            L_ROWS,
            {"two": two},
        )


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize("n", [0, 1, 3])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lid, val FROM __THIS__ CROSS JOIN s0",
        "SELECT 7 AS o FROM __THIS__ JOIN s0 ON TRUE",
        "SELECT lid FROM __THIS__ LEFT JOIN s0 ON TRUE",
    ],
)
def test_a_join_that_reads_no_static_column_keeps_its_row_count(sql, n, shape):
    # float32 is opaque to confit, so the join's map has no key and no value.
    # The row count of the static table still decides the output. Such a map
    # failed IR verification (campaign seeds 5101642, 5101719).
    s0 = pa.table({"c0": pa.array([None] * n, pa.float32())})
    rows = static(L, L_ROWS)
    if n > 1 and shape is None:
        v = assert_parity(sql, rows, statics={"s0": s0}, expect="REFUSED")
        assert "has no equality key" in v.detail
    else:
        assert_parity(sql, rows, statics={"s0": s0}, shape=shape)


@pytest.mark.parametrize(
    "residual",
    [
        # Every node RECOGNISED (Arith under a Cmp) and every one on one
        # side, so the scan's answer is complete and the trapping
        # single-side rule really is what binds. Both sides, so neither
        # gets the diagnosis by accident of which one the scan looks at.
        "r.budget + 1 > 0",
        "lid + 1 > 0",
    ],
)
def test_single_side_trapping_residual_rejects_cleanly(residual):
    with pytest.raises(ValueError, match="trapping"):
        duck_check(
            f"SELECT lid FROM __THIS__ JOIN r ON lid = r.id AND {residual}",
            L,
            L_ROWS,
            {"r": R},
        )


@pytest.mark.parametrize(
    "residual",
    [
        # Both sides visible OUTSIDE the unrecognised node.
        "r.budget + lid > log(lid)",
        # Only ONE side visible outside it: the other is buried in the node
        # the scan refuses to descend into. The sides the scan reports are
        # therefore the sides outside that node, which say nothing about
        # what the condition really reads -- diagnosing off them tells the
        # reader "single-side" about a residual that plainly reads both.
        "r.budget > log(lid)",
        "lid > log(r.budget)",
        # Neither side visible outside it.
        "log(lid + r.budget) > 0",
    ],
)
def test_unclassifiable_residual_names_the_classifier(residual):
    # What stops these is a node the residual scan does not recognise, not
    # the trapping single-side rule. The refusal has to say so -- the reader
    # who is told "single-side" goes looking at the columns instead of at
    # the scan.
    with pytest.raises(ValueError, match="does not recognise"):
        duck_check(
            f"SELECT lid FROM __THIS__ JOIN r ON lid = r.id AND {residual}",
            L,
            L_ROWS,
            {"r": R},
        )
