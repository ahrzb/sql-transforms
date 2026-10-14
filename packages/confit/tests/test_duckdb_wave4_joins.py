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

from confit.oracle import Oracle, Trap  # noqa: E402
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


# --- Comma joins: the order of the WHERE terms -------------------------------
# DuckDB (optimizer off) joins comma tables as a cross product, then runs the
# WHERE terms in written order, each over the row pairs that passed the terms
# before it. Confit probes each comma table with its key terms, then runs the
# other terms over the matched pairs. The traps agree when a term that can trap
# comes after every key term, and a key term that can trap comes right after
# the key terms of the earlier comma tables. The other orders refuse by name
# (`check_comma_order` in src/specializer/frontend/from.rs).

BIG = 9223372036854775807  # the largest BIGINT: `BIG + 1` overflows
PC = {"pid": "int", "c": "int"}
S = static({"id": "int", "z": "int"}, [{"id": 5, "z": 0}, {"id": 6, "z": 1}])
S1 = static({"x": "int"}, [{"x": 5}])
S2 = static({"y": "int"}, [{"y": 7}])
S3 = static({"y": "int"}, [{"y": 5}])


def _comma_sql(where: str, statics: dict) -> str:
    return f"SELECT pid FROM {', '.join(['__THIS__', *statics])} WHERE {where}"


def _duck(sql: str, rows: list[dict], statics: dict) -> str | int:
    """DuckDB's answer: "trap", or its row count."""
    with Oracle() as o:
        o.table("__THIS__", "pid BIGINT, c BIGINT", [(r["pid"], r["c"]) for r in rows])
        for name, t in statics.items():
            o.load(name, t)
        got = o.try_answer(sql)
    return "trap" if isinstance(got, Trap) else got.num_rows


COMMA_REFUSED = [
    # A term that can trap before a key: DuckDB evaluates it over every pair,
    # so the pair with s.id = 6 overflows. Confit's key matches no pair.
    pytest.param(
        f"({BIG} + s.z) > 0 AND pid = s.id",
        [{"pid": 1, "c": 0}],
        {"s": S},
        "trap",
        id="term-before-key",
    ),
    pytest.param(
        f"({BIG} + c) > 0 AND pid = s.id",
        [{"pid": 1, "c": 1}],
        {"s": S},
        "trap",
        id="row-term-before-key",
    ),
    pytest.param(
        f"(s.z = 0 OR ({BIG} + s.z) > 0) AND pid = s.id",
        [{"pid": 1, "c": 0}],
        {"s": S},
        "trap",
        id="or-term-before-key",
    ),
    # BETWEEN's lower half stays in place; NOT BETWEEN is one term.
    pytest.param(
        f"({BIG} + s.z) BETWEEN 0 AND 10 AND pid = s.id",
        [{"pid": 1, "c": 0}],
        {"s": S},
        "trap",
        id="between-lower-half",
    ),
    pytest.param(
        f"s.z NOT BETWEEN -1 AND ({BIG} + s.z) AND pid = s.id",
        [{"pid": 1, "c": 0}],
        {"s": S},
        "trap",
        id="not-between",
    ),
    # Between two keys of one table: DuckDB evaluates it where the first key
    # matched (s.id = 6, s.z = 1). Confit's two keys together match no row.
    pytest.param(
        f"pid = s.id AND ({BIG} + s.z) > 0 AND c = s.z",
        [{"pid": 6, "c": 0}],
        {"s": S},
        "trap",
        id="term-between-keys",
    ),
    # A key that can trap after another term: DuckDB evaluates it only where
    # the term before it passed, here nowhere. Confit evaluates it per row.
    pytest.param(
        f"c > 0 AND (pid + {BIG}) = s.id",
        [{"pid": 1, "c": 0}],
        {"s": S},
        0,
        id="key-after-term",
    ),
    pytest.param(
        f"pid = s.id AND (c + {BIG}) = s.z",
        [{"pid": 1, "c": 1}],
        {"s": S},
        0,
        id="key-after-key",
    ),
    # A key that can trap before the key of an earlier comma table: DuckDB
    # evaluates it for every pair. Confit evaluates it where s1 matched.
    pytest.param(
        f"(c + {BIG}) = s2.y AND pid = s1.x",
        [{"pid": 1, "c": 1}],
        {"s1": S1, "s2": S2},
        "trap",
        id="key-before-earlier-key",
    ),
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(("where", "rows", "statics", "duck"), COMMA_REFUSED)
def test_a_comma_join_refuses_a_term_order_that_traps_on_other_rows(
    where, rows, statics, duck, shape
):
    sql = _comma_sql(where, statics)
    # What DuckDB answers, against what confit's key-first plan would answer:
    # no rows where DuckDB traps, a trap where DuckDB answers no rows.
    assert _duck(sql, rows, statics) == duck
    v = assert_parity(
        sql, static(PC, rows), statics=statics, shape=shape, expect="REFUSED"
    )
    assert "comma join" in v.detail


COMMA_SERVED = [
    # Terms after the keys, and terms that cannot trap, go anywhere.
    pytest.param(
        f"pid = s.id AND ({BIG} + s.z) > 0",
        [{"pid": 5, "c": 0}, {"pid": 1, "c": 1}],
        {"s": S},
        "AGREE",
        id="term-after-key",
    ),
    pytest.param(
        f"pid = s.id AND ({BIG} + c) > 0",
        [{"pid": 5, "c": 0}, {"pid": 1, "c": 1}],
        {"s": S},
        "AGREE",
        id="row-term-after-key",
    ),
    pytest.param(
        f"pid = s.id AND c = s.z AND ({BIG} + s.z) > 0",
        [{"pid": 6, "c": 0}, {"pid": 5, "c": 0}],
        {"s": S},
        "AGREE",
        id="term-after-two-keys",
    ),
    pytest.param(
        "s.z > 0 AND pid = s.id",
        [{"pid": 6, "c": 0}, {"pid": 5, "c": 0}],
        {"s": S},
        "AGREE",
        id="term-that-cannot-trap",
    ),
    pytest.param(
        "(1e308 * s.z) > 0 AND pid = s.id",
        [{"pid": 6, "c": 0}, {"pid": 5, "c": 0}],
        {"s": S},
        "AGREE",
        id="double-term",
    ),
    # DuckDB's planner moves BETWEEN's upper half to the end of the WHERE.
    # Its optimizer evaluates it over every row of s, and traps: confit
    # follows the reading without the optimizer (DIVERGE_OPT).
    pytest.param(
        f"s.z BETWEEN -1 AND ({BIG} + s.z) AND pid = s.id",
        [{"pid": 1, "c": 0}, {"pid": 5, "c": 0}],
        {"s": S},
        "DIVERGE_OPT",
        id="between-upper-half",
    ),
    # A key that can trap, first: both engines evaluate it for every row.
    # DuckDB's optimizer moves `c > 0` before it, and answers no rows.
    pytest.param(
        f"(pid + {BIG}) = s.id AND c > 0",
        [{"pid": 1, "c": 0}],
        {"s": S},
        "DIVERGE_OPT",
        id="trapping-key-first",
    ),
    pytest.param(
        f"(c + {BIG}) = s.z AND pid = s.id",
        [{"pid": 1, "c": 1}],
        {"s": S},
        "DIVERGE_OPT",
        id="trapping-key-before-key",
    ),
    # Right after the key of the earlier comma table: both engines evaluate
    # it where s1 matched.
    pytest.param(
        f"pid = s1.x AND (c + {BIG}) = s2.y",
        [{"pid": 1, "c": 1}, {"pid": 5, "c": 0}, {"pid": 5, "c": 7 - BIG}],
        {"s1": S1, "s2": S2},
        "AGREE",
        id="keys-in-from-order",
    ),
    # A key that reads the key column of an earlier comma table reads its
    # static value on DuckDB, which cannot trap: it goes anywhere.
    pytest.param(
        f"(c + {BIG}) = s1.x AND c < 0 AND s1.x = s3.y",
        [{"pid": 1, "c": 5 - BIG}, {"pid": 2, "c": 0}],
        {"s1": S1, "s3": S3},
        "AGREE",
        id="key-reads-an-earlier-key",
    ),
]


@pytest.mark.parametrize("shape", [None, "many"])
@pytest.mark.parametrize(("where", "rows", "statics", "expect"), COMMA_SERVED)
def test_a_comma_join_serves_a_term_order_that_traps_on_the_same_rows(
    where, rows, statics, expect, shape
):
    sql = _comma_sql(where, statics)
    assert_parity(sql, static(PC, rows), statics=statics, shape=shape, expect=expect)


def test_join_on_serves_the_refused_comma_order():
    # The refusal's advice: under JOIN ... ON, DuckDB runs the WHERE over the
    # joined rows, as confit does. So the term that traps under the comma
    # join (above) is not evaluated here.
    assert_parity(
        f"SELECT pid FROM __THIS__ JOIN s ON pid = s.id WHERE ({BIG} + s.z) > 0",
        static(PC, [{"pid": 1, "c": 0}, {"pid": 5, "c": 0}]),
        statics={"s": S},
        expect="AGREE",
    )
