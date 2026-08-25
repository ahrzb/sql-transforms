"""Stage-B multiplicity vs the duckdb oracle (TASK-59), multiset parity.

Pins: docs/superpowers/specs/2026-07-28-stageB-multiplicity-pins.md —
DuckDB's join output ORDER is a hash-join accident, so comparison is
SORTED; the engine's own order (probe outer, insertion inner) is a
documented contract of its own.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

T = pa.schema([pa.field("pid", pa.int64())])
ROWS = [{"pid": 1}, {"pid": 2}, {"pid": 3}, {"pid": None}]
DIM = pa.table({"id": [1, 2, 1, 2, 1], "v": ["a", "b", "c", "d", "e"]})


def _many_check(sql: str):
    """Engine (shape='many') vs DuckDB, sorted-row multiset."""
    fn = DuckDBInferFn(
        sql.replace("__THIS__", "__THIS__"),
        row_tables={"__THIS__": T},
        static_tables={"d": DIM},
        shape="many",
    )
    got = [tuple(r.values()) for r in fn.infer_rows(ROWS)]

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (pid BIGINT)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?)", [r["pid"]])
    con.register("__arrow_d", DIM)
    con.execute('CREATE TABLE d AS SELECT * FROM "__arrow_d"')
    want = con.execute(sql).fetchall()
    key = lambda t: tuple((x is None, x) for x in t)  # noqa: E731
    assert sorted(got, key=key) == sorted(want, key=key), f"{sql}\n{got}\n{want}"


def test_dup_key_fanout_vs_oracle():
    _many_check("SELECT pid, v FROM __THIS__ JOIN d ON pid = d.id")
    _many_check("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id")
    _many_check("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id AND d.v > 'b'")
    _many_check("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id WHERE v IS NULL")
    _many_check("SELECT upper(v) AS u FROM __THIS__ JOIN d ON pid = d.id WHERE pid > 1")


def test_cross_and_inequality_vs_oracle():
    _many_check("SELECT pid, id, v FROM __THIS__, d")
    _many_check("SELECT pid, id FROM __THIS__ JOIN d ON pid > d.id")
    _many_check("SELECT pid, id FROM __THIS__ LEFT JOIN d ON pid > d.id")
    _many_check("SELECT pid, id FROM __THIS__ LEFT JOIN d ON NULL = 2")
    _many_check("SELECT pid, id FROM __THIS__, d WHERE pid >= id AND v <> 'c'")


def test_engine_order_contract():
    # The engine's OWN documented deterministic order: probe rows in input
    # order, matches contiguous in build INSERTION order, null-extension
    # in place.
    fn = DuckDBInferFn(
        "SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id",
        row_tables={"__THIS__": T},
        static_tables={"d": DIM},
        shape="many",
    )
    got = [tuple(r.values()) for r in fn.infer_rows(ROWS)]
    assert got == [
        (1, "a"),
        (1, "c"),
        (1, "e"),
        (2, "b"),
        (2, "d"),
        (3, None),
        (None, None),
    ]


def test_value_lanes_from_both_column_sources_vs_oracle():
    """A map's value slots are laid out from ONE of two column lists, and
    which one is a property of the join, not of the layout rule: a MULTIMAP
    reads the static catalog, a BATCHMAP (the stage-B self-join) reads the
    caller's own row schema. A nullable column rides as a validity+payload
    PAIR on both, so each source needs a NULL that survives the round trip.

    `test_dup_key_fanout_vs_oracle` already covers the multimap half with a
    non-NULL value; what was unpinned is a NULL through the multimap and the
    batchmap value lane at all.
    """
    schema = pa.schema(
        [pa.field("pid", pa.int64(), nullable=False), pa.field("tag", pa.string())]
    )
    rows = [{"pid": 1, "tag": "a"}, {"pid": 1, "tag": None}, {"pid": 2, "tag": "b"}]
    dim = pa.table({"id": [1, 1, 2], "v": ["x", None, "y"]})

    def check(sql: str, statics: dict):
        fn = DuckDBInferFn(
            sql,
            row_tables={"__THIS__": schema},
            static_tables=statics,
            shape="many",
        )
        got = [tuple(r.values()) for r in fn.infer_rows(rows)]
        con = duckdb.connect()
        con.execute("CREATE TABLE __THIS__ (pid BIGINT NOT NULL, tag VARCHAR)")
        for r in rows:
            con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["pid"], r["tag"]])
        for name, tbl in statics.items():
            con.register(f"__arrow_{name}", tbl)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
        want = con.execute(sql).fetchall()
        key = lambda t: tuple((x is None, x) for x in t)  # noqa: E731
        assert sorted(got, key=key) == sorted(want, key=key), f"{sql}\n{got}\n{want}"

    # MultiMap: the value lanes come from the static catalog's columns.
    check("SELECT pid, d.v AS g FROM __THIS__ JOIN d ON pid = d.id", {"d": dim})
    # BatchMap: the value lanes come from the caller's in_cols instead.
    check(
        "SELECT __THIS__.pid AS p, t2.tag AS g "
        "FROM __THIS__ JOIN __THIS__ t2 ON __THIS__.pid = t2.pid",
        {},
    )


@pytest.mark.parametrize(
    "expr",
    [
        "COALESCE(d.v, 'z')",
        "CASE WHEN d.v = 'a' THEN 'A' ELSE d.v END",
        "NULLIF(d.v, 'a')",
    ],
)
def test_split_over_a_joined_column_under_many(expr):
    """TASK-68: the many-join's probe CACHE lives per block while its values
    ride the live stack, so a CFG split used to drop it and the joined column
    fell through to the scalar `Inst::Probe` — rejected by verify with '@N is
    a multimap: use probe.range'."""
    _many_check(f"SELECT {expr} AS v FROM __THIS__ AS t JOIN d ON t.pid = d.id")


@pytest.mark.parametrize(
    "expr",
    [
        "COALESCE(d.v, 'z')",
        "CASE WHEN d.v = 'a' THEN 'A' ELSE d.v END",
    ],
)
def test_split_over_a_joined_column_left_join(expr):
    """The null-extended path seeds the cache with hit=false and a different
    lane set, so it is a separate seed site from the matched path."""
    _many_check(f"SELECT {expr} AS v FROM __THIS__ AS t LEFT JOIN d ON t.pid = d.id")


def test_split_in_a_later_output_column_only():
    """The cache is re-created per output expression, so a split in the SECOND
    column exercises a seed that a first-column-only test would not."""
    _many_check(
        "SELECT d.v AS a, CASE WHEN d.v = 'a' THEN 'A' ELSE d.v END AS b "
        "FROM __THIS__ AS t JOIN d ON t.pid = d.id"
    )


def test_split_in_the_where_clause():
    """WHERE is emitted through its own seed site (`filter_pred`), on both the
    matched and the null-extended path."""
    _many_check(
        "SELECT d.v AS v FROM __THIS__ AS t JOIN d ON t.pid = d.id "
        "WHERE COALESCE(d.v, 'z') <> 'b'"
    )
    _many_check(
        "SELECT d.v AS v FROM __THIS__ AS t LEFT JOIN d ON t.pid = d.id "
        "WHERE CASE WHEN d.v IS NULL THEN TRUE ELSE d.v <> 'b' END"
    )


def test_split_reading_the_joined_column_twice_across_a_split():
    """Two reads of the same joined column, one before and one inside the
    split: the second must reuse the re-created cache entry rather than
    emitting a second probe."""
    _many_check(
        "SELECT d.v AS a, d.id AS i, "
        "CASE WHEN d.id > 1 THEN d.v ELSE 'z' END AS b "
        "FROM __THIS__ AS t JOIN d ON t.pid = d.id"
    )


def test_nested_splits_over_a_joined_column():
    """A split inside a split: the innermost block is two transitions from the
    seed, so a fix that only restores one level fails here."""
    _many_check(
        "SELECT CASE WHEN d.id > 1 THEN "
        "  (CASE WHEN d.v = 'b' THEN 'B' ELSE COALESCE(d.v, 'z') END) "
        "ELSE d.v END AS v "
        "FROM __THIS__ AS t JOIN d ON t.pid = d.id"
    )


def test_split_inside_a_multi_operand_expression():
    """The seed has to remember WHERE the join's lanes sit, not assume they
    are the trailing ones. Here `||` pushes its first operand onto the live
    stack before the CASE splits, so the join's lanes are no longer at the
    tail — reading `live[len - nd..]` would seed the cache with the wrong
    registers and silently score the wrong column."""
    _many_check(
        "SELECT d.v || (CASE WHEN d.id > 1 THEN 'x' ELSE d.v END) AS v "
        "FROM __THIS__ AS t JOIN d ON t.pid = d.id"
    )
    _many_check(
        "SELECT COALESCE(d.v, 'z') || d.v || "
        "(CASE WHEN d.id > 1 THEN d.v ELSE 'y' END) AS v "
        "FROM __THIS__ AS t JOIN d ON t.pid = d.id"
    )


def test_split_in_the_join_condition():
    """The ON residual is its own seed site, emitted before the output
    columns exist. A CASE there that reads the joined column is the case the
    per-expression reseed could not reach.

    The bare `AND CASE ... END` spelling on an INNER join is refused earlier
    by an unrelated pre-existing rule ("single-side residual with trapping
    ops"), so the reachable forms are a LEFT join and a comparison.
    """
    _many_check(
        "SELECT pid, v FROM __THIS__ AS t LEFT JOIN d "
        "ON t.pid = d.id AND CASE WHEN d.id > 1 THEN TRUE ELSE FALSE END"
    )
    _many_check(
        "SELECT pid, v FROM __THIS__ AS t JOIN d "
        "ON t.pid = d.id AND (CASE WHEN d.id > 1 THEN d.v ELSE 'a' END) <> 'b'"
    )
