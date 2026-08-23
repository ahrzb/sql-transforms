"""Divergences we intend to CLOSE — one xfail-strict pin each, ticket named.

It emptied on 2026-08-17 when TASK-115, 116, 117, 118 and 119 all closed, and
refilled the same day from the campaign that followed the oracle change --
which is the intended rhythm, not churn. Adding a pin here is how a new
divergence gets recorded; emptying it again is what closing one looks like.

The split from `known_divergences/` is by INTENT, not by severity:

    known_divergences/   behaviour we have decided to KEEP. Every entry
                         states the ground for keeping it, and its tests
                         PASS — they are regression pins on a settled
                         answer.

    this file            behaviour we have decided to CHANGE. Every entry is
                         xfail(strict=True) and names the task that closes
                         it. When the fix lands the pin flips loudly, and
                         the entry is deleted rather than edited.

Why the separation is worth a second file: mixing the two makes "is this on
purpose?" unanswerable at a glance, and a reader who assumes the wrong one
either implements something we chose not to have, or leaves a real bug
sitting under a paragraph explaining why it is fine. The census on
2026-08-16 found both mistakes already present.

strict=True is the load-bearing part. A pin that silently starts passing is
worse than no pin: it certifies work nobody did.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn


# ---------------------------------------------------------------------------
# TASK-121: a bare struct path whose HEAD name also binds in a join scope is
# ambiguous on DuckDB and binds here. 16 of the 28 findings in the 2026-08-17
# campaign, and the largest single class the fuzzer sees.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-121: `ON (c0.f0 = s0.c0)` where __THIS__.c0 is a struct and "
    "s0.c0 is a column -- DuckDB refuses as ambiguous before it looks at .f0; "
    "we resolve to the struct and answer. The BARE-column case already refuses; "
    "the struct-path route does not consult the join scopes.",
)
def test_an_ambiguous_struct_path_refuses():
    row = pa.schema(
        [
            pa.field("c0", pa.struct([("f0", pa.int64())])),
            pa.field("v", pa.int64(), nullable=False),
        ]
    )
    static = pa.table({"c0": pa.array([1], pa.int64()), "w": pa.array([9], pa.int64())})
    sql = "SELECT v AS o FROM __THIS__ LEFT JOIN s0 ON (c0.f0 = s0.c0)"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (c0 STRUCT(f0 BIGINT), v BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES ({'f0': 1}, 2)")
    con.register("sa", static)
    con.execute("CREATE TABLE s0 AS SELECT * FROM sa")
    with pytest.raises(duckdb.Error, match="[Aa]mbiguous"):
        con.execute(sql).fetchall()  # oracle refuses; if this stops, remeasure

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s0": static})


# ---------------------------------------------------------------------------
# TASK-124: `in_filter` is a STATEMENT-level flag, but DuckDB's laziness is a
# property of the CONTEXT a boolean sits in, and that context recurses. Four
# divergences, both directions, all one nesting level below the top-level AND
# spine that `dfb3a99` measured and got right.
# ---------------------------------------------------------------------------
_SC_SCHEMA = pa.schema([pa.field("b", pa.bool_()), pa.field("s", pa.string())])
_SC_ROWS = [
    {"b": None, "s": "abc"},
    {"b": True, "s": "1.5"},
    {"b": False, "s": "abc"},
]
_SC_TRAP = "CAST(s AS DOUBLE) > 1"


def _sc_duck(sql: str, rows: list[dict]) -> list[tuple]:
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (b BOOLEAN, s VARCHAR)")
    for r in rows:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["b"], r["s"]])
    return con.execute(sql).fetchall()


@pytest.mark.xfail(
    strict=True,
    reason="TASK-124: selection context stops at the top-level AND spine. A "
    "conjunction nested under OR, or used as a CASE condition (in a PROJECTION, "
    "where `in_filter` is false), loses its laziness and evaluates a trapping "
    "operand DuckDB never reaches. We trap at runtime where BOTH readings serve.",
)
@pytest.mark.parametrize(
    "sql",
    [
        # nested one level under OR
        f"SELECT s AS o FROM __THIS__ WHERE (b AND {_SC_TRAP}) OR TRUE",
        # a CASE condition in a projection -- `in_filter` reports false here
        f"SELECT CASE WHEN (b AND {_SC_TRAP}) THEN 1 ELSE 2 END AS o FROM __THIS__",
        # ... and a CASE condition under a filter, which is still not the spine
        f"SELECT s AS o FROM __THIS__ WHERE CASE WHEN (b AND {_SC_TRAP}) "
        "THEN TRUE ELSE TRUE END",
    ],
)
def test_a_nested_conjunct_keeps_its_laziness(sql):
    want = _sc_duck(sql, _SC_ROWS)  # oracle serves; if this raises, remeasure
    assert len(want) == 3

    shape = "filter" if " WHERE " in sql else None
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _SC_SCHEMA}, static_tables={}, shape=shape
    )
    assert [tuple(x.values()) for x in fn.infer_rows(_SC_ROWS)] == want


@pytest.mark.xfail(
    strict=True,
    reason="TASK-124: the mirror. NOT and IS NULL read their operand as a "
    "VALUE, so DuckDB evaluates the conjunction eagerly and overflows even "
    "though the left operand is FALSE. `in_filter` blankets the whole predicate "
    "tree, so we short-circuit and ANSWER a query DuckDB refuses.",
)
@pytest.mark.parametrize(
    "pred", [f"NOT (b AND {_SC_TRAP})", f"(b AND {_SC_TRAP}) IS NULL"]
)
def test_a_boolean_in_value_context_is_eager(pred):
    sql = f"SELECT s AS o FROM __THIS__ WHERE {pred}"
    rows = [{"b": False, "s": "abc"}]

    with pytest.raises(duckdb.Error, match="Conversion Error"):
        _sc_duck(sql, rows)  # oracle refuses; if this stops, remeasure

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _SC_SCHEMA}, static_tables={}, shape="filter"
    )
    with pytest.raises(Exception, match="[Cc]onversion|cast"):
        fn.infer_rows(rows)


@pytest.mark.xfail(
    strict=True,
    reason="TASK-124, the many path: `flatten_and` runs only on the scalar "
    "route, so the join filter lowers its predicate whole and `kleene_shortcut` "
    "drops only on a DECIDING operand, never on a NULL one. The identical "
    "predicate without the join serves correctly.",
)
@pytest.mark.parametrize("join", ["JOIN", "LEFT JOIN"])
def test_a_null_conjunct_in_a_many_join_filter_drops_the_row(join):
    row = pa.schema(
        [
            pa.field("c0", pa.int64()),
            pa.field("b", pa.bool_()),
            pa.field("s", pa.string()),
        ]
    )
    static = pa.table(
        {"k": pa.array([1, 1], pa.int64()), "v": pa.array([10, 20], pa.int64())}
    )
    rows = [{"c0": 1, "b": None, "s": "abc"}, {"c0": 1, "b": True, "s": "1.5"}]
    sql = (
        f"SELECT s0.v AS o FROM __THIS__ {join} s0 ON c0 = s0.k "
        f"WHERE (b AND {_SC_TRAP})"
    )

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (c0 BIGINT, b BOOLEAN, s VARCHAR)")
    for r in rows:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?, ?)", [r["c0"], r["b"], r["s"]])
    con.execute("CREATE TABLE s0 (k BIGINT, v BIGINT)")
    con.execute("INSERT INTO s0 VALUES (1, 10), (1, 20)")
    want = con.execute(sql).fetchall()  # oracle serves; if this raises, remeasure
    # sorted: the two READINGS disagree on row order here, so order is not what
    # this pin is about -- the trap is.
    assert sorted(want) == [(10,), (20,)]

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": row}, static_tables={"s0": static}, shape="many"
    )
    assert sorted(tuple(x.values()) for x in fn.infer_rows(rows)) == sorted(want)
