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
# TASK-122: narrow `%` by -1 at the width's MIN. The one narrow overflow
# TASK-118's result-range check structurally cannot see, because the overflow
# is in the OPERATION and the result (0) is in range.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-122: `i % -1` over INT32_MIN overflows the checked division "
    "on DuckDB. We compute in the i64 lane, where it is 0, and the range check "
    "on the result sees nothing wrong with 0. `i // -1` already traps because "
    "its result leaves the width.",
)
@pytest.mark.parametrize(
    ("arrow_ty", "ddl", "lo"),
    [
        (pa.int8(), "TINYINT", -128),
        (pa.int16(), "SMALLINT", -32768),
        (pa.int32(), "INTEGER", -2147483648),
    ],
)
def test_narrow_modulo_by_minus_one_at_min_overflows(arrow_ty, ddl, lo):
    row = pa.schema([pa.field("i", arrow_ty, nullable=False)])
    sql = "SELECT (i % -1) AS o FROM __THIS__"

    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ (i {ddl})")
    con.execute("INSERT INTO __THIS__ VALUES (?)", [lo])
    with pytest.raises(duckdb.Error, match="Overflow"):
        con.execute(sql).fetchall()  # oracle overflows; if this stops, remeasure

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={})
    with pytest.raises(Exception, match="[Oo]verflow|range"):
        fn.infer_rows([{"i": lo}])


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


# ---------------------------------------------------------------------------
# TASK-125/126/127: the relation, again. `972fd72` made the DESTRUCTURE of
# `TableFactor::Table` exhaustive; these are the three places the clause is
# still lost one level below that -- in the star, in the alias's consumer, and
# in the name lookup.
# ---------------------------------------------------------------------------
_REL_ROW = pa.schema([pa.field("k", pa.int64(), nullable=False)])
_REL_ROWS = [{"k": 1}]
_S_STRUCT = pa.table(
    {
        "id": pa.array([1], pa.int64()),
        "w": pa.array(
            [{"mean": 1.5, "sd": 0.25}],
            pa.struct([("mean", pa.float64()), ("sd", pa.float64())]),
        ),
        "z": pa.array([7], pa.int64()),
    }
)
_S_OPAQUE = pa.table(
    {
        "id": pa.array([1], pa.int64()),
        "ts": pa.array([0], pa.timestamp("us")),
        "z": pa.array([7], pa.int64()),
    }
)


def _rel_duck(sql: str, ddl: str, insert: str):
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (1)")
    con.execute(ddl)
    con.execute(insert)
    res = con.execute(sql)
    return [d[0] for d in res.description], res.fetchall()


_STRUCT_DDL = "CREATE TABLE s (id BIGINT, w STRUCT(mean DOUBLE, sd DOUBLE), z BIGINT)"
_STRUCT_INS = "INSERT INTO s VALUES (1, {'mean': 1.5, 'sd': 0.25}, 7)"
_OPAQUE_DDL = "CREATE TABLE s (id BIGINT, ts TIMESTAMP, z BIGINT)"
_OPAQUE_INS = "INSERT INTO s VALUES (1, TIMESTAMP '1970-01-01', 7)"


@pytest.mark.xfail(
    strict=True,
    reason="TASK-125: the STATIC star iterates the table's lanes with no "
    "StarLane::Opaque interleave (the row star at frontend.rs:1952 has one). A "
    "struct column expands into `w.mean`/`w.sd` phantom columns; an opaque one "
    "is dropped from the output with no refusal at all.",
)
@pytest.mark.parametrize("star", ["s.*", "*"])
@pytest.mark.parametrize(
    ("static", "ddl", "insert", "unservable"),
    [
        (_S_STRUCT, _STRUCT_DDL, _STRUCT_INS, "w"),
        (_S_OPAQUE, _OPAQUE_DDL, _OPAQUE_INS, "ts"),
    ],
    ids=["struct", "opaque"],
)
def test_a_static_star_refuses_a_column_it_cannot_serve(
    star, static, ddl, insert, unservable
):
    sql = f"SELECT {star} FROM __THIS__ JOIN s ON s.id = __THIS__.k"
    names, _ = _rel_duck(sql, ddl, insert)  # oracle serves; if it raises, remeasure
    assert unservable in names

    # We serve no struct and no TIMESTAMP value, so the only contract-legal
    # answer is a refusal that NAMES the column -- never a different column set.
    with pytest.raises(ValueError, match=unservable):
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": _REL_ROW}, static_tables={"s": static}
        )
        fn.infer_rows(_REL_ROWS)


@pytest.mark.xfail(
    strict=True,
    reason="TASK-126: the JOIN arm (frontend.rs:730) and the comma arm (:925) "
    "take `alias.name.value` and drop `alias.columns`. The driving-table arm "
    "consumes it properly, so the rename works there and nowhere else.",
)
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT x.p AS o FROM __THIS__ JOIN s AS x(p, q) ON x.p = __THIS__.k",
        "SELECT x.p AS o FROM __THIS__, s AS x(p, q) WHERE x.p = __THIS__.k",
    ],
    ids=["join", "comma"],
)
def test_a_relation_column_list_alias_renames_positionally(sql):
    static = pa.table({"a": pa.array([1], pa.int64()), "b": pa.array([99], pa.int64())})
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (1)")
    con.execute("CREATE TABLE s (a BIGINT, b BIGINT)")
    con.execute("INSERT INTO s VALUES (1, 99)")
    want = con.execute(sql).fetchall()  # oracle serves; if this raises, remeasure
    assert want == [(1,)]

    shape = "filter" if " WHERE " in sql else None
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _REL_ROW}, static_tables={"s": static}, shape=shape
    )
    assert [tuple(x.values()) for x in fn.infer_rows(_REL_ROWS)] == want


@pytest.mark.xfail(
    strict=True,
    reason="TASK-126: the dropped column list also makes us ANSWER two queries "
    "DuckDB rejects -- an arity mismatch, and a reference to a name the rename "
    "took away.",
)
@pytest.mark.parametrize(
    ("sql", "oracle_msg"),
    [
        (
            "SELECT x.a AS o FROM __THIS__ JOIN s AS x(p, q, r) ON x.a = __THIS__.k",
            "columns specified",
        ),
        (
            "SELECT x.b AS o FROM __THIS__ JOIN s AS x(p, q) ON x.b = __THIS__.k",
            "does not have a column named",
        ),
    ],
    ids=["too-many-names", "renamed-away"],
)
def test_a_bad_relation_column_list_alias_refuses(sql, oracle_msg):
    static = pa.table({"a": pa.array([1], pa.int64()), "b": pa.array([99], pa.int64())})
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (1)")
    con.execute("CREATE TABLE s (a BIGINT, b BIGINT)")
    con.execute("INSERT INTO s VALUES (1, 99)")
    with pytest.raises(duckdb.Error, match=oracle_msg):
        con.execute(sql).fetchall()  # oracle refuses; if this stops, remeasure

    with pytest.raises(ValueError):
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": _REL_ROW}, static_tables={"s": static}
        )
        fn.infer_rows(_REL_ROWS)


@pytest.mark.xfail(
    strict=True,
    reason="TASK-127: flattening a static struct to `w.mean`/`w.sd` lanes takes "
    "`w` out of the name space, so EXCLUDE cannot name the column it is meant "
    "to remove. Severity 4 -- a refusal, not a wrong answer.",
)
def test_exclude_can_name_a_static_struct_column():
    sql = "SELECT s.* EXCLUDE (w) FROM __THIS__ JOIN s ON s.id = __THIS__.k"
    names, want = _rel_duck(sql, _STRUCT_DDL, _STRUCT_INS)
    assert names == ["id", "z"] and want == [(1, 7)]

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _REL_ROW}, static_tables={"s": _S_STRUCT}
    )
    assert [tuple(x.values()) for x in fn.infer_rows(_REL_ROWS)] == want
