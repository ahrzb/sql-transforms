"""Wave-5 structural + dialect forms vs the duckdb oracle.

Pins: docs/superpowers/specs/2026-07-26-wave5-structural-pins.md — colon
prefix aliases (token pre-rewrite), slices, extended subscripts, bitwise
ops, ^@/GLOB, star forms, duplicate-name contract, binder tail. The file
grows stage by stage with TASK-52.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
from confit import DuckDBInferFn
from test_duckdb_interpreter import duck_check, static

T = {"a": "int", "s": "str?"}
T_ROWS = [
    {"a": 3, "s": "hello"},
    {"a": -1, "s": None},
    {"a": 0, "s": ""},
    {"a": 7, "s": "héllo"},
]


def test_null_value_statics_vs_oracle():
    # TASK-55: NULL values in static tables flow through joins as NULL
    # (INNER and LEFT, incl. under residual predicates); NULL keys drop.
    dim = static(
        {"id": "int", "v": "int?", "w": "str?"},
        [
            {"id": 3, "v": None, "w": "x"},
            {"id": 7, "v": 70, "w": None},
        ],
    )
    duck_check(
        "SELECT a, v, w FROM __THIS__ JOIN dim ON a = dim.id",
        T,
        T_ROWS,
        {"dim": dim},
    )
    duck_check(
        "SELECT a, v, w, v + 1 AS v1 FROM __THIS__ LEFT JOIN dim ON a = dim.id",
        T,
        T_ROWS,
        {"dim": dim},
    )
    duck_check(
        "SELECT a, v FROM __THIS__ LEFT JOIN dim ON a = dim.id AND a > 2",
        T,
        T_ROWS,
        {"dim": dim},
    )
    duck_check(
        "SELECT * FROM __THIS__ LEFT JOIN dim ON a = dim.id",
        T,
        T_ROWS,
        {"dim": dim},
    )


def test_schema_qualified_relations_vs_oracle():
    # TASK-55: schema qualifiers are registry-noise; suffix match binds.
    duck_check("SELECT main.__THIS__.a FROM main.__THIS__", T, T_ROWS)


def test_binder_tail_vs_oracle():
    # NULL <op> NULL typing, lateral aliases (real column wins), main.
    # qualifier, t AS u(x,y), NATURAL JOIN, mixed-literal IN/BETWEEN.
    duck_check(
        "SELECT NULL + NULL AS s, NULL / NULL AS d, NULL = NULL AS e FROM __THIS__",
        T,
        T_ROWS,
    )
    duck_check("SELECT a % 2 AS k, k * 2 AS d FROM __THIS__ WHERE k = 1", T, T_ROWS)
    duck_check("SELECT a FROM main.__THIS__", T, T_ROWS)
    duck_check("SELECT x + 1 AS p FROM __THIS__ AS u(x)", T, T_ROWS)
    dim = static(
        {"a": "int", "label": "str"},
        [{"a": 3, "label": "three"}, {"a": 7, "label": "seven"}],
    )
    duck_check(
        "SELECT * FROM __THIS__ NATURAL JOIN dim",
        T,
        T_ROWS,
        {"dim": dim},
    )
    duck_check(
        "SELECT a IN ('1', 3) AS x, a BETWEEN '0' AND '5' AS r, "
        "true IN (1, 0) AS b FROM __THIS__",
        T,
        T_ROWS,
    )


def test_star_filters_replace_rename_vs_oracle():
    # Name filters / REPLACE / RENAME against the oracle (values + names).
    duck_check("SELECT * LIKE 'a%' FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * ILIKE 'A%' FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * NOT LIKE 's%' FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * GLOB '[as]*' FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * REPLACE (a * 2 AS a) FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * RENAME (a AS q) FROM __THIS__", T, T_ROWS)
    duck_check("SELECT * EXCLUDE (s) LIKE 'a%' FROM __THIS__", T, T_ROWS)


def test_dup_names_match_duckdb_df_contract():
    # The pinned contract: our field names == DuckDB's own .df() dedup
    # (which equals its subquery-boundary rename). Values stay positional.
    con = duckdb.connect()
    dim = static(
        {"id": "int", "v": "int"},
        [{"id": 1, "v": 10}, {"id": 2, "v": 20}],
    )
    schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(
        "SELECT * FROM __THIS__ JOIN dim ON __THIS__.id = dim.id",
        row_tables={"__THIS__": schema},
        static_tables={"dim": dim},
    )
    got = fn.infer_rows([{"id": 1}])
    con.execute("CREATE TABLE t (id BIGINT); INSERT INTO t VALUES (1)")
    con.execute("CREATE TABLE dim (id BIGINT, v BIGINT)")
    con.execute("INSERT INTO dim VALUES (1, 10), (2, 20)")
    df = con.execute("SELECT * FROM t JOIN dim ON t.id = dim.id").df()
    assert list(got[0].keys()) == list(df.columns)
    assert list(got[0].values()) == [x.item() for x in df.iloc[0].values]


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


def test_caret_at_and_glob_vs_oracle():
    duck_check("SELECT s ^@ 'h' AS p, s ^@ '' AS e FROM __THIS__", T, T_ROWS)
    duck_check(
        "SELECT s GLOB 'h*' AS g1, s GLOB 'h?llo' AS g2, s GLOB '[hé]*' AS g3, "
        "s GLOB '[!h]*' AS g4, s GLOB '[^h]ello' AS g5, s GLOB '[a-]' AS dead, "
        "s GLOB '' AS empty, NOT (s GLOB 'h*') AS neg FROM __THIS__",
        T,
        T_ROWS,
    )


def test_bitwise_ops_vs_oracle():
    # Flat precedence tier + arithmetic-vs-shift interplay + dynamic
    # operands (negatives only where << can't trap in either engine).
    duck_check(
        "SELECT 4 | 1 & 1 AS p1, 1 & 3 << 1 AS p2, 8 >> 2 | 1 AS p3, "
        "1 << 1 + 1 AS p4, xor(5, 3) AS x FROM __THIS__",
        T,
        T_ROWS,
    )
    duck_check(
        "SELECT a << 1 AS s, a >> 1 AS r, a & 3 AS n, a | 8 AS o, "
        "xor(a, 5) AS x FROM __THIS__ WHERE a >= 0",
        T,
        T_ROWS,
    )
    duck_check("SELECT a >> 65 AS z, a >> -1 AS zn FROM __THIS__", T, T_ROWS)


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


def test_joined_relation_column_list_alias_vs_oracle():
    # TASK-126: `AS x(p, q)` on a JOINED or comma relation renames
    # positionally over the DECLARED columns, exactly like the driving arm
    # (t AS u(x, y) above). The clause was silently DROPPED before, which
    # answered with the wrong names in scope in all three directions.
    dim = static({"a": "int", "b": "int"}, [{"a": 3, "b": 30}, {"a": 7, "b": 70}])
    for sql in [
        # the rename works, old spellings below refuse
        "SELECT x.p + x.q AS o FROM __THIS__ JOIN s AS x(p, q) ON x.p = a",
        # comma form
        "SELECT x.q AS o FROM __THIS__, s AS x(p, q) WHERE x.p = a",
        # PARTIAL list: prefix renames, the rest keep their names
        "SELECT x.b AS o FROM __THIS__ JOIN s AS x(p) ON x.p = a",
        # star and EXCLUDE see the NEW names
        "SELECT x.* FROM __THIS__ JOIN s AS x(p, q) ON x.p = a",
        "SELECT x.* EXCLUDE (q) FROM __THIS__ JOIN s AS x(p, q) ON x.p = a",
        # USING pairs through the new name
        "SELECT x.q AS o FROM __THIS__ JOIN s AS x(a, q) USING (a)",
    ]:
        duck_check(sql, T, T_ROWS, {"s": dim})


def test_joined_relation_column_list_alias_refusals():
    # Two directions DuckDB refuses and we must refuse too (they SERVED
    # before TASK-126), plus one deliberate cost.
    row = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    stat = pa.table({"a": pa.array([3], pa.int64()), "b": pa.array([30], pa.int64())})

    def refuses(sql, match):
        con = duckdb.connect()
        con.execute("CREATE TABLE __THIS__ (a BIGINT)")
        con.execute("INSERT INTO __THIS__ VALUES (3)")
        con.execute("CREATE TABLE s (a BIGINT, b BIGINT)")
        con.execute("INSERT INTO s VALUES (3, 30)")
        try:
            con.execute(sql).fetchall()
            oracle_served = True
        except duckdb.Error:
            oracle_served = False
        try:
            DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": stat})
            raise AssertionError(f"built: {sql}")
        except ValueError as e:
            assert match in str(e), f"{sql}: {e}"
        return oracle_served

    # arity: DuckDB's own bind error, same wording
    assert not refuses(
        "SELECT x.a AS o FROM __THIS__ JOIN s AS x(p, q, r) ON x.p = __THIS__.a",
        "columns specified",
    )
    # a renamed-away original name
    assert not refuses(
        "SELECT x.b AS o FROM __THIS__ JOIN s AS x(p, q) ON x.p = __THIS__.a",
        "does not exist",
    )
    # DELIBERATE COST (severity 4): duplicate names in the list -- DuckDB
    # serves the first match, we refuse as ambiguous. Measured 2026-08-19.
    assert refuses(
        "SELECT x.p AS o FROM __THIS__ JOIN s AS x(p, p) ON x.p = __THIS__.a",
        "ambiguous",
    )
