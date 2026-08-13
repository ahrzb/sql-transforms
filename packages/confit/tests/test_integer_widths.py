"""The integer-widths feature (m-8 phase 2, TASK-79).

DuckDB types in a width lattice (TINYINT..HUGEINT); this engine typed every
integer BIGINT, so infer_arrow's schema diverged wherever DuckDB says
INTEGER (literals, ascii, CASE over literals, ::INTEGER casts). Phase 2
types the widths for real in the frontend, erased to the i64 lane at
compute; the width is observable exactly where DuckDB's is — the trap
threshold (phase 3) and the Arrow schema (here).

Every schema expectation below is the live oracle, never a hardcoded width:
the assert is `ours == DuckDB's`, so a wrong row here is impossible.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from pydantic import create_model

In = create_model("In", k=(int, ...), s=(str, ...))
ROWS = [{"k": 0, "s": "ab"}, {"k": 2, "s": "Z"}, {"k": 30000, "s": "!"}]

# The measured catalogue (spec 2026-08-11 + probe 2026-08-13). Mix of rows
# that must NARROW (int32/int16/int8) and controls that must stay int64 or
# double — the oracle decides which is which.
CATALOGUE = [
    "SELECT 1 AS o FROM __THIS__",
    "SELECT 2147483647 AS o FROM __THIS__",
    "SELECT 2147483648 AS o FROM __THIS__",
    "SELECT -2147483647 AS o FROM __THIS__",
    "SELECT -2147483648 AS o FROM __THIS__",  # BIGINT: parsed -(2147483648)
    "SELECT CASE WHEN k > 1 THEN 1 ELSE 0 END AS o FROM __THIS__",
    "SELECT CASE WHEN k > 1 THEN 1 ELSE k END AS o FROM __THIS__",
    "SELECT 1 + 2 AS o FROM __THIS__",
    "SELECT 2000000000 + 2000000 AS o FROM __THIS__",
    "SELECT 7 // 2 AS o FROM __THIS__",
    "SELECT 7 % 2 AS o FROM __THIS__",
    "SELECT 7 / 2 AS o FROM __THIS__",
    "SELECT 1 & 2 AS o FROM __THIS__",
    "SELECT 1 | 2 AS o FROM __THIS__",
    "SELECT 1 << 2 AS o FROM __THIS__",
    "SELECT xor(1, 2) AS o FROM __THIS__",
    "SELECT k + 1 AS o FROM __THIS__",
    "SELECT k & 1 AS o FROM __THIS__",
    "SELECT -(3) AS o FROM __THIS__",
    "SELECT ascii(s) AS o FROM __THIS__",
    "SELECT unicode(s) AS o FROM __THIS__",
    "SELECT ord(s) AS o FROM __THIS__",
    "SELECT length(s) AS o FROM __THIS__",
    "SELECT bit_length(s) AS o FROM __THIS__",
    "SELECT strpos(s, 'a') AS o FROM __THIS__",
    "SELECT levenshtein(s, 'ab') AS o FROM __THIS__",
    "SELECT abs(-5) AS o FROM __THIS__",
    "SELECT abs(k) AS o FROM __THIS__",
    "SELECT round(1) AS o FROM __THIS__",
    "SELECT trunc(1) AS o FROM __THIS__",
    "SELECT round(1, 0) AS o FROM __THIS__",
    "SELECT greatest(1, 2) AS o FROM __THIS__",
    "SELECT least(1, 2) AS o FROM __THIS__",
    "SELECT greatest(1, k) AS o FROM __THIS__",
    "SELECT coalesce(1, k) AS o FROM __THIS__",
    "SELECT coalesce(1, 2) AS o FROM __THIS__",
    "SELECT nullif(1, k) AS o FROM __THIS__",
    "SELECT nullif(k, 1) AS o FROM __THIS__",
    "SELECT nullif(2, 3) AS o FROM __THIS__",
    "SELECT NULL AS o FROM __THIS__",
    "SELECT nullif(NULL, 84.754e0) AS o FROM __THIS__",
    "SELECT nullif(NULL, k) AS o FROM __THIS__",
    "SELECT nullif(NULL, NULL) AS o FROM __THIS__",
    "SELECT struct_pack(a := ascii(s), b := 1, c := NULL) AS o FROM __THIS__",
    "SELECT - NULL AS o FROM __THIS__",
    "SELECT + NULL AS o FROM __THIS__",
    "SELECT -(CAST(NULL AS INTEGER)) AS o FROM __THIS__",
    # The -2147483648 corners: the only value that is BIGINT-typed as a
    # spelling yet fits int32, so DuckDB's value-fits promotion is visible
    # on it alone (9-probe matrix 2026-08-13, all binder-level).
    "SELECT unicode(s) % -2147483648 AS o FROM __THIS__",
    "SELECT (2147483647 % -2147483648) AS o FROM __THIS__",
    "SELECT ((49 % 9007199254740991) % unicode(s)) AS o FROM __THIS__",
    "SELECT -2147483648 % CAST(-15 AS INTEGER) AS o FROM __THIS__",
    "SELECT (CAST(-45 AS INTEGER) * NULL) - -2147483648 AS o FROM __THIS__",
    "SELECT coalesce(-2147483648, (14 - -13)) AS o FROM __THIS__",
    "SELECT CASE WHEN NULL THEN 11 WHEN NULL THEN -2147483648 ELSE -24 END"
    " AS o FROM __THIS__",
    "SELECT CASE WHEN NULL THEN -2147483648 ELSE -24 END AS o FROM __THIS__",
    "SELECT CASE WHEN NULL THEN -24 ELSE -2147483648 END AS o FROM __THIS__",
    "SELECT CASE WHEN NULL THEN -2147483648 WHEN NULL THEN 11 ELSE -24 END"
    " AS o FROM __THIS__",
    "SELECT CASE WHEN NULL THEN -2147483648 ELSE (14 - -13) END AS o FROM __THIS__",
    "SELECT CASE WHEN NULL THEN (14 - -13) ELSE -2147483648 END AS o FROM __THIS__",
    "SELECT CASE WHEN k > 1 THEN unicode(s) ELSE -2147483648 END AS o FROM __THIS__",
    # Adversarial-fleet pins (2026-08-13): the CASE fold seeds from the
    # ELSE; hints are syntactic-only; SQLNULL propagates through nullif.
    "SELECT CASE WHEN k > 1 THEN -2147483648 WHEN k > 0 THEN (2147483647 + -13)"
    " END AS o FROM __THIS__",
    "SELECT coalesce(CASE WHEN k > 9 THEN 5 ELSE 44 END, 9007199254740993)"
    " AS o FROM __THIS__",
    "SELECT greatest(-2147483648, 9007199254740993, unicode(s)) AS o FROM __THIS__",
    "SELECT CAST(k AS SMALLINT) * (0 - 1000) AS o FROM __THIS__",
    "SELECT CAST(k AS SMALLINT) * +2 AS o FROM __THIS__",
    "SELECT - nullif(NULL, 1) AS o FROM __THIS__",
    "SELECT nullif(NULL, 1) * CAST(1 AS SMALLINT) AS o FROM __THIS__",
    "SELECT CAST(TRUE AS INTEGER) % coalesce(-2147483648, 7) AS o FROM __THIS__",
    # DECIMAL literal (+|-|*|%) bare NULL folds to SQLNULL = INTEGER on
    # DuckDB; division stays DOUBLE (campaign seed 617691).
    "SELECT (-2.681 + NULL) AS o FROM __THIS__",
    "SELECT 2.5 * NULL AS o FROM __THIS__",
    "SELECT NULL - 2.681 AS o FROM __THIS__",
    "SELECT 2.5 / NULL AS o FROM __THIS__",
    "SELECT NULL + 1.5e0 AS o FROM __THIS__",
    "SELECT CAST(k AS INTEGER) AS o FROM __THIS__",
    "SELECT CAST(k AS SMALLINT) AS o FROM __THIS__",
    "SELECT CAST(1 AS TINYINT) AS o FROM __THIS__",
    "SELECT CAST(k AS BIGINT) AS o FROM __THIS__",
    "SELECT TRY_CAST(k AS INTEGER) AS o FROM __THIS__",
    "SELECT CASE WHEN k > 1 THEN ascii(s) ELSE 0 END AS o FROM __THIS__",
    "SELECT 1 BETWEEN 0 AND k AS o FROM __THIS__",
    "SELECT CAST(k AS INTEGER) % 24 AS o FROM __THIS__",
]


def _duck(sql: str) -> pa.Table:
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["k"], r["s"]])
    return con.execute(sql).to_arrow_table()


def _ours(sql: str) -> pa.Table:
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    return fn.infer_arrow(pa.Table.from_pylist(ROWS))


@pytest.mark.parametrize("sql", CATALOGUE)
def test_output_width_matches_duckdb(sql):
    got, want = _ours(sql), _duck(sql)
    assert got.to_pylist() == want.to_pylist(), sql
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"


def test_try_cast_to_integer_nulls_out_of_range():
    """TRY_CAST out of the target's range is NULL on DuckDB — not a trap, so
    it is phase-2 value semantics, not phase-3 trap work."""
    sql = "SELECT TRY_CAST(k AS INTEGER) AS o FROM __THIS__"
    big = [{"k": 9007199254740993, "s": "x"}, {"k": 5, "s": "y"}]
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    got = fn.infer({"__THIS__": [In(**r) for r in big]})
    assert [r["o"] for r in got] == [None, 5]


def test_row_and_arrow_boundaries_agree_on_narrow_widths():
    """Fleet 2026-08-13: infer() served a value infer_arrow refused. The
    width contract holds on EVERY boundary — both refuse an out-of-range
    narrow value (DuckDB traps the same input; our trap is phase 3)."""
    sql = "SELECT CAST(k AS TINYINT) AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    ok = [{"k": 5, "s": "a"}]
    assert [r["o"] for r in fn.infer({"__THIS__": [In(**r) for r in ok]})] == [5]
    assert fn.infer_arrow(pa.Table.from_pylist(ok)).to_pylist() == [{"o": 5}]
    bad = [{"k": 300, "s": "a"}]
    with pytest.raises(ValueError, match="TINYINT"):
        fn.infer({"__THIS__": [In(**r) for r in bad]})
    with pytest.raises(ValueError, match="TINYINT"):
        fn.infer_arrow(pa.Table.from_pylist(bad))


def test_struct_children_hold_narrow_widths_on_both_boundaries():
    """A struct_pack child is an arbitrary expression, so it can carry a
    narrow width (int32 here via ::INTEGER). The width contract holds for
    struct children exactly as for scalar columns: BOTH boundaries refuse
    an out-of-range value by name (the row path served it silently; the
    arrow path refused with pyarrow's wording instead of ours)."""
    sql = "SELECT struct_pack(v := CAST(k AS INTEGER)) AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    ok = [{"k": 5, "s": "a"}]
    assert [r["o"] for r in fn.infer({"__THIS__": [In(**r) for r in ok]})] == [{"v": 5}]
    assert fn.infer_arrow(pa.Table.from_pylist(ok)).to_pylist() == [{"o": {"v": 5}}]
    bad = [{"k": 3000000000, "s": "a"}]
    with pytest.raises(ValueError, match="INTEGER"):
        fn.infer({"__THIS__": [In(**r) for r in bad]})
    with pytest.raises(ValueError, match="INTEGER"):
        fn.infer_arrow(pa.Table.from_pylist(bad))


def test_unary_plus_refuses_non_numerics():
    """DuckDB's + is a real unary function over numerics; +'a' is a binder
    error there (fleet 2026-08-13 — we built and served it)."""
    for sql in [
        "SELECT + s AS o FROM __THIS__",
        "SELECT +('a') AS o FROM __THIS__",
    ]:
        with pytest.raises(ValueError, match=r"\+\("):
            DuckDBInferFn(sql, row_tables={"__THIS__": In}, static_tables={})


def test_out_of_range_dynamic_int32_refuses_at_emit_not_wraps():
    """CAST(k AS INTEGER) on an out-of-range k TRAPS on DuckDB. Our dynamic
    trap is phase 3; until then the int32 EMIT must refuse by name rather
    than wrap — every input this refuses is an input DuckDB errors on too."""
    sql = "SELECT CAST(k AS INTEGER) AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    with pytest.raises(Exception, match="int32|INT32|INTEGER"):
        fn.infer_arrow(pa.Table.from_pylist([{"k": 9007199254740993, "s": "x"}]))


# ---- Decided divergences, pinned xfail-strict until their PRs land ----
# (AmirHossein 2026-08-13: "do what duckdb does" — bug-for-bug.)


@pytest.mark.xfail(
    strict=True,
    reason="TASK-101 decided 2026-08-13: DuckDB executes a pure "
    "(side_effects=False, its default) UDF at bind when a fold context "
    "asks for its constant-args value — field access is one — and a None "
    "result is SQLNULL/int32. We mirror: pure-by-default UDF bind fold, "
    "side_effects=True opt-out. Flips when the fold lands.",
)
def test_pure_udf_bind_fold_matches_duckdb_schema():
    """Seed 601418, the last width finding: `(udf(1, NULL)).f1` types int32
    on DuckDB (the bind fold RUNS the udf — special null handling honored —
    and this udf returns None for NULL args) but int64 here (declared field
    type; we never execute user code at build yet)."""

    class U:
        name = "udf9"
        takes = pa.schema([("a", pa.int64()), ("b", pa.int64())])
        returns = pa.struct([("f1", pa.int64())])

        def __call__(self, a, b):
            if a is None or b is None:
                return None
            return (a + b,)

    sql = "SELECT (udf9(1, NULL)).f1 AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": In}, static_tables={}, udfs=[U()])
    ours = fn.infer_arrow(pa.Table.from_pylist(ROWS)).schema

    con = duckdb.connect()
    u = U()
    con.create_function(
        "udf9",
        lambda a, b: None if (r := u(a, b)) is None else {"f1": r[0]},
        ["BIGINT", "BIGINT"],
        "STRUCT(f1 BIGINT)",
        null_handling="special",
    )
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["k"], r["s"]])
    duck = con.execute(sql).to_arrow_table().schema
    assert duck.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert ours == duck, f"{ours} != {duck}"


@pytest.mark.xfail(
    strict=True,
    reason="TASK-102 decided 2026-08-13: || with a bind-foldable "
    "constant-NULL operand is SQLNULL/int32 on DuckDB (concat-specific; "
    "+, LIKE, functions keep their promoted type). Ours types VARCHAR — "
    "the old §5 contract row, now decided to align. Flips when the fold "
    "arm lands.",
)
def test_concat_with_foldable_null_operand_is_sqlnull():
    sql = "SELECT s || NULL AS o FROM __THIS__"
    got, want = _ours(sql), _duck(sql)
    assert want.schema.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
