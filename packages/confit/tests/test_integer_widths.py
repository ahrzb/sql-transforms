"""The integer-widths feature (m-8 phase 2).

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

IN_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)
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
    "SELECT round(1.567e0, CAST(2 AS TINYINT)) AS o FROM __THIS__",
    "SELECT trunc(1.567e0, CAST(k AS INTEGER)) AS o FROM __THIS__",
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
    # DOUBLE-typed foldable NULL does NOT collapse (control for the fold).
    "SELECT (- (CASE WHEN FALSE THEN 1.5e0 END)) AS o FROM __THIS__",
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})
    got = fn.infer_rows(big)
    assert [r["o"] for r in got] == [None, 5]


def test_row_and_arrow_boundaries_agree_on_narrow_widths():
    """Fleet 2026-08-13: infer() served a value infer_arrow refused. The
    width contract holds on EVERY boundary — both refuse an out-of-range
    narrow value (DuckDB traps the same input; our trap is phase 3)."""
    sql = "SELECT CAST(k AS TINYINT) AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})
    ok = [{"k": 5, "s": "a"}]
    assert [r["o"] for r in fn.infer_rows(ok)] == [5]
    assert fn.infer_arrow(pa.Table.from_pylist(ok)).to_pylist() == [{"o": 5}]
    bad = [{"k": 300, "s": "a"}]
    with pytest.raises(ValueError, match="int8"):
        fn.infer_rows(bad)
    with pytest.raises(ValueError, match="int8"):
        fn.infer_arrow(pa.Table.from_pylist(bad))


def test_struct_children_hold_narrow_widths_on_both_boundaries():
    """A struct_pack child is an arbitrary expression, so it can carry a
    narrow width (int32 here via ::INTEGER). The width contract holds for
    struct children exactly as for scalar columns: BOTH boundaries refuse
    an out-of-range value by name (the row path served it silently; the
    arrow path refused with pyarrow's wording instead of ours)."""
    sql = "SELECT struct_pack(v := CAST(k AS INTEGER)) AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})
    ok = [{"k": 5, "s": "a"}]
    assert [r["o"] for r in fn.infer_rows(ok)] == [{"v": 5}]
    assert fn.infer_arrow(pa.Table.from_pylist(ok)).to_pylist() == [{"o": {"v": 5}}]
    bad = [{"k": 3000000000, "s": "a"}]
    with pytest.raises(ValueError, match="int32"):
        fn.infer_rows(bad)
    with pytest.raises(ValueError, match="int32"):
        fn.infer_arrow(pa.Table.from_pylist(bad))


def test_unary_plus_refuses_non_numerics():
    """DuckDB's + is a real unary function over numerics; +'a' is a binder
    error there (fleet 2026-08-13 — we built and served it)."""
    for sql in [
        "SELECT + s AS o FROM __THIS__",
        "SELECT +('a') AS o FROM __THIS__",
    ]:
        with pytest.raises(ValueError, match=r"\+\("):
            DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})


def test_out_of_range_dynamic_int32_refuses_at_emit_not_wraps():
    """CAST(k AS INTEGER) on an out-of-range k TRAPS on DuckDB. Our dynamic
    trap is phase 3; until then the int32 EMIT must refuse by name rather
    than wrap — every input this refuses is an input DuckDB errors on too."""
    sql = "SELECT CAST(k AS INTEGER) AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})
    with pytest.raises(Exception, match="int32|INT32|INTEGER"):
        fn.infer_arrow(pa.Table.from_pylist([{"k": 9007199254740993, "s": "x"}]))


# ---- Decided divergences, pinned against the live oracle ----
# (AmirHossein 2026-08-13: "do what duckdb does" — bug-for-bug.)


# Decided 2026-08-13 (spec 2026-08-13-bind-fold-alignment):
# DuckDB executes a pure (side_effects=False, its default) UDF at BIND when
# a fold context asks for its constant-args value, honoring special null
# handling — the real result is used, never assumed. A whole-call-None
# result is SQLNULL/int32 under field access and ||; a real struct keeps
# its declared field types (NULL fields included); a raising callable
# fails the BUILD under field access but is swallowed under || (measured).


class _StructUdf:
    """The seed-601418 family: None on any NULL arg, else a real struct."""

    name = "udf9"
    takes = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    returns = pa.struct([("f1", pa.int64())])

    def __init__(self, on_null=None):
        self.calls = 0
        self.on_null = on_null

    def __call__(self, a, b):
        self.calls += 1
        if a is None or b is None:
            return self.on_null
        return (a + b,)


def _ours_udf(sql, *udfs):
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={}, udfs=list(udfs)
    )
    return fn.infer_arrow(pa.Table.from_pylist(ROWS))


def _duck_udf(sql, *udfs):
    con = duckdb.connect()
    for u in udfs:
        names = [f.name for f in u.returns] if pa.types.is_struct(u.returns) else None
        duck_ret = (
            "STRUCT(" + ", ".join(f"{n} BIGINT" for n in names) + ")"
            if names
            else "VARCHAR"
        )
        con.create_function(
            u.name,
            (
                lambda uu, nn: (
                    lambda a, b: (
                        None
                        if (r := uu(a, b)) is None
                        else dict(zip(nn, r, strict=True))
                        if nn
                        else r[0]
                    )
                )
            )(u, names),
            ["BIGINT"] * len(u.takes),
            duck_ret,
            null_handling="special",
        )
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["k"], r["s"]])
    return con.execute(sql).to_arrow_table()


def test_pure_udf_bind_fold_matches_duckdb_schema():
    """Whole-call None under field access is SQLNULL/int32 — both engines."""
    sql = "SELECT (udf9(1, NULL)).f1 AS o FROM __THIS__"
    ours, duck = _ours_udf(sql, _StructUdf()), _duck_udf(sql, _StructUdf())
    assert duck.schema.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert ours.to_pylist() == duck.to_pylist()
    assert ours.schema == duck.schema, f"{ours.schema} != {duck.schema}"


def test_pure_udf_fold_uses_the_real_result():
    """Special null handling holds through the fold: a udf answering 99 for
    NULL args folds to 99/BIGINT — executed once at build, never assumed."""
    u = _StructUdf(on_null=(99,))
    sql = "SELECT (udf9(1, NULL)).f1 AS o FROM __THIS__"
    ours = _ours_udf(sql, u)
    assert u.calls == 1, "the bind fold runs the callable exactly once"
    duck = _duck_udf(sql, _StructUdf(on_null=(99,)))
    assert duck.schema.field("o").type == pa.int64(), "oracle moved — remeasure"
    assert ours.to_pylist() == duck.to_pylist() == [{"o": 99}] * len(ROWS)
    assert ours.schema == duck.schema


def test_pure_udf_fold_keeps_declared_type_for_null_fields():
    """A valid call whose FIELD is None keeps the declared BIGINT (measured:
    the SQLNULL collapse is whole-call-None only)."""
    sql = "SELECT (udf9(1, NULL)).f1 AS o FROM __THIS__"
    ours = _ours_udf(sql, _StructUdf(on_null=(None,)))
    duck = _duck_udf(sql, _StructUdf(on_null=(None,)))
    assert duck.schema.field("o").type == pa.int64(), "oracle moved — remeasure"
    assert ours.schema == duck.schema
    assert ours.to_pylist() == duck.to_pylist()


def test_side_effects_udf_is_never_executed_at_build():
    """side_effects=True (DuckDB's own flag and default-False semantics)
    opts out of the fold entirely: no build-time call, declared schema."""
    u = _StructUdf()
    u.side_effects = True
    sql = "SELECT (udf9(1, NULL)).f1 AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={}, udfs=[u]
    )
    assert u.calls == 0, "a side-effectful udf must never run at build"
    ours = fn.infer_arrow(pa.Table.from_pylist(ROWS))
    assert ours.schema.field("o").type == pa.int64()


def test_raising_pure_udf_keeps_the_runtime_call():
    """DuckDB's fold SWALLOWS a raising callable UNIFORMLY (review
    2026-08-13: DESCRIBE succeeds typed by the declaration, a zero-row
    batch answers empty, the error fires at RUN with rows — an earlier
    'errors at bind' probe was FROM-less eager evaluation, not the
    binder). So: build succeeds, schema stays int64, rows raise at run."""

    class Boom(_StructUdf):
        def __call__(self, a, b):
            self.calls += 1
            raise ValueError("user code exploded")

    u = Boom()
    sql = "SELECT (udf9(1, 2)).f1 AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={}, udfs=[u]
    )
    assert u.calls == 1, "the fold TRIED (and swallowed the raise)"
    empty = pa.Table.from_pylist(
        [], schema=pa.schema([("k", pa.int64()), ("s", pa.string())])
    )
    ours_schema = fn.infer_arrow(empty).schema
    # The non-raising twin folds args (1, 2) to the VALUE 3 — BIGINT; our
    # unfolded runtime call types the declared int64. Same schema.
    duck = _duck_udf(sql, _StructUdf())
    assert duck.schema.field("o").type == pa.int64(), "oracle moved"
    assert ours_schema.field("o").type == pa.int64()  # unfolded, declared
    with pytest.raises(Exception, match="user code exploded"):
        fn.infer_arrow(pa.Table.from_pylist(ROWS))


ADOPTION_BATTERY = [
    # SQLNULL re-promotes by signature on DuckDB: these are BIGINT there,
    # NOT int32 — the fold result must ride the adoptable-NULL channel
    # (review 2026-08-13; `abs(CAST(NULL AS INTEGER))` is the int32
    # control, a genuinely TYPED null).
    "SELECT abs((udf9(1, NULL)).f1) AS o FROM __THIS__",
    "SELECT - ((udf9(1, NULL)).f1) AS o FROM __THIS__",
    "SELECT + ((udf9(1, NULL)).f1) AS o FROM __THIS__",
    "SELECT abs(s || NULL) AS o FROM __THIS__",
    "SELECT - (s || NULL) AS o FROM __THIS__",
]


@pytest.mark.parametrize("sql", ADOPTION_BATTERY)
def test_sqlnull_fold_results_adopt_like_bare_null(sql):
    ours, duck = _ours_udf(sql, _StructUdf()), _duck_udf(sql, _StructUdf())
    assert duck.schema.field("o").type == pa.int64(), "oracle moved — remeasure"
    assert ours.to_pylist() == duck.to_pylist(), sql
    assert ours.schema == duck.schema, f"{sql}: {ours.schema} != {duck.schema}"


class _ScalarStrUdf:
    name = "us9"
    takes = pa.schema([("a", pa.int64()), ("b", pa.int64())])
    returns = pa.string()

    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    def __call__(self, a, b):
        self.calls += 1
        if self.result == "raise":
            raise ValueError("boom")
        return self.result


def test_pure_scalar_udf_null_under_concat_collapses():
    """A pure scalar udf operand folding to None makes || SQLNULL/int32 —
    the concat collapse reads through the bind-time udf fold."""
    sql = "SELECT us9(1, 2) || s AS o FROM __THIS__"
    ours = _ours_udf(sql, _ScalarStrUdf(result=None))
    duck = _duck_udf(sql, _ScalarStrUdf(result=None))
    assert duck.schema.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert ours.to_pylist() == duck.to_pylist()
    assert ours.schema == duck.schema


def test_raising_pure_udf_under_concat_stays_runtime():
    """The uniform swallow, || spelling: the fold TRIES (one call), the
    raise is swallowed, the runtime call stays and errors at RUN."""
    u = _ScalarStrUdf(result="raise")
    sql = "SELECT us9(1, 2) || s AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={}, udfs=[u]
    )
    assert u.calls == 1, "the fold tried once and swallowed"
    with pytest.raises(Exception, match="boom"):
        fn.infer_arrow(pa.Table.from_pylist(ROWS))


def test_pure_scalar_udf_value_under_concat_bakes_once():
    """A folded VALUE bakes in as a literal: DuckDB executes the udf once
    at bind and never per row — matching call counts and values."""
    u = _ScalarStrUdf(result=("x",))
    sql = "SELECT us9(1, 2) || s AS o FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={}, udfs=[u]
    )
    assert u.calls == 1
    ours = fn.infer_arrow(pa.Table.from_pylist(ROWS))
    assert u.calls == 1, "the baked literal never re-executes the udf"
    duck = _duck_udf(sql, _ScalarStrUdf(result=("x",)))
    assert ours.to_pylist() == duck.to_pylist() == [{"o": "x" + r["s"]} for r in ROWS]
    assert ours.schema == duck.schema


# Measured 2026-08-19: the bind-fold finishes what DuckDB's does
# on these spellings — Abs and upper/lower fold over literals (same
# kernels as the runtime), and a pure extern bakes under a stack of
# unary wrappers, so the || SQLNULL collapse sees through upper(us9(..)).
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT upper(us9(1, 2)) || s AS o FROM __THIS__",
        "SELECT (udf9(abs(-3), NULL)).f1 AS o FROM __THIS__",
    ],
)
def test_bind_fold_composition_gaps(sql):
    ours = _ours_udf(sql, _StructUdf(), _ScalarStrUdf(result=None))
    duck = _duck_udf(sql, _StructUdf(), _ScalarStrUdf(result=None))
    assert duck.schema.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert ours.schema == duck.schema, f"{sql}: {ours.schema} != {duck.schema}"


# Decided 2026-08-13: || with an operand that FOLDS to NULL is an SQLNULL
# constant on DuckDB — int32 at the boundary, the column side
# notwithstanding (|| propagates NULL to every row). Concat-specific:
# +, LIKE, unary minus and function calls keep their promoted type, and
# concat() the function skips NULLs. Measured bind-time (DESCRIBE agrees).
CONCAT_NULL_BATTERY = [
    "SELECT s || NULL AS o FROM __THIS__",
    "SELECT s || CAST(NULL AS VARCHAR) AS o FROM __THIS__",
    "SELECT upper(NULL) || s AS o FROM __THIS__",
    "SELECT nullif('a', 'a') || s AS o FROM __THIS__",
    "SELECT (CASE WHEN 1 = 0 THEN 'x' END) || s AS o FROM __THIS__",
    "SELECT (NULL || 'a') || s AS o FROM __THIS__",
]


@pytest.mark.parametrize("sql", CONCAT_NULL_BATTERY)
def test_concat_with_foldable_null_operand_is_sqlnull(sql):
    got, want = _ours(sql), _duck(sql)
    assert want.schema.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert got.to_pylist() == want.to_pylist(), sql
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"


def test_concat_with_unfoldable_null_operand_stays_varchar():
    """The foldability boundary: a column inside the CASE blocks the fold,
    so DuckDB keeps the bound VARCHAR type — and so must we."""
    sql = "SELECT (CASE WHEN 1 = 0 THEN s END) || 'a' AS o FROM __THIS__"
    got, want = _ours(sql), _duck(sql)
    assert want.schema.field("o").type == pa.string(), "oracle moved — remeasure"
    assert got.to_pylist() == want.to_pylist(), sql
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (- (CASE WHEN FALSE THEN 1.25 END)) AS o FROM __THIS__",
        "SELECT (1.25 + (CASE WHEN FALSE THEN 1.25 END)) AS o FROM __THIS__",
    ],
)
def test_decimal_arith_over_foldable_null_collapses(sql):
    got, want = _ours(sql), _duck(sql)
    assert want.schema.field("o").type == pa.int32(), "oracle moved — remeasure"
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"


# The round/trunc DIGITS slot accepts INTEGER-or-narrower on DuckDB; a
# BIGINT digits expression (column or wide literal) is a binder error
# there. Both sides live-oracle.
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT round(1.567e0, k) AS o FROM __THIS__",
        "SELECT trunc(1.567e0, k) AS o FROM __THIS__",
        "SELECT round(1.567e0, 9007199254740993) AS o FROM __THIS__",
    ],
)
def test_round_trunc_bigint_digits_refuse_like_duckdb(sql):
    with pytest.raises(duckdb.BinderException, match="No function matches"):
        _duck(sql)
    with pytest.raises(ValueError, match="no function matches"):
        DuckDBInferFn(sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={})


# A STATIC column's width is its arrow declaration, exactly as a row column's
# is. A catalogue path with a parser of its own collapses every integer width
# to int64 — a joined int32 payload then emits int64 where DuckDB emits int32.
# Two parsers for one physical vocabulary is the whole defect.
@pytest.mark.parametrize(
    ("arrow_ty", "name"),
    [
        (pa.int8(), "int8"),
        (pa.int16(), "int16"),
        (pa.int32(), "int32"),
        (pa.int64(), "int64"),
        (pa.float64(), "double"),
        (pa.string(), "string"),
        (pa.bool_(), "bool"),
    ],
)
def test_static_column_types_at_its_arrow_width(arrow_ty, name):
    payload = {
        "int8": 7,
        "int16": 7,
        "int32": 7,
        "int64": 7,
        "double": 7.5,
        "string": "x",
        "bool": True,
    }[name]
    static = pa.table(
        {"id": pa.array([5], pa.int64()), "v": pa.array([payload], arrow_ty)}
    )
    sql = "SELECT s.v AS o FROM __THIS__ JOIN s ON k = s.id"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["k"], r["s"]])
    con.register("s_arrow", static)
    con.execute("CREATE TABLE s AS SELECT * FROM s_arrow")
    want = con.execute(sql).to_arrow_table()

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": IN_SCHEMA}, static_tables={"s": static}
    )
    got = fn.infer_arrow(pa.Table.from_pylist(ROWS))
    assert want.schema.field("o").type == arrow_ty, "oracle moved — remeasure"
    assert got.schema == want.schema, f"{name}: {got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


# The same rule for a KEY column, which the value-column rule above does not
# reach. A projected static key is reconstructed from the DYNAMIC side
# (measured: on a match the two are equal; on a LEFT miss it is NULL, never
# coalesced), so a reconstruction that adopts the ROW column's declaration is
# wrong in BOTH directions — which is why both are parametrized here: DuckDB
# compares across widths NUMERICALLY, so a match already proves the value
# fits both declarations and only the declared width was ever in question.
_KEY_DDL = {pa.int8(): "TINYINT", pa.int32(): "INTEGER", pa.int64(): "BIGINT"}


@pytest.mark.parametrize("kind", ["JOIN", "LEFT JOIN"])
@pytest.mark.parametrize(
    ("row_ty", "static_ty"),
    [
        (pa.int8(), pa.int64()),  # narrow row key, wide static
        (pa.int64(), pa.int8()),  # wide row key, narrow static
        (pa.int32(), pa.int32()),  # control: same width, unchanged
    ],
)
def test_projected_static_key_takes_the_static_columns_width(row_ty, static_ty, kind):
    row = pa.schema([pa.field("k", row_ty, nullable=False)])
    static = pa.table({"c0": pa.array([5], static_ty), "v": pa.array([7], pa.int64())})
    sql = f"SELECT s.c0 AS o FROM __THIS__ {kind} s ON k = s.c0"

    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ (k {_KEY_DDL[row_ty]})")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = con.execute(sql).to_arrow_table()

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": static})
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=row))
    assert want.schema.field("o").type == static_ty, "oracle moved — remeasure"
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


def test_a_left_miss_on_the_key_is_still_null():
    """The re-declaration must not disturb what it rides on: a LEFT miss is
    NULL on the key column, never coalesced to the probe's value."""
    row = pa.schema([pa.field("k", pa.int8(), nullable=False)])
    static = pa.table({"c0": pa.array([5], pa.int64()), "v": pa.array([7], pa.int64())})
    sql = "SELECT s.c0 AS o FROM __THIS__ LEFT JOIN s ON k = s.c0"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k TINYINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5), (6)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = con.execute(sql).to_arrow_table()

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": static})
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}, {"k": 6}], schema=row))
    assert want.to_pylist() == [{"o": 5}, {"o": None}], "oracle moved — remeasure"
    assert got.to_pylist() == want.to_pylist()
    assert got.schema == want.schema


# ===========================================================================
# The one pairing re-declaration cannot fix — a DOUBLE probe against an
# INTEGER static key. The comparison happens in double space, so
# reconstructing the key from the probe holds a double, and a double does not
# name one i64. The answer is not a smarter conversion: the static column,
# when projected, is read as a real probe VALUE lane (its own declared type,
# its own real values) and never rebuilt from the comparison.
#
# Measured 2026-08-25 against optimizer-off DuckDB, static {2**53, 2**53+1}
# (both float() to 9007199254740992.0), probe 9007199254740992.0:
#
#   SELECT s.c0, s.v   ->  (9007199254740993, 71), (9007199254740992, 70)
#
# i.e. TWO output rows, each carrying its OWN i64 — the fan-out our 'many'
# shape already produces, and the duplicate-key refusal the unique shapes
# already give. Only the projection was ever wrong.
_P53 = 9007199254740992  # 2**53
_P53_1 = 9007199254740993  # 2**53+1, float() == float(_P53)
_P53_2 = 9007199254740994  # 2**53+2, exactly representable

_KEY_DDL[pa.int16()] = "SMALLINT"  # the width parametrization needs all four

_D_ROW = pa.schema([pa.field("k", pa.float64(), nullable=False)])
_D_ONE = pa.table(
    {"c0": pa.array([_P53_1], pa.int64()), "v": pa.array([7], pa.int64())}
)
_D_PAIR = pa.table(
    {
        "c0": pa.array([_P53, _P53_1], pa.int64()),
        "v": pa.array([70, 71], pa.int64()),
    }
)


def _double_probe_duck(sql, static, ddl="k DOUBLE", probe=((float(_P53),),)):
    """The live oracle for a DOUBLE-probe join (optimizer-off per conftest)."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ ({ddl})")
    for r in probe:
        marks = ", ".join(["?"] * len(r))
        con.execute(f"INSERT INTO __THIS__ VALUES ({marks})", list(r))
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    return con.execute(sql).to_arrow_table()


def test_a_double_probe_against_an_integer_key_serves_the_static_value():
    """AC #4, the rewrite: the pairing that used to refuse now serves."""
    static = pa.table({"c0": pa.array([5], pa.int64()), "v": pa.array([7], pa.int64())})
    sql = "SELECT s.c0 AS o FROM __THIS__ JOIN s ON k = s.c0"
    want = _double_probe_duck(sql, static, probe=((5.0,),))

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _D_ROW}, static_tables={"s": static}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5.0}], schema=_D_ROW))
    assert want.schema.field("o").type == pa.int64(), "oracle moved — remeasure"
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


def test_a_double_probe_key_serves_the_i64_its_double_cannot_name():
    """THE regression test: one static row, so no multiplicity is involved —
    only whether the projected key is the REAL i64 or the probe's double read
    back. The oracle says 9007199254740993; a reconstruction says ...992."""
    # The premise the whole ticket rides on: above 2**53 an i64 shares its
    # double with a neighbour, so no double names one i64.
    assert float(_P53) == float(_P53_1) != float(_P53_2), "the 2**53 boundary moved"
    sql = "SELECT s.c0 AS o FROM __THIS__ JOIN s ON k = s.c0"
    want = _double_probe_duck(sql, _D_ONE)
    assert want.to_pylist() == [{"o": _P53_1}], "oracle moved — remeasure"

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _D_ROW}, static_tables={"s": _D_ONE}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": float(_P53)}], schema=_D_ROW))
    assert got.to_pylist() == want.to_pylist()
    assert got.schema == want.schema


@pytest.mark.parametrize("kind", ["JOIN", "LEFT JOIN"])
@pytest.mark.parametrize("proj", ["s.c0", "s.v"])
@pytest.mark.parametrize("static_ty", [pa.int8(), pa.int16(), pa.int32(), pa.int64()])
def test_a_double_probe_serves_every_integer_static_key_width(static_ty, proj, kind):
    """`promote_key` used to serve only the i64 arm, so int8/int16/int32
    static keys refused the WHOLE join. Schema equality carries the width."""
    static = pa.table({"c0": pa.array([5], static_ty), "v": pa.array([7], pa.int64())})
    sql = f"SELECT {proj} AS o FROM __THIS__ {kind} s ON k = s.c0"
    want = _double_probe_duck(sql, static, probe=((5.0,),))
    expect = static_ty if proj == "s.c0" else pa.int64()
    assert want.schema.field("o").type == expect, "oracle moved — remeasure"

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _D_ROW}, static_tables={"s": static}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5.0}], schema=_D_ROW))
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


def test_f64_colliding_static_keys_fan_out_under_shape_many():
    """Two i64 build rows sharing one double are TWO rows, each with its own
    key value. Sorted-multiset compare, the stage-B multiplicity contract."""
    sql = "SELECT s.c0 AS o, s.v AS w FROM __THIS__ JOIN s ON k = s.c0"
    want = _double_probe_duck(sql, _D_PAIR)
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": _D_ROW},
        static_tables={"s": _D_PAIR},
        shape="many",
    )
    got = fn.infer_rows([{"k": float(_P53)}])
    key = lambda r: tuple(r.values())  # noqa: E731
    assert sorted(want.to_pylist(), key=key) == [
        {"o": _P53, "w": 70},
        {"o": _P53_1, "w": 71},
    ], "oracle moved — remeasure"
    assert sorted(got, key=key) == sorted(want.to_pylist(), key=key)


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [({}, "JOIN"), ({"shape": "map"}, "LEFT JOIN")],
)
def test_f64_colliding_static_keys_still_refuse_under_the_unique_shapes(kwargs, kind):
    """The fix must not weaken the shape contract: the unique shapes still
    refuse a static whose keys collide, value-only projection included."""
    with pytest.raises(ValueError, match="duplicate map key"):
        DuckDBInferFn(
            f"SELECT s.v AS w FROM __THIS__ {kind} s ON k = s.c0",
            row_tables={"__THIS__": _D_ROW},
            static_tables={"s": _D_PAIR},
            **kwargs,
        ).infer_rows([{"k": float(_P53)}])


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM __THIS__ JOIN s ON k = s.c0",
        "SELECT c0 AS o FROM __THIS__ JOIN s ON k = s.c0",
        "SELECT s.c0 + 1 AS o FROM __THIS__ JOIN s ON k = s.c0",
        "SELECT s.c0 = 9007199254740993 AS o FROM __THIS__ JOIN s ON k = s.c0",
        "SELECT s.c0 AS o FROM __THIS__ JOIN s ON k = s.c0 AND s.c0 > 3",
    ],
)
def test_a_double_probe_key_projects_through_star_and_expressions(sql):
    """The real value has to reach every consumer of the name, not just the
    qualified projection: star, the bare name, arithmetic, a comparison
    against the literal only the i64 can equal, and an ON residual."""
    want = _double_probe_duck(sql, _D_ONE)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _D_ROW}, static_tables={"s": _D_ONE}
    )
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": float(_P53)}], schema=_D_ROW))
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist(), sql


@pytest.mark.parametrize(
    ("sql", "want_ty"),
    [
        ("SELECT c0 AS o FROM __THIS__ JOIN s USING (c0)", pa.float64()),
        ("SELECT s.c0 AS o FROM __THIS__ JOIN s USING (c0)", pa.int64()),
    ],
)
def test_a_using_join_keeps_merging_the_double_probe_key(sql, want_ty):
    """The shadow value lane must stay invisible to NAME resolution: a USING
    key merges into the LEFT occurrence (the probe's DOUBLE), while the
    qualified static spelling still reaches the real i64."""
    row = pa.schema([pa.field("c0", pa.float64(), nullable=False)])
    want = _double_probe_duck(sql, _D_ONE, ddl="c0 DOUBLE")
    assert want.schema.field("o").type == want_ty, "oracle moved — remeasure"

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": _D_ONE})
    got = fn.infer_arrow(pa.Table.from_pylist([{"c0": float(_P53)}], schema=row))
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


def test_a_using_star_keeps_one_merged_double_column():
    """`SELECT *` over USING: one merged c0 at the probe's DOUBLE, and the
    shadow lane must not surface as a phantom second column."""
    row = pa.schema([pa.field("c0", pa.float64(), nullable=False)])
    sql = "SELECT * FROM __THIS__ JOIN s USING (c0)"
    want = _double_probe_duck(sql, _D_ONE, ddl="c0 DOUBLE")
    assert [f.name for f in want.schema] == ["c0", "v"], "oracle moved — remeasure"

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": _D_ONE})
    got = fn.infer_arrow(pa.Table.from_pylist([{"c0": float(_P53)}], schema=row))
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


def test_a_left_miss_on_a_double_probe_key_is_still_null():
    """The lossy pairing's mirror of `test_a_left_miss_on_the_key_is_still_null`:
    a miss is NULL on the key column, never coalesced to the probe's value."""
    sql = "SELECT s.c0 AS o FROM __THIS__ LEFT JOIN s ON k = s.c0"
    want = _double_probe_duck(sql, _D_ONE, probe=((float(_P53),), (1.5,)))
    assert want.to_pylist() == [{"o": _P53_1}, {"o": None}], "oracle moved — remeasure"

    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _D_ROW}, static_tables={"s": _D_ONE}
    )
    got = fn.infer_arrow(
        pa.Table.from_pylist([{"k": float(_P53)}, {"k": 1.5}], schema=_D_ROW)
    )
    assert got.to_pylist() == want.to_pylist()
    assert got.schema == want.schema


# A static column whose arrow type the engine does not serve at that type
# must REFUSE BY NAME, not get widened into a neighbouring lane. Measured
# 2026-08-15, the widening was a live divergence in both directions:
#
#   float32 static, s.v * 3.0   duck 0.30000001192092896 FLOAT
#                               ours 0.30000000447034836 DOUBLE
#   uint64  static, s.v         duck 7 UINT64   ours 7 INT64
#
# The ROW path already refuses both for exactly this reason; only the
# catalogue widened them, because it used to run its own parser.
_STATIC_ROW = pa.schema([pa.field("k", pa.int64(), nullable=False)])


def _static_fn(arrow_ty, val, expr="s.v"):
    static = pa.table({"id": pa.array([5], pa.int64()), "v": pa.array([val], arrow_ty)})
    return DuckDBInferFn(
        f"SELECT {expr} AS o FROM __THIS__ JOIN s ON k = s.id",
        row_tables={"__THIS__": _STATIC_ROW},
        static_tables={"s": static},
    )


@pytest.mark.parametrize(
    ("arrow_ty", "val"),
    [
        (pa.float32(), 0.1),
        (pa.uint8(), 7),
        (pa.uint16(), 7),
        (pa.uint32(), 7),
        (pa.uint64(), 7),
    ],
)
def test_unserved_static_type_refuses_by_name(arrow_ty, val):
    with pytest.raises(ValueError, match="'v'") as e:
        _static_fn(arrow_ty, val)
    msg = str(e.value)
    # "does not exist" is the lie the catalogue used to tell: it dropped the
    # column, so the binder truthfully could not find a column that IS there.
    assert "does not exist" not in msg, msg
    assert str(arrow_ty) in msg, msg


def test_unserved_static_type_unreferenced_still_builds():
    """Opaque-unless-referenced, the same rule the row path follows."""
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "v": pa.array([0.1], pa.float32()),
            "ok": pa.array([2], pa.int32()),
        }
    )
    fn = DuckDBInferFn(
        "SELECT s.ok AS o FROM __THIS__ JOIN s ON k = s.id",
        row_tables={"__THIS__": _STATIC_ROW},
        static_tables={"s": static},
    )
    assert fn.infer_rows([{"k": 5}]) == [{"o": 2}]


def test_large_string_static_still_serves():
    """Measured equivalent — DuckDB normalises large_string to VARCHAR, so
    this one is NOT a divergence and stays served."""
    assert _static_fn(pa.large_string(), "x", "upper(s.v)").infer_rows([{"k": 5}]) == [
        {"o": "X"}
    ]


# ===========================================================================
# The range trap (m-8 phase 3, the trap half of the width feature)
#
# The erase strategy — narrow widths compute in the i64 lane — is sound
# exactly while the range trap fires wherever DuckDB's does. Checking only
# the OUTPUT boundary leaves a narrow result that leaves through a WIDER
# type unchecked entirely: `CAST((i + 1) AS BIGINT)` over INT32_MAX then
# serves 2147483648, a value DuckDB never produces, with no refusal
# anywhere. A comparison, a function argument and a float promotion
# hid it the same way.
#
# The check now lands on the RESULT, at the point of production, which
# immediately raises the harder half: DuckDB's optimizer SIMPLIFIES
# `x ± c <cmp> k` to `x <cmp> k∓c`, so the addition never runs there and
# `(i + 1) > 5` serves where `(i + 1)` alone traps. That rewrite is
# reproduced in the frontend — it is exact arithmetic, not an approximation,
# and its guard (the shifted constant must stay in the subject's width) is
# what DuckDB's is.
#
# Every row below is `ours == DuckDB`, trap included, so there is no
# hardcoded expectation to go stale.
_W_ROW = pa.schema(
    [
        pa.field("i", pa.int32(), nullable=False),
        pa.field("j", pa.int32(), nullable=False),
        pa.field("t", pa.int8(), nullable=False),
        pa.field("h", pa.int16(), nullable=False),
        pa.field("b", pa.int64(), nullable=False),
    ]
)
_W_DDL = "CREATE TABLE __THIS__ (i INTEGER, j INTEGER, t TINYINT, h SMALLINT, b BIGINT)"
_W_ROWS = [{"i": 2147483647, "j": 3, "t": 127, "h": 32767, "b": 9223372036854775807}]


def _width_duck(sql):
    con = duckdb.connect()
    con.execute(_W_DDL)
    con.execute(
        "INSERT INTO __THIS__ VALUES (?, ?, ?, ?, ?)",
        list(_W_ROWS[0].values()),
    )
    try:
        return ("rows", con.execute(sql).to_arrow_table().to_pylist())
    except duckdb.Error:
        return ("trap", None)


def _width_ours(sql):
    try:
        fn = DuckDBInferFn(sql, row_tables={"__THIS__": _W_ROW}, static_tables={})
        return ("rows", fn.infer_rows(_W_ROWS))
    except Exception:  # noqa: BLE001 — refusal and trap are both "no answer"
        return ("trap", None)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    "sql",
    [
        # ---- the trap must survive every consumer (AC #1) ----
        "SELECT CAST((i + 1) AS BIGINT) AS o FROM __THIS__",  # the reported one
        "SELECT (i + 1) AS o FROM __THIS__",
        "SELECT abs(i + 1) AS o FROM __THIS__",
        "SELECT nullif(i + 1, 5) AS o FROM __THIS__",
        "SELECT (i + 1) * 1.0e0 AS o FROM __THIS__",
        "SELECT (i + j) > 5 AS o FROM __THIS__",  # non-constant: no rewrite
        "SELECT (i * 2) > 5 AS o FROM __THIS__",  # multiplication: no rewrite
        "SELECT (i + 1) IN (5, 6) AS o FROM __THIS__",  # IN: no rewrite
        # DuckDB materialises `i + 1` for the second item, so the first one
        # traps with it -- our per-item evaluation lands in the same place
        "SELECT (i + 1) > 5 AS a, (i + 1) AS b FROM __THIS__",
        # ---- int8 and int16 by the same rule, not just int32 (AC #3) ----
        "SELECT CAST((t + 1) AS BIGINT) AS o FROM __THIS__",
        "SELECT CAST((t * 2) AS BIGINT) AS o FROM __THIS__",
        "SELECT CAST((h + 1) AS BIGINT) AS o FROM __THIS__",
        "SELECT (t + 1) > 5 AS o FROM __THIS__",
        "SELECT (h + 1) > 5 AS o FROM __THIS__",
        # ---- the comparison rewrite, every predicate and both orders ----
        "SELECT (i + 1) > 5 AS o FROM __THIS__",
        "SELECT (i + 1) >= 5 AS o FROM __THIS__",
        "SELECT (i + 1) < 5 AS o FROM __THIS__",
        "SELECT (i + 1) <= 5 AS o FROM __THIS__",
        "SELECT (i + 1) = 5 AS o FROM __THIS__",
        "SELECT (i + 1) <> 5 AS o FROM __THIS__",
        "SELECT (i - 1) > 5 AS o FROM __THIS__",
        "SELECT (1 + i) > 5 AS o FROM __THIS__",
        "SELECT (1 - i) > 5 AS o FROM __THIS__",  # subject swaps sides
        "SELECT 5 < (i + 1) AS o FROM __THIS__",  # constant on the left
        "SELECT (i + 1 - 1) > 5 AS o FROM __THIS__",  # peels one term a pass
        "SELECT ((i + 1) + 1) > 5 AS o FROM __THIS__",
        "SELECT (i + 1) BETWEEN 5 AND 9 AS o FROM __THIS__",  # through BETWEEN
        "SELECT (b + 1) > 5 AS o FROM __THIS__",  # BIGINT too, not just narrow
        "SELECT (b * 2) > 5 AS o FROM __THIS__",
        # ---- the rewrite's own guards ----
        # shifted constant leaves the width: the addition runs and traps
        "SELECT (i + 2) > -2147483648 AS o FROM __THIS__",
        # the constant does not fit INTEGER, so the comparison is at BIGINT
        # and the INT32 addition is left alone
        "SELECT (i + 1) = 2147483648 AS o FROM __THIS__",
        # ... but a BIGINT SPELLING whose value fits does rewrite
        "SELECT (i + 1) > CAST(5 AS BIGINT) AS o FROM __THIS__",
        # ---- in-range arithmetic still serves, on every consumer (AC #4) ----
        "SELECT CAST((i - 1) AS BIGINT) AS o FROM __THIS__",
        "SELECT CAST((t - 1) AS BIGINT) AS o FROM __THIS__",
        "SELECT (i - 1) AS o FROM __THIS__",
        "SELECT (i - 1) + 1 AS o FROM __THIS__",
        "SELECT (j + 1) AS o FROM __THIS__",
        "SELECT (j + 1) > 5 AS o FROM __THIS__",
        "SELECT abs(j + 1) AS o FROM __THIS__",
        "SELECT i AS o FROM __THIS__",
        "SELECT CASE WHEN j > 1 THEN 1 ELSE 0 END AS o FROM __THIS__",
    ],
)
def test_narrow_overflow_traps_exactly_where_duckdbs_does(sql, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    assert _width_ours(sql) == _width_duck(sql), sql


def test_the_narrow_trap_names_the_width():
    """A refusal has to be usable: name the width both ways a reader might
    know it (DuckDB's spelling and the arrow one they passed in)."""
    fn = DuckDBInferFn(
        "SELECT CAST((i + 1) AS BIGINT) AS o FROM __THIS__",
        row_tables={"__THIS__": _W_ROW},
        static_tables={},
    )
    with pytest.raises(ValueError, match="INTEGER") as e:
        fn.infer_rows(_W_ROWS)
    assert "int32" in str(e.value), str(e.value)


def test_a_null_narrow_value_does_not_trap():
    """The range check is flag-gated: a NULL row has a masked default payload
    and DuckDB has no value there to overflow either."""
    row = pa.schema([pa.field("i", pa.int32())])
    sql = "SELECT CAST((i + 1) AS BIGINT) AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={})
    assert fn.infer_rows([{"i": None}]) == [{"o": None}]


# The range trap has to stay invisible in one more place, found by a
# 4000-seed differential campaign (seeds 1564, 2174):
# `<arithmetic> IS [NOT] NULL` reads only the OPERANDS' nullness on DuckDB,
# so the arithmetic never runs and never overflows. Rewriting it as the
# disjunction of the leaves' nullness is exact for a strict operator — "the
# result is NULL" and "some operand is NULL" are the same statement — and it
# also closes a divergence that predates the trap, since DuckDB elides an
# i64 overflow under IS NULL too.
#
# The rows here are deliberately NULL-FREE. DuckDB's elision is driven by the
# column's null STATISTICS, so a batch containing a NULL makes it evaluate and
# trap instead; that split is data-dependent, unrepresentable in a
# compile-once artifact, and pinned as a kept divergence in
# known_divergences/test_trap_elision.py.
_NULLNESS_ROW = pa.schema(
    [
        pa.field("c0", pa.int8()),  # declared nullable, but no NULLs below
        pa.field("h", pa.int16(), nullable=False),
        pa.field("b", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)
_NULLNESS_ROWS = [
    {"c0": -128, "h": -32768, "b": 9223372036854775807, "s": "abc"},
    {"c0": 7, "h": 1, "b": 1, "s": "2.5"},
]


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    "sql",
    [
        # elided: the operand overflows its width and is never evaluated
        "SELECT (c0 * 32) IS NOT NULL AS o FROM __THIS__",
        "SELECT (c0 * 32) IS NULL AS o FROM __THIS__",
        "SELECT ((c0 * 32) + 1) IS NOT NULL AS o FROM __THIS__",  # nested
        "SELECT (- h) IS NOT NULL AS o FROM __THIS__",  # int16 MIN negation
        "SELECT abs(c0 * 32) IS NOT NULL AS o FROM __THIS__",
        "SELECT (c0 * 1.5e0) IS NOT NULL AS o FROM __THIS__",  # through promotion
        "SELECT (b + 1) IS NOT NULL AS o FROM __THIS__",  # BIGINT: no narrow trap
        "SELECT (c0 / 0) IS NOT NULL AS o FROM __THIS__",  # not even a zero divisor
        "SELECT 1 AS o FROM __THIS__ WHERE ((c0 * 32) IS NOT NULL)",
        # NOT elided — outside the measured vocabulary, so it evaluates and
        # traps on both engines
        "SELECT nullif(c0 * 32, 3) IS NOT NULL AS o FROM __THIS__",
        "SELECT CAST(s AS DOUBLE) IS NOT NULL AS o FROM __THIS__",
        # and the plain forms still answer about nullness, elision or not
        "SELECT c0 IS NULL AS o FROM __THIS__",
        "SELECT c0 IS NOT NULL AS o FROM __THIS__",
        "SELECT NULL IS NULL AS o FROM __THIS__",
        "SELECT s IS NOT NULL AS o FROM __THIS__",
    ],
)
def test_is_null_over_arithmetic_reads_the_operands_not_the_result(
    sql, backend, monkeypatch
):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (c0 TINYINT, h SMALLINT, b BIGINT, s VARCHAR)")
    for r in _NULLNESS_ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?, ?, ?)", list(r.values()))
    try:
        want = ("rows", con.execute(sql).to_arrow_table().to_pylist())
    except duckdb.Error:
        want = ("trap", None)
    try:
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": _NULLNESS_ROW}, static_tables={}
        )
        got = ("rows", fn.infer_rows(_NULLNESS_ROWS))
    except Exception:  # noqa: BLE001 — refusal and trap are both "no answer"
        got = ("trap", None)
    assert got == want, sql


# Measured 2026-08-19. `MIN % -1` is the one narrow overflow a
# RESULT-range check structurally cannot see: the mathematical result (0) is
# in range, but DuckDB computes the modulo through the checked division,
# which overflows at the width. The guard is on the OPERATION -- dividend at
# the width's MIN and divisor -1 -- and the fold declines to fold exactly
# that shape, so the constant spelling traps identically.
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
    with pytest.raises(ValueError, match="Overflow in division"):
        fn.infer_rows([{"i": lo}])

    # the guard is exact: MIN+1, and any other divisor, still serve
    assert fn.infer_rows([{"i": lo + 1}]) == [{"o": 0}]
    fn2 = DuckDBInferFn(
        "SELECT (i % -2) AS o FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
    )
    assert fn2.infer_rows([{"i": lo}]) == [{"o": 0}]


def test_constant_narrow_modulo_at_min_traps_like_the_column():
    """The fold declines this shape, so both spellings reach the same trap."""
    row = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(
        "SELECT (-128)::TINYINT % (-1)::TINYINT AS o FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
    )
    with pytest.raises(ValueError, match="Overflow in division of -128 / -1"):
        fn.infer_rows([{"k": 1}])

    # the mixed-width CONTROL: TINYINT % SMALLINT computes at SMALLINT,
    # where -128 / -1 = 128 fits -- serves 0 on both engines (measured)
    fn2 = DuckDBInferFn(
        "SELECT (-128)::TINYINT % (-1)::SMALLINT AS o FROM __THIS__",
        row_tables={"__THIS__": row},
        static_tables={},
    )
    assert fn2.infer_rows([{"k": 1}]) == [{"o": 0}]


# ---------------------------------------------------------------------------
# A bare NULL arm and an integer literal, meeting inside a fold.
#
# DuckDB's CASE types by folding ELSE first, then the THEN arms in order,
# through TryGetMaxLogicalType (bind_case_expression.cpp). The one rule our
# fold missed (types.cpp): Max(SQLNULL, X) = NormalizeType(X) -- an
# INTEGER_LITERAL that meets a NULL is normalized to its base type
# (INTEGER), losing its shrink-to-fit. So NULL alone never widens, a
# literal alone shrinks into a column's width, but a NULL meeting a
# LITERAL-seeded accumulator hardens it: that is the whole "width floor".
# COALESCE folds the same way but seeds from its first argument.
# Measured 2026-08-24, source-verified against the v1.5.5 checkout.
# ---------------------------------------------------------------------------
_T131_SCHEMA = pa.schema(
    [
        pa.field("i8", pa.int8(), nullable=True),
        pa.field("i16", pa.int16(), nullable=True),
        pa.field("i32", pa.int32(), nullable=True),
        pa.field("i64", pa.int64(), nullable=True),
    ]
)
_T131_ROWS = [{"i8": 1, "i16": 2, "i32": 3, "i64": 4}]

_W = "WHEN k > 0 THEN"
_T131_GRID = [
    # NULL hardens the literal seed: int32 floors
    f"SELECT (CASE {_W} NULL {_W} i16 ELSE -22 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} NULL {_W} i8 ELSE -1 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} NULL {_W} i16 ELSE 5 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} NULL {_W} i8 ELSE 5 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} NULL {_W} i32 ELSE -1 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} NULL {_W} i64 ELSE -1 END) AS o FROM __THIS__",
    # the seed 12745 shape (constant TRUE arms)
    "SELECT (CASE WHEN TRUE THEN NULL WHEN TRUE THEN i16 ELSE -22 END) AS o"
    " FROM __THIS__",
    # NULL after the literal already shrank: no floor
    f"SELECT (CASE {_W} i16 {_W} NULL ELSE -22 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} i16 {_W} -22 ELSE NULL END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} -22 {_W} NULL ELSE i16 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} NULL {_W} -22 ELSE i16 END) AS o FROM __THIS__",
    # SQLNULL ELSE seed normalizes the first arm: floor without any NULL arm
    f"SELECT (CASE {_W} -22 {_W} i16 ELSE NULL END) AS o FROM __THIS__",
    # NULL beside columns only: width-neutral
    f"SELECT (CASE {_W} NULL {_W} i16 ELSE i16 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} i16 ELSE NULL END) AS o FROM __THIS__",
    # typed NULL keeps its width, no floor
    f"SELECT (CASE {_W} CAST(NULL AS SMALLINT) {_W} i16 ELSE -22 END) AS o"
    " FROM __THIS__",
    f"SELECT (CASE {_W} CAST(NULL AS TINYINT) {_W} i16 ELSE -22 END) AS o"
    " FROM __THIS__",
    # nested: the inner CASE is a plain computed arm outside
    f"SELECT (CASE {_W} NULL {_W} (CASE {_W} NULL ELSE i16 END) ELSE -22 END)"
    " AS o FROM __THIS__",
    # literal-vs-column controls (no NULL): shrink-to-fit lives
    f"SELECT (CASE {_W} i16 ELSE -22 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} i8 ELSE -1 END) AS o FROM __THIS__",
    f"SELECT (CASE {_W} 200 ELSE i8 END) AS o FROM __THIS__",
    # COALESCE seeds from its FIRST argument
    "SELECT coalesce(NULL, i16) AS o FROM __THIS__",
    "SELECT coalesce(NULL, i8) AS o FROM __THIS__",
    "SELECT coalesce(NULL, i16, -22) AS o FROM __THIS__",
    "SELECT coalesce(NULL, -22, i16) AS o FROM __THIS__",
    "SELECT coalesce(-22, NULL, i16) AS o FROM __THIS__",
    "SELECT coalesce(-22, i16) AS o FROM __THIS__",
    "SELECT coalesce(i16, -22) AS o FROM __THIS__",
    # the if / ifnull / nullif desugars ride the CASE fold
    "SELECT ifnull(NULL, i16) AS o FROM __THIS__",
    "SELECT if(k > 0, NULL, i16) AS o FROM __THIS__",
    "SELECT nullif(i16, NULL) AS o FROM __THIS__",
]


@pytest.mark.parametrize("sql", _T131_GRID)
def test_null_meets_literal_width_matches_duckdb(sql):
    duck_sql = sql.replace("__THIS__", "t")
    con = duckdb.connect()
    con.execute("PRAGMA disable_optimizer")
    con.execute(
        "CREATE TABLE t (i8 TINYINT, i16 SMALLINT, i32 INTEGER, i64 BIGINT, k BIGINT)"
    )
    con.execute("INSERT INTO t VALUES (1, 2, 3, 4, 1)")
    want = con.execute(duck_sql).to_arrow_table()

    schema = _T131_SCHEMA.append(pa.field("k", pa.int64(), nullable=False))
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    got = fn.infer_arrow(
        pa.Table.from_pylist([{**_T131_ROWS[0], "k": 1}], schema=schema)
    )
    assert got.schema == want.schema, (
        f"{got.schema.field('o').type} != {want.schema.field('o').type}"
    )
    assert got.to_pylist() == want.to_pylist()
