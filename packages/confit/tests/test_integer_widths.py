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
    # DOUBLE-typed foldable NULL does NOT collapse (control for TASK-103).
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


# TASK-101 (decided 2026-08-13, spec 2026-08-13-bind-fold-alignment):
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
        sql, row_tables={"__THIS__": In}, static_tables={}, udfs=list(udfs)
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": In}, static_tables={}, udfs=[u])
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": In}, static_tables={}, udfs=[u])
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
    the TASK-102 collapse reads through the TASK-101 fold."""
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": In}, static_tables={}, udfs=[u])
    assert u.calls == 1, "the fold tried once and swallowed"
    with pytest.raises(Exception, match="boom"):
        fn.infer_arrow(pa.Table.from_pylist(ROWS))


def test_pure_scalar_udf_value_under_concat_bakes_once():
    """A folded VALUE bakes in as a literal: DuckDB executes the udf once
    at bind and never per row — matching call counts and values."""
    u = _ScalarStrUdf(result=("x",))
    sql = "SELECT us9(1, 2) || s AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": In}, static_tables={}, udfs=[u])
    assert u.calls == 1
    ours = fn.infer_arrow(pa.Table.from_pylist(ROWS))
    assert u.calls == 1, "the baked literal never re-executes the udf"
    duck = _duck_udf(sql, _ScalarStrUdf(result=("x",)))
    assert ours.to_pylist() == duck.to_pylist() == [{"o": "x" + r["s"]} for r in ROWS]
    assert ours.schema == duck.schema


# The bind-fold evaluates strictly LESS than DuckDB's — our fold keeps
# runtime-only ops runtime and composition stops at the || operand
# itself. Same family as the DECIMAL foldability gap: TASK-103.
@pytest.mark.xfail(
    strict=True,
    reason="TASK-103 (widened by the 2026-08-13 review): DuckDB folds the "
    "WHOLE constant operand subtree — a pure udf under upper()/arith/CASE "
    "still collapses || there, and an argument like abs(-3) still folds "
    "before the udf executes. Our bind fold stops at spellings fold.rs "
    "can finish.",
)
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


# TASK-102 (decided 2026-08-13): || with an operand that FOLDS to NULL is
# an SQLNULL constant on DuckDB — int32 at the boundary, the column side
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


@pytest.mark.xfail(
    strict=True,
    reason="TASK-103 (campaign 2026-08-13, seed 20275804): the DECIMAL "
    "(+|-|*|%) bare-NULL SQLNULL fold keys on literal SPELLING; DuckDB's "
    "rule is operand FOLDABILITY - a constant CASE folding to NULL "
    "collapses the same way, unary minus included. Generalize the arm "
    "over bind_foldable.",
)
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
