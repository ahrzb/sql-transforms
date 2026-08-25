"""WHERE short-circuit and three-valued logic.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from _helpers import _node, _tree_udf, duck
from confit import DuckDBInferFn

# ------------------------------------------- WHERE does not short-circuit --
#
# `fn kleene` was "branchless Kleene AND/OR from flag algebra" and emitted
# BOTH operands unconditionally. `fn case`, immediately below it, DOES branch
# — which is why the same trapping call inside a never-taken CASE arm was
# correctly skipped. So a guard that excluded every row still evaluated the
# thing it was written to guard, and its trap killed the whole request.
#
# FIXED 2026-08-08. The branchless form is kept — it is what makes
# three-valued NULL semantics cheap — and is now used only when the RIGHT
# operand cannot trap, which is the overwhelmingly common case (`a > 1 AND
# b < 2` is still entirely branchless). When it can trap, AND/OR lowers to a
# branch that evaluates the right operand only on rows the left one does not
# already decide: a definite FALSE decides an AND, a definite TRUE decides an
# OR, and a NULL decides nothing — so the right operand still runs there, and
# still traps there, exactly as DuckDB does.
#
# "Can this trap" is `plan::may_trap`, the same predicate the JOIN ON residual
# rule uses. One definition, so the two cannot drift apart.
#
# The branch carries a flag param only when the result is NULLABLE, exactly as
# `FB::case` does. That is not bookkeeping: the null-lane discipline says a
# non-nullable SExpr lowers to a bare payload with no flag anywhere, and
# `emit_stores` asserts it. A first cut of this fix always carried one, which
# passed the entire suite in RELEASE — `debug_assert!` compiles out — and
# panicked on `BETWEEN` in debug. Run the suite against a debug build too.


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k FROM __THIS__ WHERE k = 0 AND 9223372036854775807 + k > 0",
        "SELECT k FROM __THIS__ WHERE k > 0 OR 9223372036854775807 + k > 0",
        # the guard in the middle of a longer conjunction
        "SELECT k FROM __THIS__ WHERE k > 0 AND k = 0 AND 9223372036854775807 + k > 0",
        # a trapping CAST rather than an overflow
        "SELECT k FROM __THIS__ WHERE k = 0 AND CAST(1e19 * k AS BIGINT) > 0",
    ],
)
@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_where_and_or_short_circuits_like_duckdb(sql, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    schema = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    assert fn.backend == backend
    got = [tuple(r.values()) for r in fn.infer_rows([{"k": 1}, {"k": 2}])]
    assert got == duck(sql, "CREATE TABLE __THIS__ (k BIGINT)", [(1,), (2,)])


_3VL_ROWS = [(k, x) for k in (None, 0, 1) for x in (-1.5, 1.5)]


@pytest.mark.parametrize(
    "right",
    [
        "CAST(x AS BIGINT) > 0",  # may trap -> the new branching path
        "x > 0",  # cannot trap -> the original branchless path
    ],
)
def test_short_circuit_preserves_three_valued_logic(right):
    """The branchless form was chosen because Kleene NULL semantics fall out
    of flag algebra for free, and the branch must not regress them.

    The full truth table — left in {NULL, FALSE, TRUE} against a right that
    is FALSE and TRUE — evaluated both ways and checked against DuckDB. The
    two parametrisations differ ONLY in which lowering path they take.
    """
    schema = pa.schema(
        [pa.field("k", pa.int64()), pa.field("x", pa.float64(), nullable=False)]
    )
    sql = f"SELECT k, (k > 0 AND {right}) AS aa, (k > 0 OR {right}) AS oo FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    got = [
        tuple(r.values())
        for r in fn.infer_rows([{"k": k, "x": x} for k, x in _3VL_ROWS])
    ]
    want = duck(sql, "CREATE TABLE __THIS__ (k BIGINT, x DOUBLE)", _3VL_ROWS)
    assert got == want
    # The table really does exercise all three left-hand values.
    assert {r[1] for r in got} == {None, True, False}


def test_where_guard_skips_an_unknown_model_trap():
    """The other half of the guarantee: scoring an id with no model raises,
    and a guard that excludes every row must stop it from ever being called.
    DuckDB cannot be the oracle here — it has no native tree scoring — so
    the assertion is the empty result the guard implies."""
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), nullable=False),
            pa.field("mid", pa.int64(), nullable=False),
            pa.field("x", pa.float64(), nullable=False),
        ]
    )
    sql = "SELECT k FROM __THIS__ WHERE k = 0 AND m(mid, x) > 0"
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[_tree_udf([_node(0, -1, 0.0, -1, -1, value=1.0)])],
    )
    rows = [{"k": 1, "mid": 999, "x": 0.0}, {"k": 2, "mid": 999, "x": 0.0}]
    assert fn.infer_rows(rows) == []
    # ... and the trap is still real for a row the guard lets through.
    with pytest.raises(ValueError, match="model"):
        fn.infer_rows([{"k": 0, "mid": 999, "x": 0.0}])


# ===========================================================================
# Selection context, the full measured matrix (2026-08-19 spec,
# docs/superpowers/specs/2026-08-19-selection-context-design.md).
#
# The model: AND is the ONLY lazy operator -- its LEFT always runs, its RIGHT
# is skipped when the left is not TRUE, recursively. OR always evaluates both
# sides but passes the context through. Selection context enters at exactly
# two places -- the WHERE root and every CASE WHEN condition (projections
# included) -- and exits at NOT / IS NULL / comparisons / function arguments
# and at CASE ARMS (a taken arm is eager, even under WHERE). Everything below
# is checked against the LIVE oracle, so if DuckDB moves, this remeasures.
# ===========================================================================
_SC124_SCHEMA = pa.schema([pa.field("b", pa.bool_()), pa.field("s", pa.string())])
_SC124_ROWS = [
    {"b": None, "s": "abc"},
    {"b": True, "s": "1.5"},
    {"b": False, "s": "abc"},
]
_T124 = "CAST(s AS DOUBLE) > 1"


@pytest.mark.parametrize(
    "sql",
    [
        # --- the lazy side: AND's right is skipped on a not-TRUE left ---
        f"SELECT s AS o FROM __THIS__ WHERE b AND {_T124}",
        f"SELECT s AS o FROM __THIS__ WHERE (b AND {_T124}) OR TRUE",
        f"SELECT s AS o FROM __THIS__ WHERE (b AND {_T124}) AND TRUE",
        f"SELECT s AS o FROM __THIS__ WHERE ((b AND {_T124}) OR TRUE) AND TRUE",
        f"SELECT s AS o FROM __THIS__ WHERE ((b OR TRUE) AND (b AND {_T124})) OR TRUE",
        # ... and through a CASE condition, filter and projection
        f"SELECT s AS o FROM __THIS__ WHERE CASE WHEN (b AND {_T124}) "
        "THEN TRUE ELSE TRUE END",
        f"SELECT CASE WHEN (b AND {_T124}) THEN 1 ELSE 2 END AS o FROM __THIS__",
        f"SELECT s AS o FROM __THIS__ WHERE CASE WHEN ((b AND {_T124}) OR TRUE) "
        "THEN TRUE ELSE TRUE END",
        f"SELECT CASE WHEN ((b AND {_T124}) OR TRUE) THEN 1 ELSE 2 END AS o "
        "FROM __THIS__",
        # --- the eager side: everything else traps exactly like DuckDB ---
        f"SELECT s AS o FROM __THIS__ WHERE {_T124} AND b",
        f"SELECT s AS o FROM __THIS__ WHERE coalesce(b, TRUE) OR {_T124}",
        f"SELECT s AS o FROM __THIS__ WHERE b OR {_T124}",
        f"SELECT s AS o FROM __THIS__ WHERE NOT (b AND {_T124})",
        f"SELECT s AS o FROM __THIS__ WHERE (b AND {_T124}) IS NULL",
        f"SELECT s AS o FROM __THIS__ WHERE (b AND {_T124}) = TRUE",
        f"SELECT s AS o FROM __THIS__ WHERE coalesce((b AND {_T124}), TRUE)",
        f"SELECT s AS o FROM __THIS__ WHERE nullif((b AND {_T124}), FALSE)",
        f"SELECT s AS o FROM __THIS__ WHERE CASE WHEN TRUE THEN (b AND {_T124}) "
        "ELSE TRUE END",
        f"SELECT s AS o FROM __THIS__ WHERE CASE WHEN FALSE THEN TRUE "
        f"ELSE (b AND {_T124}) END",
        f"SELECT (b AND {_T124}) AS o FROM __THIS__",
        f"SELECT ((b AND {_T124}) OR TRUE) AS o FROM __THIS__",
    ],
)
def test_selection_context_matches_the_oracle(sql):
    import duckdb as _duck

    con = _duck.connect()
    con.execute("CREATE TABLE __THIS__ (b BOOLEAN, s VARCHAR)")
    for r in _SC124_ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["b"], r["s"]])
    try:
        want = ("S", con.execute(sql).fetchall())
    except _duck.Error:
        want = ("T", None)

    shape = "filter" if " WHERE " in sql else None
    try:
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": _SC124_SCHEMA}, static_tables={}, shape=shape
        )
    except ValueError:
        # A named BUILD refusal is contract-legal exactly where the oracle
        # does not serve either ((b AND t) = TRUE and the nullif spelling
        # refuse "comparison on BOOLEAN" today). Where the oracle SERVES, a
        # refusal is a cost this matrix must show, so only "T" absorbs it.
        assert want[0] == "T", f"{sql}: refused where the oracle serves"
        return
    try:
        got = ("S", [tuple(x.values()) for x in fn.infer_rows(_SC124_ROWS)])
    except ValueError:
        got = ("T", None)
    assert got == want, f"{sql}: {got} != {want}"


@pytest.mark.parametrize("join", ["JOIN", "LEFT JOIN"])
def test_selection_context_reaches_the_many_join_filter(join):
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
    sql = f"SELECT s0.v AS o FROM __THIS__ {join} s0 ON c0 = s0.k WHERE (b AND {_T124})"
    import duckdb as _duck

    con = _duck.connect()
    con.execute("CREATE TABLE __THIS__ (c0 BIGINT, b BOOLEAN, s VARCHAR)")
    for r in rows:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?, ?)", [r["c0"], r["b"], r["s"]])
    con.execute("CREATE TABLE s0 (k BIGINT, v BIGINT)")
    con.execute("INSERT INTO s0 VALUES (1, 10), (1, 20)")
    want = con.execute(sql).fetchall()
    # order under 'many' is the documented multiset (join-order accident)
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": row}, static_tables={"s0": static}, shape="many"
    )
    assert sorted(tuple(x.values()) for x in fn.infer_rows(rows)) == sorted(want)
