"""Trap elision and the constant folder (TASK-85, TASK-87), and the proof
that this class is syntactic rather than semantic (TASK-117's ground).

Split out of test_known_divergences.py 2026-08-16; see that file's docstring
for what belongs here (kept behaviour + its ground) versus in
test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# TASK-85 (fuzz campaign 2026-08-11, ~20 findings). DuckDB constant-folds a
# STRICT operator with a literal-NULL operand to NULL at optimize time, so a
# trapping sibling under it simply never executes -- `ln(c) + (c - NULL)`
# returns NULL rows where the engine eagerly evaluated ln(negative) and
# trapped. Measured before fixing: the elimination is literal-NULL only; a
# RUNTIME NULL operand does NOT elide the trap on DuckDB either (`ln(c) + d`
# with d NULL still errors there), so eager evaluation already matches for
# those and only the build-time fold was missing. AND/OR are untouched: they
# are not strict (NULL AND FALSE is FALSE), and they never routed through the
# folded sites.

_NF_SCHEMA = pa.schema(
    [
        pa.field("x", pa.float64(), nullable=False),
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    ("sql", "want"),
    [
        ("SELECT (ln(x) + (x - NULL)) AS o FROM __THIS__", None),
        ("SELECT ((9223372036854775807 + k) * (k - NULL)) AS o FROM __THIS__", None),
        ("SELECT (ln(x) < NULL) AS o FROM __THIS__", None),
        # the giant count is DYNAMIC here: the literal spelling refuses at
        # build since TASK-88, and this pin is about the ELISION
        ("SELECT (lpad(s, CAST(k AS INTEGER), s) = NULL) AS o FROM __THIS__", None),
        # control: the trap itself is still live when nothing folds it away
    ],
)
def test_a_null_operand_elides_a_trapping_sibling(sql, want, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _NF_SCHEMA}, static_tables={})
    rows = [{"x": -2.0, "k": 1, "s": "a"}]
    got = fn.infer_rows(rows)
    assert got == [{"o": want}], got

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (x DOUBLE, k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (-2.0, 1, 'a')")
    assert con.execute(sql).fetchall() == [(want,)]


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_the_trap_stays_live_without_a_null_to_fold(backend, monkeypatch):
    """The fold must not grow into lazy evaluation: a RUNTIME NULL does not
    elide the trap on DuckDB, and must not here either."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    schema = pa.schema(
        [pa.field("x", pa.float64(), nullable=False), pa.field("d", pa.float64())]
    )
    fn = DuckDBInferFn(
        "SELECT (ln(x) + d) AS o FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
    )
    with pytest.raises(Exception, match="logarithm"):
        fn.infer_rows([{"x": -2.0, "d": None}])


# TASK-87 (fuzz round 2, 2026-08-11). DuckDB's constant folder decides
# trap-or-serve at PLAN time; the engine decided per row. Four faces, each
# re-measured directly (two first inferences were wrong -- the measurements
# in the ticket are the authority): a trapping constant errors there over
# ZERO rows; literal integer comparisons fold through wide arithmetic there;
# a dead-range BETWEEN is eliminated in WHERE (and ONLY in WHERE); a folded
# constant CASE landing on NULL joins the TASK-85 elision.

_CF_SCHEMA = pa.schema(
    [
        pa.field("s", pa.string(), nullable=False),
        pa.field("x", pa.float64(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        # A: trapping constant -> duck errors on every execution
        "SELECT nullif(CAST('one' AS DOUBLE), -1.5e0) AS o FROM __THIS__",
        # B: literal i64 overflow reaching a non-comparison context
        "SELECT (9223372036854775807 * -50) AS o FROM __THIS__",
    ],
)
def test_a_trapping_constant_refuses_at_build(sql):
    with pytest.raises(ValueError, match="constant|overflows"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _CF_SCHEMA}, static_tables={})


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    ("sql", "rows", "want"),
    [
        # B: comparison over literal arithmetic folds like duck's wide math
        (
            "SELECT (9223372036854775807 > (9223372036854775807 * -50)) AS o"
            " FROM __THIS__",
            [{"s": "one", "x": -2.0}],
            [{"o": True}],
        ),
        # C: dead-range BETWEEN in WHERE eliminates the trapping operand
        (
            "SELECT s AS o FROM __THIS__ WHERE (CAST(s AS BIGINT) BETWEEN 22 AND 10)",
            [{"s": "one", "x": -2.0}],
            [],
        ),
        # D: constant CASE folds to NULL, sqrt sibling eliminated (TASK-85)
        (
            "SELECT ((CASE WHEN TRUE THEN NULL WHEN FALSE THEN -2.5e0 END)"
            " * sqrt(-83.025e0)) AS o FROM __THIS__",
            [{"s": "one", "x": -2.0}],
            [{"o": None}],
        ),
    ],
)
def test_plan_time_folds_match_duckdb(sql, rows, want, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _CF_SCHEMA}, static_tables={})
    got = fn.infer_rows(rows)
    assert got == want, got

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (s VARCHAR, x DOUBLE)")
    con.executemany(
        "INSERT INTO __THIS__ VALUES (?, ?)", [(r["s"], r["x"]) for r in rows]
    )
    assert con.execute(sql).to_arrow_table().to_pylist() == want


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_a_projection_dead_range_still_traps_like_duckdb(backend, monkeypatch):
    """MEASURED split: the dead-range elimination is FILTER-side only —
    in a projection DuckDB evaluates the operand and traps, so must we."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fn = DuckDBInferFn(
        "SELECT (CAST(s AS BIGINT) BETWEEN 22 AND 10) AS o FROM __THIS__",
        row_tables={"__THIS__": _CF_SCHEMA},
        static_tables={},
    )
    with pytest.raises(Exception, match="cast|convert"):
        fn.infer_rows([{"s": "one", "x": 1.0}])


# TASK-117 (fuzz seed 1667, fixed 2026-08-17). The last live member of the
# 2026-08-13 triage's §2. TASK-85 elides a NULL's SIBLING; here the trapping
# expression is the SUBJECT being compared, which nothing elided:
# `x BETWEEN lo AND NULL` desugars to `x >= lo AND x <= NULL`, so the NULL
# folds one comparison and leaves the other holding the trap.
#
# The rule, measured below in both directions: a conjunction with a
# statically-NULL conjunct is NULL whenever it is not FALSE, so as a FILTER it
# selects nothing whatever the other conjuncts say — and DuckDB, having proved
# that, deletes the filter without evaluating them. AND-only (an OR with a
# NULL side can still be TRUE) and FILTER-only (a projection must produce a
# value), which is the same split TASK-87 face C found for the dead range.

_S117 = pa.schema(
    [
        pa.field("s", pa.string(), nullable=False),
        pa.field("x", pa.float64(), nullable=False),
    ]
)
_R117 = [{"s": "abc", "x": -2.0}]  # 'abc' is uncastable, ln(-2.0) is a domain trap
_DDL117 = "CREATE TABLE __THIS__ (s VARCHAR, x DOUBLE)"


def _duck117(sql):
    con = duckdb.connect()
    con.execute(_DDL117)
    con.execute("INSERT INTO __THIS__ VALUES ('abc', -2.0)")
    try:
        return ("rows", con.execute(sql).fetchall())
    except duckdb.Error:
        return ("trap", None)


def _ours117(sql):
    try:
        fn = DuckDBInferFn(sql, row_tables={"__THIS__": _S117}, static_tables={})
        return ("rows", [tuple(r.values()) for r in fn.infer_rows(_R117)])
    except Exception:
        return ("trap", None)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    "sql",
    [
        # the reported case, and its mirror on the low bound
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) BETWEEN 61.591e0 AND NULL",
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) BETWEEN NULL AND 61.591e0",
        "SELECT 1 AS o FROM __THIS__ WHERE ln(x) BETWEEN 1 AND NULL",
        # the same rule without BETWEEN, both conjunct positions
        "SELECT 1 AS o FROM __THIS__ WHERE (CAST(s AS DOUBLE) > 1) AND NULL",
        "SELECT 1 AS o FROM __THIS__ WHERE NULL AND (CAST(s AS DOUBLE) > 1)",
        "SELECT 1 AS o FROM __THIS__ WHERE ln(x) > 0 AND CAST(s AS DOUBLE) > NULL",
        # a bare comparison against NULL is already a NULL conjunct
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) > NULL",
        # the NULL may be FOLDED rather than spelled (TASK-85's strict-op rule
        # produces it), and the elision must see it either way
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) > (1e0 - NULL)",
        "SELECT 1 AS o FROM __THIS__ "
        "WHERE CAST(s AS DOUBLE) BETWEEN 61.591e0 AND (1e0 - NULL)",
        # --- the guard must not become a blanket suppression (AC #4) ---
        # OR, not AND: a NULL side can still let the other side be TRUE
        "SELECT 1 AS o FROM __THIS__ WHERE (CAST(s AS DOUBLE) > 1) OR NULL",
        # a live range with no NULL in it still evaluates the subject
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) BETWEEN 1e0 AND 99e0",
        "SELECT 1 AS o FROM __THIS__ WHERE CAST(s AS DOUBLE) > 1",
        # a PROJECTION has to produce a value, so it evaluates and traps
        "SELECT (CAST(s AS DOUBLE) > 1) AND NULL AS o FROM __THIS__",
        # and an ordinary predicate still selects
        "SELECT 1 AS o FROM __THIS__ WHERE x < 1",
        "SELECT 1 AS o FROM __THIS__ WHERE x > 1 AND NULL",
    ],
)
def test_a_null_conjunct_elides_the_whole_filter(sql, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    assert _ours117(sql) == _duck117(sql), sql


# ===========================================================================
# TRAP ELISION IS NOT A SEMANTIC RULE, SO IT CANNOT BE MATCHED SEMANTICALLY
#
# This pin and its ticket (TASK-117) are bounded by a result worth stating
# once, here, rather than re-deriving per ticket: DuckDB's decision to
# evaluate or skip a trapping subexpression is not a function of what the
# query MEANS. Proof, all lines measured against DuckDB 1.5.5 on 2026-08-16.
#
# Take two queries over the same table (s='abc' uncastable, n IS NULL):
#
#   P1  WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND NULL   -> rows=[]
#   P2  WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND n      -> TRAP
#
# The two predicates have the same value on every row (NULL), therefore
# select the same rows (none), therefore denote the same relation. P1 and P2
# are semantically identical. Suppose some rule R decides "evaluate the
# subject or not" as a function of the query's meaning. Then R(P1) = R(P2),
# so both trap or neither does. DuckDB traps on exactly one. Therefore no
# such R exists: the decision reads the SYNTAX — whether the operand is a
# literal the constant folder can see — not the semantics.
#
# The same fact from the other side, showing it is about fold visibility and
# evaluation ORDER rather than about NULL:
#
#   WHERE FALSE AND trap      -> rows=[]      folded away at plan time
#   WHERE keep  AND trap      -> rows=[(1,)]  per-row short-circuit, L-to-R
#   WHERE trap  AND keep      -> TRAP         same operands, other order
#
# We already match the second and third (TASK-75's flag lanes): those ARE
# semantic — left-to-right short-circuit is in the language. Only the
# fold-visible rows differ.
#
# WHAT MATCHING WOULD COST. Since the rule is syntactic, agreement is not
# "implement NULL propagation correctly" but "make our constant folder's
# reachable set EQUAL DuckDB's". Equal, not merely correct, and in both
# directions: a folder weaker than theirs traps where they serve (this pin);
# a folder stronger than theirs serves where they trap. That target is
# undocumented, is an optimizer's internals rather than an answer, and moves
# whenever their rewriter improves. Our contract names DuckDB's ANSWERS; it
# cannot name this.
#
# WHY WE STILL FIXED THIS ONE. The unmatchable part is the folder-equality
# class, not this instance. TASK-85 already built the fold-to-NULL machinery
# for a strict op whose SIBLING is NULL; TASK-117 (landed 2026-08-17, pinned
# above) extended the same mechanism to the operand being compared, by
# deciding it at the FILTER instead of at the operator. That was bounded work
# on a mechanism we own — one predicate-level rule, no knowledge of DuckDB's
# folder required.
#
# WHEN TO STOP FIXING. Instance-fixing a class we cannot close is unbounded,
# so it gets a stopping rule rather than a habit: if a campaign turns up a
# SECOND fold-visibility mismatch that TASK-117's mechanism does not cover,
# stop matching folds and refuse at build instead — "a trapping expression
# whose reachability depends on constant folding" is decidable from our side
# alone, needs no knowledge of theirs, and turns an open-ended chase into one
# named refusal. Trapping at RUNTIME where DuckDB serves is the only outcome
# the contract has no room for; a build-time no is always available.
# ===========================================================================
def test_duckdbs_trap_elision_is_syntactic_not_semantic():
    """The premises of the proof above, executable.

    If DuckDB ever makes these two agree, the argument for bounding TASK-117
    (and for the stopping rule) has lost its basis and must be re-derived —
    so this fails loudly rather than the reasoning quietly going stale. It
    asserts DuckDB alone; confit is not involved.
    """
    con = duckdb.connect()
    con.execute("CREATE TABLE t (s VARCHAR, n DOUBLE, keep BOOLEAN)")
    con.execute("INSERT INTO t VALUES ('abc', NULL, false), ('1.5', 2.0, true)")

    def duck(sql):
        try:
            return ("rows", con.execute(sql).fetchall())
        except duckdb.Error:
            return ("trap", None)

    # P1 and P2 denote the same relation: the predicate is NULL on every row,
    # so both select nothing. Only the SPELLING of the upper bound differs.
    p1 = duck("SELECT 1 FROM t WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND NULL")
    p2 = duck("SELECT 1 FROM t WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND n")
    assert p1 == ("rows", []), p1
    assert p2 == ("trap", None), p2
    assert p1 != p2, "same meaning, same behaviour — the proof no longer holds"

    # And the ordering half: identical operands, opposite outcomes by position.
    assert duck("SELECT 1 FROM t WHERE FALSE AND CAST(s AS DOUBLE) > 1")[0] == "rows"
    assert duck("SELECT 1 FROM t WHERE keep AND CAST(s AS DOUBLE) > 1")[0] == "rows"
    assert duck("SELECT 1 FROM t WHERE CAST(s AS DOUBLE) > 1 AND keep")[0] == "trap"
