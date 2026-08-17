"""Trap elision: which folds are the BINDER's (so we match them) and which
were the OPTIMIZER's (so we no longer do).

Rewritten 2026-08-17 when the oracle became DuckDB with the optimizer off.
Most of this file used to argue that trap elision is a syntactic,
optimizer-shaped class we could not match as a class and had to chase
instance by instance. Naming the optimizer-off reading dissolved that: the
folds split cleanly into two piles, and only one is ours to reproduce.

  BINDER, so we match it -- survives `PRAGMA disable_optimizer`:
    a literal-NULL operand of an ARITHMETIC op folds the op to NULL, which
    elides a trapping sibling (TASK-85's real half); a constant-condition
    CASE folding to NULL does the same (TASK-87 face D); constant integer
    arithmetic errors at bind.

  OPTIMIZER, so we do NOT -- disappears with the optimizer off:
    comparison against a literal NULL (TASK-85's other half); the i128
    comparison fold (face B); the dead-range BETWEEN in a filter (face C);
    the statically-NULL-conjunct filter elision (TASK-117); the constant
    shift `x ± c <cmp> k` (TASK-118); the `IS NULL` nullness rewrite.

  EXECUTION, so we match it and it was never about folding:
    a filter short-circuits left to right and drops the row as soon as a
    conjunct is not TRUE -- NULL included -- per row.

The two long argument blocks near the end are kept deliberately. They are no
longer about the contract; they are the evidence for why the optimizer is
excluded from the oracle, and both opt the optimizer back ON to say so.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

# TASK-85 (fuzz campaign 2026-08-11, ~20 findings), CORRECTED 2026-08-17 when
# the oracle became DuckDB with the optimizer off.
#
# The original reading was that DuckDB folds a STRICT operator with a
# literal-NULL operand to NULL "at optimize time", so a trapping sibling never
# executes -- `ln(-2.0) + (x - NULL)` is NULL rather than a domain error. Half
# of that is right and half was the optimizer:
#
#   SELECT ln(x) + (x - NULL)   oracle NULL      opt-on NULL    <- the BINDER
#   SELECT ln(x) < NULL         oracle TRAP      opt-on NULL    <- the OPTIMIZER
#
# ARITHMETIC with a literal-NULL operand folds in the binder and survives the
# optimizer being switched off, so the elision is real and we keep it.
# COMPARISON against a literal NULL does not: that fold is a plan rewrite, and
# under the oracle both operands run. Two rules that looked like one.
#
# A RUNTIME NULL operand elides nothing on either reading (`ln(x) + d` with d
# NULL still errors), so eager evaluation always matched for those. AND/OR are
# untouched here: they are not strict, and the filter-side left-to-right
# short-circuit lives in lowering, not in this fold.

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
        # a comparison whose operands cannot trap: nothing to elide, so the
        # NULL simply propagates and both readings agree
        # (the giant count is DYNAMIC here; the literal spelling refuses at
        # build since TASK-88)
        ("SELECT (lpad(s, CAST(k AS INTEGER), s) = NULL) AS o FROM __THIS__", None),
    ],
)
def test_a_null_arith_operand_elides_a_trapping_sibling(
    sql, want, backend, monkeypatch
):
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
def test_a_null_comparison_operand_does_not_elide_its_sibling(backend, monkeypatch):
    """The half of TASK-85 that turned out to be the optimizer. `ln(-2.0) <
    NULL` is NULL on optimizer-ON DuckDB and a domain error on the oracle, so
    the engine evaluates and traps -- unlike the arithmetic form above."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    sql = "SELECT (ln(x) < NULL) AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _NF_SCHEMA}, static_tables={})
    with pytest.raises(Exception, match="logarithm"):
        fn.infer_rows([{"x": -2.0, "k": 1, "s": "a"}])

    con = duckdb.connect()  # the oracle: optimizer off (conftest)
    con.execute("CREATE TABLE __THIS__ (x DOUBLE, k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (-2.0, 1, 'a')")
    with pytest.raises(duckdb.Error, match="logarithm"):
        con.execute(sql).fetchall()


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


# TASK-87 (fuzz round 2, 2026-08-11), RE-SPLIT 2026-08-17 by the oracle.
# Four faces were pinned as folds we match. Two of them are the BINDER and
# still are: A, a trapping constant that errors over ZERO rows, and D, a
# folded constant CASE landing on NULL joining TASK-85's elision. The other
# two were plan rewrites and now evaluate -- B, the wide-arithmetic literal
# comparison, and C, the dead-range BETWEEN in a WHERE. See
# `test_a_fold_that_was_only_the_optimizer_now_evaluates` below.

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
        # D: constant CASE folds to NULL, sqrt sibling eliminated (TASK-85).
        # The only one of TASK-87's four faces that is the BINDER rather than
        # the optimizer, so the only one still here -- see the block below for
        # where B and C went.
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
@pytest.mark.parametrize(
    ("sql", "match"),
    [
        # B: a comparison over literal arithmetic. Optimizer-ON DuckDB answers
        # it through wide range analysis without performing the overflowing
        # multiply; the oracle performs it. The operand ALONE errors at bind on
        # both readings, so the comparison wrapper was the whole difference.
        (
            "SELECT (9223372036854775807 > (9223372036854775807 * -50)) AS o"
            " FROM __THIS__",
            "verflow|constant|out of range",
        ),
        # C: a dead-range BETWEEN in a WHERE. Optimizer-ON folds `lo > hi` to
        # FALSE and drops the subject; the oracle evaluates it. Its PROJECTION
        # twin (below) evaluated under both readings all along, which is what
        # made this one the optimizer's doing rather than the language's.
        (
            "SELECT s AS o FROM __THIS__ WHERE (CAST(s AS BIGINT) BETWEEN 22 AND 10)",
            "cast|convert",
        ),
    ],
)
def test_a_fold_that_was_only_the_optimizer_now_evaluates(
    sql, match, backend, monkeypatch
):
    """TASK-87 faces B and C, which this file used to pin as folds we match.
    Both were plan rewrites, so under the oracle they evaluate and trap -- and
    so do we. Kept as pins because the engine reproduced each of them for a
    while, and would again by accident."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    rows = [{"s": "one", "x": -2.0}]
    with pytest.raises(Exception, match=match):
        fn = DuckDBInferFn(sql, row_tables={"__THIS__": _CF_SCHEMA}, static_tables={})
        fn.infer_rows(rows)

    con = duckdb.connect()  # the oracle: optimizer off (conftest)
    con.execute("CREATE TABLE __THIS__ (s VARCHAR, x DOUBLE)")
    con.execute("INSERT INTO __THIS__ VALUES ('one', -2.0)")
    with pytest.raises(duckdb.Error):
        con.execute(sql).fetchall()


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


# TASK-117 (fuzz seed 1667). Fixed 2026-08-16 by eliding the whole filter when
# a top-level conjunct folded to NULL; RETIRED 2026-08-17 when the oracle
# became optimizer-off DuckDB, which evaluates it:
#
#   WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND NULL
#   oracle: Conversion Error: Could not convert string 'abc' to DOUBLE
#   opt-on: []
#
# So the plan-time elision is gone. What SURVIVES, and is pinned below, is the
# part that was never the optimizer: a filter short-circuits left to right and
# drops the row as soon as a conjunct is not TRUE, NULL included, so a later
# conjunct never runs. That is execution, not rewriting -- it holds with the
# optimizer off -- and it is per ROW rather than a constant fold:
#
#   WHERE b AND (CAST(s AS DOUBLE) > 1)   b NULL on the uncastable row
#   oracle: the uncastable row is dropped, the other row is emitted
#
# A projection has no row to drop, so it evaluates and traps. The engine
# lowers the top-level AND spine one conjunct at a time for exactly this.

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
def test_a_filter_short_circuits_left_to_right_on_a_null_conjunct(
    sql, backend, monkeypatch
):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    assert _ours117(sql) == _duck117(sql), sql


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_the_filter_short_circuit_is_per_row_not_a_constant_fold(backend, monkeypatch):
    """The evidence that this is execution and not rewriting: the left operand
    is a nullable COLUMN, NULL on one row and TRUE on another, and the oracle
    drops the first row while emitting the second. A plan-time elision could
    only have taken the whole filter or none of it."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    schema = pa.schema(
        [pa.field("s", pa.string(), nullable=False), pa.field("b", pa.bool_())]
    )
    rows = [{"s": "abc", "b": None}, {"s": "1.5", "b": True}]
    sql = "SELECT 1 AS o FROM __THIS__ WHERE b AND (CAST(s AS DOUBLE) > 1)"

    con = duckdb.connect()  # the oracle: optimizer off (conftest)
    con.execute("CREATE TABLE __THIS__ (s VARCHAR, b BOOLEAN)")
    con.execute("INSERT INTO __THIS__ VALUES ('abc', NULL), ('1.5', true)")
    want = con.execute(sql).fetchall()
    assert want == [(1,)], "oracle moved — remeasure"

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    assert [tuple(r.values()) for r in fn.infer_rows(rows)] == want

    # ... and the reverse order still traps, because the trapping conjunct now
    # runs first. Same operands, so this is ORDER, not nullness.
    rev = "SELECT 1 AS o FROM __THIS__ WHERE (CAST(s AS DOUBLE) > 1) AND b"
    with pytest.raises(duckdb.Error):
        con.execute(rev).fetchall()
    rfn = DuckDBInferFn(rev, row_tables={"__THIS__": schema}, static_tables={})
    with pytest.raises(Exception, match="cast|convert"):
        rfn.infer_rows(rows)


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
# HOW IT WAS ACTUALLY RESOLVED, 2026-08-17. Not by matching the folder and not
# by refusing: by changing WHICH DuckDB the contract names. The oracle is now
# DuckDB with the optimizer off, and every fold in this argument belongs to the
# optimizer, so the whole class collapses -- P1 and P2 BOTH trap under the
# oracle and there is nothing left to be unmatchable about.
#
# That is why this test opts the optimizer back ON. It is now a statement about
# DuckDB's optimizer rather than about the contract, and it is kept for two
# reasons: it is the evidence that the fold-visibility class is real (so nobody
# reintroduces an emulation thinking it is free), and it is the measurement
# that would have to be re-derived if the oracle ever moved back.
#
# The stopping rule this block used to carry -- "on a SECOND fold-visibility
# mismatch, refuse at build" -- is retired along with the class. The oracle
# makes the question moot: there is no fold to chase, so there is nothing to
# stop chasing. What replaced it is narrower and checkable: when a finding
# turns out to be an optimizer pass, name the pass (the campaign does this
# mechanically now) and match the oracle, which does not have it.
# ===========================================================================
def test_duckdbs_trap_elision_is_syntactic_not_semantic():
    """The premises of the proof above, executable.

    If DuckDB ever makes these two agree, the argument for bounding TASK-117
    (and for the stopping rule) has lost its basis and must be re-derived —
    so this fails loudly rather than the reasoning quietly going stale. It
    asserts DuckDB alone; confit is not involved.
    """
    con = duckdb.connect()
    # ABOUT the optimizer, so it opts back into it (conftest hands out the
    # oracle, optimizer OFF, by default). With the optimizer off both P1 and
    # P2 simply trap and there is no split to prove.
    con.execute("PRAGMA enable_optimizer")
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


# ===========================================================================
# AND ONE DISCRIMINANT WEAKER STILL: NOT THE QUERY, NOT THE ROWS, THE HISTORY
#
# The proof above shows DuckDB's trap-or-serve decision is not a function of
# the query's MEANING. This one lands on the same class from further below.
# For `<arithmetic> IS [NOT] NULL` the decision is not a function of the query
# at all, and — the part that settles it — not a function of the ROWS either.
#
# `IS NOT NULL` over a strict expression folds to constant TRUE when the
# column's stored null STATISTIC says the column holds no NULLs; the folded
# expression is then deleted and never evaluated, so its overflow never
# happens. Measured 2026-08-17 on DuckDB 1.5.5, one query
# (`SELECT (c0 * 32) IS NOT NULL FROM t`), `c0` a nullable TINYINT, and
# `c0 * 32` overflowing TINYINT on the -128 row in every line:
#
#   [-128]                       -> [True]        no NULL: folds, deleted
#   [-128, 7]                    -> [True, True]  still no NULL
#   [-128, NULL]                 -> TRAP          the fold is withdrawn
#   [NULL, -128]                 -> TRAP          position is irrelevant
#   c0 declared TINYINT NOT NULL -> [True, True]  the DECLARATION alone does it
#
# So far that is "depends on the data", which a per-batch decision could in
# principle chase. These two lines close that door:
#
#   INSERT [-128, NULL] then DELETE the NULL row  -> TRAP
#   ... and WHERE c0 IS NOT NULL in the query too -> TRAP
#
# After the DELETE the table's ROWS are identical to the `[-128]` case, which
# serves. Same query, same schema, same rows, opposite answers — the only
# surviving difference is a statistic left behind by an insert that no longer
# has a row. And filtering the NULL out inside the query does not help either:
# the fold is decided from the base column's statistic, upstream of the
# filter.
#
# WHY THAT MAKES IT STRUCTURAL, NOT MERELY UNDOCUMENTED. There is nothing in
# confit's world that corresponds to "a NULL was here once". We compile ONCE
# against a schema and serve many batches; we do not own the table, we never
# see its history, and there is no build-time "the data" to compute a
# statistic over. Even the fallback of deciding per batch — which would be
# wrong anyway, since it makes an input row's answer depend on its NEIGHBOURS
# in the batch, breaking the one property callers rely on — cannot reproduce
# the post-DELETE line, because by then the evidence is gone.
#
# WHAT WE DO ABOUT IT. Nothing, now, and that is the point of the oracle. For
# one day (2026-08-17) the engine rewrote `IS [NOT] NULL` over arithmetic into
# the disjunction of its operands' nullness, so it never evaluated and never
# trapped — an emulation of `statistics_propagation`, chosen because it agreed
# with the reading a user sees on the common no-NULL batch. The oracle is
# optimizer-off DuckDB, which has no statistics pass and simply evaluates:
#
#   SELECT (c0 * 32) IS NOT NULL FROM t   -- c0 TINYINT, one row -128
#   oracle: Out of Range Error: Overflow in multiplication of INT8 (-128 * 32)!
#
# so the engine evaluates and traps too, on every batch, and the rewrite is
# gone. The measurements above are kept because they are the reason this pass
# is one we must NOT chase: it is the sharpest example in the record of an
# optimizer whose answer is not a function of the query, and the argument for
# excluding the optimizer from the oracle rests partly on it.
#
# The cost is stated plainly rather than hidden: a user running DuckDB with the
# optimizer on WILL be served `true` here where we trap. That is the standing
# price of naming the optimizer-off reading, accepted deliberately, and it is
# not specific to this pass.
_IS_NN = "SELECT (c0 * 32) IS NOT NULL AS o FROM t"


def test_duckdbs_is_null_elision_is_not_a_function_of_the_query_or_the_rows():
    """The premises above, executable. If DuckDB ever makes these agree, the
    ground for eliding unconditionally has gone and must be re-derived."""

    def duck(setup, decl="TINYINT", q=_IS_NN):
        con = duckdb.connect()
        # This test is ABOUT the optimizer, so it opts back into it -- conftest
        # hands every connection the oracle (optimizer OFF) by default.
        con.execute("PRAGMA enable_optimizer")
        con.execute(f"CREATE TABLE t (c0 {decl})")
        for s in setup:
            con.execute(s)
        try:
            return ("rows", con.execute(q).fetchall())
        except duckdb.Error:
            return ("trap", None)

    ins = "INSERT INTO t VALUES "
    assert duck([ins + "(-128)"]) == ("rows", [(True,)])
    assert duck([ins + "(-128), (7)"]) == ("rows", [(True,), (True,)])
    assert duck([ins + "(-128), (NULL)"]) == ("trap", None)
    assert duck([ins + "(NULL), (-128)"]) == ("trap", None)
    # the DECLARATION alone does it -- the half a compile-once engine COULD
    # have matched, since our arrow field carries exactly this bit
    assert duck([ins + "(-128), (7)"], decl="TINYINT NOT NULL") == (
        "rows",
        [(True,), (True,)],
    )

    # The two that make it unmatchable rather than merely awkward. After the
    # DELETE the ROWS are identical to the first line above, which serves.
    assert duck([ins + "(-128), (NULL)", "DELETE FROM t WHERE c0 IS NULL"]) == (
        "trap",
        None,
    )
    # ... and the query's own filter is downstream of the fold decision
    assert duck(
        [ins + "(-128), (NULL)"],
        q="SELECT (c0 * 32) IS NOT NULL AS o FROM t WHERE c0 IS NOT NULL",
    ) == ("trap", None)

    # Control: the fold is what removes the expression, not a widening of the
    # multiplication -- projected on its own, the same row traps.
    assert duck([ins + "(-128)"], q="SELECT c0 * 32 AS o FROM t") == ("trap", None)


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_we_evaluate_is_null_over_arithmetic_like_the_oracle(backend, monkeypatch):
    """Our side, stated rather than implied: no nullness rewrite, so the
    arithmetic runs and the narrow overflow traps -- on every batch, matching
    the oracle, and NOT matching what a user with the optimizer on would see."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    schema = pa.schema([pa.field("c0", pa.int8())])
    sql = "SELECT (c0 * 32) IS NOT NULL AS o FROM __THIS__"
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})

    con = duckdb.connect()  # the oracle: optimizer off (conftest)
    con.execute("CREATE TABLE __THIS__ (c0 TINYINT)")
    con.execute("INSERT INTO __THIS__ VALUES (-128), (7)")
    with pytest.raises(duckdb.Error, match="Overflow"):
        con.execute(sql).fetchall()
    with pytest.raises(Exception, match="range|Overflow"):
        fn.infer_rows([{"c0": -128}, {"c0": 7}])

    # and a batch with nothing out of range serves on both
    con2 = duckdb.connect()
    con2.execute("CREATE TABLE __THIS__ (c0 TINYINT)")
    con2.execute("INSERT INTO __THIS__ VALUES (3), (NULL)")
    want = con2.execute(sql).fetchall()
    assert want == [(True,), (False,)], "oracle moved — remeasure"
    got = fn.infer_rows([{"c0": 3}, {"c0": None}])
    assert [tuple(r.values()) for r in got] == want
