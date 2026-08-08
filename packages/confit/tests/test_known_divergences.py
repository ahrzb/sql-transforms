"""The 2026-08-08 adversarial sweep — the durable record of every finding.

The engine's contract is: **either it matches DuckDB bit-for-bit, or it refuses
at build with a named error. There is no third mode.** Most of what follows was
a breach of that contract rather than an exotic edge case — SQL that DuckDB
runs and the engine silently answered differently.

Started as nine xfail-strict pins. As each was fixed its marker came off and
the section above it became the account of what the fix was and why, so this
file reads as history rather than as a list of complaints. `strict=True` on
what remains means it can neither silently start passing nor silently stop
failing. Tickets are TASK-69..TASK-78 in `backlog/tasks/`.

Still red, and honestly so:

* the integer-WIDTH schema difference below, found while fixing TASK-72 and
  deliberately not folded into it — it needs its own ticket;
* TASK-77 (an integer feature above 2**53), pinned in
  `sql_transform/_trees_test.py` because it is a packer-side question.

Two of the findings were adjudicated by hand after the sweep's own verifiers
split on them, and they went opposite ways: TASK-78 was real and is fixed;
TASK-76 was a defect in the SPEC, not the code, and the spec was corrected —
see the model-table section at the bottom, which now checks every OTHER
refusal the spec claims by construction rather than assuming them.

Provenance: 6 finder agents over distinct surfaces, then two independent
refuters per finding, each required to build its own construction and to
default to "refuted". 18 raw findings, 12 verified, 9 confirmed, 2 disputed,
1 refuted. Four of the nine were reproduced by hand before being written down.

Two tests here run in a SUBPROCESS because the failure is a process death
(stack overflow), not an exception: observed from inside the session it would
take the whole test run with it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from pydantic import create_model

# --------------------------------------------------------------- helpers --

PROBE_PRELUDE = """
import pyarrow as pa
from confit import DuckDBInferFn
from pydantic import create_model
"""


def probe(body: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in its own process and hand back the result.

    A build that dies of stack overflow returns 0xC00000FD on Windows and a
    signal elsewhere; either way it is not catchable in-process.
    """
    return subprocess.run(  # noqa: S603 — the source is this file's own literal
        [sys.executable, "-c", PROBE_PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def duck(sql: str, ddl: str, rows: list[tuple]) -> list[tuple]:
    con = duckdb.connect()
    con.execute(ddl)
    for r in rows:
        con.execute(f"INSERT INTO __THIS__ VALUES ({', '.join('?' * len(r))})", list(r))
    return con.execute(sql).fetchall()


# ------------------------------------------- silently discarded clauses --
#
# The frontend validates `query.with`, `order_by`, `limit_clause`,
# `select.distinct`, `group_by` and `having` — and refuses LIMIT by name. It
# never looks at `select.qualify`, nor at the FETCH spelling of LIMIT. Those
# two clauses are parsed and dropped, so every input row is emitted.
#
# Worse than a wrong row count: `one_row_blocker` sees no Filter node, so
# `shape='map'` — the exactly-one-row-out-per-row-in PROOF — also builds and
# certifies a query whose entire purpose is to drop rows.
#
# Reproduced by hand 2026-08-08. TASK-69.

_QUAL_DDL = "CREATE TABLE __THIS__ (k BIGINT, ts BIGINT)"
_QUAL_ROWS = [(1, 1), (1, 2), (2, 5)]
QualRow = create_model("QualRow", k=(int, ...), ts=(int, ...))


def _run(sql: str, model, rows: list[dict]) -> list[tuple]:
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": model}, static_tables={}, output="dict"
    )
    return [
        tuple(r.values()) for r in fn.infer({"__THIS__": [model(**r) for r in rows]})
    ]


# FIXED 2026-08-08 (TASK-69). The resolution is REFUSAL, which is half the
# contract: match DuckDB or refuse by name. Ignoring the clause was the third
# mode that is not supposed to exist.
#
# Fixed as a class, not as two instances: `refuse_unhandled_query` and
# `refuse_unhandled_select` destructure their AST node EXHAUSTIVELY, with no
# `..` pattern, so a clause added to sqlparser breaks the build instead of the
# answers. That immediately caught `Select::flavor`, which this audit had
# missed by hand.


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        (
            "SELECT k, ts FROM __THIS__ "
            "QUALIFY row_number() OVER (PARTITION BY k ORDER BY ts DESC) = 1",
            "QUALIFY",
        ),
        ("SELECT k, ts FROM __THIS__ FETCH FIRST 1 ROWS ONLY", "FETCH"),
        ("SELECT TOP 1 k, ts FROM __THIS__", "TOP"),
        ("SELECT k, ts FROM __THIS__ LIMIT 1", "LIMIT"),
        ("SELECT k, ts FROM __THIS__ QUALIFY k > 1", "QUALIFY"),
    ],
)
def test_row_limiting_clauses_are_refused_not_dropped(sql, match):
    """Each of these silently emitted every input row before TASK-69. LIMIT is
    the control: it was always refused, and the others are its synonyms."""
    with pytest.raises(ValueError, match=match):
        _run(sql, QualRow, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])


def test_ordinary_query_still_builds():
    """The audit refuses by exhaustive destructure, so the risk is refusing
    something that used to work."""
    sql = "SELECT k, ts FROM __THIS__ WHERE k = 1"
    got = _run(sql, QualRow, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])
    assert got == duck(sql, _QUAL_DDL, _QUAL_ROWS)


# ------------------------------------------------- CAST rounding mode --
#
# `lower::cast` emitted `Inst::Ftoi { mode: RoundMode::Round }` under the
# comment "ftoi.round matches DuckDB CAST rounding". It did not. Both backends
# implemented RoundMode::Round as Rust `f64::round()` — half AWAY from zero —
# while DuckDB's DOUBLE->BIGINT cast is half-to-EVEN.
#
# FIXED 2026-08-08 (TASK-70). The mode is now `RoundMode::Nearest`
# (`ftoi.nearest` in the IR text), half-to-even on both backends. Only CAST
# and TRY_CAST ever emitted it, so no other op moved.
#
# TWO SEPARATE ROUNDINGS LIVE HERE AND THEY ARE EASY TO CONFUSE:
#
#   CAST(DOUBLE AS BIGINT)   half to even        -2.5 -> -2
#   CAST(DECIMAL AS BIGINT)  half away from zero -2.5 -> -3
#   round(DOUBLE)            half away from zero -2.5 -> -3.0
#
# Two pre-existing Rust pins asserted half-away-from-zero for the DOUBLE cast
# and had to be corrected. Both were written from a DuckDB query on a bare
# `-2.5` literal — which DuckDB types DECIMAL(2,1), not DOUBLE. Measure a
# DOUBLE cast with a DOUBLE column or an explicit `::DOUBLE`, never a literal.
# (Decimal literals binding as f64 is a separate, deliberate v0 divergence;
# see docs/known-limitations.md.)

CastRow = create_model("CastRow", f=(float, ...))
_CAST_F = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 2.6, -2.6, 1e19]


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_cast_double_to_bigint_rounds_half_to_even(backend, monkeypatch):
    """Every exactly-representable half-integer used to differ by 1. `1e19` is
    on the end to keep the range-guarded TRY_CAST path (a second `Ftoi` site)
    in the same comparison — it overflows BIGINT and must become NULL."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    sql = "SELECT TRY_CAST(f AS BIGINT) AS i, round(f) AS r FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": CastRow}, static_tables={}, output="dict"
    )
    assert fn.backend == backend
    got = [
        (r["i"], r["r"])
        for r in fn.infer({"__THIS__": [CastRow(f=v) for v in _CAST_F]})
    ]
    want = duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [(v,) for v in _CAST_F])
    assert got == want
    # And the contrast, stated rather than implied: the two columns disagree
    # on every tie, so this test fails if the cast ever adopts round()'s mode.
    assert [(i, r) for i, r in got if i is not None and float(i) != r] != []


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_plain_cast_double_to_bigint_traps_out_of_range(backend, monkeypatch):
    """The non-TRY path shares the rounding but keeps its own range trap."""
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    sql = "SELECT CAST(f AS BIGINT) AS i FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": CastRow}, static_tables={}, output="dict"
    )
    assert fn.backend == backend
    fine = [v for v in _CAST_F if v != 1e19]
    got = [r["i"] for r in fn.infer({"__THIS__": [CastRow(f=v) for v in fine]})]
    want = [
        r[0]
        for r in duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [(v,) for v in fine])
    ]
    assert got == want
    with pytest.raises(ValueError, match="range"):
        fn.infer({"__THIS__": [CastRow(f=1e19)]})


# ------------------------------------------------- the infer_arrow path --
#
# Three documented entry points — infer, infer_rows, infer_arrow — are supposed
# to be the same function behind different boundaries. Two ways they were not.
#
# FIXED 2026-08-08 (TASK-71, TASK-72).
#
# TASK-71 is resolved by REFUSAL, the other half of the contract. `infer_arrow`
# builds no Python rows — that is its entire reason to exist — so there is
# nothing to call `model_validate` on. Running the rows through pydantic anyway
# would make the columnar path exactly as slow as `infer`, which is to say
# pointless; silently skipping it gave two answers from one function. So it
# now refuses when an `output_model` was SUPPLIED. A synthesized one (the
# default) carries no validators, defaults or coercion, so the columnar path
# stays available for it, which is the common case.
#
# TASK-72 is resolved by matching DuckDB: `pa.string()`, 32-bit offsets. The
# 2 GiB-per-batch ceiling that comes with them is refused by name rather than
# wrapped.


def test_infer_arrow_refuses_a_supplied_output_model():
    """Each of the three things a supplied model can do — a validator, a
    defaulted field, a coercion — silently did NOT happen on the columnar
    path. The refusal names itself and points at the entry point that can."""
    from pydantic import field_validator

    In = create_model("In", x=(int, ...))

    class Out(create_model("OutBase", x=(int, ...), tag=(str, "constant"))):
        @field_validator("x")
        @classmethod
        def _cap(cls, v: int) -> int:
            return min(v, 15)

    fn = DuckDBInferFn(
        "SELECT x * 5 AS x FROM __THIS__",
        row_tables={"__THIS__": In},
        static_tables={},
        output_model=Out,
    )
    by_row = [r.model_dump() for r in fn.infer({"__THIS__": [In(x=2), In(x=4)]})]
    # the validator capped 20 -> 15, and `tag` was defaulted in
    assert by_row == [
        {"x": 10, "tag": "constant"},
        {"x": 15, "tag": "constant"},
    ]
    with pytest.raises(ValueError, match="output_model is not applied"):
        fn.infer_arrow(pa.table({"x": [2, 4]}))


def test_infer_arrow_without_an_output_model_still_works():
    """The refusal keys on SUPPLIED, not on the field being populated — the
    synthesized model is always there and must not block the fast path."""
    In = create_model("In", x=(int, ...))
    fn = DuckDBInferFn(
        "SELECT x * 5 AS y FROM __THIS__", row_tables={"__THIS__": In}, static_tables={}
    )
    assert fn.infer_arrow(pa.table({"x": [2, 4]})).to_pylist() == [
        {"y": 10},
        {"y": 20},
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT upper(s) AS u FROM __THIS__",
        # a NULL in the column, so the validity buffer is exercised too
        "SELECT NULLIF(s, 'a') AS u FROM __THIS__",
        # several string lanes at once
        "SELECT s AS a, lower(s) AS b, s || s AS c FROM __THIS__",
    ],
)
def test_infer_arrow_string_type_matches_duckdb(sql):
    In = create_model("In", s=(str, ...))
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    got = fn.infer_arrow(pa.table({"s": ["a", "bb", "ccc"]}))
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES ('a'), ('bb'), ('ccc')")
    want = con.execute(sql).to_arrow_table()
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()
    # The point of the schema agreeing: the two stack.
    assert pa.concat_tables([want, got]).num_rows == 6


# --------------------------- integer width in the arrow output schema --
#
# Found 2026-08-08 while fixing TASK-72, by widening its scenario sweep from
# "the string column" to "the whole output schema". NOT part of TASK-72 —
# a different type, a different cause — so it is pinned rather than folded in.
#
# DuckDB types a bare integer literal INTEGER, so `CASE WHEN .. THEN 1 ELSE 0
# END` is int32 in its arrow output and int64 in ours. This is the
# arrow-visible face of the documented "narrow integer widths don't exist"
# limitation (docs/known-limitations.md), which until now was only ever
# discussed as an arithmetic concern. It has the same consequence TASK-72 had:
# `pa.concat_tables([duck_out, ours])` raises.
#
# It bites the `titanic` serving scenario, whose `multi_cabin` column is
# exactly that CASE — so this is not hypothetical SQL.
#
# Reproduced by hand 2026-08-08. TASK-79.


@pytest.mark.xfail(
    strict=True,
    reason="A bare integer literal is INTEGER in DuckDB and BIGINT for us, so "
    "an integer-literal CASE comes back int32 there and int64 here. Values "
    "agree; the schemas do not stack.",
)
def test_infer_arrow_integer_width_matches_duckdb():
    In = create_model("In", k=(int, ...))
    sql = "SELECT CASE WHEN k > 1 THEN 1 ELSE 0 END AS c FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    got = fn.infer_arrow(pa.table({"k": [0, 2]}))
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (0), (2)")
    want = con.execute(sql).to_arrow_table()
    assert got.to_pylist() == want.to_pylist()  # values already agree
    assert got.schema == want.schema


def test_infer_arrow_string_output_feeds_back_in():
    """Our own output is a valid input — `ingest` takes 32-bit offsets."""
    In = create_model("In", s=(str, ...))
    fn = DuckDBInferFn(
        "SELECT upper(s) AS s FROM __THIS__",
        row_tables={"__THIS__": In},
        static_tables={},
        output="dict",
    )
    once = fn.infer_arrow(pa.table({"s": ["a", "bb"]}))
    assert fn.infer_arrow(once).to_pylist() == [{"s": "A"}, {"s": "BB"}]


# ------------------------------------- the join ON residual, three ways --
#
# All three live in the same corner: a one-sided residual on a JOIN ON clause.
#
# TASK-73 is the serious one and it is a REGRESSION IN MY OWN REASONING. When
# fixing TASK-68 I wrote, in the ticket and in the commit message, that a
# scalar join losing its probe cache across a CFG split was "correct, and free
# when the split is a branch (only one arm runs)". That is false when the split
# is inside the join's OWN residual: the cache miss re-enters emit_probe, which
# re-emits the residual, which contains the split, which misses again —
# unbounded recursion that kills the process at build time. I asserted it
# without testing it.
#
# TASK-73 reproduced by hand 2026-08-08 (exit 0xC00000FD, both join kinds).
# TASK-74 relayed from the sweep.

_ONRES_BODY = """
Row = create_model("Row", k=(int, ...), n=(int, ...))
r = pa.table({{"id": pa.array([0], pa.int64()), "bud": pa.array([100], pa.int64())}})
fn = DuckDBInferFn({sql!r}, row_tables={{"__THIS__": Row}},
                   static_tables={{"r": r}}, output="dict")
print("BUILT", [tuple(x.values()) for x in fn.infer({{"__THIS__": [Row(k=0, n=1)]}})])
"""


# FIXED 2026-08-08 (TASK-73). The scalar probe cache is now re-created on
# every block transition, exactly as TASK-68 did for the many-join cache, plus
# a re-entry guard so a FUTURE cache hole raises a named error instead of
# recursing to death.
#
# Still run in a SUBPROCESS: if this ever regresses it goes back to killing the
# interpreter, and a subprocess turns that into a failed test rather than a
# dead suite.


@pytest.mark.parametrize("join", ["JOIN", "LEFT JOIN"])
@pytest.mark.parametrize(
    "residual",
    [
        "n + COALESCE(r.bud, 0) > 50",
        "n + (CASE WHEN r.bud > 1 THEN r.bud ELSE 0 END) > 50",
        "n + COALESCE(NULLIF(r.bud, 7), 0) > 50",
    ],
)
def test_split_in_the_on_residual_builds_and_is_correct(join, residual):
    sql = (
        f"SELECT n, r.bud AS b FROM __THIS__ AS t {join} r ON t.k = r.id AND {residual}"
    )
    p = probe(_ONRES_BODY.format(sql=sql))
    code = hex(p.returncode & 0xFFFFFFFF)
    assert p.returncode == 0, (
        f"exit {p.returncode} ({code})\n{p.stdout}\n{p.stderr[-1500:]}"
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, n BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (0, 1)")
    con.execute("CREATE TABLE r (id BIGINT, bud BIGINT)")
    con.execute("INSERT INTO r VALUES (0, 100)")
    want = con.execute(sql).fetchall()
    assert p.stdout.strip().splitlines()[-1] == f"BUILT {want}", p.stdout


# FIXED 2026-08-08 (TASK-74). `scan_residual` no longer decides
# trap-freeness at all: that question moved to `plan::may_trap`, one
# definition shared with Kleene lowering (TASK-75) so the two cannot drift.
# A CASE is trap-free exactly when all of its arms are.

_ONESIDED = pa.table(
    {"id": pa.array([0, 1], pa.int64()), "cat": pa.array([1, 2], pa.int64())}
)
_ONESIDED_DDL = "CREATE TABLE r (id BIGINT, cat BIGINT)"
_ONESIDED_ROWS = [(0, 1), (1, 2)]


def _one_sided(residual: str, rows: list[tuple[int, int]]) -> list[tuple]:
    """Run `JOIN r ON t.k = r.id AND <residual>` and check it against DuckDB.

    `residual` mentions only the dynamic side, which is the case the wave-4
    rule guards: DuckDB scan-pushes a single-side residual, so trap TIMING
    would differ from our hit-guarded lowering if it could trap at all.
    """
    Row = create_model("Row", k=(int, ...), n=(int, ...))
    sql = f"SELECT n, r.cat AS c FROM __THIS__ AS t JOIN r ON t.k = r.id AND {residual}"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": Row}, static_tables={"r": _ONESIDED}, output="dict"
    )
    got = [
        tuple(x.values())
        for x in fn.infer({"__THIS__": [Row(k=k, n=n) for k, n in rows]})
    ]
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, n BIGINT)")
    for k, n in rows:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [k, n])
    con.execute(_ONESIDED_DDL)
    for r in _ONESIDED_ROWS:
        con.execute("INSERT INTO r VALUES (?, ?)", list(r))
    assert got == con.execute(sql).fetchall()
    return got


@pytest.mark.parametrize(
    "residual",
    [
        # every arm an integer literal — nothing here can trap
        "(CASE WHEN n > 1 THEN 1 ELSE 0 END) = 1",
        # no ELSE: the implicit NULL default is not a trap either
        "(CASE WHEN n > 1 THEN 1 END) = 1",
        # COALESCE and NULLIF desugar to CASE and were refused the same way
        "COALESCE(n, 0) > 1",
        "NULLIF(n, 7) > 1",
        # nested, and with a comparison in the arm rather than the condition
        "(CASE WHEN n > 1 THEN (CASE WHEN n > 3 THEN 1 ELSE 0 END) ELSE 0 END) = 1",
    ],
)
def test_trap_free_case_in_a_one_sided_on_residual_builds(residual):
    _one_sided(residual, [(0, 5), (1, 0), (0, 1)])


@pytest.mark.parametrize(
    "residual",
    [
        # an arm that really can overflow
        "(CASE WHEN n > 1 THEN 9223372036854775807 + n ELSE 0 END) = 1",
        # ... and one in the CONDITION rather than the result
        "(CASE WHEN 9223372036854775807 + n > 1 THEN 1 ELSE 0 END) = 1",
        # ... and in the ELSE
        "(CASE WHEN n > 1 THEN 0 ELSE 9223372036854775807 + n END) = 1",
        # bare arithmetic, the case that always was refused
        "9223372036854775807 + n > 1",
    ],
)
def test_a_genuinely_trapping_one_sided_on_residual_is_still_refused(residual):
    """The guard exists for a real reason. Widening it to trap-free CASEs must
    not widen it to CASEs that trap."""
    with pytest.raises(ValueError, match="single-side residual with trapping ops"):
        _one_sided(residual, [(0, 5)])


# ------------------------------------------- WHERE does not short-circuit --
#
# `fn kleene` was "branchless Kleene AND/OR from flag algebra" and emitted
# BOTH operands unconditionally. `fn case`, immediately below it, DOES branch
# — which is why the same trapping call inside a never-taken CASE arm was
# correctly skipped. So a guard that excluded every row still evaluated the
# thing it was written to guard, and its trap killed the whole request.
#
# FIXED 2026-08-08 (TASK-75). The branchless form is kept — it is what makes
# three-valued NULL semantics cheap — and is now used only when the RIGHT
# operand cannot trap, which is the overwhelmingly common case (`a > 1 AND
# b < 2` is still entirely branchless). When it can trap, AND/OR lowers to a
# branch that evaluates the right operand only on rows the left one does not
# already decide: a definite FALSE decides an AND, a definite TRUE decides an
# OR, and a NULL decides nothing — so the right operand still runs there, and
# still traps there, exactly as DuckDB does.
#
# "Can this trap" is `plan::may_trap`, the same predicate the JOIN ON residual
# rule uses (TASK-74). One definition, so the two cannot drift apart.
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
    Row = create_model("Row", k=(int, ...))
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": Row}, static_tables={}, output="dict"
    )
    assert fn.backend == backend
    got = [tuple(r.values()) for r in fn.infer({"__THIS__": [Row(k=1), Row(k=2)]})]
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
    """AC #3: the branchless form was chosen because Kleene NULL semantics
    fall out of flag algebra for free, and the branch must not regress them.

    The full truth table — left in {NULL, FALSE, TRUE} against a right that
    is FALSE and TRUE — evaluated both ways and checked against DuckDB. The
    two parametrisations differ ONLY in which lowering path they take.
    """
    Row = create_model("Row", k=(int | None, None), x=(float, ...))
    sql = f"SELECT k, (k > 0 AND {right}) AS aa, (k > 0 OR {right}) AS oo FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": Row}, static_tables={}, output="dict"
    )
    got = [
        tuple(r.values())
        for r in fn.infer({"__THIS__": [Row(k=k, x=x) for k, x in _3VL_ROWS]})
    ]
    want = duck(sql, "CREATE TABLE __THIS__ (k BIGINT, x DOUBLE)", _3VL_ROWS)
    assert got == want
    # The table really does exercise all three left-hand values.
    assert {r[1] for r in got} == {None, True, False}


def test_where_guard_skips_an_unknown_model_trap():
    """AC #2's other half: scoring an id with no model raises, and a guard
    that excludes every row must stop it from ever being called. DuckDB
    cannot be the oracle here — it has no native tree scoring — so the
    assertion is the empty result the guard implies."""
    Row = create_model("Row", k=(int, ...), mid=(int, ...), x=(float, ...))
    sql = "SELECT k FROM __THIS__ WHERE k = 0 AND m(mid, x) > 0"
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": Row},
        static_tables={},
        output="dict",
        udfs=[_tree_udf([_node(0, -1, 0.0, -1, -1, value=1.0)])],
    )
    rows = [Row(k=1, mid=999, x=0.0), Row(k=2, mid=999, x=0.0)]
    assert list(fn.infer({"__THIS__": rows})) == []
    # ... and the trap is still real for a row the guard lets through.
    with pytest.raises(ValueError, match="model"):
        fn.infer({"__THIS__": [Row(k=0, mid=999, x=0.0)]})


# ------------------------------------------ model-table structure checks --
#
# DISPUTED by the sweep's own verifiers — one refuter broke it, one did not,
# and I have not adjudicated by hand. Pinned anyway so the question cannot be
# lost; if it turns out the behaviour is correct, delete the test and say so
# in the ticket rather than weakening it. TASK-76.

NODE_SCHEMA = pa.schema(
    [
        pa.field("model_id", pa.int64(), nullable=False),
        pa.field("tree_id", pa.int64(), nullable=False),
        pa.field("node_id", pa.int64(), nullable=False),
        pa.field("feature", pa.int32(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=False),
        pa.field("left", pa.int32(), nullable=False),
        pa.field("right", pa.int32(), nullable=False),
        pa.field("missing_left", pa.bool_(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
)
MODEL_SCHEMA = pa.schema(
    [
        pa.field("model_id", pa.int64(), nullable=False),
        pa.field("base", pa.float64(), nullable=False),
        pa.field("agg", pa.string(), nullable=False),
        pa.field("link", pa.string(), nullable=False),
    ]
)


def _node(nid, feature, threshold, left, right, value=0.0):
    return {
        "model_id": 0,
        "tree_id": 0,
        "node_id": nid,
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "missing_left": True,
        "value": value,
    }


ModelRow = create_model("ModelRow", id=(int, ...), x=(float, ...))


class _TreeUDF:
    """A tree transform straight from Arrow — the engine protocol without a
    packer behind it."""

    def __init__(self, nodes, headers, n_features):
        self.name = "m"
        self.takes = pa.schema([(f"f{i}", pa.float64()) for i in range(n_features)])
        self.returns = pa.float64()
        self.instances = {0: None}
        self._t = (nodes, headers, "float32")

    def tree_tables(self):
        return self._t


def _tree_udf(nodes, agg="sum", link="identity", n_features=1):
    return _TreeUDF(
        pa.Table.from_pylist(nodes, schema=NODE_SCHEMA),
        pa.Table.from_pylist(
            [{"model_id": 0, "base": 0.0, "agg": agg, "link": link}],
            schema=MODEL_SCHEMA,
        ),
        n_features,
    )


def _model_fn(nodes, agg="sum", link="identity", features=("x",)):
    return DuckDBInferFn(
        "SELECT m(id, x) AS p FROM __THIS__",
        row_tables={"__THIS__": ModelRow},
        static_tables={},
        output="dict",
        udfs=[_tree_udf(nodes, agg, link, len(features))],
    )


# ADJUDICATED 2026-08-08, then FIXED (TASK-76). Two separate things.
#
# The spec's bullet was wrong and is corrected: it read "a cycle: a node
# reachable from two parents, or unreachable from its tree's root", and a node
# with two parents is NOT a cycle. Children are already forced to strictly
# follow their parent, which rules out cycles by construction and is what makes
# traversal terminate without a depth counter. Under that rule a shared child
# is a decision DAG — the walk from the root still takes exactly one path, and
# it scores exactly what the table names. It was never a wrong-answer bug.
#
# It is refused anyway, because the validator was half-checking tree-ness.
# Given children that strictly follow their parent, a table is a tree exactly
# when every non-root node has ONE parent: one parent each makes the parent
# function total, and the ordering makes walking parents strictly decrease, so
# every node has a unique path back to node 0. The validator already kept a
# saturating parent count (it called it `reachable`) and rejected the ZERO
# case; rejecting zero but allowing two was an arbitrary place to stop. The
# other end is the same array, so full tree-ness costs one line and no extra
# pass.


def test_a_shared_child_is_refused_as_not_a_tree():
    """Scores fine — one path, terminating — and is refused anyway, because
    the table is not a tree and nothing we target emits one."""
    nodes = [
        _node(0, 0, 0.5, 1, 2),
        _node(1, 0, 0.25, 3, 4),
        _node(2, 0, 0.75, 3, 5),  # <- node 3 has two parents
        _node(3, -1, 0.0, -1, -1, value=10.0),
        _node(4, -1, 0.0, -1, -1, value=20.0),
        _node(5, -1, 0.0, -1, -1, value=30.0),
    ]
    with pytest.raises(ValueError, match="child 3 already has a parent"):
        _model_fn(nodes)
    # ... and the same shape with node 3 duplicated into a genuine tree builds,
    # so what is refused is the SHARING, not the shape around it.
    nodes[2] = _node(2, 0, 0.75, 6, 5)
    nodes.append(_node(6, -1, 0.0, -1, -1, value=10.0))
    fn = _model_fn(nodes)
    rows = [ModelRow(id=0, x=v) for v in (0.1, 0.4, 0.6, 0.9)]
    assert [r["p"] for r in fn.infer({"__THIS__": rows})] == [10.0, 20.0, 10.0, 30.0]


# TASK-76 AC #4: every OTHER refusal the spec claims, checked by construction
# rather than assumed. All nine hold.
@pytest.mark.parametrize(
    ("what", "nodes", "kw", "match"),
    [
        (
            "child index out of range",
            [_node(0, 0, 0.5, 1, 9), _node(1, -1, 0.0, -1, -1, value=1.0)],
            {},
            "child 9 out of range",
        ),
        (
            "child precedes its parent (how a cycle would have to be spelled)",
            [
                _node(0, 0, 0.5, 1, 2),
                _node(1, 0, 0.5, 0, 2),
                _node(2, -1, 0.0, -1, -1, value=1.0),
            ],
            {},
            "must follow its parent",
        ),
        (
            "node unreachable from its tree's root",
            [
                _node(0, 0, 0.5, 1, 2),
                _node(1, -1, 0.0, -1, -1, value=1.0),
                _node(2, -1, 0.0, -1, -1, value=2.0),
                _node(3, -1, 0.0, -1, -1, value=3.0),
            ],
            {},
            "unreachable from its tree's root",
        ),
        (
            "leaf with children",
            [_node(0, -1, 0.0, 1, 1), _node(1, -1, 0.0, -1, -1)],
            {},
            "leaf .* with children",
        ),
        (
            "split node missing a child",
            [_node(0, 0, 0.5, 1, -1), _node(1, -1, 0.0, -1, -1, value=1.0)],
            {},
            "split node missing a child",
        ),
        (
            "feature beyond the declared width",
            [
                _node(0, 5, 0.5, 1, 2),
                _node(1, -1, 0.0, -1, -1),
                _node(2, -1, 0.0, -1, -1),
            ],
            {},
            "beyond the declared width",
        ),
        (
            "node id out of dense order",
            [
                _node(0, 0, 0.5, 1, 2),
                _node(7, -1, 0.0, -1, -1),
                _node(2, -1, 0.0, -1, -1),
            ],
            {},
            "out of dense order",
        ),
        (
            "unknown agg",
            [_node(0, -1, 0.0, -1, -1, value=1.0)],
            {"agg": "median"},
            "unknown agg",
        ),
        (
            "unknown link",
            [_node(0, -1, 0.0, -1, -1, value=1.0)],
            {"link": "probit"},
            "unknown link",
        ),
    ],
)
def test_every_claimed_model_table_refusal_holds(what, nodes, kw, match):
    with pytest.raises(ValueError, match=match):
        _model_fn(nodes, **kw)
