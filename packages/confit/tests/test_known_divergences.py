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

Feature pins do NOT live here — they live with their feature's tests
(`test_decimals.py`, the width pin in `test_infer_arrow.py`). TASK-77 (an
integer feature above 2**53) is pinned in `sql_transform/_trees_test.py`
as a packer-side question.

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

# --------------------------------------------------------------- helpers --

PROBE_PRELUDE = """
import pyarrow as pa
from confit import DuckDBInferFn
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
QUAL_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
    ]
)


def _run(sql: str, schema: pa.Schema, rows: list[dict]) -> list[tuple]:
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    return [tuple(r.values()) for r in fn.infer_rows(rows)]


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
        _run(sql, QUAL_SCHEMA, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])


def test_ordinary_query_still_builds():
    """The audit refuses by exhaustive destructure, so the risk is refusing
    something that used to work."""
    sql = "SELECT k, ts FROM __THIS__ WHERE k = 1"
    got = _run(sql, QUAL_SCHEMA, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])
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

CAST_SCHEMA = pa.schema([pa.field("f", pa.float64(), nullable=False)])
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": CAST_SCHEMA}, static_tables={})
    assert fn.backend == backend
    got = [(r["i"], r["r"]) for r in fn.infer_rows([{"f": v} for v in _CAST_F])]
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
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": CAST_SCHEMA}, static_tables={})
    assert fn.backend == backend
    fine = [v for v in _CAST_F if v != 1e19]
    got = [r["i"] for r in fn.infer_rows([{"f": v} for v in fine])]
    want = [
        r[0]
        for r in duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [(v,) for v in fine])
    ]
    assert got == want
    with pytest.raises(ValueError, match="range"):
        fn.infer_rows([{"f": 1e19}])


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
#
# MIGRATION-NOTE (2026-08-13, arrow-schema-api): TASK-71's refusal fired only
# when an `output_model` was SUPPLIED to a pydantic-surface build — that whole
# kwarg, and the synthesized-model machinery it stood next to, is deleted by
# this migration (spec 2026-08-13-arrow-schema-api-design.md: "the
# `output_model` refusal on `infer_arrow` ... exist[s] only to serve pydantic
# out. Dict-out deletes the machinery and the limitation."). The test that
# pinned it (`test_infer_arrow_refuses_a_supplied_output_model`) has no
# construction left to express — `output_model=` now raises TypeError at
# `DuckDBInferFn.__init__` itself, for every entry point, which is already
# covered by `test_arrow_schema_api.py::test_infer_and_output_model_are_gone`.
# Removed rather than mistranslated.


def test_infer_arrow_without_an_output_model_still_works():
    """infer_arrow needs nothing extra supplied to serve — the fast path
    that TASK-71/72 protected is exercised directly here."""
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False)])
    fn = DuckDBInferFn(
        "SELECT x * 5 AS y FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
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
    schema = pa.schema([pa.field("s", pa.string(), nullable=False)])
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
    got = fn.infer_arrow(pa.table({"s": ["a", "bb", "ccc"]}))
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES ('a'), ('bb'), ('ccc')")
    want = con.execute(sql).to_arrow_table()
    assert got.schema == want.schema
    assert got.to_pylist() == want.to_pylist()
    # The point of the schema agreeing: the two stack.
    assert pa.concat_tables([want, got]).num_rows == 6


# ---------------------------------------- arrow output round-trips --


def test_infer_arrow_string_output_feeds_back_in():
    """Our own output is a valid input — `ingest` takes 32-bit offsets."""
    schema = pa.schema([pa.field("s", pa.string(), nullable=False)])
    fn = DuckDBInferFn(
        "SELECT upper(s) AS s FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
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
schema = pa.schema([pa.field("k", pa.int64(), nullable=False),
                    pa.field("n", pa.int64(), nullable=False)])
r = pa.table({{"id": pa.array([0], pa.int64()), "bud": pa.array([100], pa.int64())}})
fn = DuckDBInferFn({sql!r}, row_tables={{"__THIS__": schema}},
                   static_tables={{"r": r}})
print("BUILT", [tuple(x.values()) for x in fn.infer_rows([{{"k": 0, "n": 1}}])])
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
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), nullable=False),
            pa.field("n", pa.int64(), nullable=False),
        ]
    )
    sql = f"SELECT n, r.cat AS c FROM __THIS__ AS t JOIN r ON t.k = r.id AND {residual}"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": schema}, static_tables={"r": _ONESIDED}
    )
    got = [
        tuple(x.values()) for x in fn.infer_rows([{"k": k, "n": n} for k, n in rows])
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
    """AC #3: the branchless form was chosen because Kleene NULL semantics
    fall out of flag algebra for free, and the branch must not regress them.

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
    """AC #2's other half: scoring an id with no model raises, and a guard
    that excludes every row must stop it from ever being called. DuckDB
    cannot be the oracle here — it has no native tree scoring — so the
    assertion is the empty result the guard implies."""
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


MODEL_ROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("x", pa.float64(), nullable=False),
    ]
)


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
        row_tables={"__THIS__": MODEL_ROW_SCHEMA},
        static_tables={},
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
    rows = [{"id": 0, "x": v} for v in (0.1, 0.4, 0.6, 0.9)]
    assert [r["p"] for r in fn.infer_rows(rows)] == [10.0, 20.0, 10.0, 30.0]


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


# TASK-81 (fuzz campaign 2026-08-11, ~570 of 963 findings). A call-node
# modifier that DuckDB refuses -- OVER () on a scalar function, FILTER, IGNORE
# NULLS -- was parsed and silently DROPPED on the builtin path, so the bare
# call was served where the oracle errors. The udf path already screened
# FILTER / IGNORE NULLS / WITHIN GROUP (frontend review round) but not OVER;
# the builtin path screened nothing. One screen now runs where every scalar
# call dispatches.

_MOD_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


class _ModUdf:
    name = "mudf"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.float64()

    def __call__(self, x):
        return (x,)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT abs(k) OVER () AS c FROM __THIS__",
        "SELECT abs(k) FILTER (WHERE TRUE) AS c FROM __THIS__",
        # both IGNORE NULLS spellings: in-paren parses to the argument
        # list's clauses (already refused before this fix), post-paren to
        # the call node's null_treatment (silently dropped before it)
        "SELECT upper(s IGNORE NULLS) AS c FROM __THIS__",
        "SELECT ltrim(lower(s) IGNORE NULLS) AS c FROM __THIS__",
        "SELECT coalesce(k, 1) OVER () AS c FROM __THIS__",
        "SELECT mudf(1.0e0) OVER () AS c FROM __THIS__",
        "SELECT mudf(1.0e0) FILTER (WHERE TRUE) AS c FROM __THIS__",
    ],
)
def test_scalar_call_modifiers_are_refused_not_dropped(sql):
    """DuckDB refuses each of these (CatalogException for OVER on a scalar,
    InvalidInput for FILTER, ParserException for IGNORE NULLS); before the
    fix the engine built them and served the bare call."""
    with pytest.raises(ValueError, match="modifier|argument clauses"):
        DuckDBInferFn(
            sql,
            row_tables={"__THIS__": _MOD_SCHEMA},
            static_tables={},
            udfs=[_ModUdf()],
        )


def test_unmodified_scalar_calls_still_build():
    """The screen must not catch the bare spellings of the same calls."""
    fn = DuckDBInferFn(
        "SELECT abs(k) AS a, upper(s) AS u, mudf(1.0e0) AS m FROM __THIS__",
        row_tables={"__THIS__": _MOD_SCHEMA},
        static_tables={},
        udfs=[_ModUdf()],
    )
    rows = fn.infer_rows([{"k": -2, "s": "x"}])
    assert rows == [{"a": 2, "u": "X", "m": 1.0}]


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


# TASK-82 (fuzz campaign 2026-08-11, 169 of 963 findings). DuckDB's lpad and
# rpad take INTEGER, and its binder does NOT implicitly downcast: a BIGINT
# count -- a row column, or even 2::BIGINT -- is a binder error there, while
# this engine's single integer width bound it happily and served what the
# oracle refuses. The count now binds only when it is spelled a way DuckDB
# types INTEGER or narrower: an int32-range literal (possibly under
# +,-,*,%,parens), or an EXPLICIT cast to INTEGER or narrower -- the
# documented spelling for a column count. A bare column or a BIGINT cast
# refuses. repeat and substr take BIGINT on DuckDB and are untouched.

_PAD_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lpad(s, k, 'x') AS o FROM __THIS__",
        "SELECT rpad(s, k, 'x') AS o FROM __THIS__",
        "SELECT lpad(s, CAST(2 AS BIGINT), 'x') AS o FROM __THIS__",
        "SELECT lpad(s, 3000000000, 'x') AS o FROM __THIS__",
    ],
)
def test_a_bigint_pad_count_refuses_like_duckdb(sql):
    with pytest.raises(ValueError, match="lpad|rpad"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _PAD_SCHEMA}, static_tables={})

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (2, 'ab')")
    with pytest.raises(Exception, match="No function matches|out of range"):
        con.execute(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lpad(s, 4, 'x') AS o FROM __THIS__",
        "SELECT lpad(s, (1 + 3), 'x') AS o FROM __THIS__",
        "SELECT rpad(s, -2, 'x') AS o FROM __THIS__",
        "SELECT lpad(s, CAST(k AS INTEGER), 'x') AS o FROM __THIS__",
        "SELECT repeat(s, k) AS o FROM __THIS__",
    ],
)
def test_integer_shaped_counts_still_bind_and_match(sql):
    """The spellings DuckDB types INTEGER keep building — and keep matching:
    literals, constant arithmetic, negatives; repeat's count is BIGINT on
    DuckDB and stays column-friendly."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _PAD_SCHEMA}, static_tables={})
    rows = [{"k": 2, "s": "ab"}]
    got = fn.infer_rows(rows)

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (2, 'ab')")
    want = con.execute(sql).to_arrow_table().to_pylist()
    assert got == want, f"{got} != {want}"


# TASK-86 (fuzz campaign 2026-08-11, 11 schema findings + 5 downstream
# binder splits). DuckDB types a bare NULL argument FIRST (INTEGER, or the
# BLOB overload), and lets IT drive the signature -- so nullif(NULL, 84.7e0)
# comes back int32 there and double here, and repeat(NULL, n) is BLOB there
# and string here, which then splits every OUTER call binding a BLOB
# (strpos/ltrim/lower/levenshtein/LIKE -- the campaign's five singleton
# "No function matches" findings). Values all NULL, schemas apart, so
# concat_tables against the oracle raises: the TASK-72/79 consequence
# through a different door. The two divergent adopters now refuse by name;
# CAST(NULL AS ...) stays the documented spelling, and adopters that agree
# with DuckDB (upper(NULL), coalesce(NULL, x), nullif(x, NULL)) are
# untouched.

_BN_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        # The nullif face closed with m-8 phase 2 (int32 is real; parity
        # pinned in test_integer_widths.py). These two are the BLOB face.
        "SELECT repeat(NULL, 3) AS o FROM __THIS__",
        "SELECT ltrim(repeat(NULL, k)) AS o FROM __THIS__",
    ],
)
def test_a_divergently_typed_bare_null_argument_refuses(sql):
    with pytest.raises(ValueError, match="NULL"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _BN_SCHEMA}, static_tables={})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT nullif(CAST(NULL AS DOUBLE), 84.754e0) AS o FROM __THIS__",
        "SELECT nullif(84.754e0, NULL) AS o FROM __THIS__",
        "SELECT upper(NULL) AS o FROM __THIS__",
        "SELECT repeat('ab', NULL) AS o FROM __THIS__",
        "SELECT coalesce(NULL, 2.5e0) AS o FROM __THIS__",
    ],
)
def test_agreeing_null_adopters_still_bind_and_match(sql):
    """Everywhere the adopted type EQUALS DuckDB's inference, bare NULL keeps
    working -- schema compared too, that being the whole point."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _BN_SCHEMA}, static_tables={})
    got = fn.infer_arrow(
        pa.table({"k": pa.array([2], pa.int64()), "s": pa.array(["ab"], pa.string())})
    )

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (2, 'ab')")
    want = con.execute(sql).to_arrow_table()
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"
    assert got.to_pylist() == want.to_pylist()


# TASK-84 (fuzz campaign 2026-08-11, 16 DIVERGE_TRAP findings). DuckDB types
# integer literals INTEGER and computes their arithmetic in 32 bits, so
# `-6 * (- 2147483647)` ERRORS there -- while the engine's single i64 width
# served 12884901882 where the oracle traps. Literal-shaped integer
# arithmetic is now evaluated at build in checked int32, DuckDB's own
# semantics, and a subtree that would trap refuses by name. The residual --
# `CAST(k AS INTEGER) * 2` trapping data-dependently at row time -- needs
# the declared-width design (TASK-79) and stays ticketed there; a BIGINT
# operand anywhere in the expression keeps 64-bit math on both engines.

_OV_SCHEMA = pa.schema([pa.field("k", pa.int64(), nullable=False)])


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (-6 * (- 2147483647)) AS o FROM __THIS__",
        "SELECT ((2000000000 + 2000000000) - 2000000000) AS o FROM __THIS__",
        "SELECT (2147483647 + 1) AS o FROM __THIS__",
    ],
)
def test_int32_literal_overflow_refuses_where_duckdb_traps(sql):
    with pytest.raises(ValueError, match="INTEGER|int32|32"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _OV_SCHEMA}, static_tables={})

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (1)")
    with pytest.raises(Exception, match="[Oo]verflow"):
        con.execute(sql).fetchall()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (-6 * CAST(-2147483647 AS BIGINT)) AS o FROM __THIS__",
        "SELECT (k * 2147483647) AS o FROM __THIS__",
        "SELECT (2 + 3) AS o FROM __THIS__",
        "SELECT (2000000000 % 0 + 1) AS o FROM __THIS__",
    ],
)
def test_bigint_and_in_range_literal_arithmetic_still_matches(sql):
    """A BIGINT operand keeps 64-bit math on BOTH engines; in-range literal
    arithmetic and the measured INTEGER%0->NULL stay served and matching."""
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _OV_SCHEMA}, static_tables={})
    got = fn.infer_rows([{"k": 2}])

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (2)")
    want = con.execute(sql).to_arrow_table().to_pylist()
    key = lambda r: sorted((c, repr(v)) for c, v in r.items())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, want)), f"{got} != {want}"


# TASK-80 (fuzz campaign 2026-08-11, 113 of 963 findings). Unary minus was
# lowered as `0 - x` -- the comment on that lowering even said so -- and IEEE
# `0.0 - 0.0` is +0.0, so the sign of negative zero vanished everywhere it
# could arise: the folded literal `-0.0e0`, runtime `(- x)` at x = 0.0, and
# any product with a signed zero operand fed through the fold. Observable at
# any magnitude through division (the sign of infinity) and as text through
# CAST AS VARCHAR. The fix subtracts from -0.0 for FLOAT operands, which is
# exact IEEE negation for every double; the integer path keeps 0 - x and its
# i64::MIN trap, matching DuckDB.

_NEG_SCHEMA = pa.schema([pa.field("x", pa.float64(), nullable=False)])


@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
@pytest.mark.parametrize(
    ("sql", "rows"),
    [
        ("SELECT -0.0e0 AS o0 FROM __THIS__", [{"x": 1.0}]),
        ("SELECT (- x) AS o0 FROM __THIS__", [{"x": 0.0}]),
        ("SELECT (- (x * -1.5e0)) AS o0 FROM __THIS__", [{"x": -0.0}]),
        ("SELECT (1.0e0 / (x * -0.0e0)) AS o0 FROM __THIS__", [{"x": 1.0}]),
        ("SELECT CAST((- x) AS VARCHAR) AS o0 FROM __THIS__", [{"x": 0.0}]),
    ],
)
def test_negative_zero_keeps_its_sign(sql, rows, backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _NEG_SCHEMA}, static_tables={})
    got = fn.infer_rows(rows)

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (x DOUBLE)")
    con.executemany("INSERT INTO __THIS__ VALUES (?)", [(r["x"],) for r in rows])
    want = con.execute(sql).to_arrow_table().to_pylist()
    key = lambda r: sorted((k, repr(v)) for k, v in r.items())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, want)), f"{got} != {want}"


# TASK-82 follow-up (certification campaign 2026-08-11, seed 1589): the
# count check ran AFTER the NULL short-circuit, so a bare-NULL string let a
# BIGINT count slip through -- lpad(NULL, c1, 'x') served NULL where DuckDB
# still binder-errors on the count. The count check now runs first. A NULL
# string with an INTEGER-shaped count stays served: DuckDB types that
# VARCHAR (measured -- lpad has no BLOB overload, unlike repeat).


def test_a_null_string_does_not_smuggle_a_bigint_pad_count():
    with pytest.raises(ValueError, match="lpad"):
        DuckDBInferFn(
            "SELECT lpad(NULL, k, 'x') AS o FROM __THIS__",
            row_tables={"__THIS__": _PAD_SCHEMA},
            static_tables={},
        )
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (2, 'ab')")
    with pytest.raises(Exception, match="No function matches"):
        con.execute("SELECT lpad(NULL, k, 'x') AS o FROM __THIS__")


def test_a_null_string_with_an_integer_count_still_serves():
    fn = DuckDBInferFn(
        "SELECT lpad(NULL, 3, 'x') AS o, rpad(NULL, 3, 'x') AS p FROM __THIS__",
        row_tables={"__THIS__": _PAD_SCHEMA},
        static_tables={},
    )
    got = fn.infer_rows([{"k": 2, "s": "ab"}])
    assert got == [{"o": None, "p": None}]


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


# TASK-88 (fuzz rounds 1+2, ~7 findings + the campaign timeouts). The
# engine's string builder traps at 1 GiB; DuckDB sometimes serves the 2GB
# result and sometimes errors its own builder message, so a giant literal
# pad/repeat count is a run-time coin flip ACROSS engines. A literal count
# that can exceed the budget now refuses at build by name. Data-driven
# counts (a column, CAST(k AS INTEGER)) keep the documented runtime cap.

_SB_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT lpad(s, 2000000000, 'x') AS o FROM __THIS__",
        "SELECT rpad(s, 1500000000, 'x') AS o FROM __THIS__",
        "SELECT repeat(s, 2000000000) AS o FROM __THIS__",
    ],
)
def test_a_budget_breaking_literal_count_refuses(sql):
    with pytest.raises(ValueError, match="builder|GiB"):
        DuckDBInferFn(sql, row_tables={"__THIS__": _SB_SCHEMA}, static_tables={})


def test_a_large_but_bounded_count_still_serves_and_matches():
    sql = (
        "SELECT length(lpad(s, 100000, 'x')) AS o,"
        " length(repeat(s, 50000)) AS p FROM __THIS__"
    )
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": _SB_SCHEMA}, static_tables={})
    got = fn.infer_rows([{"k": 1, "s": "ab"}])

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES (1, 'ab')")
    want = con.execute(sql).to_arrow_table().to_pylist()
    assert got == want, f"{got} != {want}"


# ---------------------------------------------------------------------------
# TASK-115: a projected static column takes the JOIN KEY's width instead of
# its own. Found by the widened fuzzer (seed 1379) while verifying TASK-96.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-115: projecting the static side of a join KEY reconstructs "
    "it from the dynamic side, so it takes the ROW column's width instead of "
    "the static column's own. int8 row key vs int64 static key emits int8; "
    "DuckDB emits int64. Non-key static columns are unaffected.",
)
def test_projected_static_key_keeps_its_own_width():
    row = pa.schema([pa.field("k", pa.int8(), nullable=False)])
    static = pa.table({"c0": pa.array([5], pa.int64()), "v": pa.array([7], pa.int64())})
    sql = "SELECT s.c0 AS o FROM __THIS__ LEFT JOIN s ON k = s.c0"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k TINYINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = con.execute(sql).to_arrow_table()

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": static})
    got = fn.infer_arrow(pa.Table.from_pylist([{"k": 5}], schema=row))
    assert want.schema.field("o").type == pa.int64(), "oracle moved — remeasure"
    assert got.schema == want.schema, f"{got.schema} != {want.schema}"


# ---------------------------------------------------------------------------
# TASK-116: a struct column is lanes in a row table and unserved in a static
# one. Reported 2026-08-15; #157 made the refusal name the type, this pin is
# for actually serving it.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="TASK-116: struct columns serve in a ROW table (TASK-56 flattens "
    "them to lanes) and are unserved in a STATIC one, so the same column "
    "binds or refuses depending only on which table it sits in.",
)
def test_struct_static_column_serves_its_lanes():
    row = pa.schema([pa.field("k", pa.int64(), nullable=False)])
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "w": pa.array([{"mean": 2.0}], pa.struct([("mean", pa.float64())])),
        }
    )
    sql = "SELECT s.w.mean AS o FROM __THIS__ JOIN s ON k = s.id"

    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT)")
    con.execute("INSERT INTO __THIS__ VALUES (5)")
    con.register("sa", static)
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    want = con.execute(sql).to_arrow_table().to_pylist()

    fn = DuckDBInferFn(sql, row_tables={"__THIS__": row}, static_tables={"s": static})
    assert fn.infer_rows([{"k": 5}]) == want
