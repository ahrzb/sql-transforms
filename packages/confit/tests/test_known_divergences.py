"""Known divergences from the 2026-08-08 adversarial sweep — each pinned xfail-strict.

Every test here FAILS today and states exactly why. `strict=True` means none of
them can silently start passing (a fix must delete the marker) or silently stop
failing. This file is the durable record of the sweep; the tickets are
TASK-69..TASK-78 in `backlog/tasks/`.

The engine's contract is: **either it matches DuckDB bit-for-bit, or it refuses
at build with a named error. There is no third mode.** Most of what follows is a
breach of that contract rather than an exotic edge case — SQL that DuckDB runs
and the engine silently answers differently.

Provenance: 6 finder agents over distinct surfaces, then two independent
refuters per finding, each required to build its own construction and to default
to "refuted". 18 raw findings, 12 verified, 9 confirmed, 2 disputed, 1 refuted.
Four of the nine were then reproduced by hand before being written down; those
say so. The two DISPUTED ones are marked as such — one refuter each could not
break them, one could, and neither has been adjudicated by hand yet.

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


@pytest.mark.xfail(
    strict=True,
    reason="TASK-69: QUALIFY is parsed and then silently discarded, so every "
    "row is emitted. `QUALIFY row_number() OVER (PARTITION BY k ORDER BY ts "
    "DESC) = 1` is the standard dedupe-to-latest-per-key idiom. LIMIT is "
    "refused by name, so the refusal exists — QUALIFY just is not in it.",
)
def test_qualify_is_not_silently_dropped():
    sql = (
        "SELECT k, ts FROM __THIS__ "
        "QUALIFY row_number() OVER (PARTITION BY k ORDER BY ts DESC) = 1"
    )
    got = _run(sql, QualRow, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])
    assert got == duck(sql, _QUAL_DDL, _QUAL_ROWS)


@pytest.mark.xfail(
    strict=True,
    reason="TASK-69: FETCH FIRST n ROWS ONLY is silently discarded while the "
    "exactly equivalent LIMIT n is refused with 'unsupported: LIMIT/OFFSET'. "
    "Same clause, two spellings, opposite behaviour.",
)
def test_fetch_first_is_not_silently_dropped():
    sql = "SELECT k, ts FROM __THIS__ FETCH FIRST 1 ROWS ONLY"
    got = _run(sql, QualRow, [{"k": k, "ts": t} for k, t in _QUAL_ROWS])
    assert got == duck(sql, _QUAL_DDL, _QUAL_ROWS)


# ------------------------------------------------- CAST rounding mode --
#
# `lower::cast` emits `Inst::Ftoi { mode: RoundMode::Round }` under the comment
# "ftoi.round matches DuckDB CAST rounding". It does not. Both backends
# implement RoundMode::Round as Rust `f64::round()` — half AWAY from zero —
# while DuckDB's DOUBLE->BIGINT cast is half-to-EVEN.
#
# The confusion is real and worth stating: the SQL `round()` FUNCTION *is*
# half-away-from-zero and is correctly pinned that way in the wave-1 builtin
# pins. The CAST is a different operation and needs its own mode. Cranelift's
# `nearest` instruction is already half-to-even.
#
# Relayed from the sweep, not reproduced by hand. TASK-70.

CastRow = create_model("CastRow", f=(float, ...))
_CAST_F = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]


@pytest.mark.xfail(
    strict=True,
    reason="TASK-70: CAST(DOUBLE AS BIGINT) rounds half away from zero; "
    "DuckDB rounds half to even. Every exactly-representable half-integer "
    "differs by 1.",
)
@pytest.mark.parametrize("backend", ["cranelift", "interpreter"])
def test_cast_double_to_bigint_rounds_half_to_even(backend, monkeypatch):
    if backend == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    sql = "SELECT CAST(f AS BIGINT) AS i FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": CastRow}, static_tables={}, output="dict"
    )
    assert fn.backend == backend
    got = [r["i"] for r in fn.infer({"__THIS__": [CastRow(f=v) for v in _CAST_F]})]
    want = [
        r[0]
        for r in duck(sql, "CREATE TABLE __THIS__ (f DOUBLE)", [(v,) for v in _CAST_F])
    ]
    assert got == want


# ------------------------------------------------- the infer_arrow path --
#
# Three documented entry points — infer, infer_rows, infer_arrow — are supposed
# to be the same function behind different boundaries. Two ways they are not.
#
# Relayed from the sweep, not reproduced by hand. TASK-71, TASK-72.


@pytest.mark.xfail(
    strict=True,
    reason="TASK-71: run_rows pushes every output row through "
    "`output_model.model_validate` (validators, coercion, defaulted fields); "
    "infer_arrow goes straight from arrow::emit to a pa.Table and never "
    "touches output_model. Same fn, same rows, different answers.",
)
def test_infer_arrow_honours_output_model():
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
    rows = [In(x=2), In(x=4)]
    by_row = [r.model_dump() for r in fn.infer({"__THIS__": rows})]
    by_arrow = fn.infer_arrow(pa.table({"x": [2, 4]})).to_pylist()
    assert by_arrow == by_row


@pytest.mark.xfail(
    strict=True,
    reason="TASK-72: arrow::emit hard-codes pa.large_string() for every Str "
    "output lane; DuckDB's own .arrow() returns pa.string() (32-bit offsets). "
    "Values agree, schemas do not — pa.concat_tables([duck_out, confit_out]) "
    "raises ArrowInvalid, and so does any pinned-schema writer.",
)
def test_infer_arrow_string_type_matches_duckdb():
    In = create_model("In", s=(str, ...))
    sql = "SELECT upper(s) AS u FROM __THIS__"
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": In}, static_tables={}, output="dict"
    )
    got = fn.infer_arrow(pa.table({"s": ["a", "bb"]}))
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (s VARCHAR)")
    con.execute("INSERT INTO __THIS__ VALUES ('a'), ('bb')")
    want = con.execute(sql).arrow()
    assert got.schema == want.schema


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


@pytest.mark.xfail(
    strict=True,
    reason="TASK-73: a CFG split inside a join's own ON residual re-enters "
    "emit_probe without bound — the build dies with STATUS_STACK_OVERFLOW "
    "(0xC00000FD) rather than returning or raising. A build-time input must "
    "never be able to kill the process.",
)
@pytest.mark.parametrize("join", ["JOIN", "LEFT JOIN"])
def test_split_in_the_on_residual_does_not_kill_the_process(join):
    sql = (
        f"SELECT n, r.bud AS b FROM __THIS__ AS t {join} r "
        "ON t.k = r.id AND n + COALESCE(r.bud, 0) > 50"
    )
    p = probe(_ONRES_BODY.format(sql=sql))
    code = hex(p.returncode & 0xFFFFFFFF)
    assert p.returncode == 0, (
        f"exit {p.returncode} ({code})\n{p.stdout}\n{p.stderr[-1500:]}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="TASK-74: scan_residual clears `total` for any SKind::Case without "
    "looking at the arms, so a CASE whose every arm is an integer literal is "
    "refused as a 'single-side residual with trapping ops' — naming a trapping "
    "op the expression does not contain. COALESCE and NULLIF desugar to CASE, "
    "so they are refused the same way.",
)
def test_trap_free_case_in_a_one_sided_on_residual_builds():
    Row = create_model("Row", k=(int, ...), n=(int, ...))
    r = pa.table(
        {"id": pa.array([0, 1], pa.int64()), "cat": pa.array([1, 2], pa.int64())}
    )
    sql = (
        "SELECT n, r.cat AS c FROM __THIS__ AS t JOIN r "
        "ON t.k = r.id AND (CASE WHEN n > 1 THEN 1 ELSE 0 END) = 1"
    )
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": Row}, static_tables={"r": r}, output="dict"
    )
    got = [tuple(x.values()) for x in fn.infer({"__THIS__": [Row(k=0, n=5)]})]
    assert got == [(5, 1)]


# ------------------------------------------- WHERE does not short-circuit --
#
# `fn kleene` is "branchless Kleene AND/OR from flag algebra" and emits BOTH
# operands unconditionally. `fn case`, immediately below it, DOES branch —
# which is why the same trapping call inside a never-taken CASE arm is
# correctly skipped. So a guard that excludes every row still evaluates the
# thing it was written to guard, and its trap kills the whole request.
#
# Not tree-specific: a native BIGINT overflow in the same position diverges
# from DuckDB too, which makes DuckDB the oracle for the second case.
#
# Relayed from the sweep, not reproduced by hand. TASK-75.


@pytest.mark.xfail(
    strict=True,
    reason="TASK-75: WHERE's AND is branchless, so `WHERE k = 0 AND <trapping "
    "expr>` evaluates the right side on every row even though the left is "
    "false for all of them. DuckDB short-circuits and returns []; the engine "
    "traps and the whole batch fails.",
)
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT k FROM __THIS__ WHERE k = 0 AND 9223372036854775807 + k > 0",
        "SELECT k FROM __THIS__ WHERE k > 0 OR 9223372036854775807 + k > 0",
    ],
)
def test_where_and_or_short_circuits_like_duckdb(sql):
    Row = create_model("Row", k=(int, ...))
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": Row}, static_tables={}, output="dict"
    )
    got = [tuple(r.values()) for r in fn.infer({"__THIS__": [Row(k=1), Row(k=2)]})]
    assert got == duck(sql, "CREATE TABLE __THIS__ (k BIGINT)", [(1,), (2,)])


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


@pytest.mark.xfail(
    strict=True,
    reason="TASK-76 (DISPUTED, not adjudicated by hand): the spec lists 'a "
    "node reachable from two parents' among the build-time refusals, but a "
    "DAG-shaped node table builds and scores. Either the refusal is missing "
    "or the spec overstates it.",
)
def test_dag_shaped_node_table_is_refused():
    # nodes 1 and 2 both point their children at node 3 — a DAG, not a tree
    nodes = [
        _node(0, 0, 0.5, 1, 2),
        _node(1, 0, 0.25, 3, 4),
        _node(2, 0, 0.75, 3, 5),  # <- 3 has two parents
        _node(3, -1, 0.0, -1, -1, value=10.0),
        _node(4, -1, 0.0, -1, -1, value=20.0),
        _node(5, -1, 0.0, -1, -1, value=30.0),
    ]
    entry = {
        "nodes": pa.Table.from_pylist(nodes, schema=NODE_SCHEMA),
        "models": pa.Table.from_pylist(
            [{"model_id": 0, "base": 0.0, "agg": "sum", "link": "identity"}],
            schema=MODEL_SCHEMA,
        ),
        "features": ["x"],
    }
    Row = create_model("Row", id=(int, ...), x=(float, ...))
    with pytest.raises(ValueError, match="two parents|reachable|not a tree"):
        DuckDBInferFn(
            "SELECT tree_predict('m', id, struct_pack(x := x)) AS p FROM __THIS__",
            row_tables={"__THIS__": Row},
            static_tables={},
            models={"m": entry},
        )
