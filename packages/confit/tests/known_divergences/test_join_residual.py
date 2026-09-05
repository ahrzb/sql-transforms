"""The join ON residual, three ways.

Split out of test_known_divergences.py 2026-08-16; see README.md for what
belongs here (kept behaviour + its ground) versus in
../test_open_divergences.py (behaviour we intend to change).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from _helpers import probe
from confit import DuckDBInferFn
from confit.oracle import Oracle

# ------------------------------------- the join ON residual, three ways --
#
# All three live in the same corner: a one-sided residual on a JOIN ON clause.
#
# The RECURSION one is the serious one and it is a REGRESSION IN MY OWN
# REASONING. When fixing the many-join probe cache I wrote, in the ticket and in
# the commit message, that a scalar join losing its probe cache across a CFG
# split was "correct, and free when the split is a branch (only one arm runs)".
# That is false when the split is inside the join's OWN residual: the cache miss
# re-enters emit_probe, which re-emits the residual, which contains the split,
# which misses again — unbounded recursion that kills the process at build time.
# I asserted it without testing it.
#
# Reproduced by hand 2026-08-08 (exit 0xC00000FD, both join kinds). The
# trap-freeness one below was relayed from the sweep.

_ONRES_BODY = """
schema = pa.schema([pa.field("k", pa.int64(), nullable=False),
                    pa.field("n", pa.int64(), nullable=False)])
r = pa.table({{"id": pa.array([0], pa.int64()), "bud": pa.array([100], pa.int64())}})
fn = DuckDBInferFn({sql!r}, row_tables={{"__THIS__": schema}},
                   static_tables={{"r": r}})
print("BUILT", [tuple(x.values()) for x in fn.infer_rows([{{"k": 0, "n": 1}}])])
"""


# FIXED 2026-08-08. The scalar probe cache is now re-created on every block
# transition, exactly as the many-join cache already was, plus a re-entry guard
# so a FUTURE cache hole raises a named error instead of recursing to death.
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
def test_split_in_the_on_residual_builds_and_is_correct(join, residual, oracle):
    sql = (
        f"SELECT n, r.bud AS b FROM __THIS__ AS t {join} r ON t.k = r.id AND {residual}"
    )
    p = probe(_ONRES_BODY.format(sql=sql))
    code = hex(p.returncode & 0xFFFFFFFF)
    assert p.returncode == 0, (
        f"exit {p.returncode} ({code})\n{p.stdout}\n{p.stderr[-1500:]}"
    )
    oracle.table("__THIS__", "k BIGINT, n BIGINT", [(0, 1)])
    oracle.table("r", "id BIGINT, bud BIGINT", [(0, 100)])
    want = oracle.execute(sql).fetchall()
    assert p.stdout.strip().splitlines()[-1] == f"BUILT {want}", p.stdout


# FIXED 2026-08-08. `scan_residual` no longer decides trap-freeness at all:
# that question moved to `plan::may_trap`, which the JOIN ON residual rule is
# the sole consumer of. A CASE is trap-free exactly when all of its arms are.

_ONESIDED = pa.table(
    {"id": pa.array([0, 1], pa.int64()), "cat": pa.array([1, 2], pa.int64())}
)
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
    o = Oracle()
    o.table("__THIS__", "k BIGINT, n BIGINT", rows)
    o.table("r", "id BIGINT, cat BIGINT", _ONESIDED_ROWS)
    assert got == o.execute(sql).fetchall()
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
    not widen it to CASEs that trap.

    The last two spent one day (2026-08-17) out of this list, on the grounds
    that `c + n > 1` cannot overflow because DuckDB rewrites it to
    `n > 1 - c`. That rewrite is `expression_rewriter`, and the ORACLE is
    DuckDB with the optimizer off, which performs the addition and overflows:

        SELECT 9223372036854775807 + n > 1 FROM t   -- n = 5
        oracle: Out of Range Error: Overflow in addition of INT64

    so they can trap after all, and refusing them is right."""
    with pytest.raises(ValueError, match="single-side residual with trapping ops"):
        _one_sided(residual, [(0, 5)])


# The other arm of the same rule: a residual mentioning BOTH sides may trap,
# because DuckDB evaluates it per candidate pair. Reaching that arm needs the
# residual to be CLASSIFIABLE -- `scan_residual` walks it to see which sides
# it touches, and a node the walk does not know is refused rather than
# guessed at. So every shape the frontend can put in a residual has to be a
# shape the walk knows; unary minus on a DOUBLE is one of them.
_NEG_STATIC = pa.table(
    {"id": pa.array([0, 1], pa.int64()), "v": pa.array([1.0, 2.0], pa.float64())}
)


@pytest.mark.parametrize(
    "residual",
    [
        "r.v > (- t.d)",
        "(- r.v) < t.d",
        "(- t.d) + r.v > 0.0e0",
        "r.v > (- (t.d * 2.0e0))",
    ],
)
def test_both_sides_residual_over_a_negated_double_builds(residual, oracle):
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), nullable=False),
            pa.field("d", pa.float64(), nullable=False),
        ]
    )
    rows = [(0, -3.0), (1, 0.5), (0, 2.0)]
    sql = (
        "SELECT t.d AS d, r.v AS v FROM __THIS__ AS t "
        f"JOIN r ON t.k = r.id AND {residual}"
    )
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": schema}, static_tables={"r": _NEG_STATIC}
    )
    got = [
        tuple(x.values()) for x in fn.infer_rows([{"k": k, "d": d} for k, d in rows])
    ]
    oracle.table("__THIS__", "k BIGINT, d DOUBLE", rows)
    oracle.table("r", "id BIGINT, v DOUBLE", [(0, 1.0), (1, 2.0)])
    assert got == oracle.execute(sql).fetchall()
