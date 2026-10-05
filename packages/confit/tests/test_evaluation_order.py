"""What DuckDB evaluates, and in which order, where the nightly campaign
found confit disagreeing (issue ahrzb/sql-transforms#305, six nights from
2026-09-29). Each case is the campaign's own verdict against the
optimizer-off oracle (`fuzz.parity`).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

ROWS = table(
    {"t": "int8", "a": "int?", "m": "int", "k": "int32", "s": "str"},
    [
        {"t": 41, "a": 2147483647, "m": -9223372036854775808, "k": -13, "s": "x"},
        {"t": -6, "a": None, "m": 1, "k": 2, "s": "y"},
    ],
)


def _q(expr: str) -> str:
    return f"SELECT {expr} AS o FROM __THIS__"


# ------------------------------------------------- bind-time evaluation
# DuckDB's binder evaluates a foldable operand, and a NULL result replaces
# the call over it, so a trapping sibling never runs. confit evaluates the
# closed constants its own fold leaves unfinished the same way.


@pytest.mark.parametrize(
    "expr",
    [
        "ln(-50.025e0) - (CASE WHEN ('0' LIKE 'a_c') THEN -1.0e0 END)",
        "ln(-37.852e0) + (CASE WHEN ('' LIKE 'a_c') THEN 68.724 END)",
        "pow(CAST('a' AS DOUBLE), TRY_CAST(ltrim('  pad  ') AS DOUBLE))",
        "TRY_CAST(CAST(FALSE AS VARCHAR) AS INT1) * CAST(m AS INTEGER)",
    ],
)
def test_a_closed_null_operand_spares_its_trapping_sibling(expr):
    assert_parity(_q(expr), ROWS, expect="AGREE")


def test_a_closed_value_operand_keeps_the_trap():
    # A value, unlike a NULL, decides nothing at bind: the sibling runs.
    assert_parity(
        _q("ln(-50.025e0) - (CASE WHEN ('a' LIKE 'a') THEN -1.0e0 END)"),
        ROWS,
        trap="logarithm",
    )


# ------------------------------------------------- narrow-width constants
# A constant CASE arm escapes into a TINYINT operator: the fold must not
# compute it in i64 and hide DuckDB's overflow (nightly seeds 2729519,
# 3829756).


@pytest.mark.parametrize(
    "expr",
    [
        "(100 * (CASE WHEN TRUE THEN 100 ELSE t END)) IN (11, 2)",
        "repeat('0', 23 * (CASE WHEN TRUE THEN -13 ELSE t END))",
    ],
)
def test_a_narrow_constant_overflow_traps_like_duckdb(expr):
    # The texts differ: confit's narrow check reports the TINYINT range.
    assert_parity(_q(expr), ROWS, expect="AGREE_TRAP")


# ------------------------------------------------- WHERE conjunct order
# DuckDB binds `x BETWEEN l AND u` to `(x >= l) AND (x <= u)`, and its
# planner's SplitPredicates moves `x <= u` to the END of the WHERE's
# conjunct list (nightly seed 3510602).


# Optimizer-on DuckDB reorders these filters and returns rows, so a trap
# that matches the optimizer-off oracle reads as DIVERGE_OPT.
TRAPS = ("AGREE_TRAP", "DIVERGE_OPT")


@pytest.mark.parametrize(
    "where, expect",
    [
        # the upper bound is checked last, after the multiplication
        ("(a BETWEEN a AND 46) AND (m * k)", TRAPS),
        ("(m * k <> 0) AND (a BETWEEN a AND 46)", TRAPS),
        # the lower bound stays in place and still short-circuits
        ("(a BETWEEN 2147483648 AND a) AND (m * k)", "AGREE"),
        ("(a BETWEEN a AND 46) AND (a < 0) AND (m * k)", "AGREE"),
        # NOT BETWEEN is not a conjunction, and a CASE is not a filter
        ("(a NOT BETWEEN 0 AND 4611686018427387904) AND (m * k)", "AGREE"),
        ("CASE WHEN (a BETWEEN a AND 46) AND (m * k) THEN TRUE END", "AGREE"),
    ],
)
def test_a_where_between_splits_like_duckdbs_planner(where, expect):
    assert_parity(f"SELECT 1 AS o FROM __THIS__ WHERE {where}", ROWS, expect=expect)


# ------------------------------------------------- joins


def _static(schema, rows):
    return table(schema, rows)


S0 = {
    "s0": _static(
        {"c0": "int", "c1": "str"},
        [{"c0": 1, "c1": "1"}],
    )
}
JROWS = table({"c0": "int", "c1": "int"}, [{"c0": 1, "c1": 1}])


@pytest.mark.parametrize(
    "on",
    [
        "c0 + 1 = s0.c0",
        "s0.c0 = c0 + 1",
        "CAST(c0 AS VARCHAR) = s0.c1",
    ],
)
def test_a_shared_bare_name_inside_a_key_expression_is_ambiguous(on):
    # DuckDB already sees the joined table while it binds the ON.
    v = assert_parity(
        f"SELECT s0.c1 AS o FROM __THIS__ JOIN s0 ON ({on})",
        JROWS,
        statics=S0,
        expect="REFUSED",
    )
    assert "ambiguous" in v.detail


def test_a_static_only_trapping_conjunct_through_the_key_column_refuses():
    # `(s0.f - s0.v) < s0.k` names only s0, so DuckDB filters s0 with it
    # before the join and traps over every s0 row (nightly seed 2269747).
    statics = {
        "s0": _static(
            {"k": "int", "f": "int8", "v": "int8"},
            [{"k": -5, "f": -128, "v": 20}, {"k": 1, "f": 1, "v": 1}],
        )
    }
    rows = table({"c0": "int"}, [{"c0": 1}])
    v = assert_parity(
        "SELECT 1 AS o FROM __THIS__ JOIN s0 "
        "ON (__THIS__.c0 = s0.k) AND ((s0.f - s0.v) < s0.k)",
        rows,
        statics=statics,
        expect="REFUSED",
    )
    assert "single-side residual" in v.detail


@pytest.mark.parametrize("shape", [None, "filter"])
def test_a_double_key_projects_the_static_sides_bits(shape):
    # 0.0 matches -0.0, and the key column projects the static value
    # (nightly seed 2678684).
    statics = {"s0": _static({"k": "float"}, [{"k": -0.0}, {"k": 2.5}])}
    rows = table({"x": "float"}, [{"x": 0.0}, {"x": 2.5}])
    assert_parity(
        "SELECT s0.k AS o FROM __THIS__ JOIN s0 ON (x = s0.k)",
        rows,
        statics=statics,
        shape=shape,
        expect="AGREE",
    )
    assert_parity(
        "SELECT s0.* FROM __THIS__ LEFT JOIN s0 ON (x = s0.k)",
        rows,
        statics=statics,
        expect="AGREE",
    )
