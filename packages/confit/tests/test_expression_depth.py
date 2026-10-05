"""Expression depth: DuckDB's parser refuses an expression nested past
`max_expression_depth` (1000), so a long `+` chain refuses on both engines,
by name and without walking the tree it refuses. AND/OR chains are n-ary on
DuckDB and do not count."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity  # noqa: E402

ROWS = pa.table({"x": [1.0, None, -2.5], "k": [1, 2, None]})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT " + " + ".join(["x"] * 1500) + " AS o FROM __THIS__",
        "SELECT " + " * ".join(["k"] * 1500) + " AS o FROM __THIS__",
    ],
)
def test_a_deep_expression_refuses_on_both_engines(sql):
    v = assert_parity(sql, ROWS, expect="REFUSED")
    assert "expression depth" in v.detail


@pytest.mark.parametrize("op", ["AND", "OR"])
def test_a_long_and_or_chain_serves(op):
    # Past 64 terms the chain binds as a balanced tree (depth log n): a
    # 6000-term chain overflowed the stack before.
    terms = [f"x > {i % 7 - 3}" for i in range(6000)]
    sql = f"SELECT {f' {op} '.join(terms)} AS o FROM __THIS__"
    assert_parity(sql, ROWS)


def test_a_long_chain_keeps_its_traps():
    terms = ["x > 0"] * 100 + ["k + 9223372036854775807 > 0"] + ["x < 9"] * 100
    sql = f"SELECT {' AND '.join(terms)} AS o FROM __THIS__"
    # Optimizer-off DuckDB traps on the overflow too; its optimizer prunes
    # the conjunct (the ungated DIVERGE_OPT class).
    assert_parity(sql, ROWS, expect="DIVERGE_OPT")


def test_a_very_deep_expression_refuses_without_walking_it():
    # 50000 levels: the refusal comes from the depth guard, before any
    # per-level walk of the whole subtree (which overflowed the stack).
    sql = "SELECT " + " + ".join(["x"] * 50_000) + " AS o FROM __THIS__"
    with pytest.raises(ValueError, match="expression depth"):
        DuckDBInferFn(sql, row_tables={"__THIS__": ROWS.schema}, static_tables={})


def test_int_literal_overflow_still_refuses():
    with pytest.raises(ValueError, match="overflows INTEGER"):
        DuckDBInferFn(
            "SELECT 2147483647 + 1 + x AS o FROM __THIS__",
            row_tables={"__THIS__": ROWS.schema},
            static_tables={},
        )
