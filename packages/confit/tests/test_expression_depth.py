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


def test_a_long_and_chain_still_serves():
    sql = "SELECT " + " AND ".join(["x > 0"] * 1500) + " AS o FROM __THIS__"
    assert_parity(sql, ROWS)


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
