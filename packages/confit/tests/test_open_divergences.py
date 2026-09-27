"""Divergences we intend to CLOSE — one xfail-strict pin each.

Adding a pin here is how a new divergence gets recorded, and deleting it is
what closing one looks like; the file may be empty.

The split from `known_divergences/` is by INTENT, not by severity:

    known_divergences/   behaviour we have decided to KEEP. Every entry
                         states the ground for keeping it, and its tests
                         PASS — they are regression pins on a settled
                         answer.

    this file            behaviour we have decided to CHANGE. Every entry is
                         xfail(strict=True). When the divergence closes the
                         pin flips loudly, and the entry is deleted rather
                         than edited.

Why the separation is worth a second file: mixing the two makes "is this on
purpose?" unanswerable at a glance, and a reader who assumes the wrong one
either implements something we chose not to have, or leaves a real bug
sitting under a paragraph explaining why it is fine.

strict=True is the load-bearing part. A pin that silently starts passing is
worse than no pin: it certifies work nobody did.
"""

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle

_S = pa.schema([pa.field("a", pa.int64()), pa.field("s", pa.string())])


@pytest.mark.xfail(
    strict=True,
    reason="an UNALIASED expression's output name: DuckDB prints the bound "
    "expression, parenthesizing operators ('(a + 1)', '-(a)', "
    "'((a * 2) + 1)'); confit echoes the SQL text ('a + 1', '-a', "
    "'a * 2 + 1'). Found serving derived tables, where the name is what an "
    "outer level references; it holds at the top level too.",
)
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a + 1, -a, a * 2 + 1 FROM __THIS__",
        "SELECT * FROM (SELECT a + 1, -a FROM __THIS__)",
    ],
)
def test_unaliased_expression_names(sql):
    o = Oracle()
    o.load("__THIS__", pa.table({"a": [1], "s": ["x"]}, schema=_S))
    want = o.answer(sql).schema.names
    got = DuckDBInferFn(sql, row_tables={"__THIS__": _S}, static_tables={})
    assert got.output_schema.names == want
