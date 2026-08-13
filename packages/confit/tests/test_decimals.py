"""The decimals feature (m-8 Dec lane).

Ordinary fits produce DECIMAL statics — sum(BIGINT) is decimal128(38,0) —
so serving them exactly is a real demand path, not an edge case. TASK-91
lands exact serving; until then the build refuses inexact decimal statics
by name (decided 2026-08-13: a named no beats the silent off-by-one it
replaced). The xfail below pins the feature's end state and flips loudly
when TASK-91 lands, taking the refusal out with it.
"""

from __future__ import annotations

import decimal

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from pydantic import create_model

_DecRow = create_model("_DecRow", gid=(str, ...))


def _dec_static(val: str) -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a"], pa.string()),
            "sk": pa.array([decimal.Decimal(val)], pa.decimal128(38, 0)),
        }
    )


@pytest.mark.xfail(
    strict=True,
    reason="we refuse inexact DECIMAL statics for now; TASK-91 (m-8 Dec"
    " lane, first slice) implements exact serving",
)
def test_an_inexact_decimal_static_serves_exactly():
    """2^53+1 in a decimal static comes back as ITSELF."""
    fn = DuckDBInferFn(
        "SELECT sk AS o FROM __THIS__ LEFT JOIN p ON gid = p.g",
        row_tables={"__THIS__": _DecRow},
        static_tables={"p": _dec_static("9007199254740993")},
    )
    got = [r.model_dump() for r in fn.infer({"__THIS__": [_DecRow(gid="a")]})]
    assert got == [{"o": decimal.Decimal("9007199254740993")}]


def test_an_exact_decimal_static_still_serves():
    fn = DuckDBInferFn(
        "SELECT sk AS o FROM __THIS__ LEFT JOIN p ON gid = p.g",
        row_tables={"__THIS__": _DecRow},
        static_tables={"p": _dec_static("9007199254740992")},
    )
    got = [r.model_dump() for r in fn.infer({"__THIS__": [_DecRow(gid="a")]})]
    assert got == [{"o": 9007199254740992.0}]
