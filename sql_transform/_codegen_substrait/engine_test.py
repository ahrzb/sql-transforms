"""Focused tests for the Substrait front-end. Broad parity is covered by the
differential suite (the `substrait` backend in tests/differential.py); here we
pin the bits it doesn't touch: the raw-plan-bytes entry point, and clean
deferral of unsupported surfaces."""

import pytest
from pydantic import BaseModel

from sql_transform._codegen.engine import CodegenFn
from sql_transform._codegen_substrait import (
    CodegenSubstraitFn,
    Context,
    UnsupportedInCodegen,
)
from sql_transform._codegen_substrait.engine import sql_to_substrait


class Row(BaseModel):
    amount: float
    age: float


SQL = "SELECT age / 271.0 AS age_r, amount - age AS d FROM t WHERE amount > 5.0"
ROWS = [Row(amount=123.0, age=37.0), Row(amount=4.0, age=9.0)]
CTX = Context(row_tables={"t": Row})


def _dump(fn):
    return [r.model_dump() for r in fn.infer({"t": ROWS})]


def test_raw_substrait_plan_bytes_is_the_input():
    """The conditional: a caller passes a Substrait plan (bytes), not SQL."""
    plan_bytes = sql_to_substrait(SQL, CTX)
    assert isinstance(plan_bytes, bytes)
    sub = CodegenSubstraitFn(CTX, plan_bytes)
    assert _dump(sub) == _dump(CodegenFn(SQL, row_tables={"t": Row}, static_tables={}))


def test_from_sql_matches_sql_codegen():
    sub = CodegenSubstraitFn.from_sql(SQL, CTX)
    assert _dump(sub) == _dump(CodegenFn(SQL, row_tables={"t": Row}, static_tables={}))


def test_where_filter_is_applied():
    out = _dump(CodegenSubstraitFn.from_sql(SQL, CTX))
    assert len(out) == 1  # amount=4.0 row filtered out
    assert out[0]["d"] == 86.0


def test_unsupported_relation_defers():
    """A surface the basic consumer doesn't cover (a join) defers cleanly."""

    class Other(BaseModel):
        k: int

    ctx = Context(row_tables={"t": Row, "u": Other})
    with pytest.raises(UnsupportedInCodegen):
        CodegenSubstraitFn.from_sql("SELECT amount FROM t CROSS JOIN u", ctx)
