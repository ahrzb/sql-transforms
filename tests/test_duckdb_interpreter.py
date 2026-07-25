"""DuckDBInferFn is a stub: it parses in the DuckDB dialect and nothing else."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sql_transform._interpreter import DuckDBInferFn


class Row(BaseModel):
    age: int


def test_builds_and_infer_raises():
    fn = DuckDBInferFn(
        "SELECT age FROM __THIS__", row_tables={"__THIS__": Row}, static_tables={}
    )
    with pytest.raises(NotImplementedError):
        fn.infer({"__THIS__": [Row(age=1)]})


def test_bad_sql_is_a_build_error():
    with pytest.raises(ValueError, match="SQL parse error"):
        DuckDBInferFn("SELECT FROM", row_tables={}, static_tables={})
