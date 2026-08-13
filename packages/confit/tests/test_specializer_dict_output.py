"""The dict row-output surface: DuckDBInferFn.infer_rows returns per-row
dicts. Dict is now the only mode (the pydantic typed contract and its
opt-in `output="dict"` are both deleted); these tests assert the
dict-shape invariants that survive the mode collapse — fresh dicts per
call, join/NULL field values, constant-engine serving.
"""

from __future__ import annotations

import pyarrow as pa
from confit import DuckDBInferFn
from test_duckdb_interpreter import static

Row = pa.schema([pa.field("a", pa.int64(), nullable=False), pa.field("s", pa.string())])
SQL = "SELECT a * 2 AS d, upper(coalesce(s, 'x')) AS u FROM __THIS__ WHERE a > 0"


def test_dict_mode_with_join_and_nulls():
    dim = static(
        {"id": "int", "name": "str"},
        [{"id": 1, "name": "one"}, {"id": 3, "name": "three"}],
    )
    fn = DuckDBInferFn(
        "SELECT k, name FROM __THIS__ LEFT JOIN dim ON k = dim.id",
        row_tables={"__THIS__": pa.schema([pa.field("k", pa.int64(), nullable=False)])},
        static_tables={"dim": dim},
    )
    got = fn.infer_rows([{"k": 1}, {"k": 2}])
    assert got == [{"k": 1, "name": "one"}, {"k": 2, "name": None}]


def test_dict_mode_returns_fresh_dicts_per_call():
    fn = DuckDBInferFn(SQL, row_tables={"__THIS__": Row}, static_tables={})
    rows = [{"a": 1, "s": None}]
    first = fn.infer_rows(rows)
    first[0]["d"] = "mutated"
    again = fn.infer_rows(rows)
    assert again == [{"d": 2, "u": "X"}]


def test_dict_mode_on_constant_engine():
    fn = DuckDBInferFn(
        "SELECT 1 AS one, 'x' AS s",
        row_tables={"__THIS__": Row},
        static_tables={},
    )
    got = fn.infer_rows([])
    assert all(type(r) is dict for r in got)
    # Mutating a returned dict must not leak into the next call.
    if got:
        got[0]["one"] = "mutated"
        assert fn.infer_rows([]) != got
