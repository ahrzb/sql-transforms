"""Field access over struct_pack: `(struct_pack(a := e)).a` and
`struct_extract(struct_pack(a := e), 'a')` are a pure bind-time desugar —
extracting a field of a just-packed struct IS binding that field's
expression. Case-insensitive matching, DuckDB's missing-key wording, and
the bare-NULL field rides the adoptable-SQLNULL channel (measured:
`- (struct_pack(a := NULL)).a` is BIGINT on DuckDB, the bare field
INTEGER). Every expectation is the live oracle.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

IN = pa.schema(
    [
        pa.field("k", pa.int64(), nullable=False),
        pa.field("s", pa.string(), nullable=False),
    ]
)
ROWS = [{"k": 5, "s": "ab"}, {"k": -3, "s": "Z"}]

BATTERY = [
    "SELECT (struct_pack(a := 1)).a AS o FROM __THIS__",
    "SELECT struct_extract(struct_pack(a := 1), 'a') AS o FROM __THIS__",
    "SELECT (struct_pack(aB := k + 1)).Ab AS o FROM __THIS__",
    "SELECT struct_extract(struct_pack(aB := s), 'AB') AS o FROM __THIS__",
    "SELECT (struct_pack(a := NULL)).a AS o FROM __THIS__",
    "SELECT - (struct_pack(a := NULL)).a AS o FROM __THIS__",
    "SELECT (struct_pack(a := struct_pack(b := k))).a.b AS o FROM __THIS__",
    "SELECT struct_pack(a := 1).a AS o FROM __THIS__",
    "SELECT (struct_pack(a := k > 1, b := 2)).a AS o FROM __THIS__",
]


def _duck(sql: str) -> pa.Table:
    con = duckdb.connect()
    con.execute("CREATE TABLE __THIS__ (k BIGINT, s VARCHAR)")
    for r in ROWS:
        con.execute("INSERT INTO __THIS__ VALUES (?, ?)", [r["k"], r["s"]])
    return con.execute(sql).to_arrow_table()


def _ours(sql: str) -> pa.Table:
    fn = DuckDBInferFn(sql, row_tables={"__THIS__": IN}, static_tables={})
    return fn.infer_arrow(pa.Table.from_pylist(ROWS, schema=IN))


@pytest.mark.parametrize("sql", BATTERY)
def test_struct_pack_field_access_matches_duckdb(sql):
    got, want = _ours(sql), _duck(sql)
    assert got.to_pylist() == want.to_pylist(), sql
    assert got.schema == want.schema, f"{sql}: {got.schema} != {want.schema}"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT (struct_pack(a := 1)).z AS o FROM __THIS__",
        "SELECT struct_extract(struct_pack(a := 1), 'z') AS o FROM __THIS__",
    ],
)
def test_missing_key_refuses_with_duckdb_wording(sql):
    with pytest.raises(Exception, match='Could not find key "z" in struct'):
        DuckDBInferFn(sql, row_tables={"__THIS__": IN}, static_tables={})
