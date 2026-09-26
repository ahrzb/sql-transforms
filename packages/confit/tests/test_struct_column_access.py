"""Field access over a struct COLUMN in every spelling DuckDB serves:
`s['f']`, `struct_extract(s, 'f')`, `(s).f`, chains mixing them with the
dot form, and `v.*` over a joined static struct. Each reads exactly what
`s.f` reads. Every expectation is the live oracle, names included (DuckDB
names `s['n'].x` as `(s['n']).x`).
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn, compare
from confit.oracle import Oracle

_S = pa.struct(
    [("a", pa.int64()), ("B", pa.string()), ("n", pa.struct([("x", pa.float64())]))]
)
ROW = pa.schema(
    [
        pa.field("k", pa.int64()),
        pa.field("s", _S),
        pa.field("w", pa.struct([("a", pa.int64())])),
    ]
)
ROWS = [
    {"k": 1, "s": {"a": 1, "B": "q", "n": {"x": 1.5}}, "w": {"a": 9}},
    {"k": 2, "s": None, "w": None},
    {"k": 3, "s": {"a": None, "B": None, "n": None}, "w": {"a": None}},
]
D = pa.table(
    {
        "id": pa.array([1, 3], pa.int64()),
        "v": pa.array(
            [{"x": 1, "y": "p"}, None],
            pa.struct([("x", pa.int64()), ("y", pa.string())]),
        ),
        "w": pa.array([{"a": 5}, None], pa.struct([("a", pa.int64())])),
        "n": pa.array(
            [{"m": {"q": 1}, "r": 2}, None],
            pa.struct([("m", pa.struct([("q", pa.int64())])), ("r", pa.int64())]),
        ),
    }
)


def _duck(sql):
    with Oracle() as o:
        o.load("__THIS__", pa.Table.from_pylist(ROWS, schema=ROW))
        o.load("d", D)
        return o.try_answer(sql)


def _build(sql):
    return DuckDBInferFn(sql, row_tables={"__THIS__": ROW}, static_tables={"d": D})


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT s['a'] FROM __THIS__",
        "SELECT s['b'] FROM __THIS__",
        "SELECT s['A'] FROM __THIS__",
        "SELECT s['n']['x'] FROM __THIS__",
        "SELECT s.n['x'] FROM __THIS__",
        "SELECT s['n'].x FROM __THIS__",
        "SELECT (s['n']).x FROM __THIS__",
        "SELECT (s).a FROM __THIS__",
        "SELECT __THIS__.s['a'] FROM __THIS__",
        "SELECT struct_extract(s, 'a') FROM __THIS__",
        "SELECT struct_extract(s, 'A') FROM __THIS__",
        "SELECT struct_extract(s.n, 'x') FROM __THIS__",
        "SELECT struct_extract(struct_extract(s, 'n'), 'x') FROM __THIS__",
        "SELECT struct_extract(s['n'], 'x') AS o FROM __THIS__",
        "SELECT s['a'] + 1 AS o FROM __THIS__ WHERE s['a'] IS NOT NULL",
        "SELECT coalesce(s['a'], w['a']) AS o FROM __THIS__",
        "SELECT v['x'], struct_extract(v, 'y') FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT d.v['x'] AS o FROM __THIS__ LEFT JOIN d ON k = id",
        # struct star over a joined static struct
        "SELECT v.* FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT v.* EXCLUDE (y), k FROM __THIS__ LEFT JOIN d ON k = id",
        "SELECT v.* FROM __THIS__ JOIN d ON k = v.x",
        "SELECT v.* FROM __THIS__ LEFT JOIN d USING (w)",
        "SELECT n.* EXCLUDE (m) FROM __THIS__ LEFT JOIN d ON k = id",
    ],
)
def test_struct_column_access_matches_duckdb(sql):
    want = _duck(sql)
    assert isinstance(want, pa.Table), f"oracle moved: {want}"
    fn = _build(sql)
    got = fn.infer_arrow(pa.Table.from_pylist(ROWS, schema=ROW))
    compare.assert_schema(got.schema, want.schema, ctx=sql)
    # DuckDB's join output order is a hash-join accident: compare as a
    # multiset (the pinned join parity contract).
    compare.assert_rows(compare.rows(got), compare.rows(want), ordered=False, ctx=sql)
    assert fn.infer_rows(ROWS) == got.to_pylist()


@pytest.mark.parametrize(
    ("sql", "needle"),
    [
        ("SELECT s['z'] FROM __THIS__", 'Could not find key "z" in struct'),
        ("SELECT struct_extract(s, 'z') FROM __THIS__", 'Could not find key "z"'),
        # an integer key on a NAMED struct is DuckDB's binder error
        ("SELECT s[1] FROM __THIS__", "struct column 's'"),
        ("SELECT struct_extract(s, 1) FROM __THIS__", "struct_extract"),
        (
            "SELECT w.* FROM __THIS__ LEFT JOIN d ON k = id",
            "ambiguous column 'w'",
        ),
    ],
)
def test_struct_column_access_refuses_where_duckdb_does(sql, needle):
    assert not isinstance(_duck(sql), pa.Table), "oracle moved: DuckDB serves"
    with pytest.raises(ValueError, match=needle):
        _build(sql)


@pytest.mark.parametrize(
    "sql",
    [
        # a nested struct field is a whole STRUCT value, as for `s.n`
        "SELECT s['n'] FROM __THIS__",
        "SELECT n.* FROM __THIS__ LEFT JOIN d ON k = id",
        # `s` names the relation here: `s['a']` is NOT `s.a` (the table's
        # column), so the subscript form keeps the refusing path
        "SELECT s['a'] FROM __THIS__ AS s",
    ],
)
def test_struct_column_access_refusals_duckdb_serves(sql):
    assert isinstance(_duck(sql), pa.Table), "oracle moved: DuckDB refuses"
    with pytest.raises(ValueError, match="unsupported"):
        _build(sql)
