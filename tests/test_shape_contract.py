"""The row-shape contract flag (TASK-58): shape="map" | "filter" | "many".

"map" is a STATIC build-time proof of exactly one output row per input row
(out[i] <-> in[i]) — the serving-path guarantee. "filter" is the default
0..1 behavior, byte-identical to before the flag existed. "many" is
reserved for join multiplicity (stage B) and is the only shape under which
those constructs will ever build.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from pydantic import create_model

from sql_transform._interpreter import DuckDBInferFn

T = create_model("T", a=(int, ...), s=(str | None, None))
DIM = pa.table({"id": [1, 2], "v": [10, 20]})


def build(sql, shape=None, statics=None):
    kwargs = {"output": "dict"}
    if shape is not None:
        kwargs["shape"] = shape
    return DuckDBInferFn(
        sql, row_tables={"__THIS__": T}, static_tables=statics or {}, **kwargs
    )


def test_map_serves_projections_and_left_joins():
    fn = build("SELECT a + 1 AS b, upper(s) AS u FROM __THIS__", shape="map")
    assert fn.shape == "map"
    got = fn.infer({"__THIS__": [T(a=1, s="x"), T(a=2, s=None)]})
    assert [r["b"] for r in got] == [2, 3]  # exactly one out per in, in order

    fn = build(
        "SELECT a, v FROM __THIS__ LEFT JOIN d ON a = d.id",
        shape="map",
        statics={"d": DIM},
    )
    got = fn.infer({"__THIS__": [T(a=1), T(a=99)]})
    assert [r["v"] for r in got] == [10, None]  # a miss maps, never drops


def test_map_rejects_row_dropping_constructs():
    with pytest.raises(ValueError, match="shape='map'.*WHERE"):
        build("SELECT a FROM __THIS__ WHERE a > 0", shape="map")
    with pytest.raises(ValueError, match="shape='map'.*INNER JOIN 'd'"):
        build(
            "SELECT a, v FROM __THIS__ JOIN d ON a = d.id",
            shape="map",
            statics={"d": DIM},
        )
    # A static-only query emits fixed rows unrelated to the input.
    with pytest.raises(ValueError, match="shape='map'.*static-tables-only"):
        build("SELECT max(id) FROM d", shape="map", statics={"d": DIM})


def test_filter_default_unchanged():
    for kwargs in [{}, {"shape": "filter"}]:
        fn = build("SELECT a FROM __THIS__ WHERE a > 1", **kwargs)
        assert fn.shape == "filter"
        got = fn.infer({"__THIS__": [T(a=1), T(a=2)]})
        assert [r["a"] for r in got] == [2]


def test_many_is_reserved_and_bad_values_are_named():
    with pytest.raises(ValueError, match="stage B"):
        build("SELECT a FROM __THIS__", shape="many")
    with pytest.raises(ValueError, match="must be 'map', 'filter', or 'many'"):
        build("SELECT a FROM __THIS__", shape="projection")
