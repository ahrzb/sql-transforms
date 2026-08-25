"""The row-shape contract flag: shape="map" | "filter" | "many".

"map" is a STATIC build-time proof of exactly one output row per input row
(out[i] <-> in[i]) — the serving-path guarantee. "filter" is the default
0..1 behavior, byte-identical to before the flag existed. "many" is
reserved for join multiplicity (stage B) and is the only shape under which
those constructs will ever build.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from confit import DuckDBInferFn

T = pa.schema([pa.field("a", pa.int64(), nullable=False), pa.field("s", pa.string())])
DIM = pa.table({"id": [1, 2], "v": [10, 20]})


def build(sql, shape=None, statics=None):
    kwargs = {}
    if shape is not None:
        kwargs["shape"] = shape
    return DuckDBInferFn(
        sql, row_tables={"__THIS__": T}, static_tables=statics or {}, **kwargs
    )


def test_map_serves_projections_and_left_joins():
    fn = build("SELECT a + 1 AS b, upper(s) AS u FROM __THIS__", shape="map")
    assert fn.shape == "map"
    got = fn.infer_rows([{"a": 1, "s": "x"}, {"a": 2, "s": None}])
    assert [r["b"] for r in got] == [2, 3]  # exactly one out per in, in order

    fn = build(
        "SELECT a, v FROM __THIS__ LEFT JOIN d ON a = d.id",
        shape="map",
        statics={"d": DIM},
    )
    got = fn.infer_rows([{"a": 1, "s": None}, {"a": 99, "s": None}])
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
        got = fn.infer_rows([{"a": 1, "s": None}, {"a": 2, "s": None}])
        assert [r["a"] for r in got] == [2]


def test_many_enables_multiplicity_and_bad_values_are_named():
    # 'many' is the stage-B opt-in: dup-key joins build ONLY under it.
    dup = pa.table({"id": [1, 1], "v": [10, 11]})
    with pytest.raises(ValueError, match="duplicate map key"):
        build("SELECT a, v FROM __THIS__ JOIN d ON a = d.id", statics={"d": dup})
    fn = build(
        "SELECT a, v FROM __THIS__ JOIN d ON a = d.id",
        shape="many",
        statics={"d": dup},
    )
    assert fn.shape == "many"
    got = fn.infer_rows([{"a": 1, "s": None}, {"a": 2, "s": None}])
    assert sorted(r["v"] for r in got) == [10, 11]
    with pytest.raises(ValueError, match="must be 'map', 'filter', or 'many'"):
        build("SELECT a FROM __THIS__", shape="projection")
