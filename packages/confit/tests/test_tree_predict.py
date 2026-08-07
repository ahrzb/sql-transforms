"""The `models=` boundary: two pyarrow tables in, native scoring out.

DuckDB has no `tree_predict`, so there is no differential oracle here — the
values are asserted directly against the tree the test itself builds. Parity
against sklearn is a separate gate that lives in sql-transform.

Every scoring case runs on BOTH backends: `duckdb/mod.rs` discards the
cranelift compile error and falls back to the interpreter silently, so a
cranelift-only break would otherwise pass green.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from test_duckdb_interpreter import _row_model

NODE_SCHEMA = pa.schema(
    [
        pa.field("model_id", pa.int64(), nullable=False),
        pa.field("tree_id", pa.int64(), nullable=False),
        pa.field("node_id", pa.int64(), nullable=False),
        pa.field("feature", pa.int32(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=False),
        pa.field("left", pa.int32(), nullable=False),
        pa.field("right", pa.int32(), nullable=False),
        pa.field("missing_left", pa.bool_(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
)
MODEL_SCHEMA = pa.schema(
    [
        pa.field("model_id", pa.int64(), nullable=False),
        pa.field("base", pa.float64(), nullable=False),
        pa.field("agg", pa.string(), nullable=False),
        pa.field("link", pa.string(), nullable=False),
    ]
)


def split(
    model_id, tree_id, node_id, feature, threshold, left, right, missing_left=True
):
    return {
        "model_id": model_id,
        "tree_id": tree_id,
        "node_id": node_id,
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "missing_left": missing_left,
        "value": 0.0,
    }


def leaf(model_id, tree_id, node_id, value):
    return {
        "model_id": model_id,
        "tree_id": tree_id,
        "node_id": node_id,
        "feature": -1,
        "threshold": 0.0,
        "left": -1,
        "right": -1,
        "missing_left": True,
        "value": value,
    }


def ensemble(
    nodes: list[dict[str, Any]],
    features: list[str],
    headers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    headers = headers or [
        {"model_id": 0, "base": 0.0, "agg": "sum", "link": "identity"}
    ]
    return {
        "nodes": pa.Table.from_pylist(nodes, schema=NODE_SCHEMA),
        "models": pa.Table.from_pylist(headers, schema=MODEL_SCHEMA),
        "features": features,
    }


# One model, one tree: x <= 0.5 -> 10.0, else 20.0. Missing goes left.
STUMP = ensemble(
    [
        split(0, 0, 0, feature=0, threshold=0.5, left=1, right=2),
        leaf(0, 0, 1, 10.0),
        leaf(0, 0, 2, 20.0),
    ],
    features=["x"],
)


_EXPECT: list[str] = []


@pytest.fixture(params=["cranelift", "interpreter"])
def backend(request, monkeypatch):
    if request.param == "interpreter":
        monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    else:
        monkeypatch.delenv("SPECIALIZER_FORCE_INTERP", raising=False)
    _EXPECT.append(request.param)
    yield request.param
    _EXPECT.pop()


def check_backend(fn):
    """The cranelift failure path falls back to the interpreter SILENTLY, so
    asserting the engine is the only thing that stops the [cranelift] runs
    from quietly becoming a second interpreter suite."""
    if _EXPECT:
        assert fn.backend == _EXPECT[-1]


def run(sql, row_schema, rows, models, **kw):
    model = _row_model(row_schema)
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": model},
        static_tables={},
        models=models,
        **kw,
    )
    check_backend(fn)
    return [r.model_dump() for r in fn.infer({"__THIS__": [model(**r) for r in rows]})]


SQL = "SELECT tree_predict('trees', id, struct_pack(x := x)) AS p FROM __THIS__"


def test_stump_scores_both_leaves(backend):
    got = run(
        SQL,
        {"id": "int", "x": "float"},
        [{"id": 0, "x": 0.0}, {"id": 0, "x": 1.0}, {"id": 0, "x": 0.5}],
        {"trees": STUMP},
    )
    assert [r["p"] for r in got] == [10.0, 20.0, 10.0]


def test_null_feature_takes_the_missing_branch(backend):
    """A NULL feature is a value the model handles — only a missing MODEL
    nulls the output. `missing_left` decides, per node."""
    right = ensemble(
        [
            split(
                0, 0, 0, feature=0, threshold=0.5, left=1, right=2, missing_left=False
            ),
            leaf(0, 0, 1, 10.0),
            leaf(0, 0, 2, 20.0),
        ],
        features=["x"],
    )
    rows = [{"id": 0, "x": None}]
    assert run(SQL, {"id": "int", "x": "float?"}, rows, {"trees": STUMP}) == [
        {"p": 10.0}
    ]
    assert run(SQL, {"id": "int", "x": "float?"}, rows, {"trees": right}) == [
        {"p": 20.0}
    ]


def test_unknown_model_id_traps(backend):
    """A NULL feature is data; a model id naming no model is a bug in the
    caller's params table. It raises rather than quietly nulling."""
    with pytest.raises(ValueError, match="no model with id 7"):
        run(
            SQL,
            {"id": "int", "x": "float"},
            [{"id": 0, "x": 0.0}, {"id": 7, "x": 0.0}],
            {"trees": STUMP},
        )


def test_null_id_yields_null(backend):
    got = run(
        SQL,
        {"id": "int?", "x": "float"},
        [{"id": None, "x": 0.0}],
        {"trees": STUMP},
    )
    assert got == [{"p": None}]


def test_forest_averages_and_boosted_sums_the_base(backend):
    trees = [
        split(0, t, 0, feature=0, threshold=0.5, left=1, right=2) for t in range(2)
    ]
    nodes = [
        trees[0],
        leaf(0, 0, 1, 10.0),
        leaf(0, 0, 2, 20.0),
        trees[1],
        leaf(0, 1, 1, 30.0),
        leaf(0, 1, 2, 40.0),
    ]
    mean = ensemble(
        nodes, ["x"], [{"model_id": 0, "base": 1.0, "agg": "mean", "link": "identity"}]
    )
    total = ensemble(
        nodes, ["x"], [{"model_id": 0, "base": 1.0, "agg": "sum", "link": "identity"}]
    )
    rows = [{"id": 0, "x": 0.0}]
    assert run(SQL, {"id": "int", "x": "float"}, rows, {"trees": mean}) == [{"p": 21.0}]
    assert run(SQL, {"id": "int", "x": "float"}, rows, {"trees": total}) == [
        {"p": 41.0}
    ]


def test_sigmoid_link(backend):
    sig = ensemble(
        [
            split(0, 0, 0, feature=0, threshold=0.5, left=1, right=2),
            leaf(0, 0, 1, 0.0),
            leaf(0, 0, 2, 1.0),
        ],
        ["x"],
        [{"model_id": 0, "base": 0.0, "agg": "sum", "link": "sigmoid"}],
    )
    got = run(
        SQL,
        {"id": "int", "x": "float"},
        [{"id": 0, "x": 0.0}, {"id": 0, "x": 1.0}],
        {"trees": sig},
    )
    assert got[0]["p"] == pytest.approx(0.5)
    assert got[0 + 1]["p"] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_two_models_score_independently(backend):
    two = ensemble(
        [
            leaf(0, 0, 0, 10.0),
            leaf(1, 0, 0, 20.0),
        ],
        ["x"],
        [
            {"model_id": 0, "base": 0.0, "agg": "sum", "link": "identity"},
            {"model_id": 1, "base": 0.0, "agg": "sum", "link": "identity"},
        ],
    )
    got = run(
        SQL,
        {"id": "int", "x": "float"},
        [{"id": 0, "x": 0.0}, {"id": 1, "x": 0.0}],
        {"trees": two},
    )
    assert [r["p"] for r in got] == [10.0, 20.0]


def test_features_resolve_by_name_not_position(backend):
    """Declared order is ['a', 'b']; the call site writes them backwards."""
    two_feat = ensemble(
        [
            split(0, 0, 0, feature=1, threshold=0.5, left=1, right=2),
            leaf(0, 0, 1, 10.0),
            leaf(0, 0, 2, 20.0),
        ],
        features=["a", "b"],
    )
    got = run(
        "SELECT tree_predict('trees', id, struct_pack(b := b, a := a)) AS p "
        "FROM __THIS__",
        {"id": "int", "a": "float", "b": "float"},
        [{"id": 0, "a": 9.0, "b": 0.0}, {"id": 0, "a": 0.0, "b": 9.0}],
        {"trees": two_feat},
    )
    assert [r["p"] for r in got] == [10.0, 20.0]


def test_int_feature_promotes_to_double(backend):
    got = run(
        SQL,
        {"id": "int", "x": "int"},
        [{"id": 0, "x": 0}, {"id": 0, "x": 1}],
        {"trees": STUMP},
    )
    assert [r["p"] for r in got] == [10.0, 20.0]


# ------------------------------------------------------------ refusals --


def build(models, sql=SQL, row_schema=None):
    model = _row_model(row_schema or {"id": "int", "x": "float"})
    return DuckDBInferFn(
        sql, row_tables={"__THIS__": model}, static_tables={}, models=models
    )


def test_unnamed_model_set_refuses():
    with pytest.raises(Exception, match="not provided to prepare"):
        build({"other": STUMP})


def test_feature_name_mismatch_refuses():
    with pytest.raises(Exception, match="feature"):
        build({"trees": ensemble(STUMP["nodes"].to_pylist(), features=["zzz"])})


def test_child_index_out_of_range_refuses():
    bad = ensemble(
        [
            split(0, 0, 0, feature=0, threshold=0.5, left=1, right=99),
            leaf(0, 0, 1, 10.0),
        ],
        features=["x"],
    )
    with pytest.raises(Exception, match="99|child"):
        build({"trees": bad})


def test_unknown_agg_refuses():
    bad = ensemble(
        [leaf(0, 0, 0, 1.0)],
        ["x"],
        [{"model_id": 0, "base": 0.0, "agg": "median", "link": "identity"}],
    )
    with pytest.raises(Exception, match="median"):
        build({"trees": bad})


def test_feature_index_beyond_declared_width_refuses():
    bad = ensemble(
        [
            split(0, 0, 0, feature=3, threshold=0.5, left=1, right=2),
            leaf(0, 0, 1, 10.0),
            leaf(0, 0, 2, 20.0),
        ],
        features=["x"],
    )
    with pytest.raises(Exception, match="feature"):
        build({"trees": bad})


def test_null_in_a_node_column_refuses():
    nodes = pa.table(
        {
            "model_id": pa.array([0], pa.int64()),
            "tree_id": pa.array([0], pa.int64()),
            "node_id": pa.array([0], pa.int64()),
            "feature": pa.array([-1], pa.int32()),
            "threshold": pa.array([0.0], pa.float64()),
            "left": pa.array([-1], pa.int32()),
            "right": pa.array([-1], pa.int32()),
            "missing_left": pa.array([True], pa.bool_()),
            "value": pa.array([None], pa.float64()),
        }
    )
    with pytest.raises(Exception, match="NULL"):
        build({"trees": {**STUMP, "nodes": nodes}})


def test_malformed_model_set_entry_refuses():
    with pytest.raises(Exception, match="nodes"):
        build({"trees": {"models": STUMP["models"], "features": ["x"]}})


def test_missing_node_column_refuses():
    with pytest.raises(Exception, match="threshold"):
        build({"trees": {**STUMP, "nodes": STUMP["nodes"].drop_columns(["threshold"])}})


def test_model_set_alongside_a_static_join(backend):
    """Model statics are appended AFTER every join static — if they were
    interleaved, the probe's `@N` would shift and this would read garbage."""
    params = pa.table(
        {"k": pa.array([1], pa.int64()), "est": pa.array([0], pa.int64())}
    )
    model = _row_model({"k": "int", "x": "float"})
    fn = DuckDBInferFn(
        "SELECT tree_predict('trees', p.est, struct_pack(x := t.x)) AS p "
        "FROM __THIS__ AS t LEFT JOIN params AS p ON t.k = p.k",
        row_tables={"__THIS__": model},
        static_tables={"params": params},
        models={"trees": STUMP},
    )
    check_backend(fn)
    rows = [{"k": 1, "x": 0.0}, {"k": 1, "x": 1.0}, {"k": 2, "x": 0.0}]
    got = [r.model_dump() for r in fn.infer({"__THIS__": [model(**r) for r in rows]})]
    assert [r["p"] for r in got] == [10.0, 20.0, None]


def test_unused_model_set_costs_nothing(backend):
    got = run(
        "SELECT x AS p FROM __THIS__",
        {"id": "int", "x": "float"},
        [{"id": 0, "x": 3.0}],
        {"trees": STUMP},
    )
    assert got == [{"p": 3.0}]
