"""Tests for the DataFusion batch execution path."""

import pyarrow as pa

from sql_transform._batch import run_batch
from sql_transform._schema import synthesize_state_model


def _state(values: dict[str, float]):
    model = synthesize_state_model(values)
    return model(**values)


def test_run_batch_applies_frozen_state():
    state = _state({"avg_age": 30.0})
    table = pa.table({"age": [30.0, 60.0]})
    sql = (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )
    out = run_batch(sql, table, state)
    assert out.column("age_norm").to_pylist() == [1.0, 2.0]


def test_run_batch_empty_state_uses_placeholder():
    state = _state({})  # no window aggregates -> zero-field state model
    table = pa.table({"age": [1, 2, 3]})
    sql = "SELECT __THIS__.age AS age FROM __THIS__ CROSS JOIN __STATE__"
    out = run_batch(sql, table, state)
    assert out.column("age").to_pylist() == [1, 2, 3]
    assert out.schema.names == ["age"]  # placeholder marker column absent


def test_run_batch_empty_batch_preserves_schema():
    state = _state({"avg_age": 30.0})
    table = pa.table({"age": pa.array([], type=pa.float64())})
    sql = (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )
    out = run_batch(sql, table, state)
    assert out.num_rows == 0
    assert out.schema.names == ["age_norm"]
