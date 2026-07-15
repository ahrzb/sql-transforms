"""Tests for the DataFusion batch execution path."""

import pyarrow as pa

from sql_transform._batch import run_batch


def test_run_batch_applies_frozen_state():
    state_tables = {"__STATE__": pa.table({"avg_age": [30.0]})}
    table = pa.table({"age": [30.0, 60.0]})
    sql = (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )
    out = run_batch(sql, table, state_tables)
    assert out.column("age_norm").to_pylist() == [1.0, 2.0]


def test_run_batch_no_state_tables():
    table = pa.table({"age": [1, 2, 3]})
    sql = "SELECT __THIS__.age AS age FROM __THIS__"
    out = run_batch(sql, table, {})
    assert out.column("age").to_pylist() == [1, 2, 3]


def test_run_batch_empty_batch_preserves_schema():
    state_tables = {"__STATE__": pa.table({"avg_age": [30.0]})}
    table = pa.table({"age": pa.array([], type=pa.float64())})
    sql = (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )
    out = run_batch(sql, table, state_tables)
    assert out.num_rows == 0
    assert out.schema.names == ["age_norm"]


def test_run_batch_preserves_input_row_order_with_partitioned_state():
    """A LEFT JOIN's physical plan may build its hash table from either side,
    so this guards against the join scrambling output row order relative to
    __THIS__'s input order (see run_batch's row-id/ORDER BY trick)."""
    state_tables = {
        "__STATE_BY_city__": pa.table({"city": ["a", "b"], "avg_target": [1.5, 3.5]})
    }
    table = pa.table({"city": ["a", "b", "zzz"], "target": [0.0, 0.0, 0.0]})
    sql = (
        "SELECT __STATE_BY_city__.avg_target AS enc FROM __THIS__ "
        "LEFT JOIN __STATE_BY_city__ ON __THIS__.city = __STATE_BY_city__.city"
    )
    for _ in range(20):
        out = run_batch(sql, table, state_tables)
        assert out.column("enc").to_pylist() == [1.5, 3.5, None]
