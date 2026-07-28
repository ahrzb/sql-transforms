"""Tests for building typed per-partition state tables."""

import datafusion
import pytest

from sql_transform._sql import find_window_aggregates, parse_and_validate
from sql_transform._state import build_state_tables, state_key, state_table_name


def _windows(sql: str):
    return find_window_aggregates(parse_and_validate(sql))


def _ctx(data: dict, name: str = "__THIS__"):
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name=name)
    return ctx


def test_state_key_lowercases():
    assert state_key("AVG", "age") == "avg_age"
    assert state_key("avg", "AGE") == "avg_age"


def test_state_table_name_global_and_partitioned():
    assert state_table_name(()) == "__STATE__"
    assert state_table_name(("city",)) == "__STATE_BY_city__"
    assert state_table_name(("city", "region")) == "__STATE_BY_city_region__"


def test_global_state_table_has_marker_and_value():
    ctx = _ctx({"age": [25, 30, 35]})
    windows = _windows("SELECT age / MEAN(age) OVER () AS x FROM __THIS__")
    tables = build_state_tables(windows, ctx, "__THIS__")

    assert set(tables) == {"__STATE__"}
    t = tables["__STATE__"]
    assert t.num_rows == 1
    assert t.column("avg_age").to_pylist() == [30.0]
    assert t.column("__state_marker__").to_pylist() == [0]


def test_partition_state_table_per_key():
    ctx = _ctx({"city": ["a", "b", "a", "b"], "target": [1.0, 3.0, 2.0, 4.0]})
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")

    assert set(tables) == {"__STATE_BY_city__"}
    t = tables["__STATE_BY_city__"]
    got = dict(
        zip(
            t.column("city").to_pylist(),
            t.column("avg_target").to_pylist(),
            strict=True,
        )
    )
    assert got == {"a": 1.5, "b": 3.5}
    assert "__state_marker__" not in t.schema.names


def test_partition_value_type_preserved_for_count():
    import pyarrow as pa

    ctx = _ctx({"city": ["a", "a", "b"], "target": [1, 2, 3]})
    windows = _windows(
        "SELECT COUNT(target) OVER (PARTITION BY city) AS n FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    t = tables["__STATE_BY_city__"]
    # COUNT is an integer count-encoding, not a float.
    assert pa.types.is_integer(t.column("count_target").type)


def test_dedup_repeated_aggregate_in_group():
    ctx = _ctx({"city": ["a", "b"], "target": [1.0, 2.0]})
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS a, "
        "MEAN(target) OVER (PARTITION BY city) AS b FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    t = tables["__STATE_BY_city__"]
    # One value column despite two projections.
    assert [n for n in t.schema.names if n != "city"] == ["avg_target"]


def test_distinct_key_sets_distinct_tables():
    ctx = _ctx(
        {
            "city": ["a", "b"],
            "region": ["x", "y"],
            "target": [1.0, 2.0],
        }
    )
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS a, "
        "SUM(target) OVER (PARTITION BY region) AS b FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    assert set(tables) == {"__STATE_BY_city__", "__STATE_BY_region__"}


def test_composite_partition_key():
    ctx = _ctx(
        {
            "city": ["a", "a"],
            "region": ["x", "x"],
            "target": [2.0, 4.0],
        }
    )
    windows = _windows(
        "SELECT MEAN(target) OVER (PARTITION BY city, region) AS e FROM __THIS__"
    )
    tables = build_state_tables(windows, ctx, "__THIS__")
    t = tables["__STATE_BY_city_region__"]
    assert set(t.schema.names) == {"city", "region", "avg_target"}
    assert t.column("avg_target").to_pylist() == [3.0]


def test_no_windows_returns_empty_dict():
    ctx = _ctx({"age": [1, 2, 3]})
    windows = _windows("SELECT age AS x FROM __THIS__")
    assert build_state_tables(windows, ctx, "__THIS__") == {}


def test_order_by_still_not_implemented():
    ctx = _ctx({"age": [1, 2, 3]})
    windows = _windows("SELECT MEAN(age) OVER (ORDER BY age) AS r FROM __THIS__")
    with pytest.raises(NotImplementedError):
        build_state_tables(windows, ctx, "__THIS__")


def test_case_collision_raises():
    ctx = _ctx({"age": [1.0], "Age": [2.0]})
    windows = _windows(
        'SELECT MEAN(age) OVER () + MEAN("Age") OVER () AS c FROM __THIS__'
    )
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        build_state_tables(windows, ctx, "__THIS__")
