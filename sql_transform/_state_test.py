"""Tests for state extraction from DataFusion logical plans."""

import datafusion

from sql_transform._state import extract_state


def test_extract_constant_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = "SELECT age / MEAN(age) OVER () AS age_norm FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "age_norm" in state
    assert state["age_norm"] == 30.0


def test_extract_partitioned_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
        name="data",
    )

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert "city_enc" in state
    assert state["city_enc"] == {
        "lookup": {"a": 2.0, "b": 3.0},
        "partition_col": "city",
    }


def test_multiple_window_aggs():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"age": [25, 30, 35], "score": [10, 20, 30]},
        name="data",
    )

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state["age_norm"] == 30.0
    assert state["score_norm"] == 60.0
