"""Tests for state extraction from DataFusion logical plans."""

import datafusion
import pytest

from sql_transform._state import extract_state, state_key


def test_state_key_lowercases_and_strips_qualifier():
    assert state_key("AVG", "age") == "avg_age"
    assert state_key("avg", "AGE") == "avg_age"


def test_extract_constant_window_agg():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = "SELECT age / MEAN(age) OVER () AS age_norm FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.avg_age == 30.0


def test_extract_dedups_repeated_aggregate():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "MEAN(age) OVER () AS age_avg FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    # Both projections reference the same (fn, col) pair -> one field.
    # DataFusion normalizes MEAN to avg internally, so the key is avg_age.
    assert state.model_dump() == {"avg_age": 30.0}


def test_extract_multiple_distinct_aggregates():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35], "score": [10, 20, 30]}, name="data")

    sql = (
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.avg_age == 30.0
    assert state.sum_score == 60.0


def test_extract_no_aggregates_returns_empty_state():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [1, 2, 3]}, name="data")

    sql = "SELECT age FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.model_dump() == {}


def test_extract_partitioned_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
        name="data",
    )

    sql = "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    with pytest.raises(NotImplementedError):
        extract_state(plan, ctx, "data")


def test_extract_multi_column_partitioned_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {
            "city": ["a", "b", "a", "b"],
            "region": ["x", "y", "x", "y"],
            "target": [1.0, 2.0, 3.0, 4.0],
        },
        name="data",
    )

    sql = "SELECT MEAN(target) OVER (PARTITION BY city, region) AS enc FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    with pytest.raises(NotImplementedError):
        extract_state(plan, ctx, "data")


def test_extract_order_by_window_agg_raises_not_implemented():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"age": [25, 30, 35]}, name="data")

    sql = "SELECT MEAN(age) OVER (ORDER BY age) AS running_avg FROM data"
    df = ctx.sql(sql)
    plan = df.logical_plan()

    with pytest.raises(NotImplementedError):
        extract_state(plan, ctx, "data")


def test_extract_preserves_column_case_in_query():
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"Age": [25.0, 30.0, 35.0]}, name="data")

    sql = 'SELECT "Age" / MEAN("Age") OVER () AS age_norm FROM data'
    df = ctx.sql(sql)
    plan = df.logical_plan()

    state = extract_state(plan, ctx, "data")

    assert state.avg_age == 30.0


def test_extract_case_differing_columns_raises_ambiguous_error():
    ctx = datafusion.SessionContext()
    ctx.from_pydict(
        {"age": [25.0, 30.0, 35.0], "Age": [100.0, 200.0, 300.0]}, name="data"
    )

    sql = (
        'SELECT age / MEAN(age) OVER () + "Age" / MEAN("Age") OVER () '
        "AS combo FROM data"
    )
    df = ctx.sql(sql)
    plan = df.logical_plan()

    with pytest.raises(ValueError, match="Ambiguous window aggregate"):
        extract_state(plan, ctx, "data")
