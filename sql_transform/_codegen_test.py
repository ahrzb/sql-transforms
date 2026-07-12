"""Tests for Python code generation from DataFusion logical plans."""

import datafusion

from sql_transform._codegen import generate_infer_fn
from sql_transform._state import extract_state


def _setup(sql: str, data: dict) -> tuple:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    df = ctx.sql(sql)
    plan = df.logical_plan()
    state = extract_state(plan, ctx, "data")
    infer_fn = generate_infer_fn(plan, state)
    return infer_fn, state


def test_generate_constant_window_agg():
    infer_fn, state = _setup(
        "SELECT age / MEAN(age) OVER () AS age_norm FROM data",
        {"age": [25, 30, 35]},
    )
    result = infer_fn({"age": 40})
    assert result == {"age_norm": 40.0 / 30.0}


def test_generate_partitioned_window_agg():
    infer_fn, _ = _setup(
        "SELECT MEAN(target) OVER (PARTITION BY city) AS city_enc FROM data",
        {"city": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]},
    )
    result = infer_fn({"city": "a", "target": 10.0})
    assert result == {"city_enc": 2.0}


def test_generate_multiple_transforms():
    infer_fn, _ = _setup(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data",
        {"age": [25, 30, 35], "score": [10, 20, 30]},
    )
    result = infer_fn({"age": 40, "score": 5})
    assert result["age_norm"] == 40.0 / 30.0
    assert result["score_norm"] == 5.0 / 60.0


def test_generate_simple_column_pass_through():
    infer_fn, state = _setup(
        "SELECT age AS just_age FROM data",
        {"age": [1, 2, 3]},
    )
    result = infer_fn({"age": 42})
    assert result == {"just_age": 42}
