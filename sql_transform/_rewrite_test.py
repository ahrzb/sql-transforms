"""Tests for SQL rewriting from DataFusion logical plans."""

import datafusion
import pytest

from sql_transform._rewrite import rewrite_sql


def _plan(sql: str, data: dict) -> datafusion.plan.LogicalPlan:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).logical_plan()


def test_rewrite_simple_column_pass_through():
    plan = _plan("SELECT age AS just_age FROM data", {"age": [1, 2, 3]})
    sql = rewrite_sql(plan)
    assert sql == "SELECT __THIS__.age AS just_age FROM __THIS__, __STATE__"


def test_rewrite_constant_window_agg():
    plan = _plan(
        "SELECT age / MEAN(age) OVER () AS age_norm FROM data",
        {"age": [25, 30, 35]},
    )
    sql = rewrite_sql(plan)
    # DataFusion normalizes MEAN to avg internally, so the key is avg_age.
    assert sql == (
        "SELECT (__THIS__.age / __STATE__.avg_age) AS age_norm FROM __THIS__, __STATE__"
    )


def test_rewrite_bare_window_agg_alias():
    plan = _plan(
        "SELECT MEAN(age) OVER () AS age_avg FROM data",
        {"age": [25, 30, 35]},
    )
    sql = rewrite_sql(plan)
    assert sql == "SELECT __STATE__.avg_age AS age_avg FROM __THIS__, __STATE__"


def test_rewrite_multiple_projections():
    plan = _plan(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM data",
        {"age": [25, 30, 35], "score": [10, 20, 30]},
    )
    sql = rewrite_sql(plan)
    assert sql == (
        "SELECT (__THIS__.age / __STATE__.avg_age) AS age_norm, "
        "(__THIS__.score / __STATE__.sum_score) AS score_norm "
        "FROM __THIS__, __STATE__"
    )


def test_rewrite_unaliased_expression_raises_clear_error():
    plan = _plan(
        "SELECT age / MEAN(age) OVER () FROM data",
        {"age": [25, 30, 35]},
    )
    with pytest.raises(ValueError, match="needs an alias"):
        rewrite_sql(plan)


def test_rewrite_unaliased_bare_window_agg_raises_clear_error():
    plan = _plan(
        "SELECT MEAN(age) OVER () FROM data",
        {"age": [25, 30, 35]},
    )
    with pytest.raises(ValueError, match="not a valid identifier"):
        rewrite_sql(plan)
