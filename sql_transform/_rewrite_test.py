"""Tests for SQL rewriting via the sqlglot AST."""

import pytest

from sql_transform._rewrite import rewrite_sql
from sql_transform._sql import find_window_aggregates, parse_and_validate


def _rewrite(sql: str) -> str:
    tree = parse_and_validate(sql)
    windows = find_window_aggregates(tree)
    return rewrite_sql(tree, windows)


def test_rewrite_simple_column_pass_through():
    sql = _rewrite("SELECT age AS just_age FROM __THIS__")
    assert sql == "SELECT __THIS__.age AS just_age FROM __THIS__ CROSS JOIN __STATE__"


def test_rewrite_constant_window_agg():
    sql = _rewrite("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )


def test_rewrite_bare_window_agg_alias():
    sql = _rewrite("SELECT MEAN(age) OVER () AS age_avg FROM __THIS__")
    assert sql == "SELECT __STATE__.avg_age AS age_avg FROM __THIS__ CROSS JOIN __STATE__"


def test_rewrite_multiple_projections():
    sql = _rewrite(
        "SELECT age / MEAN(age) OVER () AS age_norm, "
        "score / SUM(score) OVER () AS score_norm FROM __THIS__"
    )
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm, "
        "__THIS__.score / __STATE__.sum_score AS score_norm "
        "FROM __THIS__ CROSS JOIN __STATE__"
    )


def test_rewrite_unaliased_expression_raises_clear_error():
    with pytest.raises(ValueError, match="needs an alias"):
        _rewrite("SELECT age / MEAN(age) OVER () FROM __THIS__")


def test_rewrite_unaliased_bare_window_agg_raises_clear_error():
    with pytest.raises(ValueError, match="needs an alias"):
        _rewrite("SELECT MEAN(age) OVER () FROM __THIS__")


def test_rewrite_bad_column_qualifier_raises():
    with pytest.raises(ValueError, match="does not refer to __THIS__"):
        _rewrite("SELECT foo.age AS x FROM __THIS__")


def test_rewrite_already_qualified_column_stays_this():
    sql = _rewrite("SELECT __THIS__.age AS x FROM __THIS__")
    assert sql == "SELECT __THIS__.age AS x FROM __THIS__ CROSS JOIN __STATE__"
