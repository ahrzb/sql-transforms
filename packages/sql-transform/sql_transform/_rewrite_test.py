"""Tests for SQL rewriting into LEFT-joined state-table SQL."""

import pytest

from sql_transform._rewrite import rewrite_sql
from sql_transform._sql import find_window_aggregates, parse_and_validate


def _rewrite(sql: str) -> str:
    tree = parse_and_validate(sql)
    windows = find_window_aggregates(tree)
    return rewrite_sql(tree, windows)


def test_simple_column_no_state():
    # No window aggregates -> no state join at all.
    sql = _rewrite("SELECT age AS just_age FROM __THIS__")
    assert sql == "SELECT __THIS__.age AS just_age FROM __THIS__"


def test_global_agg_left_joins_marker():
    sql = _rewrite("SELECT age / MEAN(age) OVER () AS age_norm FROM __THIS__")
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS age_norm "
        "FROM __THIS__ LEFT JOIN __STATE__ ON __STATE__.__state_marker__ = 0"
    )


def test_partition_agg_left_joins_on_key():
    sql = _rewrite("SELECT MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__")
    assert sql == (
        "SELECT __STATE_BY_city__.avg_target AS enc "
        "FROM __THIS__ LEFT JOIN __STATE_BY_city__ "
        'ON __THIS__."city" = __STATE_BY_city__."city"'
    )


def test_composite_partition_key_anded():
    sql = _rewrite(
        "SELECT MEAN(target) OVER (PARTITION BY city, region) AS e FROM __THIS__"
    )
    assert sql == (
        "SELECT __STATE_BY_city_region__.avg_target AS e "
        "FROM __THIS__ LEFT JOIN __STATE_BY_city_region__ "
        'ON __THIS__."city" = __STATE_BY_city_region__."city" '
        'AND __THIS__."region" = __STATE_BY_city_region__."region"'
    )


def test_mixed_global_and_partition():
    sql = _rewrite(
        "SELECT age / MEAN(age) OVER () AS n, "
        "MEAN(target) OVER (PARTITION BY city) AS enc FROM __THIS__"
    )
    assert sql == (
        "SELECT __THIS__.age / __STATE__.avg_age AS n, "
        "__STATE_BY_city__.avg_target AS enc "
        "FROM __THIS__ "
        "LEFT JOIN __STATE__ ON __STATE__.__state_marker__ = 0 "
        'LEFT JOIN __STATE_BY_city__ ON __THIS__."city" = __STATE_BY_city__."city"'
    )


def test_unaliased_expression_raises():
    with pytest.raises(ValueError, match="needs an alias"):
        _rewrite("SELECT age / MEAN(age) OVER () FROM __THIS__")


def test_bad_column_qualifier_raises():
    with pytest.raises(ValueError, match="does not refer to __THIS__"):
        _rewrite("SELECT foo.age AS x FROM __THIS__")
