"""Tests for SQL parsing, scope validation, and window-aggregate discovery."""

import pytest
from sqlglot import exp

from sql_transform._sql import find_window_aggregates, parse_and_validate


def test_parse_valid_simple_select():
    tree = parse_and_validate("SELECT age FROM __THIS__")
    assert isinstance(tree, exp.Select)
    assert tree.sql() == "SELECT age FROM __THIS__"


def test_parse_rejects_wrong_from_table():
    with pytest.raises(ValueError, match="__THIS__"):
        parse_and_validate("SELECT age FROM data")


def test_parse_rejects_aliased_this():
    with pytest.raises(ValueError, match="__THIS__"):
        parse_and_validate("SELECT age FROM __THIS__ AS t")


def test_parse_rejects_where():
    with pytest.raises(ValueError, match="WHERE"):
        parse_and_validate("SELECT age FROM __THIS__ WHERE age > 1")


def test_parse_rejects_join():
    with pytest.raises(ValueError, match="JOIN"):
        parse_and_validate(
            "SELECT __THIS__.x FROM __THIS__ JOIN b ON __THIS__.id = b.id"
        )


def test_parse_rejects_group_by():
    with pytest.raises(ValueError, match="GROUP BY"):
        parse_and_validate("SELECT age FROM __THIS__ GROUP BY age")


def test_parse_rejects_order_by():
    with pytest.raises(ValueError, match="ORDER BY"):
        parse_and_validate("SELECT age FROM __THIS__ ORDER BY age")


def test_parse_rejects_limit():
    with pytest.raises(ValueError, match="LIMIT"):
        parse_and_validate("SELECT age FROM __THIS__ LIMIT 5")


def test_parse_rejects_multiple_statements():
    with pytest.raises(ValueError, match="one SQL statement"):
        parse_and_validate("SELECT age FROM __THIS__; SELECT age FROM __THIS__")


def test_parse_rejects_non_select():
    with pytest.raises(ValueError, match="SELECT"):
        parse_and_validate("CREATE TABLE t (id INT)")


def test_find_window_aggregates_detects_avg():
    tree = parse_and_validate("SELECT AVG(age) OVER () AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert len(windows) == 1
    assert windows[0].fn == "AVG"
    assert windows[0].col == "age"
    assert windows[0].has_partition is False
    assert windows[0].has_order is False


def test_find_window_aggregates_normalizes_mean_to_avg():
    tree = parse_and_validate("SELECT MEAN(age) OVER () AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert windows[0].fn == "AVG"


def test_find_window_aggregates_detects_partition_by():
    tree = parse_and_validate(
        "SELECT AVG(age) OVER (PARTITION BY city) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].has_partition is True


def test_find_window_aggregates_detects_order_by():
    tree = parse_and_validate("SELECT AVG(age) OVER (ORDER BY age) AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert windows[0].has_order is True


def test_find_window_aggregates_rejects_non_column_argument():
    tree = parse_and_validate("SELECT AVG(age + 1) OVER () AS x FROM __THIS__")
    with pytest.raises(ValueError, match="single plain column"):
        find_window_aggregates(tree)


def test_find_window_aggregates_empty_when_no_windows():
    tree = parse_and_validate("SELECT age FROM __THIS__")
    assert find_window_aggregates(tree) == []


def test_find_window_aggregates_multiple_distinct():
    tree = parse_and_validate(
        "SELECT AVG(age) OVER () AS a, SUM(score) OVER () AS b FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert len(windows) == 2
    assert {(w.fn, w.col) for w in windows} == {("AVG", "age"), ("SUM", "score")}


def test_find_window_aggregates_partition_cols_empty_for_bare_over():
    tree = parse_and_validate("SELECT AVG(age) OVER () AS x FROM __THIS__")
    windows = find_window_aggregates(tree)
    assert windows[0].partition_cols == ()


def test_find_window_aggregates_single_partition_col():
    tree = parse_and_validate(
        "SELECT AVG(target) OVER (PARTITION BY city) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].partition_cols == ("city",)
    assert windows[0].has_partition is True


def test_find_window_aggregates_composite_partition_cols():
    tree = parse_and_validate(
        "SELECT AVG(target) OVER (PARTITION BY city, region) AS x FROM __THIS__"
    )
    windows = find_window_aggregates(tree)
    assert windows[0].partition_cols == ("city", "region")


def test_find_window_aggregates_rejects_non_column_partition():
    tree = parse_and_validate(
        "SELECT AVG(target) OVER (PARTITION BY city || 'x') AS y FROM __THIS__"
    )
    with pytest.raises(ValueError, match="PARTITION BY"):
        find_window_aggregates(tree)
