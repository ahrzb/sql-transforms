"""Tests for SQL parsing functionality."""

import pytest

from sql_transform.context import SqlTransformContext
from sql_transform.parser import (
    AggregateFunction,
    AggregationRef,
    ColumnRef,
    Query,
    WindowSpecification,
    parse,
)


class TestParser:
    """Test SQL parsing functionality."""

    @pytest.mark.parametrize(
        "sql,expected",
        [
            (
                """
                SELECT 
                    feature1 as feature1,
                    feature2 as feature2
                FROM data
                """,
                Query(
                    columns={
                        "feature1": ColumnRef("feature1"),
                        "feature2": ColumnRef("feature2"),
                    },
                ),
            ),
            (
                """
                SELECT
                    AVG(feature1) as avg_feature1
                FROM data
                """,
                Query(
                    columns={
                        "avg_feature1": AggregationRef(0, "avg_feature1"),
                    },
                    aggregations={
                        AggregationRef(0, "avg_feature1"): AggregateFunction(
                            "avg", [ColumnRef("feature1")], over=WindowSpecification()
                        )
                    },
                ),
            ),
            (
                """
                SELECT
                    AVG(feature1) over (partition by feature2) as avg_feature1
                FROM data
                """,
                Query(
                    columns={
                        "avg_feature1": AggregationRef(0, "avg_feature1"),
                    },
                    aggregations={
                        AggregationRef(0, "avg_feature1"): AggregateFunction(
                            "avg",
                            [ColumnRef("feature1")],
                            over=WindowSpecification(
                                partition_by=[ColumnRef("feature2")]
                            ),
                        )
                    },
                ),
            ),
        ],
    )
    def test_parse_basic(self, sql, expected):
        """Test basic SQL parsing."""
        context = SqlTransformContext()
        assert parse(sql, context) == expected

    @pytest.mark.parametrize(
        "sql,expected_operation,expected_args_count,alias",
        [
            # Basic sklearn transforms
            (
                "SELECT sklearn.standardize(feature1) as std_feature1 FROM data",
                "sklearn.standardize",
                1,
                "std_feature1",
            ),
            (
                "SELECT sklearn.robust_scale(score) as robust_score FROM data",
                "sklearn.robust_scale",
                1,
                "robust_score",
            ),
            # Sklearn transforms with parameters
            (
                "SELECT sklearn.minmax_scale(feature1, 0, 1) as scaled_feature1 "
                "FROM data",
                "sklearn.minmax_scale",
                3,
                "scaled_feature1",
            ),
            # Regular functions (non-sklearn)
            (
                "SELECT unknown_func(feature1) as result FROM data",
                "unknown_func",
                1,
                "result",
            ),
        ],
    )
    def test_parse_custom_functions(
        self, sql, expected_operation, expected_args_count, alias
    ):
        """Test that custom functions are parsed correctly as aggregations."""
        context = SqlTransformContext()
        query = parse(sql, context)

        assert len(query.columns) == 1
        assert alias in query.columns

        expr = query.columns[alias]
        assert isinstance(expr, AggregationRef)
        assert expr.hint.startswith(f"{expected_operation}_")

        # Check the aggregation was registered
        assert expr in query.aggregations
        agg = query.aggregations[expr]
        assert agg.operation == expected_operation
        assert len(agg.args) == expected_args_count
