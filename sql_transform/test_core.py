"""Core tests for SQL transformation functionality."""

import re

import pyarrow as pa
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
                        "avg_feature1": AggregationRef(0, "feature1"),
                    },
                    aggregations={
                        AggregationRef(0, "feature1"): AggregateFunction(
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
                        "avg_feature1": AggregationRef(0, "feature1"),
                    },
                    aggregations={
                        AggregationRef(0, "feature1"): AggregateFunction(
                            "avg",
                            [ColumnRef("feature1")],
                            over=WindowSpecification(partition_by=[ColumnRef("feature2")]),
                        )
                    },
                ),
            ),
        ],
    )
    def test_parse_basic(self, sql, expected):
        """Test basic SQL parsing."""
        assert parse(sql) == expected

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
                "SELECT sklearn.minmax_scale(feature1, 0, 1) as scaled_feature1 FROM data",
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
        query = parse(sql)

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


class TestTransformer:
    """Test SQLTransformer functionality."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return pa.table(
            {
                "feature1": [5.1, 4.9, 4.7, 4.6, 5.0],
                "feature2": [3.5, 3.0, 3.2, 3.1, 3.6],
                "feature3": [1.4, 1.4, 1.3, 1.5, 1.4],
                "feature4": [0.2, 0.2, 0.2, 0.2, 0.2],
                "class": ["A", "B", "A", "A", "B"],
            }
        )

    def _normalize_sql(self, query: str) -> str:
        """Normalize SQL for testing."""
        return re.sub(r"\n\s+", " ", query)

    @pytest.mark.parametrize(
        "sql,expected_columns",
        [
            (
                "SELECT feature1 as feature1 FROM data",
                {"feature1": [5.1, 4.9, 4.7, 4.6, 5.0]},
            ),
            (
                "SELECT feature1 - avg(feature1) as x FROM data",
                {"x": [0.24, 0.04, -0.16, -0.26, 0.14]},
            ),
            (
                """
                SELECT
                    feature1 as feature1,
                    avg(feature1) as avg_feature1,
                    avg(feature2) as avg_feature2,
                    avg(feature1) over (partition by class) as avg_feature1_by_class,
                    avg(feature2) over (partition by class) as avg_feature2_by_class
                FROM data
                """,
                {
                    "feature1": [5.1, 4.9, 4.7, 4.6, 5.0],
                    "avg_feature1": [4.86, 4.86, 4.86, 4.86, 4.86],
                    "avg_feature2": [3.28, 3.28, 3.28, 3.28, 3.28],
                    "avg_feature1_by_class": [4.8, 4.95, 4.8, 4.8, 4.95],
                    "avg_feature2_by_class": [
                        3.266666666666667,
                        3.3,
                        3.266666666666667,
                        3.266666666666667,
                        3.3,
                    ],
                },
            ),
        ],
    )
    def test_transformer_basic(self, sample_data, sql, expected_columns):
        """Test basic transformer functionality."""
        context = SqlTransformContext()
        transformer = context.create_transformer(self._normalize_sql(sql))
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        # Sort both tables by the first column to ensure consistent ordering
        first_col = list(expected_columns.keys())[0]
        result_sorted = result.sort_by(first_col)
        expected_table = pa.table(expected_columns)
        expected_sorted = expected_table.sort_by(first_col)

        for column in expected_columns.keys():
            result_values = result_sorted[column].to_pylist()
            expected_values = expected_sorted[column].to_pylist()
            assert result_values == pytest.approx(expected_values), (
                f"Column {column} values do not match"
            )
        assert set(result.column_names) == set(expected_columns.keys())


class TestContext:
    """Test SqlTransformContext functionality."""

    def test_context_creation(self):
        """Test context creation and basic functionality."""
        context = SqlTransformContext()
        assert len(context.list_transforms()) == 0

    def test_register_transform(self):
        """Test registering transforms."""
        context = SqlTransformContext()

        class DummyTransform:
            def fit(self, data, **kwargs):
                return self

            def transform(self, data, **kwargs):
                return data

        transform = DummyTransform()
        context.register_transform("dummy", transform)

        assert "dummy" in context.list_transforms()
        assert context.get_transform("dummy") is transform

    def test_create_transformer(self):
        """Test creating transformer from context."""
        context = SqlTransformContext()
        transformer = context.create_transformer("SELECT feature1 as f1 FROM data")

        # Should be able to create transformer without errors
        assert transformer is not None
        assert hasattr(transformer, "fit")
        assert hasattr(transformer, "transform")