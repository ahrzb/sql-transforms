"""Tests for SQLTransformer functionality."""

import re

import pyarrow as pa
import pytest

from sql_transform.context import SqlTransformContext


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
        result = transformer.transform(sample_data, output_format="arrow")

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

    @pytest.mark.parametrize(
        "sql,expected_columns",
        [
            (
                """
                SELECT
                    feature1 - avg(feature1) as centered_feature1,
                    (feature1 - avg(feature1)) / stddev(feature1) as 
                        standardized_feature1
                FROM data
                """,
                {
                    "centered_feature1": [0.24, 0.04, -0.16, -0.26, 0.14],
                    "standardized_feature1": [1.1574, 0.1929, -0.7716, -1.2538, 0.6751],
                },
            ),
            (
                """
                SELECT
                    avg(feature1) as global_avg,
                    feature1 - avg(feature1) as centered,
                    stddev(feature1 - avg(feature1)) as stddev_centered
                FROM data
                """,
                {
                    "global_avg": [4.86, 4.86, 4.86, 4.86, 4.86],
                    "centered": [0.24, 0.04, -0.16, -0.26, 0.14],
                    "stddev_centered": [0.2074, 0.2074, 0.2074, 0.2074, 0.2074],
                },
            ),
        ],
    )
    def test_nested_aggregations(self, sample_data, sql, expected_columns):
        """Test nested aggregations that depend on other aggregations."""
        context = SqlTransformContext()
        transformer = context.create_transformer(self._normalize_sql(sql))
        transformer.fit(sample_data)
        result = transformer.transform(sample_data, output_format="arrow")

        # Sort both tables by the first column to ensure consistent ordering
        first_col = list(expected_columns.keys())[0]
        result_sorted = result.sort_by(first_col)
        expected_table = pa.table(expected_columns)
        expected_sorted = expected_table.sort_by(first_col)

        for column in expected_columns.keys():
            result_values = result_sorted[column].to_pylist()
            expected_values = expected_sorted[column].to_pylist()
            assert result_values == pytest.approx(expected_values, abs=1e-3), (
                f"Column {column} values do not match"
            )
        assert set(result.column_names) == set(expected_columns.keys())


class TestMultiFormatTransformer:
    """Test SqlTransformer with multiple data formats."""

    @pytest.fixture
    def sample_data_dict(self):
        """Sample data as dict."""
        return {
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [10.0, 20.0, 30.0, 40.0],
        }

    @pytest.fixture
    def sample_data_arrow(self, sample_data_dict):
        """Sample data as arrow table."""
        return pa.table(sample_data_dict)

    def test_fit_transform_dict_input(self, sample_data_dict):
        """Test fit/transform with dict input."""
        context = SqlTransformContext()
        transformer = context.create_transformer(
            "SELECT feature1, feature1 + feature2 as sum_features FROM data"
        )

        transformer.fit(sample_data_dict)
        result = transformer.transform(sample_data_dict)

        # Result should auto-convert back to dict
        assert isinstance(result, dict)
        assert "feature1" in result
        assert "sum_features" in result
        assert result["sum_features"] == [11.0, 22.0, 33.0, 44.0]

    def test_fit_transform_arrow_input(self, sample_data_arrow):
        """Test fit/transform with arrow input."""
        context = SqlTransformContext()
        transformer = context.create_transformer(
            "SELECT feature1, feature1 * 2 as doubled FROM data"
        )

        transformer.fit(sample_data_arrow)
        result = transformer.transform(sample_data_arrow)

        # Result should auto-convert back to arrow
        assert isinstance(result, pa.Table)
        assert "feature1" in result.column_names
        assert "doubled" in result.column_names

    def test_explicit_output_format(self, sample_data_dict):
        """Test explicit output format specification."""
        context = SqlTransformContext()
        transformer = context.create_transformer("SELECT feature1 FROM data")

        transformer.fit(sample_data_dict)

        # Request arrow output explicitly
        result = transformer.transform(sample_data_dict, output_format="arrow")
        assert isinstance(result, pa.Table)

        # Request dict output explicitly
        result = transformer.transform(sample_data_dict, output_format="dict")
        assert isinstance(result, dict)

    def test_pandas_end_to_end(self, sample_data_dict):
        """Test end-to-end functionality with pandas DataFrames."""
        try:
            import pandas as pd  # type: ignore

            # Convert to pandas DataFrame
            df = pd.DataFrame(sample_data_dict)

            context = SqlTransformContext()
            transformer = context.create_transformer(
                "SELECT feature1, feature2, feature1 + feature2 as sum_features "
                "FROM data"
            )

            transformer.fit(df)
            result = transformer.transform(df)

            # Result should auto-convert back to pandas
            assert isinstance(result, pd.DataFrame)
            assert "feature1" in result.columns
            assert "feature2" in result.columns
            assert "sum_features" in result.columns
            assert len(result) == len(sample_data_dict["feature1"])
        except ImportError:
            pytest.skip("pandas not available")

    def test_polars_end_to_end(self, sample_data_dict):
        """Test end-to-end functionality with polars DataFrames."""
        try:
            import polars as pl  # type: ignore

            # Convert to polars DataFrame
            df = pl.DataFrame(sample_data_dict)

            context = SqlTransformContext()
            transformer = context.create_transformer(
                "SELECT feature1, feature2, feature1 * feature2 as product FROM data"
            )

            transformer.fit(df)
            result = transformer.transform(df)

            # Result should auto-convert back to polars
            assert isinstance(result, pl.DataFrame)
            assert "feature1" in result.columns
            assert "feature2" in result.columns
            assert "product" in result.columns
            assert len(result) == len(sample_data_dict["feature1"])
        except ImportError:
            pytest.skip("polars not available")
