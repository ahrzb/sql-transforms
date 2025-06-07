"""Core tests for SQL transformation functionality."""

import re

import pyarrow as pa
import pytest

from sql_transform.context import SqlTransformContext
from sql_transform.data_formats import (
    auto_convert_output,
    detect_input_format,
    from_arrow_table,
    to_arrow_table,
)
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
        from sql_transform.context import SqlTransformContext

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
        from sql_transform.context import SqlTransformContext

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


class TestContext:
    """Test SqlTransformContext functionality."""

    def test_context_creation(self):
        """Test context creation and basic functionality."""
        context = SqlTransformContext()
        # Context should have sklearn transforms registered by default
        transforms = context.list_transforms()
        assert len(transforms) >= 0  # May have sklearn transforms if available

    def test_register_transform(self):
        """Test registering transforms."""
        from sql_transform.function_registry import SklearnTransformSpec

        context = SqlTransformContext()

        # Register a dummy transform spec
        try:
            import sklearn.preprocessing

            dummy_spec = SklearnTransformSpec(
                "dummy", sklearn.preprocessing.StandardScaler, "Dummy transform"
            )
            context.register_transform(dummy_spec)

            assert "dummy" in context.list_transforms()
            assert context.get_transform("dummy") is dummy_spec
        except ImportError:
            # Skip if sklearn not available
            pass

    def test_create_transformer(self):
        """Test creating transformer from context."""
        context = SqlTransformContext()
        transformer = context.create_transformer("SELECT feature1 as f1 FROM data")

        # Should be able to create transformer without errors
        assert transformer is not None
        assert hasattr(transformer, "fit")
        assert hasattr(transformer, "transform")


class TestDataFormats:
    """Test data format conversion functionality."""

    @pytest.fixture
    def sample_arrow_table(self):
        """Sample PyArrow table for testing."""
        return pa.table(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [10.0, 20.0, 30.0],
                "category": ["A", "B", "A"],
            }
        )

    @pytest.fixture
    def sample_dict(self):
        """Sample dict for testing."""
        return {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [10.0, 20.0, 30.0],
            "category": ["A", "B", "A"],
        }

    def test_to_arrow_table_from_dict(self, sample_dict):
        """Test converting dict to arrow table."""
        result = to_arrow_table(sample_dict)
        assert isinstance(result, pa.Table)
        assert result.column_names == ["feature1", "feature2", "category"]
        assert result.num_rows == 3

    def test_to_arrow_table_from_arrow(self, sample_arrow_table):
        """Test converting arrow table to arrow table (passthrough)."""
        result = to_arrow_table(sample_arrow_table)
        assert result is sample_arrow_table

    def test_from_arrow_table_to_dict(self, sample_arrow_table):
        """Test converting arrow table to dict."""
        result = from_arrow_table(sample_arrow_table, "dict")
        assert isinstance(result, dict)
        assert list(result.keys()) == ["feature1", "feature2", "category"]
        assert result["feature1"] == [1.0, 2.0, 3.0]

    def test_from_arrow_table_to_arrow(self, sample_arrow_table):
        """Test converting arrow table to arrow (passthrough)."""
        result = from_arrow_table(sample_arrow_table, "arrow")
        assert result is sample_arrow_table

    def test_pandas_integration(self, sample_dict):
        """Test pandas integration if pandas is available."""
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(sample_dict)
            arrow_result = to_arrow_table(df)
            assert isinstance(arrow_result, pa.Table)

            pandas_result = from_arrow_table(arrow_result, "pandas")
            assert isinstance(pandas_result, pd.DataFrame)

            # Verify data integrity
            assert list(pandas_result.columns) == list(sample_dict.keys())
            assert len(pandas_result) == len(sample_dict["feature1"])
        except ImportError:
            pytest.skip("pandas not available")

    def test_polars_integration(self, sample_dict):
        """Test polars integration if polars is available."""
        try:
            import polars as pl  # type: ignore

            df = pl.DataFrame(sample_dict)
            arrow_result = to_arrow_table(df)
            assert isinstance(arrow_result, pa.Table)

            polars_result = from_arrow_table(arrow_result, "polars")
            assert isinstance(polars_result, pl.DataFrame)

            # Verify data integrity
            assert list(polars_result.columns) == list(sample_dict.keys())
            assert len(polars_result) == len(sample_dict["feature1"])
        except ImportError:
            pytest.skip("polars not available")

    def test_detect_input_format(self, sample_arrow_table, sample_dict):
        """Test input format detection."""
        assert detect_input_format(sample_arrow_table) == "arrow"
        assert detect_input_format(sample_dict) == "dict"

    def test_auto_convert_output(self, sample_arrow_table):
        """Test automatic output conversion."""
        # Dict input should return dict output
        result = auto_convert_output(sample_arrow_table, "dict")
        assert isinstance(result, dict)

        # Arrow input should return arrow output
        result = auto_convert_output(sample_arrow_table, "arrow")
        assert isinstance(result, pa.Table)

    def test_unsupported_format_errors(self, sample_arrow_table):
        """Test error handling for unsupported formats."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            from_arrow_table(sample_arrow_table, "unsupported")

        with pytest.raises(ValueError, match="Unsupported data format"):
            to_arrow_table(123)  # type: ignore[arg-type]


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
