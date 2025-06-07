"""Tests for data format conversion functionality."""

import pyarrow as pa
import pytest

from sql_transform.data_formats import (
    auto_convert_output,
    detect_input_format,
    from_arrow_table,
    to_arrow_table,
)


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
