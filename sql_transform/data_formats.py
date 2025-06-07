"""Data format conversion utilities for supporting multiple input/output formats."""

from typing import TYPE_CHECKING, Any, Union

import pyarrow as pa

if TYPE_CHECKING:
    import datafusion  # type: ignore[import-untyped]
    import pandas as pd  # type: ignore[import-untyped]
    import polars as pl  # type: ignore[import-untyped]

# Type for supported data formats
DataInput = Union[
    pa.Table,
    dict[str, list[Any]],
    "pd.DataFrame",  # pandas
    "pl.DataFrame",  # polars
    "datafusion.DataFrame",  # datafusion
]

DataOutput = DataInput


def to_arrow_table(data: DataInput) -> pa.Table:
    """Convert various data formats to PyArrow Table for processing."""
    if isinstance(data, pa.Table):
        return data

    if isinstance(data, dict):
        # Dict of lists format - convert to proper column format for PyArrow
        return pa.table(data)

    # Check for pandas DataFrame
    if hasattr(data, "to_arrow") and hasattr(data, "columns"):
        try:
            return data.to_arrow()
        except (AttributeError, TypeError):
            # Not a pandas DataFrame or conversion failed
            pass

    # Check for polars DataFrame
    if hasattr(data, "to_arrow") and hasattr(data, "schema"):
        try:
            return data.to_arrow()
        except (AttributeError, TypeError):
            # Not a polars DataFrame or conversion failed
            pass

    # Check for datafusion DataFrame
    if hasattr(data, "to_arrow_table"):
        try:
            return data.to_arrow_table()
        except (AttributeError, TypeError):
            # Not a datafusion DataFrame or conversion failed
            pass

    # Try pandas conversion as fallback
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data)
    except ImportError:
        pass

    # Try polars conversion as fallback
    try:
        import polars as pl

        if isinstance(pl.DataFrame, type) and isinstance(data, pl.DataFrame):
            return data.to_arrow()
    except ImportError:
        pass

    raise ValueError(f"Unsupported data format: {type(data)}")


def from_arrow_table(table: pa.Table, output_format: str) -> DataOutput:
    """Convert PyArrow Table to the specified output format."""
    if output_format == "arrow" or output_format == "pyarrow":
        return table

    if output_format == "dict":
        return table.to_pydict()

    if output_format == "pandas":
        try:
            import pandas

            return table.to_pandas()
        except ImportError as e:
            raise ValueError(
                "pandas not available. Install with: pip install pandas"
            ) from e

    if output_format == "polars":
        try:
            import polars as pl

            result = pl.from_arrow(table)
            assert hasattr(result, "schema"), "Expected polars DataFrame"
            return result
        except ImportError as e:
            raise ValueError(
                "polars not available. Install with: pip install polars"
            ) from e

    if output_format == "datafusion":
        try:
            import datafusion

            ctx = datafusion.SessionContext()
            return ctx.from_arrow(table)
        except ImportError as e:
            raise ValueError("datafusion not available") from e

    raise ValueError(f"Unsupported output format: {output_format}")


def detect_input_format(data: DataInput) -> str:
    """Detect the input data format."""
    if isinstance(data, pa.Table):
        return "arrow"

    if isinstance(data, dict):
        return "dict"

    # Check pandas
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return "pandas"
    except ImportError:
        pass

    # Check polars
    try:
        import polars as pl

        if isinstance(pl.DataFrame, type) and isinstance(data, pl.DataFrame):
            return "polars"
    except ImportError:
        pass

    # Check datafusion
    if hasattr(data, "to_arrow_table"):
        return "datafusion"

    return "unknown"


def auto_convert_output(
    table: pa.Table, input_format: str, requested_format: str | None = None
) -> DataOutput:
    """Automatically convert output to match input format or requested format."""
    target_format = requested_format or input_format

    # Default to arrow if format is unknown
    if target_format == "unknown":
        target_format = "arrow"

    return from_arrow_table(table, target_format)
