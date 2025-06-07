"""Data format conversion utilities for supporting multiple input/output formats."""

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import datafusion  # type: ignore[import-untyped]
    import pandas as pd  # type: ignore[import-untyped]
    import polars as pl  # type: ignore[import-untyped]

# Type for supported data formats
type DataInput = (
    "pa.Table | dict[str, list[Any]] | pd.DataFrame | pl.DataFrame | "
    "datafusion.DataFrame"
)
type DataOutput = DataInput


def to_arrow_table(data: DataInput) -> pa.Table:  # noqa: C901
    """Convert various data formats to PyArrow Table for processing."""
    if isinstance(data, pa.Table):
        return data

    if isinstance(data, dict):
        # Dict of lists format - convert to proper column format for PyArrow
        return pa.table(data)

    # Check for pandas DataFrame with proper type checking
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return pa.Table.from_pandas(data)
    except ImportError:
        pass

    # Check for polars DataFrame with proper type checking
    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return data.to_arrow()
    except ImportError:
        pass

    # Check for datafusion DataFrame with proper type checking
    try:
        import datafusion

        if isinstance(data, datafusion.DataFrame):
            return data.to_arrow_table()
    except ImportError:
        pass

    # Check for additional PyArrow types
    try:
        if isinstance(data, pa.RecordBatch):
            return pa.Table.from_batches([data])
    except (TypeError, ValueError, pa.ArrowInvalid):
        # RecordBatch conversion failed
        pass

    raise ValueError(f"Unsupported data format: {type(data)}")


def from_arrow_table(table: pa.Table, output_format: str) -> DataOutput:
    """Convert PyArrow Table to the specified output format."""
    if output_format == "arrow" or output_format == "pyarrow":
        return table

    if output_format == "dict":
        return table.to_pydict()

    if output_format == "pandas":
        return table.to_pandas()

    if output_format == "polars":
        try:
            import polars as pl

            polars_df = pl.from_arrow(table)
            assert hasattr(polars_df, "schema"), "Expected polars DataFrame"
            return polars_df
        except ImportError as e:
            raise ValueError(
                "polars not available. Install with: pip install polars"
            ) from e

    if output_format == "datafusion":
        try:
            import datafusion

            ctx = datafusion.SessionContext()
            datafusion_df = ctx.from_arrow(table)
            assert hasattr(datafusion_df, "to_arrow_table"), (
                "Expected datafusion DataFrame"
            )
            return datafusion_df
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

        if isinstance(data, pl.DataFrame):
            return "polars"
    except ImportError:
        pass

    # Check datafusion
    try:
        import datafusion

        if isinstance(data, datafusion.DataFrame):
            return "datafusion"
    except ImportError:
        pass

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
