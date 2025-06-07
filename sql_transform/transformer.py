from typing import Optional, Self

import datafusion
import pyarrow as pa

import sql_transform.parser
from sql_transform.context import SqlTransformContext
from sql_transform.data_formats import (
    DataInput,
    DataOutput,
    auto_convert_output,
    detect_input_format,
    to_arrow_table,
)
from sql_transform.parser import AggregateFunction, Expression


class SQLTransformer:
    def __init__(self, sql: str, context: SqlTransformContext):
        self.context = context
        self.query = sql_transform.parser.parse(sql=sql, context=self.context)
        self.aggs: list[tuple[AggregateFunction, datafusion.DataFrame]] = []
        self._input_format: Optional[str] = None

    def fit(self, data: DataInput) -> Self:
        # Store input format for auto-converting output
        self._input_format = detect_input_format(data)
        
        # Convert to arrow table for processing
        arrow_table = to_arrow_table(data)
        df = self.context.datafusion_ctx.from_arrow(arrow_table)

        self.partition_keys: dict[Expression, str] = {}
        for agg in self.query.aggregations.values():
            for partition in agg.over.partition_by:
                if partition in self.partition_keys:
                    continue
                self.partition_keys[partition] = (
                    f"${partition.hint_name()}_partition{len(self.partition_keys)}"
                )

        for agg_ref, agg in self.query.aggregations.items():
            agg_df = df.aggregate(
                [
                    p.to_datafusion_expr().alias(f"_{self.partition_keys[p]}")
                    for p in agg.over.partition_by
                ],
                agg.to_datafusion_expr().alias(agg_ref.name),
            )
            self.aggs.append((agg, agg_df))

        return self

    def transform(
        self, data: DataInput, output_format: Optional[str] = None
    ) -> DataOutput:
        # Convert input to arrow table for processing
        arrow_table = to_arrow_table(data)
        df = self.context.datafusion_ctx.from_arrow(arrow_table)

        df = df.with_columns(
            [
                expr.to_datafusion_expr().alias(name)
                for expr, name in self.partition_keys.items()
            ]
        )

        for agg, aggdf in self.aggs:
            df = df.join(
                aggdf,
                left_on=[self.partition_keys[p] for p in agg.over.partition_by],
                right_on=[f"_{self.partition_keys[p]}" for p in agg.over.partition_by],
                how="left",
            ).drop(*[f"_{self.partition_keys[p]}" for p in agg.over.partition_by])

        # Process data in datafusion and get result as arrow table
        result_table = df.select(
            *[
                col.to_datafusion_expr().alias(name)
                for name, col in self.query.columns.items()
            ]
        ).to_arrow_table()
        
        # Convert output to requested format (or auto-detect from input)
        input_format = detect_input_format(data)
        return auto_convert_output(result_table, input_format, output_format)
