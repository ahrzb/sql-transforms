from collections import defaultdict
from typing import Any, Self

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
from sql_transform.parser import (
    AggregateFunction,
    AggregationRef,
    ApplyFunction,
    Expression,
    TransformFunction,
)


class SQLTransformer:
    def __init__(self, sql: str, context: SqlTransformContext):
        self.context = context
        self.query = sql_transform.parser.parse(sql=sql, context=self.context)
        self.aggs: list[tuple[AggregateFunction, datafusion.DataFrame]] = []
        self.fitted_transforms: dict[
            str, Any
        ] = {}  # Fitted sklearn transforms by agg_ref name
        self.windowed_aggs: list[tuple[AggregateFunction, str]] = []  # Window functions
        self._input_format: str | None = None
        self._execution_layers: list[list[AggregationRef]] = []  # BFS dependency layers

    def _build_dependency_graph(self) -> dict[AggregationRef, set[AggregationRef]]:
        """Build dependency graph for aggregations and transforms."""
        dependencies: dict[AggregationRef, set[AggregationRef]] = defaultdict(set)

        def find_agg_dependencies(expr: Expression) -> set[AggregationRef]:
            """Recursively find all AggregationRef dependencies in an expression."""
            deps = set()

            if isinstance(expr, AggregationRef):
                deps.add(expr)
            elif isinstance(expr, ApplyFunction):
                for arg in expr.args:
                    deps.update(find_agg_dependencies(arg))
            # Add other expression types as needed

            return deps

        # Find dependencies in aggregation arguments
        for agg_ref, agg_or_transform in self.query.aggregations.items():
            for arg in agg_or_transform.args:
                deps = find_agg_dependencies(arg)
                dependencies[agg_ref].update(deps)

            # Also check partition by expressions
            for partition_expr in agg_or_transform.over.partition_by:
                deps = find_agg_dependencies(partition_expr)
                dependencies[agg_ref].update(deps)

        return dependencies

    def _bfs_dependency_layers(
        self, dependencies: dict[AggregationRef, set[AggregationRef]]
    ) -> list[list[AggregationRef]]:
        """Use BFS to group aggregations into dependency layers.

        Enables parallel processing within each layer.
        """
        all_nodes = set(self.query.aggregations.keys())
        processed: set[AggregationRef] = set()
        layers = []

        while processed != all_nodes:
            # Find nodes with no unprocessed dependencies
            current_layer = []
            for node in all_nodes - processed:
                node_deps = dependencies.get(node, set())
                # Check if all dependencies of this node are already processed
                unprocessed_deps = node_deps - processed
                if not unprocessed_deps:
                    current_layer.append(node)

            if not current_layer:
                # Circular dependency detected
                remaining = all_nodes - processed
                raise ValueError(
                    f"Circular dependency detected in aggregations: {remaining}"
                )

            layers.append(current_layer)
            processed.update(current_layer)

        return layers

    def _resolve_and_fit_transform(
        self, transform_func: TransformFunction, agg_ref_name: str, data: pa.Table
    ):
        """Resolve and fit a transform function."""
        try:
            # Get the transform spec from context
            spec = self.context.get_transform_spec(transform_func.operation)
            if spec is None:
                raise ValueError(f"Unknown transform: {transform_func.operation}")

            # Get the column to transform (first argument)
            column_expr = transform_func.args[0]
            from sql_transform.parser import ColumnRef

            if isinstance(column_expr, ColumnRef):
                column_name = column_expr.name
            else:
                # Handle more complex expressions later
                raise NotImplementedError(
                    "Complex expressions in transforms not yet supported"
                )

            # Create fitter and fit the transform
            fitter = spec.create_fitter(transform_func.args)
            fitted_transform = fitter.fit(
                data, column_name, {}
            )  # Empty context for now

            # Store the fitted transformer
            self.fitted_transforms[agg_ref_name] = fitted_transform

        except Exception as e:
            raise ValueError(
                f"Failed to resolve transform {transform_func.operation}: {e}"
            ) from e

    def fit(self, data: DataInput) -> Self:
        # Store input format for auto-converting output
        self._input_format = detect_input_format(data)

        # Convert to arrow table for processing
        arrow_table = to_arrow_table(data)
        df = self.context.datafusion_ctx.from_arrow(arrow_table)

        # Build dependency graph and determine execution layers
        dependencies = self._build_dependency_graph()
        self._execution_layers = self._bfs_dependency_layers(dependencies)

        self.partition_keys: dict[Expression, str] = {}
        for agg in self.query.aggregations.values():
            for partition in agg.over.partition_by:
                if partition in self.partition_keys:
                    continue
                self.partition_keys[partition] = (
                    f"${partition.hint_name()}_partition{len(self.partition_keys)}"
                )

        # Process aggregations in dependency layers
        self.aggs.clear()
        agg_results: dict[AggregationRef, datafusion.DataFrame] = {}

        # We need to augment the dataframe with computed aggregations as we go
        working_df = df

        for layer in self._execution_layers:
            # All aggregations in this layer can be processed in parallel
            # since they don't depend on each other
            layer_results = []

            for agg_ref in layer:
                agg_or_transform = self.query.aggregations[agg_ref]

                if isinstance(agg_or_transform, AggregateFunction):
                    if agg_or_transform.over.partition_by:
                        # Windowed aggregation - compute now with partitioning
                        agg_df = working_df.aggregate(
                            [
                                p.to_datafusion_expr().alias(
                                    f"_{self.partition_keys[p]}"
                                )
                                for p in agg_or_transform.over.partition_by
                            ],
                            agg_or_transform.to_datafusion_expr(self.context).alias(
                                agg_ref.name
                            ),
                        )
                        self.aggs.append((agg_or_transform, agg_df))
                        agg_results[agg_ref] = agg_df
                    else:
                        # Regular aggregation - compute using current working_df
                        agg_df = working_df.aggregate(
                            [],  # No grouping for global aggregations
                            agg_or_transform.to_datafusion_expr(self.context).alias(
                                agg_ref.name
                            ),
                        )
                        self.aggs.append((agg_or_transform, agg_df))
                        agg_results[agg_ref] = agg_df
                        layer_results.append((agg_ref.name, agg_df))

                elif isinstance(agg_or_transform, TransformFunction):
                    # Custom transform - resolve and fit it now
                    self._resolve_and_fit_transform(
                        agg_or_transform, agg_ref.name, arrow_table
                    )

            # Add the computed aggregations from this layer to the working dataframe
            # so they're available for the next layer
            for agg_name, agg_df in layer_results:
                agg_value = agg_df.collect()[0]
                working_df = working_df.with_column(
                    agg_name, datafusion.literal(agg_value.column(agg_name)[0].as_py())
                )

        return self

    def transform(
        self, data: DataInput, output_format: str | None = None
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

        # Add aggregation results as columns to the dataframe
        for agg, aggdf in self.aggs:
            if agg.over.partition_by:
                # Windowed aggregation - join on partition keys
                df = df.join(
                    aggdf,
                    left_on=[self.partition_keys[p] for p in agg.over.partition_by],
                    right_on=[
                        f"_{self.partition_keys[p]}" for p in agg.over.partition_by
                    ],
                    how="left",
                ).drop(*[f"_{self.partition_keys[p]}" for p in agg.over.partition_by])
            else:
                # Global aggregation - cross join (add same value to all rows)
                agg_value = aggdf.collect()[0]  # Get the single aggregated value
                for col_name in aggdf.schema().names:
                    df = df.with_column(
                        col_name,
                        datafusion.literal(agg_value.column(col_name)[0].as_py()),
                    )

        # Apply fitted transforms
        for transform_name, fitted_transform in self.fitted_transforms.items():
            # Find the original transform function to get the column name
            transform_func = None
            for agg_ref, agg_or_transform in self.query.aggregations.items():
                if agg_ref.name == transform_name and isinstance(
                    agg_or_transform, TransformFunction
                ):
                    transform_func = agg_or_transform
                    break

            if transform_func:
                # Get the column to transform
                column_expr = transform_func.args[0]
                from sql_transform.parser import ColumnRef

                if isinstance(column_expr, ColumnRef):
                    column_name = column_expr.name

                    # Apply the fitted transform
                    current_table = df.to_arrow_table()
                    transformed_data = fitted_transform.transform(
                        current_table, column_name
                    )

                    # Add the transformed column by rebuilding the Arrow table
                    new_table = current_table.append_column(
                        transform_name, transformed_data
                    )
                    df = self.context.datafusion_ctx.from_arrow(new_table)

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
