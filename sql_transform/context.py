"""Context for registering transforms and managing SQL transformation environment."""

from typing import TYPE_CHECKING

import datafusion

from sql_transform.function_registry import (
    AggregationRegistry,
    FunctionResolver,
    create_sklearn_registry,
)

if TYPE_CHECKING:
    from sql_transform.transformer import SQLTransformer


class SqlTransformContext:
    """Context for managing SQL transformations."""

    def __init__(self):
        # Initialize function registries
        self.aggregation_registry = AggregationRegistry()
        self.transform_registry = create_sklearn_registry()
        self.function_resolver = FunctionResolver(
            self.aggregation_registry, self.transform_registry
        )

        self._datafusion_ctx: datafusion.SessionContext | None = None

    @property
    def datafusion_ctx(self) -> datafusion.SessionContext:
        """Get or create the DataFusion session context."""
        if self._datafusion_ctx is None:
            self._datafusion_ctx = datafusion.SessionContext()
        return self._datafusion_ctx

    def register_aggregation(self, aggregation_spec):
        """Register a custom aggregation function."""
        self.aggregation_registry.register(aggregation_spec)

    def register_transform(self, transform_spec):
        """Register a custom transform function."""
        self.transform_registry.register(transform_spec)

    def get_aggregation(self, name: str):
        """Get a registered aggregation."""
        return self.aggregation_registry.get(name)

    def get_transform(self, name: str):
        """Get a registered transform."""
        return self.transform_registry.get(name)

    def list_functions(self) -> dict[str, tuple[str, str]]:
        """List all registered functions with their type and description."""
        return self.function_resolver.list_all_functions()

    def list_aggregations(self) -> dict[str, str]:
        """List all registered aggregations."""
        return self.aggregation_registry.list_aggregations()

    def list_transforms(self) -> dict[str, str]:
        """List all registered transforms."""
        return self.transform_registry.list_transforms()

    def resolve_function(self, function_name: str, args: list, over=None):
        """
        Resolve function to either AggregateFunction or TransformFunction.

        Args:
            function_name: Function name (e.g., 'avg', 'sklearn.standardize')
            args: List of arguments/expressions
            over: WindowSpecification if present

        Returns:
            Either AggregateFunction or TransformFunction based on type
        """
        from sql_transform.parser import (
            AggregateFunction,
            TransformFunction,
            WindowSpecification,
        )

        if over is None:
            over = WindowSpecification()

        try:
            func_type, _spec = self.function_resolver.resolve(function_name)

            if func_type == "aggregation":
                return AggregateFunction(function_name, args, over=over)
            else:  # transform
                return TransformFunction(function_name, args, over=over)

        except ValueError:
            # Unknown function - default to TransformFunction for extensibility
            return TransformFunction(function_name, args, over=over)

    def get_aggregation_spec(self, function_name: str):
        """Get aggregation spec for DataFusion expression generation."""
        return self.aggregation_registry.get(function_name)

    def get_transform_spec(self, function_name: str):
        """Get transform spec for fitting and applying."""
        return self.transform_registry.get(function_name)

    def create_transformer(self, sql: str) -> "SQLTransformer":
        """Create a new SQLTransformer using this context."""
        from sql_transform.transformer import SQLTransformer

        return SQLTransformer(sql, self)
