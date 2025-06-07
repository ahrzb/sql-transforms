"""Context for registering transforms and managing SQL transformation environment."""

from typing import TYPE_CHECKING, Any, Protocol

import datafusion

if TYPE_CHECKING:
    from sql_transform.transformer import SQLTransformer


class TransformFunction(Protocol):
    """Protocol for transform functions that can be registered with the context."""

    def fit(self, data: Any, **kwargs) -> Any: ...
    def transform(self, data: Any, **kwargs) -> Any: ...


class SqlTransformContext:
    """Context for managing SQL transformations."""

    def __init__(self):
        self.transforms: dict[str, TransformFunction] = {}
        self._datafusion_ctx: datafusion.SessionContext | None = None
        self._sklearn_registry = None
        self._initialize_sklearn_registry()

    @property
    def datafusion_ctx(self) -> datafusion.SessionContext:
        """Get or create the DataFusion session context."""
        if self._datafusion_ctx is None:
            self._datafusion_ctx = datafusion.SessionContext()
        return self._datafusion_ctx

    def register_transform(self, name: str, transform_func: TransformFunction) -> None:
        """Register a transform function that can be used in SQL queries."""
        self.transforms[name] = transform_func

    def get_transform(self, name: str) -> TransformFunction | None:
        """Get a registered transform function by name."""
        return self.transforms.get(name)

    def list_transforms(self) -> list[str]:
        """List all registered transform names."""
        return list(self.transforms.keys())

    def _initialize_sklearn_registry(self):
        """Initialize the sklearn transform registry."""
        try:
            from sql_transform.sklearn_integration import TransformRegistry

            self._sklearn_registry = TransformRegistry()
        except ImportError:
            # sklearn_integration not available
            self._sklearn_registry = None

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

        # Built-in aggregation functions
        builtin_aggregations = {
            "avg",
            "sum",
            "count",
            "min",
            "max",
            "stddev",
            "variance",
            "first",
            "last",
            "median",
            "mode",
        }

        if function_name in builtin_aggregations:
            return AggregateFunction(function_name, args, over=over)

        # Check if it's a registered sklearn transform
        if self._sklearn_registry:
            try:
                self._sklearn_registry.get(function_name)
                # If we found it in the registry, it's a transform
                return TransformFunction(function_name, args, over=over)
            except ValueError:
                pass  # Not in sklearn registry

        # Check if it's in our custom transforms registry
        if function_name in self.transforms:
            return TransformFunction(function_name, args, over=over)

        # Unknown function - treat as transform (will fail at runtime if not resolvable)
        return TransformFunction(function_name, args, over=over)

    def get_sklearn_spec(self, function_name: str):
        """Get sklearn transform spec if available."""
        if self._sklearn_registry:
            try:
                return self._sklearn_registry.get(function_name)
            except ValueError:
                return None
        return None

    def create_transformer(self, sql: str) -> "SQLTransformer":
        """Create a new SQLTransformer using this context."""
        from sql_transform.transformer import SQLTransformer

        return SQLTransformer(sql, self)
