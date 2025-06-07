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

    def create_transformer(self, sql: str) -> "SQLTransformer":
        """Create a new SQLTransformer using this context."""
        from sql_transform.transformer import SQLTransformer

        return SQLTransformer(sql, self)
