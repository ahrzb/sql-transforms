"""
Clean function registry focused on extensibility.

Simple API for registering aggregations and transforms without premature validation.
"""

from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

# === AGGREGATION API (Simple, DataFusion-native) ===


@runtime_checkable
class AggregationSpec(Protocol):
    """Protocol for aggregation function specifications."""

    @property
    def name(self) -> str:
        """SQL name of the aggregation."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def arity(self) -> int | tuple[int, int] | None:
        """Number of arguments this function accepts."""
        ...

    def to_datafusion_expr(self, column_expr: Any, args: list[Any]) -> Any:
        """Convert to DataFusion expression for direct execution."""
        ...


class SimpleAggregation:
    """Simple aggregation that maps directly to DataFusion functions."""

    def __init__(
        self, name: str, description: str = "", arity: int | tuple[int, int] | None = 1
    ):
        self._name = name
        self._description = description
        self._arity = arity

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def arity(self) -> int | tuple[int, int] | None:
        return self._arity

    def to_datafusion_expr(self, column_expr: Any, args: list[Any]) -> Any:
        import datafusion

        mapping = {
            "avg": datafusion.functions.avg,
            "sum": datafusion.functions.sum,
            "count": datafusion.functions.count,
            "min": datafusion.functions.min,
            "max": datafusion.functions.max,
            "stddev": datafusion.functions.stddev,
            "variance": datafusion.functions.variance,
        }

        if self.name in mapping:
            return mapping[self.name](column_expr)
        else:
            raise ValueError(f"Unknown aggregation: {self.name}")


# === TRANSFORM API (Complex, stateful) ===


@runtime_checkable
class TransformSpec(Protocol):
    """Protocol for transformation function specifications."""

    @property
    def name(self) -> str:
        """SQL name of the transform."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    @property
    def arity(self) -> int | tuple[int, int] | None:
        """Number of arguments this function accepts."""
        ...

    def create_fitter(self, args: list[Any]) -> "TransformFitter":
        """Create a fitter instance for this transform with given args."""
        ...


@runtime_checkable
class TransformFitter(Protocol):
    """Protocol for fitting transforms."""

    def fit(
        self, data: pa.Table, column_name: str, context: dict[str, Any]
    ) -> "FittedTransform":
        """Fit the transform to data."""
        ...


@runtime_checkable
class FittedTransform(Protocol):
    """Protocol for a fitted transform - pure applier."""

    def transform(self, data: pa.Table, column_name: str) -> pa.Array:
        """Apply the fitted transform to data."""
        ...


# === SKLEARN IMPLEMENTATION ===


class SklearnTransformSpec:
    """Sklearn transform specification."""

    def __init__(
        self,
        name: str,
        sklearn_class: type,
        description: str = "",
        arity: int | tuple[int, int] | None = (1, None),
        param_mapping: dict[str, str] | None = None,
    ):
        self._name = name
        self._sklearn_class = sklearn_class
        self._description = description
        self._arity = arity
        self._param_mapping = param_mapping or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def arity(self) -> int | tuple[int, int] | None:
        return self._arity

    def create_fitter(self, args: list[Any]) -> "SklearnTransformFitter":
        return SklearnTransformFitter(self._sklearn_class, args, self._param_mapping)


class SklearnTransformFitter:
    """Fitter for sklearn transforms."""

    def __init__(
        self, sklearn_class: type, args: list[Any], param_mapping: dict[str, str]
    ):
        self._sklearn_class = sklearn_class
        self._args = args
        self._param_mapping = param_mapping

    def fit(
        self, data: pa.Table, column_name: str, context: dict[str, Any]
    ) -> "SklearnFittedTransform":
        # Extract parameters from args (skip first arg which is column)
        params = self._extract_params(self._args[1:])

        # Create and fit sklearn transformer
        transformer = self._sklearn_class(**params)
        column_data = data[column_name].to_numpy().reshape(-1, 1)
        transformer.fit(column_data)

        return SklearnFittedTransform(transformer)

    def _extract_params(self, param_args: list[Any]) -> dict[str, Any]:
        # Extract sklearn parameters from SQL arguments
        # Implementation depends on specific sklearn class and param_mapping
        return {}


class SklearnFittedTransform:
    """Fitted sklearn transform."""

    def __init__(self, sklearn_transformer: Any):
        self._transformer = sklearn_transformer

    def transform(self, data: pa.Table, column_name: str) -> pa.Array:
        column_data = data[column_name].to_numpy().reshape(-1, 1)
        transformed = self._transformer.transform(column_data).flatten()
        return pa.array(transformed)


# === REGISTRIES ===


class AggregationRegistry:
    """Registry for aggregation functions."""

    def __init__(self):
        self._aggregations: dict[str, AggregationSpec] = {}
        self._register_builtins()

    def register(self, aggregation: AggregationSpec) -> None:
        """Register an aggregation function."""
        self._aggregations[aggregation.name] = aggregation

    def get(self, name: str) -> AggregationSpec | None:
        """Get an aggregation by name."""
        return self._aggregations.get(name)

    def is_aggregation(self, name: str) -> bool:
        """Check if a function name is a registered aggregation."""
        return name in self._aggregations

    def list_aggregations(self) -> dict[str, str]:
        """List all aggregations with descriptions."""
        return {name: agg.description for name, agg in self._aggregations.items()}

    def _register_builtins(self) -> None:
        """Register built-in DataFusion aggregations."""
        builtins = [
            ("avg", "Calculate mean/average", 1),
            ("sum", "Calculate sum", 1),
            ("count", "Count non-null values", 1),
            ("min", "Find minimum value", 1),
            ("max", "Find maximum value", 1),
            ("stddev", "Calculate standard deviation", 1),
            ("variance", "Calculate variance", 1),
        ]

        for name, desc, arity in builtins:
            self.register(SimpleAggregation(name, desc, arity))


class TransformRegistry:
    """Registry for transformation functions."""

    def __init__(self):
        self._transforms: dict[str, TransformSpec] = {}

    def register(self, transform: TransformSpec) -> None:
        """Register a transformation function."""
        self._transforms[transform.name] = transform

    def get(self, name: str) -> TransformSpec | None:
        """Get a transform by name."""
        return self._transforms.get(name)

    def is_transform(self, name: str) -> bool:
        """Check if a function name is a registered transform."""
        return name in self._transforms

    def list_transforms(self) -> dict[str, str]:
        """List all transforms with descriptions."""
        return {name: trans.description for name, trans in self._transforms.items()}


# === FUNCTION RESOLVER ===


class FunctionResolver:
    """Resolves function names to either aggregations or transforms."""

    def __init__(
        self,
        aggregation_registry: AggregationRegistry,
        transform_registry: TransformRegistry,
    ):
        self.aggregations = aggregation_registry
        self.transforms = transform_registry

    def resolve(
        self, function_name: str
    ) -> tuple[str, AggregationSpec | TransformSpec]:
        """Resolve a function name to its type and spec.

        Returns:
            ("aggregation", spec) or ("transform", spec)

        Raises:
            ValueError if function is not found
        """
        if self.aggregations.is_aggregation(function_name):
            return "aggregation", self.aggregations.get(function_name)

        if self.transforms.is_transform(function_name):
            return "transform", self.transforms.get(function_name)

        raise ValueError(f"Unknown function: {function_name}")

    def list_all_functions(self) -> dict[str, tuple[str, str]]:
        """List all functions with their type and description.

        Returns:
            {function_name: (type, description)}
        """
        result = {}

        for name, desc in self.aggregations.list_aggregations().items():
            result[name] = ("aggregation", desc)

        for name, desc in self.transforms.list_transforms().items():
            result[name] = ("transform", desc)

        return result


# === REGISTRATION HELPERS ===


def create_sklearn_registry() -> TransformRegistry:
    """Create a transform registry with sklearn transforms."""
    registry = TransformRegistry()

    try:
        import sklearn.preprocessing as prep

        registry.register(
            SklearnTransformSpec(
                "sklearn.standardize",
                prep.StandardScaler,
                "Z-score normalization",
                arity=1,
            )
        )

        registry.register(
            SklearnTransformSpec(
                "sklearn.minmax_scale",
                prep.MinMaxScaler,
                "Scale to range",
                arity=(1, 3),  # column + optional min + optional max
            )
        )

        registry.register(
            SklearnTransformSpec(
                "sklearn.robust_scale",
                prep.RobustScaler,
                "Robust scaling using median and IQR",
                arity=1,
            )
        )

    except ImportError:
        pass  # sklearn not available

    return registry
