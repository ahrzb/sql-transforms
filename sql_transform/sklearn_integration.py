"""
Optional sklearn integration for SQL transforms.

This module provides sklearn transformer integration when sklearn is available,
but gracefully degrades when it's not installed.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    import sklearn.feature_extraction.text as text  # type: ignore
    import sklearn.feature_selection as selection  # type: ignore
    import sklearn.preprocessing as prep  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    prep = None
    text = None
    selection = None


@dataclass
class TransformSpec:
    """Specification for a transformer integration."""

    transformer_class: type | None
    sql_name: str
    param_mapping: dict[str, str] | None = None
    fit_columns: list[str] | None = None
    output_columns: str | Callable | None = None
    description: str = ""
    requires_sklearn: bool = True

    def __post_init__(self):
        if self.param_mapping is None:
            self.param_mapping = {}


class TransformRegistry:
    """Registry for transformers that can be used in SQL."""

    def __init__(self):
        self._transforms: dict[str, TransformSpec] = {}
        self._register_builtin_transforms()
        if SKLEARN_AVAILABLE:
            self._register_sklearn_transforms()

    def register(self, spec: TransformSpec):
        """Register a new transformer."""
        if spec.requires_sklearn and not SKLEARN_AVAILABLE:
            warnings.warn(
                f"Skipping {spec.sql_name}: sklearn not available", stacklevel=2
            )
            return
        self._transforms[spec.sql_name] = spec

    def get(self, sql_name: str) -> TransformSpec:
        """Get transformer spec by SQL name."""
        if sql_name not in self._transforms:
            # Check if this is a known sklearn transform that's not available
            if not SKLEARN_AVAILABLE and self._is_sklearn_transform(sql_name):
                raise ValueError(
                    f"Transform '{sql_name}' requires sklearn. "
                    "Install with: pip install scikit-learn"
                )
            raise ValueError(f"Unknown transform: {sql_name}")
        return self._transforms[sql_name]

    def _is_sklearn_transform(self, sql_name: str) -> bool:
        """Check if the given SQL name is a known sklearn transform."""
        sklearn_transforms = {
            "sklearn.standardize",
            "sklearn.minmax_scale",
            "sklearn.robust_scale",
            "sklearn.quantile_transform",
            "sklearn.kbins_discretize",
            "sklearn.onehot_encode",
            "sklearn.ordinal_encode",
            "sklearn.tfidf_vectorize",
        }
        return sql_name in sklearn_transforms

    def list_transforms(self) -> dict[str, str]:
        """List all available transforms with descriptions."""
        result = {}
        for name, spec in self._transforms.items():
            status = "" if spec.transformer_class else " (sklearn required)"
            result[name] = spec.description + status
        return result

    def _register_builtin_transforms(self):
        """Register built-in transforms that don't require sklearn."""
        # These would be our native implementations
        pass

    def _register_sklearn_transforms(self):
        """Register sklearn transformers (only if sklearn is available)."""
        if not SKLEARN_AVAILABLE:
            return

        # Scaling transforms
        self.register(
            TransformSpec(
                transformer_class=prep.StandardScaler,
                sql_name="sklearn.standardize",
                description="Z-score normalization (mean=0, std=1)",
            )
        )

        self.register(
            TransformSpec(
                transformer_class=prep.MinMaxScaler,
                sql_name="sklearn.minmax_scale",
                param_mapping={"min": "feature_range[0]", "max": "feature_range[1]"},
                description="Scale features to a given range (default 0-1)",
            )
        )

        self.register(
            TransformSpec(
                transformer_class=prep.RobustScaler,
                sql_name="sklearn.robust_scale",
                description="Scale using median and IQR (robust to outliers)",
            )
        )

        self.register(
            TransformSpec(
                transformer_class=prep.QuantileTransformer,
                sql_name="sklearn.quantile_transform",
                param_mapping={
                    "n_quantiles": "n_quantiles",
                    "distribution": "output_distribution",
                },
                description="Transform to uniform or normal distribution",
            )
        )

        # Binning transforms
        self.register(
            TransformSpec(
                transformer_class=prep.KBinsDiscretizer,
                sql_name="sklearn.kbins_discretize",
                param_mapping={
                    "n_bins": "n_bins",
                    "strategy": "strategy",
                    "encode": "encode",
                },
                description="Bin continuous features into discrete intervals",
            )
        )

        # Categorical encoding
        self.register(
            TransformSpec(
                transformer_class=prep.OneHotEncoder,
                sql_name="sklearn.onehot_encode",
                param_mapping={"drop": "drop", "sparse": "sparse_output"},
                description="One-hot encode categorical features",
            )
        )

        self.register(
            TransformSpec(
                transformer_class=prep.OrdinalEncoder,
                sql_name="sklearn.ordinal_encode",
                description="Encode categorical features as ordinal integers",
            )
        )

        # Text transforms (if available)
        self.register(
            TransformSpec(
                transformer_class=text.TfidfVectorizer,
                sql_name="sklearn.tfidf_vectorize",
                param_mapping={
                    "max_features": "max_features",
                    "ngram_range": "ngram_range",
                    "min_df": "min_df",
                    "max_df": "max_df",
                },
                description="TF-IDF text vectorization",
            )
        )


# Global registry instance
transform_registry = TransformRegistry()


def register_transform(
    transformer_class: type | None,
    sql_name: str,
    param_mapping: dict[str, str] | None = None,
    fit_columns: list[str] | None = None,
    output_columns: str | Callable | None = None,
    description: str = "",
    requires_sklearn: bool = True,
):
    """Register a transformer (decorator or direct call)."""
    spec = TransformSpec(
        transformer_class=transformer_class,
        sql_name=sql_name,
        param_mapping=param_mapping or {},
        fit_columns=fit_columns,
        output_columns=output_columns,
        description=description,
        requires_sklearn=requires_sklearn,
    )
    transform_registry.register(spec)


def create_transformer(spec: TransformSpec, **sql_params):
    """Create a transformer instance from SQL parameters."""
    if spec.transformer_class is None:
        raise ValueError(f"Transform {spec.sql_name} requires sklearn")

    # Map SQL parameters to sklearn parameters
    sklearn_params: dict[str, Any] = {}
    for sql_param, value in sql_params.items():
        if spec.param_mapping and sql_param in spec.param_mapping:
            sklearn_param = spec.param_mapping[sql_param]
            # Handle nested parameters like "feature_range[0]"
            if "[" in sklearn_param and "]" in sklearn_param:
                base_param = sklearn_param.split("[")[0]
                if base_param not in sklearn_params:
                    sklearn_params[base_param] = [None, None]
                index = int(sklearn_param.split("[")[1].split("]")[0])
                sklearn_params[base_param][index] = value
            else:
                sklearn_params[sklearn_param] = value
        else:
            sklearn_params[sql_param] = value

    # Convert lists to tuples for sklearn
    for key, value in sklearn_params.items():
        if isinstance(value, list):
            sklearn_params[key] = tuple(value)

    try:
        return spec.transformer_class(**sklearn_params)
    except TypeError as e:
        raise ValueError(f"Invalid parameters for {spec.sql_name}: {e}") from e


def check_sklearn_availability():
    """Check if sklearn is available and return helpful message."""
    if SKLEARN_AVAILABLE:
        return True, "sklearn is available"
    else:
        return False, "sklearn not available. Install with: pip install scikit-learn"


def get_example_sql():
    """Return example SQL showing transform usage."""
    available_note = (
        ""
        if SKLEARN_AVAILABLE
        else "\n-- Note: sklearn transforms require: pip install scikit-learn\n"
    )

    return f"""{available_note}
-- Scaling examples (requires sklearn)
SELECT 
    sklearn.standardize(feature1) as std_feature1,
    sklearn.minmax_scale(feature2, min=0, max=1) as scaled_feature2,
    sklearn.robust_scale(feature3) as robust_feature3
FROM data;

-- Binning example (requires sklearn)
SELECT
    sklearn.kbins_discretize(feature1, n_bins=5, strategy='uniform') as binned_feature1
FROM data;

-- Built-in aggregations (always available)
SELECT
    feature1 - avg(feature1) as centered_feature1,
    avg(feature1) over (partition by class) as class_mean
FROM data;
"""


if __name__ == "__main__":
    # Show available transforms
    print(f"sklearn available: {SKLEARN_AVAILABLE}")
    print("\nAvailable transforms:")
    for name, desc in transform_registry.list_transforms().items():
        print(f"  {name}: {desc}")

    print(get_example_sql())
