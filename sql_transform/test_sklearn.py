"""Tests for sklearn integration functionality."""

import pyarrow as pa
import pytest

from sql_transform.sklearn_integration import (
    SKLEARN_AVAILABLE,
    check_sklearn_availability,
    create_transformer,
    transform_registry,
)


class TestSklearnIntegration:
    """Test sklearn integration registry and transforms."""

    def test_sklearn_availability_check(self):
        """Test that we can check sklearn availability."""
        available, message = check_sklearn_availability()
        assert isinstance(available, bool)
        assert isinstance(message, str)

        if available:
            assert "sklearn is available" in message
        else:
            assert "Install with: pip install scikit-learn" in message

    def test_registry_lists_transforms(self):
        """Test that registry can list available transforms."""
        transforms = transform_registry.list_transforms()
        assert isinstance(transforms, dict)

        # Should have at least some transforms listed
        if SKLEARN_AVAILABLE:
            assert len(transforms) > 0
            assert "sklearn.standardize" in transforms
            assert "sklearn.minmax_scale" in transforms

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_standardize_transform_creation(self):
        """Test creating standardize transformer."""
        spec = transform_registry.get("sklearn.standardize")
        transformer = create_transformer(spec)

        # Should be a sklearn StandardScaler
        assert transformer.__class__.__name__ == "StandardScaler"

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_minmax_transform_with_params(self):
        """Test creating minmax transformer with parameters."""
        spec = transform_registry.get("sklearn.minmax_scale")
        transformer = create_transformer(spec, min=0, max=10)

        # Should be a sklearn MinMaxScaler with correct range
        assert transformer.__class__.__name__ == "MinMaxScaler"
        assert transformer.feature_range == (0, 10)

    def test_unknown_transform_error(self):
        """Test error handling for unknown transforms."""
        with pytest.raises(ValueError, match="Unknown transform"):
            transform_registry.get("nonexistent_transform")

    @pytest.mark.skipif(SKLEARN_AVAILABLE, reason="sklearn is available")
    def test_sklearn_required_error_when_not_available(self):
        """Test error when sklearn transform requested but not available."""
        with pytest.raises(ValueError, match="requires sklearn"):
            transform_registry.get("sklearn.standardize")

    def test_registry_graceful_degradation(self):
        """Test that registry works even without sklearn."""
        # Should not crash when sklearn is not available
        transforms = transform_registry.list_transforms()
        assert isinstance(transforms, dict)

        # If sklearn is not available, sklearn transforms should not be in registry
        if not SKLEARN_AVAILABLE:
            sklearn_transforms = [
                "sklearn.standardize",
                "sklearn.minmax_scale",
                "sklearn.robust_scale",
            ]
            for transform in sklearn_transforms:
                assert transform not in transforms


# TODO: End-to-end tests will be re-enabled once sklearn resolver is implemented
@pytest.mark.skip(reason="End-to-end tests require sklearn resolver implementation")
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestSklearnEndToEnd:
    """End-to-end tests for sklearn transforms in SQL queries.
    
    These tests show the expected behavior when sklearn transforms are fully integrated.
    Currently they will fail because the resolver step isn't implemented yet.
    """

    def test_standardize_transform_basic(self):
        """Test basic standardize transform end-to-end."""
        from sql_transform.context import SqlTransformContext

        # Create test data
        data = pa.table(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        sql = "SELECT standardize(feature1) as std_feature1 FROM data"

        context = SqlTransformContext()
        transformer = context.create_transformer(sql)
        transformer.fit(data)
        result = transformer.transform(data)

        # Check that result has the expected column
        assert "std_feature1" in result.column_names

        # Check that standardization was applied (mean ≈ 0, std ≈ 1)
        std_values = result["std_feature1"].to_pylist()
        mean = sum(std_values) / len(std_values)
        assert abs(mean) < 0.01  # Mean should be close to 0

        # Standard deviation calculation
        variance = sum((x - mean) ** 2 for x in std_values) / len(std_values)
        std = variance**0.5
        assert abs(std - 1.0) < 0.01  # Std should be close to 1

    def test_multiple_sklearn_transforms(self):
        """Test multiple sklearn transforms in one query."""
        from sql_transform.context import SqlTransformContext

        data = pa.table(
            {
                "income": [30000.0, 50000.0, 80000.0, 120000.0, 200000.0],
                "age": [25.0, 35.0, 45.0, 55.0, 65.0],
                "score": [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        )

        sql = """
        SELECT 
            standardize(income) as std_income,
            minmax_scale(age, 0, 1) as scaled_age,
            robust_scale(score) as robust_score
        FROM data
        """

        context = SqlTransformContext()
        transformer = context.create_transformer(sql)
        transformer.fit(data)
        result = transformer.transform(data)

        # Check all columns are present
        expected_columns = ["std_income", "scaled_age", "robust_score"]
        for col in expected_columns:
            assert col in result.column_names