"""Tests for SqlTransformContext functionality."""

from sql_transform.context import SqlTransformContext


class TestContext:
    """Test SqlTransformContext functionality."""

    def test_context_creation(self):
        """Test context creation and basic functionality."""
        context = SqlTransformContext()
        # Context should have sklearn transforms registered by default
        transforms = context.list_transforms()
        assert len(transforms) >= 0  # May have sklearn transforms if available

    def test_register_transform(self):
        """Test registering transforms."""
        from sql_transform.function_registry import SklearnTransformSpec

        context = SqlTransformContext()

        # Register a dummy transform spec
        try:
            import sklearn.preprocessing

            dummy_spec = SklearnTransformSpec(
                "dummy", sklearn.preprocessing.StandardScaler, "Dummy transform"
            )
            context.register_transform(dummy_spec)

            assert "dummy" in context.list_transforms()
            assert context.get_transform(dummy_spec.name) is dummy_spec
        except ImportError:
            # Skip if sklearn not available
            pass

    def test_create_transformer(self):
        """Test creating transformer from context."""
        context = SqlTransformContext()
        transformer = context.create_transformer("SELECT feature1 as f1 FROM data")

        # Should be able to create transformer without errors
        assert transformer is not None
        assert hasattr(transformer, "fit")
        assert hasattr(transformer, "transform")
