from string.templatelib import Template

import pyarrow as pa
import pytest

from sql_transform import SQLTransform
from sql_transform._compose import Ref, desugar_template, inline_references


def test_desugar_static_template_has_no_refs():
    # A t-string with no interpolations desugars to itself with an empty ref map.
    sql, refs = desugar_template(Template("SELECT 1 AS x"))
    assert sql == "SELECT 1 AS x"
    assert refs == {}


def test_non_transform_interpolation_errors():
    with pytest.raises(TypeError, match="fitted transformer"):
        SQLTransform(t"SELECT {42}(age) AS s FROM __THIS__")


def test_unfitted_transformer_raises_not_fitted_error():
    # Before: TypeError blaming the interpolation TYPE, which hides the real
    # cause. An unfitted transformer has .transform but no n_features_in_, so we
    # can name the actual problem.
    from sklearn.preprocessing import StandardScaler

    train = pa.table({"age": [10.0, 20.0], "income": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not fitted"):
        SQLTransform(t"SELECT {StandardScaler()}(age, income) AS o FROM __THIS__").fit(
            train
        )


# --- composition is reset: the surface refuses by name rather than half-working


def test_composing_one_transform_into_another_is_refused():
    other = SQLTransform("SELECT age AS a FROM __THIS__")
    with pytest.raises(NotImplementedError, match="not supported yet"):
        SQLTransform(t"SELECT {other}(age) AS s FROM __THIS__")


def test_inline_references_refuses_a_hand_built_ref_map():
    # The only route to a non-empty map now is a caller building Refs directly.
    with pytest.raises(NotImplementedError, match="not supported yet"):
        inline_references(
            None, {"__COMPOSE_0__": Ref(object(), frozen=False)}, None, None
        )
