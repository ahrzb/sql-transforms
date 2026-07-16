from string.templatelib import Template

import pyarrow as pa
import pytest

from sql_transform import SQLTransform
from sql_transform._compose import desugar_template


def test_desugar_static_template_has_no_refs():
    # A t-string with no interpolations desugars to itself with an empty ref map.
    sql, refs = desugar_template(Template("SELECT 1 AS x"))
    assert sql == "SELECT 1 AS x"
    assert refs == {}


def test_bare_reference_raises_fit_cascade_not_implemented():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train)
    with pytest.raises(NotImplementedError, match="fit-cascade"):
        SQLTransform(t"SELECT {scaler}(age) AS s2 FROM __THIS__").fit(train)


def test_multi_input_inner_raises_clean_value_error():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0], "city": ["a", "a", "b", "b"]})
    grouped = SQLTransform(
        "SELECT age / AVG(age) OVER (PARTITION BY city) AS e FROM __THIS__"
    ).fit(train)
    with pytest.raises(ValueError, match="one input column"):
        SQLTransform(
            t"SELECT {grouped.transform}(age) AS e2 FROM __THIS__"
        ).fit(train)


def test_frozen_reference_on_unfit_errors():
    unfit = SQLTransform("SELECT age * 2 AS d FROM __THIS__")
    train = pa.table({"age": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not fitted"):
        SQLTransform(t"SELECT {unfit.transform}(age) AS d FROM __THIS__").fit(train)


def test_reference_not_applied_to_column_errors():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train)
    with pytest.raises(ValueError, match="single input column"):
        SQLTransform(
            t"SELECT {scaler.transform}(age + 1) AS s FROM __THIS__"
        ).fit(train)


def test_non_transform_interpolation_errors():
    with pytest.raises(TypeError, match="SQLTransform"):
        SQLTransform(t"SELECT {42}(age) AS s FROM __THIS__")
