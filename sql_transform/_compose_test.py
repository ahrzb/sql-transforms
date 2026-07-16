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
