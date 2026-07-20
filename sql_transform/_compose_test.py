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


def test_bare_reference_on_fitted_transform_is_ambiguous_error():
    # A bare {a} on an already-FITTED transform is ambiguous (reuse its frozen
    # state, or re-fit?) and must error. Use {a.transform} to reuse the frozen
    # state, or pass a fresh unfit instance to {a} to fit it into scope.
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train)
    with pytest.raises(ValueError, match="already fitted"):
        SQLTransform(t"SELECT {scaler}(age) AS s2 FROM __THIS__").fit(train)


def test_multi_input_inner_raises_clean_value_error():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0], "city": ["a", "a", "b", "b"]})
    grouped = SQLTransform(
        "SELECT age / AVG(age) OVER (PARTITION BY city) AS e FROM __THIS__"
    ).fit(train)
    with pytest.raises(ValueError, match="one input column"):
        SQLTransform(t"SELECT {grouped.transform}(age) AS e2 FROM __THIS__").fit(train)


def test_frozen_reference_on_unfit_errors():
    unfit = SQLTransform("SELECT age * 2 AS d FROM __THIS__")
    train = pa.table({"age": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not fitted"):
        SQLTransform(t"SELECT {unfit.transform}(age) AS d FROM __THIS__").fit(train)


def test_bare_reference_on_fitted_transform_is_ambiguous_error_via_cascade():
    # Same ambiguity guard as above, but `b` is nested as another ref's input
    # ({a}({b}(col))) -- confirms the guard fires on the recursive fit-cascade
    # path (process_ref/resolve_arg), not just at the top level.
    train_v = pa.table({"v": [10.0, 20.0, 30.0, 40.0]})
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    b = SQLTransform("SELECT v * 2 AS d FROM __THIS__").fit(train_v)
    a = SQLTransform(
        "SELECT (v - AVG(v) OVER ()) / STDDEV(v) OVER () AS s FROM __THIS__"
    )
    with pytest.raises(ValueError, match="already fitted"):
        SQLTransform(t"SELECT {a}({b}(age)) AS z FROM __THIS__").fit(train)


def test_frozen_reference_on_unfit_errors_via_cascade():
    # Same "not fitted" guard as above, but the unfit frozen ref ({b.transform})
    # is nested inside another ref's input -- confirms it fires on the cascade
    # path too.
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    b = SQLTransform("SELECT v * 2 AS d FROM __THIS__")  # unfit
    a = SQLTransform(
        "SELECT (v - AVG(v) OVER ()) / STDDEV(v) OVER () AS s FROM __THIS__"
    )
    with pytest.raises(ValueError, match="not fitted"):
        SQLTransform(t"SELECT {a}({b.transform}(age)) AS z FROM __THIS__").fit(train)


def test_multi_input_unfit_reference_raises():
    # Bare {a} on an unfit multi-input transform hits fit_into_scope's own
    # arity guard (distinct from the frozen-path guard in _frozen_expr
    # exercised by test_multi_input_inner_raises_clean_value_error).
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0], "city": ["a", "a", "b", "b"]})
    grouped = SQLTransform(
        "SELECT age / AVG(age) OVER (PARTITION BY city) AS e FROM __THIS__"
    )  # unfit
    with pytest.raises(ValueError, match="one input column"):
        SQLTransform(t"SELECT {grouped}(age) AS e2 FROM __THIS__").fit(train)


def test_multi_output_unfit_reference_raises():
    # Bare {a} on an unfit multi-output transform hits fit_into_scope's
    # single-output guard. Multi-output fan-out is not supported this slice.
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    multi_out = SQLTransform("SELECT age * 2 AS d, age * 3 AS e FROM __THIS__")  # unfit
    with pytest.raises(ValueError, match="single-output"):
        SQLTransform(t"SELECT {multi_out}(age) AS z FROM __THIS__").fit(train)


def test_reference_not_applied_to_column_errors():
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    scaler = SQLTransform(
        "SELECT (age - AVG(age) OVER ()) / STDDEV(age) OVER () AS s FROM __THIS__"
    ).fit(train)
    with pytest.raises(ValueError, match="single input column"):
        SQLTransform(t"SELECT {scaler.transform}(age + 1) AS s FROM __THIS__").fit(
            train
        )


def test_non_transform_interpolation_errors():
    with pytest.raises(TypeError, match="SQLTransform"):
        SQLTransform(t"SELECT {42}(age) AS s FROM __THIS__")
