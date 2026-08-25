"""Names starting with ``__`` belong to the model.

``packages/confit/docs/properties.md`` P8 states the law and names ``__cf_``, the old
implementation's prefix. The new model mints ``__param_0``, ``__param_fit``,
``__param_{cte_key}`` and ``{name}__x{token}`` and reserved nothing, so a user
relation of that name silently beat the frozen parameter:

    __param_0 = pa.table({"k": [7.0]})
    t = SQLTransform("SELECT t.v * p.k AS z FROM __THIS__ t, __param_0 p, "
                     "(SELECT avg(v) AS k FROM __FIT__) s")
    run(t, D)               -> 7, 14, 21     the user's k
    t.fit(D).transform(D)   -> 2, 4, 6       the frozen k, no error

The rule is the whole prefix rather than ``__param_`` alone: one line to state,
nothing to keep in step as more names are synthesized, and it turns the older
refusal of a CTE named for a parameter into a special case of a general law.
``__FIT__`` and ``__THIS__`` remain legal — they are the two parameters.
"""

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform, TransformError, run

D = pa.table({"v": [1.0, 2.0, 3.0]})


# ------------------------------------------------------ the silent one, closed


def test_a_captured_binding_under_the_prefix_refuses():
    __param_0 = pa.table({"k": [7.0]})
    assert __param_0 is not None
    with pytest.raises(TransformError, match="reserved"):
        SQLTransform(
            "SELECT t.v * p.k AS z FROM __THIS__ t, __param_0 p, "
            "(SELECT avg(v) AS k FROM __FIT__) s"
        )


def test_a_cte_under_the_prefix_refuses():
    with pytest.raises(TransformError, match="reserved"):
        SQLTransform(
            "WITH __param_0 AS (SELECT 7.0 AS k) "
            "SELECT t.v * p.k AS z FROM __THIS__ t, __param_0 p"
        )


def test_a_catalog_table_under_the_prefix_refuses():
    con = duckdb.connect()
    con.execute("CREATE TABLE __param_0 AS SELECT 7.0 AS k")
    with pytest.raises(TransformError, match="reserved"):
        SQLTransform(
            "SELECT t.v * p.k AS z FROM __THIS__ t, __param_0 p", connection=con
        )


def test_an_alias_under_the_prefix_refuses():
    with pytest.raises(TransformError, match="reserved"):
        SQLTransform("SELECT x.v AS z FROM __THIS__ AS __x, __THIS__ x")


@pytest.mark.parametrize("name", ["__param_fit", "__PARAM_0", "__x", "___", "__cf_a"])
def test_the_whole_prefix_is_reserved_not_just_the_names_we_mint(name):
    with pytest.raises(TransformError, match="reserved"):
        SQLTransform(f'WITH "{name}" AS (SELECT 1 AS a) SELECT * FROM "{name}"')


# ------------------------------------------------- and the parameters still work


def test_the_two_parameters_are_still_legal():
    t = SQLTransform(
        "SELECT t.v / s.m AS z FROM __THIS__ t, (SELECT avg(v) m FROM __FIT__) s"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_cte_named_for_a_parameter_still_refuses_by_its_own_name():
    """That refusal is a special case of this law, but it keeps its own
    message: naming a CTE __FIT__ is a different mistake from stepping on the
    reserved prefix."""
    with pytest.raises(TransformError, match="two parameters"):
        SQLTransform("WITH __FIT__ AS (SELECT 1 AS x) SELECT * FROM __FIT__")


def test_ordinary_names_are_untouched():
    codes = pa.table({"k": [7.0]})
    assert codes is not None
    t = SQLTransform(
        "SELECT t.v * c.k AS z FROM __THIS__ t, codes c ORDER BY z", output="arrow"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_a_single_underscore_is_fine():
    _lookup = pa.table({"k": [2.0]})
    assert _lookup is not None
    t = SQLTransform("SELECT t.v * l.k AS z FROM __THIS__ t, _lookup l")
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()
