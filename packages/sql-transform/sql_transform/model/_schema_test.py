"""*Freezing is faithful* covers the schema, not only the values.

``freeze`` swapped the frozen node for ``SELECT * FROM __param_N``, and DuckDB
names an unaliased select item after its own printed text — so the column came
back called ``(SELECT * FROM __param_0)`` and ``get_feature_names_out()``, a
public sklearn surface, handed back an internal parameter name.

The decision this pins: the law compares column names too. It is recorded in
docs/properties.md next to the law itself.
"""

import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform, run

D = pa.table({"grp": ["a", "a", "b"], "price": [1.0, 3.0, 100.0]})

UNALIASED = [
    "SELECT (SELECT avg(price) FROM __FIT__), t.price FROM __THIS__ t",
    "SELECT t.price - (SELECT avg(price) FROM __FIT__) FROM __THIS__ t",
    "SELECT (SELECT max(price) FROM __FIT__), (SELECT min(price) FROM __FIT__) "
    "FROM __THIS__ t",
]


@pytest.mark.parametrize("sql", UNALIASED)
def test_freezing_keeps_the_column_names_run_would_give(sql):
    t = SQLTransform(sql)
    assert t.fit(D).transform(D).column_names == run(t, D).column_names


@pytest.mark.parametrize("sql", UNALIASED)
def test_freezing_keeps_the_values_too(sql):
    t = SQLTransform(sql)
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()


def test_no_internal_name_reaches_get_feature_names_out():
    """The sklearn surface, which is where this leaked out of the library."""
    t = SQLTransform(UNALIASED[0])
    t.fit(D)
    t.transform(D)
    assert not any("__param" in name for name in t.get_feature_names_out())


def test_an_explicit_alias_is_still_respected():
    t = SQLTransform("SELECT (SELECT avg(price) FROM __FIT__) AS m FROM __THIS__ t")
    assert t.fit(D).transform(D).column_names == ["m"]


def test_a_plain_column_is_untouched():
    t = SQLTransform("SELECT t.price, t.grp FROM __THIS__ t")
    assert t.fit(D).transform(D).column_names == ["price", "grp"]
