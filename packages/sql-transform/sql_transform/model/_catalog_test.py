"""TASK-73: passing a connection means using all of its catalog.

``_catalog`` read the supplied connection for tables and views, but the two
function catalogs called bare ``duckdb.execute`` — the module-level default
connection — and cached the answer for the process. So ``connection=`` was
half honoured: a table in your catalog resolved, a macro in the same catalog
refused at construction as an unknown free name.

Also here: ``output=`` was not checked until something was transformed, which
P7 says is too late.
"""

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform, TransformError
from sql_transform.model._transform import OUTPUTS, _as_output

D = pa.table({"v": [1.0, 2.0, 3.0]})


# ------------------------------------------------- the user's own functions


def test_a_macro_on_the_supplied_connection_resolves():
    con = duckdb.connect()
    con.execute("CREATE MACRO double_it(x) AS x * 2")
    t = SQLTransform("SELECT double_it(t.v) AS z FROM __THIS__ t", connection=con)
    assert t.fit(D).transform(D).to_pylist() == [{"z": 2.0}, {"z": 4.0}, {"z": 6.0}]


def test_a_table_macro_on_the_supplied_connection_resolves():
    con = duckdb.connect()
    con.execute("CREATE MACRO nums() AS TABLE SELECT 1 AS n UNION ALL SELECT 2")
    t = SQLTransform("SELECT count(*) AS c FROM __THIS__ t, nums()", connection=con)
    assert t.fit(D).transform(D).to_pylist() == [{"c": 6}]


def test_a_macro_defined_after_the_first_lookup_is_still_seen():
    """The catalog is not cached per connection: caching would pin the
    connection for the life of the process and miss anything defined later."""
    con = duckdb.connect()
    SQLTransform("SELECT t.v AS z FROM __THIS__ t", connection=con)  # warms any cache
    con.execute("CREATE MACRO tripled(x) AS x * 3")
    t = SQLTransform("SELECT tripled(t.v) AS z FROM __THIS__ t", connection=con)
    assert t.fit(D).transform(D).to_pylist()[0] == {"z": 3.0}


def test_an_unknown_function_still_refuses_by_name():
    con = duckdb.connect()
    with pytest.raises(TransformError, match="nowhere"):
        SQLTransform("SELECT nowhere_fit(t.v) AS z FROM __THIS__ t", connection=con)


def test_the_oracles_own_functions_still_work_without_a_connection():
    t = SQLTransform("SELECT round(t.v, 1) AS z FROM __THIS__ t")
    assert t.fit(D).transform(D).to_pylist()[0] == {"z": 1.0}


# ------------------------------------------------------ output= (F15, F16)


def test_a_bad_output_refuses_at_construction():
    with pytest.raises(TransformError, match="output must be one of"):
        SQLTransform("SELECT t.v FROM __THIS__ t", output="bogus")


def test_a_bad_output_refuses_before_any_data_exists():
    """Construction, not fit and not transform — P7."""
    with pytest.raises(TransformError):
        SQLTransform("SELECT 1 AS a FROM __THIS__", output="pandas_")


@pytest.mark.parametrize("output", OUTPUTS)
def test_every_declared_output_is_accepted(output):
    assert SQLTransform("SELECT t.v FROM __THIS__ t", output=output).output == output


@pytest.mark.parametrize("output", OUTPUTS)
def test_as_output_handles_every_declared_output(output):
    """It used to have no branch for ``duckdb``, so it raised
    'output must be one of (... duckdb ...); got duckdb'. Unreachable at the
    time, because transform() routed that case first — a landmine for the next
    person to move the routing."""
    assert _as_output(D, output, None) is not None
