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


# ------------------------------------------------- qualified names (TASK-76)
#
# Found while narrowing `_catalog` to the names a connection can bind
# *unqualified*: that listing was also what let a qualified name through, so
# narrowing it refused `side.main.far` and `information_schema.tables`, both of
# which used to work. The catalog was never the right test for those — a
# captured object is registered under a bare name, so a qualified reference can
# never mean one, whatever any listing says.


def test_a_schema_qualified_name_is_the_connections_own():
    con = duckdb.connect()
    t = SQLTransform(
        "SELECT count(*) AS c FROM __THIS__ t, information_schema.tables",
        connection=con,
    )
    assert t.captured == {}
    assert t.fit(D).transform(D).to_pylist()[0]["c"] >= 0


def test_an_attached_database_is_reachable_when_qualified(tmp_path):
    path = tmp_path / "side.db"
    side = duckdb.connect(str(path))
    side.execute("CREATE TABLE far AS SELECT 4.0 AS k")
    side.close()
    con = duckdb.connect()
    con.execute(f"ATTACH '{path}' AS side")
    t = SQLTransform(
        "SELECT t.v * s.k AS z FROM __THIS__ t, side.main.far s", connection=con
    )
    assert t.captured == {}
    assert t.fit(D).transform(D).to_pylist() == [{"z": 4.0}, {"z": 8.0}, {"z": 12.0}]


def test_a_qualified_name_cannot_reach_a_frame_object():
    """The rule stated the other way round. ``far`` is right there in the
    frame, and ``side.main.far`` still does not mean it."""
    far = pa.table({"k": [4.0]})
    assert far is not None
    with pytest.raises(TransformError, match="connection"):
        SQLTransform("SELECT t.v * s.k AS z FROM __THIS__ t, side.main.far s")


def test_a_qualified_name_without_a_connection_refuses_at_construction():
    """P7. The transform's own connection is fresh per call, so there is no
    catalog for a qualified name to be in — saying so at construction beats
    DuckDB's own message at fit."""
    with pytest.raises(TransformError, match="connection"):
        SQLTransform("SELECT count(*) AS c FROM __THIS__ t, information_schema.tables")


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
