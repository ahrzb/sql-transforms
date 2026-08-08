"""TASK-76: an identifier means whatever DuckDB says it means.

TASK-71 folded CTE keys because the binder is case-insensitive, and stopped
there. The walk compares identifiers in three other places, and each was still
comparing exact strings:

* ``_catalog`` — a supplied connection's ``Customers`` was unreachable as
  ``customers``, and a frame object of that name then quietly won a lookup the
  connection owns;
* ``_correlation`` — an inner alias differing only in case from an outer one
  produced a ``CorrelatedFit`` on a query DuckDB binds inward, and the mirror
  shape missed a real correlation and died at fit instead;
* ``_reads`` — folded already, but nothing exercised the fold, so dropping it
  stayed green.

The boundary matters as much as the fold: DuckDB's names fold, **Python's do
not**. A frame variable ``codes`` must not answer to ``Codes`` — that lookup is
``scope.get``, not a binder.
"""

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import CorrelatedFit, SQLTransform, UnknownName, run
from sql_transform.model._analysis import _reads
from sql_transform.model._ast import _parse

D = pa.table({"v": [1.0, 2.0, 3.0]})


def node(sql: str):
    return _parse(sql).statements[0].node


# ------------------------------------------------------------- the oracle
# Why a blanket fold is safe here and would be wrong in Postgres.


@pytest.mark.parametrize("used", ["weird", '"weird"', '"WEIRD"', "WeIrD"])
def test_duckdb_folds_quoted_identifiers_too(used):
    con = duckdb.connect()
    con.execute('CREATE TABLE "Weird" AS SELECT 1 AS k')
    assert con.execute(f"SELECT k FROM {used}").fetchall() == [(1,)]


# ------------------------------------------------------------ the catalog


@pytest.mark.parametrize(
    "stored,used",
    [
        ("Customers", "customers"),
        ("customers", "CUSTOMERS"),
        ("CuStOmErS", "cUsToMeRs"),
    ],
)
def test_a_connection_table_is_reachable_in_any_case(stored, used):
    """``_catalog`` returned the names as stored, so a table DuckDB binds
    happily was refused as an unknown free name.

    Both spellings vary: folding only the catalog side leaves the *reference*
    exact, and a mixed-case reference to a lowercase table stays broken.
    """
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {stored} AS SELECT 10.0 AS k")
    t = SQLTransform(f"SELECT t.v * c.k AS z FROM __THIS__ t, {used} c", connection=con)
    assert run(t, D).to_pylist() == [{"z": 10.0}, {"z": 20.0}, {"z": 30.0}]


def test_the_connection_beats_a_frame_object_spelled_differently():
    """The silent one. Passing a connection says *use my catalog*; the exact
    comparison missed the table, so the frame answered instead — same shape,
    different numbers, no error."""
    con = duckdb.connect()
    con.execute("CREATE TABLE Customers AS SELECT 10.0 AS k")
    customers = pa.table({"k": [999.0]})
    assert customers is not None
    t = SQLTransform(
        "SELECT t.v * c.k AS z FROM __THIS__ t, customers c", connection=con
    )
    assert "customers" not in t.captured
    assert run(t, D).to_pylist() == [{"z": 10.0}, {"z": 20.0}, {"z": 30.0}]


def test_a_system_view_is_not_a_name_the_connection_can_bind():
    """``_catalog`` read every row of ``duckdb_views()``, which on a fresh
    connection is 47 internal ones — ``tables``, ``columns``, ``schemata``.
    They are not reachable unqualified, so claiming them cost the caller their
    own object of that name."""
    con = duckdb.connect()
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM tables")  # the claim was false
    tables = pa.table({"k": [4.0]})
    assert tables is not None
    t = SQLTransform("SELECT t.v * s.k AS z FROM __THIS__ t, tables s", connection=con)
    assert "tables" in t.captured
    assert run(t, D).to_pylist() == [{"z": 4.0}, {"z": 8.0}, {"z": 12.0}]


def test_an_attached_databases_table_is_not_bindable_either(tmp_path):
    """The same over-claim, found while fixing the first: ``duckdb_tables()``
    lists every attached database, and only the current one and ``temp`` are on
    the search path."""
    path = tmp_path / "side.db"
    side = duckdb.connect(str(path))
    side.execute("CREATE TABLE far AS SELECT 1 AS k")
    side.close()
    con = duckdb.connect()
    con.execute(f"ATTACH '{path}' AS side")
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM far")  # the claim was false here too
    far = pa.table({"k": [4.0]})
    assert far is not None
    t = SQLTransform("SELECT t.v * s.k AS z FROM __THIS__ t, far s", connection=con)
    assert "far" in t.captured
    assert run(t, D).to_pylist() == [{"z": 4.0}, {"z": 8.0}, {"z": 12.0}]


def test_a_python_name_is_still_case_sensitive():
    """The boundary. ``scope.get`` is Python's namespace, not a binder, and
    folding there would make two distinct variables one."""
    codes = pa.table({"k": [4.0]})
    assert codes is not None
    with pytest.raises(UnknownName):
        SQLTransform("SELECT t.v * c.k AS z FROM __THIS__ t, Codes c")


# --------------------------------------------------------- the correlation


@pytest.mark.parametrize(
    "outer,inner,used", [("t", "T", "t"), ("t", "T", "T"), ("T", "t", "T")]
)
def test_an_inner_alias_shadows_an_outer_one_differing_only_in_case(outer, inner, used):
    """``used.v`` binds to the inner alias however either is spelled, so there
    is no correlation at all — the exact comparison saw the outer one and
    refused a legal query.

    Three spellings because there are three sides to fold: the names inside,
    the names outside, and the reference's own qualifier.
    """
    sql = (
        f"SELECT x.z FROM __THIS__ {outer}, "
        f"(SELECT avg({used}.v) AS z FROM __FIT__ {inner}) x"
    )
    assert run(SQLTransform(sql), D).to_pylist() == [{"z": 2.0}] * 3


@pytest.mark.parametrize("outer,used", [("T", "t"), ("t", "T"), ("Tt", "tT")])
def test_a_correlation_into_this_refuses_at_construction_whatever_the_case(outer, used):
    """The mirror. ``used.v`` reaches the outer alias, which is ``__THIS__`` —
    the one refusal, and P7 says it is named and at construction. Folding only
    one side would have left this dying at fit inside DuckDB's own message."""
    sql = (
        f"SELECT x.z FROM __THIS__ {outer}, "
        f"(SELECT avg(f.v) + {used}.v AS z FROM __FIT__ f) x"
    )
    with pytest.raises(CorrelatedFit):
        SQLTransform(sql)


# ---------------------------------------------------------------- _reads


@pytest.mark.parametrize("used", ["Live", "LIVE", "lIvE"])
def test_reads_follows_a_cte_referenced_in_another_case(used):
    """The fold in ``_reads`` was already there and nothing held it down: the
    only case gate went through ``_resolve``, so dropping it stayed green.

    The subtree must *reference* the CTE without containing it — that is how
    ``_plan`` asks, and a node carrying the definition answers from the
    definition, whatever the lookup does.
    """
    assert _reads(node(f"SELECT count(*) FROM {used}"), {"live": {"__THIS__"}}) == {
        "__THIS__"
    }


@pytest.mark.parametrize("used", ["Live", "LIVE", "lIvE"])
def test_a_subtree_reading_this_through_a_cte_in_another_case_is_not_frozen(used):
    """What that fold buys end to end, and it is the F6 shape again: the
    subquery reads ``__FIT__`` and a CTE that reads ``__THIS__``. Miss the CTE
    and it looks freezable, so it is evaluated at fit — where ``__THIS__`` is
    not bound."""
    t = SQLTransform(
        "WITH Live AS (SELECT * FROM __THIS__) "
        f"SELECT t.v, (SELECT count(*) FROM {used}, (SELECT v FROM __FIT__) f) AS c "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pylist() == run(t, D).to_pylist()
