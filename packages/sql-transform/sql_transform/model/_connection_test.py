"""Bring your own connection, and take output that has not been executed.

A transform that conjures a private connection cannot compose with anything: a
``DuckDBPyRelation`` belongs to the connection that built it and cannot be
handed to another one, not even to a cursor of the same connection. So the
connection is a parameter, and lazy output chains when both stages share one.

Sharing has a price, and the collision test below is what pays it: two
transforms on one connection both bind ``__THIS__`` and both call a parameter
``__param_0``. Eagerly that is harmless — each materialises before the next
registers — but a relation is not executed yet.
"""

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from sql_transform.model import SQLTransform, Transform, UnknownName
from sql_transform.model._freezing_test import approx

SMALL = pa.table({"v": [1.0, 2.0, 3.0]})
BIG = pa.table({"v": [50.0, 100.0, 150.0]})
LIVE = pa.table({"v": [10.0, 20.0]})

REL = (
    "SELECT round(t.v / s.m, 4) AS z FROM __THIS__ t, (SELECT avg(v) m FROM __FIT__) s"
)


def test_duckdb_output_is_a_relation_that_agrees_with_arrow():
    t = SQLTransform(REL)
    eager = t.fit(SMALL).transform(LIVE)
    lazy = SQLTransform(REL).set_output(transform="duckdb")
    lazy.fit(SMALL)
    out = lazy.transform(LIVE)
    assert isinstance(out, duckdb.DuckDBPyRelation)
    assert out.to_arrow_table().equals(eager)


def test_the_relation_is_not_executed_until_it_is_consumed():
    """The whole point of ``duckdb`` output. A foreign transform records every
    call, so *was it run yet* is a fact rather than a claim."""
    calls: list[int] = []

    def transform(instance, relation):
        calls.append(len(relation))
        return pa.table({"v": pc.multiply(relation["v"], 2.0)})

    tick = Transform(
        fit=lambda f: None, transform=transform, takes=("v",), returns=("v",)
    )
    assert tick is not None
    t = SQLTransform(
        "SELECT tick_transform(f.theta, struct_pack(v := t.v)).v AS z "
        "FROM __THIS__ t, (SELECT tick_fit(struct_pack(v := v)) AS theta "
        "FROM __FIT__) f"
    ).set_output(transform="duckdb")
    t.fit(SMALL)

    lazy = t.transform(LIVE)
    assert calls == []  # bound and planned, but nothing has run
    assert lazy.columns == ["z"]  # binding is eager; execution is not
    assert lazy.to_arrow_table().num_rows == 2
    assert calls == [2]  # ... and now it has


def test_column_names_come_free_without_executing():
    t = SQLTransform(REL).set_output(transform="duckdb")
    t.fit(SMALL)
    t.transform(LIVE)
    assert list(t.get_feature_names_out()) == ["z"]


# ---------------------------------------------------------- a shared connection


def test_two_transforms_chain_lazily_on_a_shared_connection():
    con = duckdb.connect()
    first = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    second = SQLTransform(
        "SELECT round(sum(z), 4) AS total FROM __THIS__", connection=con
    ).set_output(transform="duckdb")

    first.fit(SMALL)
    stage1 = first.transform(LIVE)
    second.fit(stage1)
    stage2 = second.transform(stage1)

    assert isinstance(stage2, duckdb.DuckDBPyRelation)
    assert stage2.to_arrow_table().to_pylist() == [{"total": 15.0}]


def test_a_lazy_relation_cannot_cross_connections_and_says_so():
    """Not silent: DuckDB refuses a relation built by another connection."""
    first = SQLTransform(REL).set_output(transform="duckdb")  # its own connection
    second = SQLTransform("SELECT sum(z) AS total FROM __THIS__")  # and another
    first.fit(SMALL)
    with pytest.raises(duckdb.Error, match="another Connection"):
        second.fit(first.transform(LIVE))


def test_a_shared_connection_does_not_let_one_stage_read_anothers_params():
    """**The gate sharing has to pass.**

    Two fits of the same text have the same parameter name. Hold both
    relations unexecuted, then consume the first: without a rename per
    execution it reads the second's parameters, and the answer is wrong with
    no error at all — same shape, plausible numbers.
    """
    con = duckdb.connect()
    small = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    big = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    small.fit(SMALL)  # mean 2
    big.fit(BIG)  # mean 100

    first = small.transform(LIVE)
    second = big.transform(LIVE)  # registers its own __param_0 and __THIS__

    assert approx(first.to_arrow_table()) == [(5.0,), (10.0,)]  # 10/2, 20/2
    assert approx(second.to_arrow_table()) == [(0.1,), (0.2,)]  # 10/100, 20/100


def test_the_supplied_connections_catalog_resolves():
    """Passing a connection is how you say *use my catalog*, so its tables are
    not free references to hunt for in the caller's frame."""
    con = duckdb.connect()
    con.execute("CREATE TABLE dim AS SELECT 3.0 AS factor")
    t = SQLTransform(
        "SELECT t.v * dim.factor AS z FROM __THIS__ t, dim ORDER BY z",
        connection=con,
    )
    assert t.captured == {}  # dim is the connection's, not a captured object
    assert approx(t.fit(SMALL)(LIVE)) == [(30.0,), (60.0,)]


def test_a_name_in_neither_the_catalog_nor_the_frame_still_refuses():
    con = duckdb.connect()
    with pytest.raises(UnknownName, match="nowhere"):
        SQLTransform("SELECT * FROM __THIS__, nowhere", connection=con)


def test_a_shared_connection_is_not_polluted_with_stray_names():
    """Every execution registers under its own names, so the readable ones
    stay yours."""
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con)
    t.fit(SMALL)
    t.transform(LIVE)
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM __THIS__").fetchall()
