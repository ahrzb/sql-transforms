"""TASK-69: nothing the model registers outlives the call that needed it.

``Fitted._bind`` mints a ``{name}__x{token}`` per execution so a shared
connection never sees two registrations under one name. ``_connect`` — the
``fit()`` and ``run()`` path — did not, and neither path ever unregistered
anything. Three failures followed:

* leftovers under readable names went into ``_catalog``, so the *next*
  transform built on that connection bound to them instead of capturing from
  its caller's frame — silently, with plausible numbers;
* every execution added registrations that were never dropped, so a serving
  loop pinned every batch it had ever seen;
* foreign halves registered under the unmangled stem, so a second ``fit()``
  with a foreign member crashed.

The gate is a *delta*: whatever the connection held before a call, it holds
after. That is stronger than naming the leaked objects, and it does not go
stale when a new one is added.
"""

import gc

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from sql_transform.model import SQLTransform, Transform, run

SMALL = pa.table({"v": [1.0, 2.0, 3.0]})
BIG = pa.table({"v": [50.0, 100.0, 150.0]})
LIVE = pa.table({"v": [10.0, 20.0]})

REL = (
    "SELECT round(t.v / s.m, 4) AS z FROM __THIS__ t, (SELECT avg(v) m FROM __FIT__) s"
)


def catalog(con) -> set[str]:
    rows = con.execute(
        "SELECT table_name FROM duckdb_tables()"
        " UNION ALL SELECT view_name FROM duckdb_views()"
    ).fetchall()
    return {name for (name,) in rows}


def functions(con) -> set[str]:
    rows = con.execute(
        "SELECT function_name FROM duckdb_functions() WHERE function_type = 'scalar'"
    ).fetchall()
    return {name for (name,) in rows}


# ------------------------------------------------------- nothing outlives a call


def test_fit_leaves_no_trace():
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con)
    before = catalog(con)
    t.fit(SMALL)
    assert catalog(con) == before


def test_transform_leaves_no_trace():
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con)
    t.fit(SMALL)
    before = catalog(con)
    t.transform(LIVE)
    assert catalog(con) == before


def test_run_leaves_no_trace():
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con)
    before = catalog(con)
    run(t, SMALL)
    assert catalog(con) == before


def test_the_training_set_does_not_stay_readable_after_fit():
    """The pointed one: ``fit`` left ``__FIT__`` bound to the whole training
    relation, on the caller's own connection, for good."""
    con = duckdb.connect()
    SQLTransform(REL, connection=con).fit(SMALL)
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM __FIT__").fetchall()


def test_repeated_transform_does_not_accumulate():
    """Serving is a loop. 400 objects for 200 calls is the shape that kills it."""
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con)
    t.fit(SMALL)
    t.transform(LIVE)
    before = catalog(con)
    for _ in range(50):
        t.transform(LIVE)
    assert catalog(con) == before


# ------------------------------------------- leftovers must not capture a name


def _built_with_own_codes(con, factor):
    """A transform whose ``codes`` is local to this call, so the two cannot be
    the same object by accident."""
    codes = pa.table({"k": ["a"], "mul": [factor]})
    assert codes is not None
    return SQLTransform(
        "SELECT t.v * c.mul AS z FROM __THIS__ t, codes c", connection=con
    )


def test_a_transform_built_after_a_fit_still_captures_from_its_own_frame():
    """F3, the silent one.

    ``fit`` left ``codes`` registered under its readable name; ``_catalog``
    then reported it as *the connection already owns this*, so the second
    transform never looked in its caller's frame and quietly served the first
    one's lookup table.
    """
    con = duckdb.connect()
    first = _built_with_own_codes(con, 1.0)
    first.fit(SMALL)
    second = _built_with_own_codes(con, 100.0)
    assert "codes" in second.captured  # not taken from the leftovers
    assert second.fit(SMALL).transform(pa.table({"v": [1.0]})).to_pylist() == [
        {"z": 100.0}
    ]


# ------------------------------------------------------------ foreign members


def _tick():
    return Transform(
        fit=lambda f: None,
        transform=lambda i, r: pa.table({"v": pc.multiply(r["v"], 2.0)}),
        takes=("v",),
        returns=("v",),
    )


FOREIGN_SQL = (
    "SELECT sc_transform(f.theta, struct_pack(v := t.v)).v AS z "
    "FROM __THIS__ t, (SELECT sc_fit(struct_pack(v := v)) AS theta FROM __FIT__) f"
)


def test_a_second_fit_with_a_foreign_member_works_on_a_shared_connection():
    """F9: the halves registered under the raw stem, so the second fit hit
    'a function by the name of sc_fit is already created'."""
    con = duckdb.connect()
    sc = _tick()
    assert sc is not None
    t = SQLTransform(FOREIGN_SQL, connection=con)
    assert t.fit(SMALL).transform(LIVE).to_pylist() == [{"z": 20.0}, {"z": 40.0}]
    assert t.fit(BIG).transform(LIVE).to_pylist() == [{"z": 20.0}, {"z": 40.0}]


def test_foreign_functions_do_not_accumulate():
    con = duckdb.connect()
    sc = _tick()
    assert sc is not None
    t = SQLTransform(FOREIGN_SQL, connection=con)
    t.fit(SMALL)
    t.transform(LIVE)
    before = functions(con)
    for _ in range(10):
        t.fit(SMALL)
        t.transform(LIVE)
    assert functions(con) == before


# ------------------------------------------------------------ the lazy path


def test_a_lazy_relation_survives_until_it_is_consumed():
    """Cleanup must not reach a relation that has not run yet — that is the
    whole point of ``duckdb`` output."""
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    t.fit(SMALL)
    lazy = t.transform(LIVE)
    t.transform(LIVE)  # a later execution must not pull it out from under
    t.fit(BIG)
    assert lazy.to_arrow_table().to_pylist() == [{"z": 5.0}, {"z": 10.0}]


def test_two_lazy_relations_stay_independent():
    con = duckdb.connect()
    small = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    big = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    small.fit(SMALL)
    big.fit(BIG)
    first, second = small.transform(LIVE), big.transform(LIVE)
    assert first.to_arrow_table().to_pylist() == [{"z": 5.0}, {"z": 10.0}]
    assert second.to_arrow_table().to_pylist() == [{"z": 0.1}, {"z": 0.2}]


def test_a_dropped_lazy_relation_releases_what_it_held():
    """The lazy path cannot clean up when the call returns, so it cleans up
    when the relation it handed back is collected."""
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con).set_output(transform="duckdb")
    t.fit(SMALL)
    t.transform(LIVE).to_arrow_table()
    gc.collect()
    before = catalog(con)
    for _ in range(20):
        t.transform(LIVE).to_arrow_table()
    gc.collect()
    assert catalog(con) == before


# --------------------------------------------------------- the user's own stuff


def test_the_users_own_objects_are_untouched():
    con = duckdb.connect()
    con.execute("CREATE TABLE dim AS SELECT 3.0 AS factor")
    t = SQLTransform(
        "SELECT t.v * dim.factor AS z FROM __THIS__ t, dim", connection=con
    )
    t.fit(SMALL)
    t.transform(LIVE)
    assert con.execute("SELECT * FROM dim").fetchall() == [(3.0,)]
