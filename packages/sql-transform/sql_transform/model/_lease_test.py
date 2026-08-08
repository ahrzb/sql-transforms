"""TASK-74 R1: a lease outlives the relation, and dies with the artifact.

``relation()`` tied its lease to the returned relation via ``weakref.finalize``.
A relation *derived* from it still needs those tables but holds no reference to
its parent — verified: the parent is collected while the derived one is alive —
so the obvious spelling crashed:

    t.transform(D).limit(2)      # parent never bound to a name

The lease now lives on the ``Fitted``, which is what the caller actually holds.
That is bounded rather than free: every outstanding lazy relation costs one
registration until the artifact is released, refit, or dropped. ``release()``
is the deterministic way out, and the eager path never accumulates at all.
"""

import gc

import duckdb
import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform

D = pa.table({"v": [1.0, 2.0, 3.0]})


def catalog(con) -> set[str]:
    rows = con.execute(
        "SELECT table_name FROM duckdb_tables()"
        " UNION ALL SELECT view_name FROM duckdb_views()"
    ).fetchall()
    return {name for (name,) in rows}


def lazy_on(con):
    t = SQLTransform(
        "SELECT t.v * 2 AS z FROM __THIS__ t", connection=con, output="duckdb"
    )
    t.fit(D)
    return t


# --------------------------------------------------------------- the crash


def test_a_derived_relation_works_when_its_parent_was_never_named():
    """The regression, in the spelling that reads most naturally."""
    con = duckdb.connect()
    t = lazy_on(con)
    assert t.transform(D).limit(2).fetchall() == [(2.0,), (4.0,)]


def test_a_derived_relation_outlives_an_explicitly_dropped_parent():
    con = duckdb.connect()
    t = lazy_on(con)
    parent = t.transform(D)
    derived = parent.limit(2)
    del parent
    gc.collect()
    assert derived.fetchall() == [(2.0,), (4.0,)]


def test_chaining_several_operations_off_one_relation():
    con = duckdb.connect()
    t = lazy_on(con)
    assert t.transform(D).filter("z > 2").limit(1).fetchall() == [(4.0,)]


# ------------------------------------------------- and it is still bounded


def test_dropping_the_artifact_gives_every_table_back():
    """Also the gate for the classic footgun: a finalizer that closes over the
    Fitted keeps it alive, so it never fires and the release never happens."""
    con = duckdb.connect()
    before = catalog(con)
    t = lazy_on(con)
    fitted = t.fitted_
    fitted.relation(D)
    fitted.relation(D)
    assert len(catalog(con)) > len(before)  # they really are registered
    del t, fitted
    gc.collect()
    assert catalog(con) == before


def test_release_gives_them_back_on_demand():
    con = duckdb.connect()
    t = lazy_on(con)
    before = catalog(con)
    for _ in range(20):
        t.transform(D).fetchall()
    assert len(catalog(con)) > len(before)
    t.fitted_.release()
    assert catalog(con) == before


def test_refitting_releases_the_previous_artifacts_leases():
    con = duckdb.connect()
    t = lazy_on(con)
    before = catalog(con)
    for _ in range(10):
        t.transform(D).fetchall()
    t.fit(D)
    gc.collect()
    assert catalog(con) == before


def test_the_eager_path_still_never_accumulates():
    con = duckdb.connect()
    t = SQLTransform("SELECT t.v * 2 AS z FROM __THIS__ t", connection=con)
    t.fit(D)
    before = catalog(con)
    for _ in range(50):
        t.transform(D)
    assert catalog(con) == before


def test_two_lazy_relations_from_one_artifact_stay_valid_together():
    con = duckdb.connect()
    t = lazy_on(con)
    first, second = t.transform(D), t.transform(D)
    assert first.fetchall() == [(2.0,), (4.0,), (6.0,)]
    assert second.limit(1).fetchall() == [(2.0,)]


def test_release_is_idempotent():
    con = duckdb.connect()
    t = lazy_on(con)
    t.transform(D)
    t.fitted_.release()
    t.fitted_.release()  # must not raise on the second pass


# ------------------------------------- R2: a leaf's refusal keeps its name


def test_a_foreign_transforms_own_refusal_reaches_the_caller():
    """TASK-71 wrapped a failing fit step in a named TransformError. It caught
    every duckdb.Error, and DuckDB rewraps a Python exception raised inside a
    UDF — so a leaf's own refusal was replaced by a message about correlated
    subqueries, which is both wrong and unactionable.

    ``_Registry`` carries the first real error precisely so a refusal keeps
    its name; the wrapper has to consult it before dressing anything up.
    """
    from sql_transform.model import Transform, TransformError

    def boom(_):
        raise TransformError("sc: the leaf's own words")

    sc = Transform(fit=boom, transform=lambda i, r: r, takes=("v",), returns=("v",))
    assert sc is not None
    t = SQLTransform(
        "SELECT sc_transform(f.theta, struct_pack(v := t.v)).v AS z "
        "FROM __THIS__ t, (SELECT sc_fit(struct_pack(v := v)) AS theta "
        "FROM __FIT__) f"
    )
    with pytest.raises(TransformError) as caught:
        t.fit(D)
    # Not merely *containing* the leaf's words: the wrapper embedded the whole
    # DuckDB error, so a substring match passed while the explanation on top
    # was wrong and unactionable.
    assert str(caught.value) == "sc: the leaf's own words"
    assert "does not stand on its own" not in str(caught.value)


def test_a_genuine_binding_error_still_gets_our_explanation():
    """The wrapper must keep doing its job for the case it was written for."""
    from sql_transform.model import TransformError

    t = SQLTransform(
        "SELECT t.cat, (SELECT avg(f.price) FROM __FIT__ f WHERE f.grp = cat) AS m "
        "FROM (SELECT grp AS cat, price FROM __THIS__) t"
    )
    with pytest.raises(TransformError, match="does not stand on its own"):
        t.fit(pa.table({"grp": ["a"], "price": [1.0]}))
