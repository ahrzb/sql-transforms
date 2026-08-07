"""Slice 3 gates: foreign transforms — the Python pair, θ handles, instances.

An sklearn transformer is already the pair: ``x_fit`` is the UDAF half,
``x_transform`` the UDF half. θ is an opaque handle into a registry of fitted
estimators, so a `Fitted` carries ``instances`` alongside ``params``: an SQL
leaf gives an inspectable params table, a fitted RandomForest gives a pointer.
"""

import sys
import threading

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from sql_transform.model import SQLTransform, Transform, TransformError, run
from sql_transform.model._freezing_test import approx, reads_fit, rows
from sql_transform.model._transform import _Registry

D = pa.table(
    {
        "cat": ["a", "a", "b", "b"],
        "price": [1.0, 3.0, 10.0, 30.0],
    }
)
UNSEEN = pa.table({"cat": ["a", "zz"], "price": [2.0, 99.0]})

# The pair supplied directly, as the spec writes it: fit sees a relation and
# returns whatever it likes; transform sees that and a relation.
sc = Transform(
    fit=lambda f: pa.table({"m": [pc.mean(f["v"]).as_py()]}),
    transform=lambda p, t: pa.table({"v": pc.divide(t["v"], p["m"][0])}),
    takes=("v",),
    returns=("v",),
)

PER_CAT = """
SELECT t.cat, sc_transform(f.theta, struct_pack(v := t.price)).v AS z
FROM __THIS__ t
LEFT JOIN (SELECT cat, sc_fit(struct_pack(v := price)) AS theta
           FROM __FIT__ GROUP BY cat) f
  ON t.cat = f.cat
ORDER BY t.cat, z
"""


def test_the_pair_fits_and_serves():
    """cat a: mean 2 -> [0.5, 1.5]; cat b: mean 20 -> [0.5, 1.5]."""
    fitted = SQLTransform(PER_CAT).fit(D)
    assert approx(fitted(D), 4) == [
        ("a", 0.5),
        ("a", 1.5),
        ("b", 0.5),
        ("b", 1.5),
    ]


def test_leaves_need_no_special_case():
    """An sklearn leaf freezes by the same rule as SQL: nothing bespoke."""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    ss = Transform.from_estimator(StandardScaler(), takes=("v",), returns=("v",))
    assert ss is not None
    sql = PER_CAT.replace("sc_", "ss_")
    t = SQLTransform(sql)

    fitted = t.fit(D)
    assert not reads_fit(fitted.sql)
    assert len(fitted.params) == 1  # the theta table, one row per category
    assert approx(fitted.transform(D), 4) == approx(run(t, D), 4)
    assert approx(fitted.transform(D), 4) == [
        ("a", -1.0),
        ("a", 1.0),
        ("b", -1.0),
        ("b", 1.0),
    ]


def test_theta_is_an_opaque_handle():
    """``Struct<type, id>`` into ``instances`` — not the estimator itself."""
    fitted = SQLTransform(PER_CAT).fit(D)
    (theta_table,) = fitted.params.values()
    thetas = theta_table.column("theta").to_pylist()
    assert sorted(t["type"] for t in thetas) == ["sc", "sc"]
    assert len(fitted.instances) == 2  # one per category, not one shared
    assert sorted(t["id"] for t in thetas) == sorted(fitted.instances)


def test_ids_are_unique_under_concurrency():
    """θ ids come from a counter under a lock, not from ``len(instances)``.

    DuckDB fits groups on several threads. A read-then-write id would let two
    of them mint the same handle, and the loser's rows would be served by the
    winner's estimator with no error at all. Gated here rather than through
    the SQL path, because an interleaving is not something a test can demand.
    """
    registry = _Registry()
    ids: list[int] = []
    barrier = threading.Barrier(8)

    def mint():
        barrier.wait()
        ids.extend(registry.add(object()) for _ in range(20))

    # Without this the whole loop finishes inside one GIL slice and a
    # read-then-write counter never collides — the gate would pass on the
    # broken version, which is the only failure mode that matters here.
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=mint) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(previous)

    assert len(ids) == 160
    assert len(set(ids)) == 160
    assert len(registry.instances) == 160


def test_the_one_null_story():
    """Unseen key ⇒ join miss ⇒ NULL θ ⇒ NULL output, never a dropped row (P14)."""
    fitted = SQLTransform(PER_CAT).fit(D)
    out = rows(fitted(UNSEEN))
    assert out == [("a", 1.0), ("zz", None)]


def test_broken_artifacts_raise():
    """A θ id present but absent from ``instances`` raises, naming it."""
    fitted = SQLTransform(PER_CAT).fit(D)
    missing = next(iter(fitted.instances))
    del fitted.instances[missing]
    with pytest.raises(TransformError, match=str(missing)):
        fitted(D)


def test_transductive_fit_over_this_is_legal():
    """``sc_fit`` over ``__THIS__`` means what it says: refit on this batch.

    Under this design you cannot write it by accident, because you had to type
    ``__THIS__``.
    """
    sql = PER_CAT.replace("FROM __FIT__ GROUP BY cat", "FROM __THIS__ GROUP BY cat")
    fitted = SQLTransform(sql).fit(D)
    assert fitted.params == {}  # nothing was learned at fit

    # Each batch is normalized by its own statistics, so the same row differs.
    first = pa.table({"cat": ["a", "a"], "price": [1.0, 3.0]})
    second = pa.table({"cat": ["a", "a"], "price": [1.0, 9.0]})
    assert approx(fitted(first), 4) == [("a", 0.5), ("a", 1.5)]
    assert approx(fitted(second), 4) == [("a", 0.2), ("a", 1.8)]


def test_a_transductive_refit_does_not_leak_between_calls():
    """Instances minted while serving are per-call, not accumulated forever."""
    sql = PER_CAT.replace("FROM __FIT__ GROUP BY cat", "FROM __THIS__ GROUP BY cat")
    fitted = SQLTransform(sql).fit(D)
    fitted(D)
    fitted(D)
    assert fitted.instances == {}


def test_a_foreign_transform_declares_its_struct():
    """DuckDB has no ANY type, so the shape has to be declared, and checked."""
    with pytest.raises(TransformError, match="width"):
        wrong = Transform(
            fit=lambda f: None,
            transform=lambda p, t: pa.table({"v": t["v"], "extra": t["v"]}),
            takes=("v",),
            returns=("v",),
        )
        assert wrong is not None
        SQLTransform(
            "SELECT wrong_transform(w.theta, struct_pack(v := price)).v AS z "
            "FROM __THIS__, (SELECT wrong_fit(struct_pack(v := price)) AS theta "
            "FROM __FIT__) w"
        ).fit(D)(D)


def test_an_unknown_scalar_function_refuses():
    from sql_transform.model import UnknownName  # noqa: PLC0415

    with pytest.raises(UnknownName, match="nope_transform"):
        SQLTransform("SELECT nope_transform(price) AS z FROM __THIS__")
