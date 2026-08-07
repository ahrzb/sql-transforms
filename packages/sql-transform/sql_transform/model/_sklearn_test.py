"""SQLTransform as an sklearn estimator.

``fit`` still returns the ``Fitted`` artifact rather than ``self`` — that is
the currying the model is built on, and it is what makes the artifact a thing
you can ship. Everything sklearn actually reaches for is here alongside it:
``fit_transform``, a stateful ``transform``, ``get_params``/``set_params`` so
``clone`` works, and ``set_output`` so a downstream estimator gets arrays.

Pipeline never consults what ``fit`` returned: it keeps the object it called
and asks it to ``transform`` later. Both spellings therefore agree, and the
first test pins that.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform.model import NotFitted, SQLTransform, run
from sql_transform.model._freezing_test import approx

D = pa.table({"store": ["S1", "S1", "S2", "S2"], "price": [10.0, 30.0, 100.0, 300.0]})
LIVE = pa.table({"store": ["S1", "S2"], "price": [20.0, 200.0]})

REL = """
SELECT round(t.price / s.m, 4) AS rel
FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s
"""


def test_every_spelling_agrees():
    """Curried, stateful and fit_transform are three views of one thing."""
    t = SQLTransform(REL)
    curried = approx(t(D).transform(D))
    stateful = approx(t.fit(D).transform(D))
    t2 = SQLTransform(REL)
    t2.fit(D)
    assert approx(t2.transform(D)) == curried
    assert approx(t2.fit_transform(D)) == curried
    assert approx(run(t, D)) == curried == stateful


def test_fit_transform_on_training_data_is_the_faithfulness_law():
    """``fit_transform(D)`` is exactly ``run(t, D)`` — not a coincidence."""
    t = SQLTransform(REL)
    assert approx(t.fit_transform(D)) == approx(run(t, D))


def test_transform_before_fit_refuses_by_name():
    with pytest.raises(NotFitted, match="fit"):
        SQLTransform(REL).transform(D)


def test_refitting_replaces_the_artifact():
    t = SQLTransform(REL)
    t.fit(D)
    first = approx(t.transform(LIVE))
    t.fit(pa.table({"store": ["S1"], "price": [1000.0]}))
    assert approx(t.transform(LIVE)) != first


# ------------------------------------------------------------------ pipelines


def test_it_goes_in_a_pipeline_with_a_real_estimator():
    frame = pd.DataFrame({"store": ["S1", "S1", "S2", "S2"], "price": D["price"]})
    pipe = make_pipeline(
        SQLTransform(REL).set_output(transform="pandas"),
        StandardScaler(),
        LinearRegression(),
    )
    pipe.fit(frame, [1.0, 2.0, 3.0, 4.0])
    assert pipe.predict(frame).shape == (4,)


def test_a_pipeline_of_two_sql_transforms():
    shift = SQLTransform("SELECT price - 5 AS price, store FROM __THIS__")
    pipe = Pipeline([("shift", shift), ("rel", SQLTransform(REL))])
    out = pipe.fit_transform(D)
    expected = SQLTransform(REL).fit_transform(shift.fit_transform(D))
    assert approx(out) == approx(expected)


def test_set_output_pandas_and_numpy():
    t = SQLTransform(REL)
    assert isinstance(t.fit_transform(D), pa.Table)
    assert isinstance(t.set_output(transform="pandas").fit_transform(D), pd.DataFrame)
    assert isinstance(t.set_output(transform="numpy").fit_transform(D), np.ndarray)
    assert isinstance(t.set_output(transform="default").fit_transform(D), pa.Table)


def test_feature_names_out():
    t = SQLTransform(REL)
    t.fit_transform(D)
    assert list(t.get_feature_names_out()) == ["rel"]


def test_feature_names_before_a_transform_refuse_by_name():
    t = SQLTransform(REL)
    t.fit(D)
    with pytest.raises(NotFitted, match="transform"):
        t.get_feature_names_out()


# ---------------------------------------------------------------------- clone


def test_clone_round_trips():
    t = SQLTransform(REL).set_output(transform="pandas")
    copy = clone(t)
    assert copy.sql == t.sql
    assert approx(pa.Table.from_pandas(copy.fit_transform(D))) == approx(
        pa.Table.from_pandas(t.fit_transform(D))
    )


def test_clone_keeps_names_that_the_frame_can_no_longer_resolve():
    """``clone`` rebuilds from ``get_params`` in sklearn's frame, where a
    member or a lookup table is not in scope. They ride along as parameters,
    so the copy resolves to the very same objects."""
    REGION = pa.table({"store": ["S1", "S2"], "region": ["n", "s"]})
    assert REGION is not None
    inner = SQLTransform("SELECT price * 2 AS price, store FROM __THIS__")
    assert inner is not None
    outer = SQLTransform("""
        SELECT r.region, sum(x.price) AS total
        FROM inner(__FIT__, __THIS__) x JOIN REGION r ON x.store = r.store
        GROUP BY r.region ORDER BY r.region
    """)
    copy = clone(outer)
    assert copy.captured.keys() == outer.captured.keys()
    assert approx(copy.fit_transform(D)) == approx(outer.fit_transform(D))


def test_get_params_and_set_params():
    t = SQLTransform(REL)
    assert set(t.get_params()) == {"sql", "output", "connection", "captured"}
    t.set_params(output="pandas")
    assert isinstance(t.fit_transform(D), pd.DataFrame)
