"""TASK-70: the sklearn boundary — index alignment, cloning, pickling.

Three independent failures where the model meets sklearn's conventions.

The silent one is the index. ``_as_output`` returned ``table.to_pandas()``,
which mints a fresh ``RangeIndex``. sklearn's own transformers carry the
caller's index through, so in a ``FeatureUnion`` pandas aligned on index and
NaN-padded the mismatch: four rows in, seven out, no error.

The other two are loud but they close doors the guide says are open —
``clone`` is how every meta-estimator copies its steps, and it deep-copied a
live connection.
"""

import pickle

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from sql_transform.model import SQLTransform, Transform

SMALL = pa.table({"v": [1.0, 2.0, 3.0]})
REL = (
    "SELECT round(t.v / s.m, 4) AS z FROM __THIS__ t, (SELECT avg(v) m FROM __FIT__) s"
)

# A deliberately non-default index: the bug is invisible under RangeIndex.
FRAME = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=[7, 3, 11, 5])


# ------------------------------------------------------------- the index (F4)


def test_pandas_output_carries_the_callers_index():
    t = SQLTransform(REL).set_output(transform="pandas")
    assert list(t.fit_transform(FRAME).index) == [7, 3, 11, 5]


def test_a_feature_union_does_not_grow_rows():
    """The silent one, in the shape it actually bites.

    ``StandardScaler`` preserves the caller's index; ours reset it. pandas
    then aligned the two on index rather than position and padded with NaN.
    """
    union = FeatureUnion([("sql", SQLTransform(REL)), ("sc", StandardScaler())])
    union.set_output(transform="pandas")
    out = union.fit_transform(FRAME)
    assert len(out) == len(FRAME)
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_an_arrow_input_still_gets_a_default_index():
    """Nothing to carry: an arrow table has no index, so none is invented."""
    t = SQLTransform(REL).set_output(transform="pandas")
    assert list(t.fit_transform(SMALL).index) == [0, 1, 2]


def test_a_transform_that_changes_cardinality_does_not_fake_an_index():
    """A SQL transform may aggregate, unlike an sklearn one. When the row
    count does not match there is no correspondence to express, so the
    caller's index is not attached rather than guessed at."""
    t = SQLTransform("SELECT sum(t.v) AS total FROM __THIS__ t").set_output(
        transform="pandas"
    )
    out = t.fit_transform(FRAME)
    assert len(out) == 1
    assert list(out.index) == [0]


# ------------------------------------------------------------- cloning (F11)


def test_clone_survives_a_shared_connection():
    con = duckdb.connect()
    t = SQLTransform(REL, connection=con)
    copy = clone(t)
    assert copy is not t
    assert copy.connection is con  # shared, not copied: it is a resource


def test_cross_val_score_runs_with_a_shared_connection():
    con = duckdb.connect()
    frame = pd.DataFrame({"v": np.arange(12.0)})
    target = np.arange(12.0)
    pipe = make_pipeline(
        SQLTransform(REL, connection=con).set_output(transform="pandas"),
        StandardScaler(),
        LinearRegression(),  # or the scores are NaN and prove nothing
    )
    scores = cross_val_score(pipe, frame, target, cv=3, scoring="r2")
    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_clone_still_carries_captured_objects():
    codes = pa.table({"k": ["a"], "mul": [10.0]})
    assert codes is not None
    t = SQLTransform("SELECT t.v * c.mul AS z FROM __THIS__ t, codes c")
    copy = clone(t)
    assert copy.captured["codes"] is codes


# ------------------------------------------------------------- pickling (F12)


def test_a_from_estimator_leaf_pickles():
    """``Transform.from_estimator`` built its two halves as closures, which are
    unpicklable — ``deepcopy`` treats functions as atomic so ``clone``
    survived, but anything that actually serialises did not."""
    leaf = Transform.from_estimator(StandardScaler(), takes=("v",), returns=("v",))
    assert pickle.loads(pickle.dumps(leaf)).takes == ("v",)  # noqa: S301 — our own objects, round-tripped


def test_a_fitted_transform_with_a_foreign_leaf_pickles():
    sc = Transform.from_estimator(StandardScaler(), takes=("v",), returns=("v",))
    assert sc is not None
    t = SQLTransform(
        "SELECT sc_transform(f.theta, struct_pack(v := t.v)).v AS z "
        "FROM __THIS__ t, (SELECT sc_fit(struct_pack(v := v)) AS theta "
        "FROM __FIT__) f"
    )
    fitted = t.fit(SMALL)
    assert pickle.loads(pickle.dumps(fitted.params)) is not None  # noqa: S301 — our own objects, round-tripped
    assert pickle.loads(pickle.dumps(sc)).returns == ("v",)  # noqa: S301 — our own objects, round-tripped


def test_the_estimator_is_still_cloned_per_fit():
    """Making the halves picklable must not accidentally share learned state
    between groups — that was the reason they were built per call."""
    leaf = Transform.from_estimator(StandardScaler(), takes=("v",), returns=("v",))
    one = leaf.fit(pa.table({"v": [1.0, 2.0, 3.0]}))
    two = leaf.fit(pa.table({"v": [100.0, 200.0, 300.0]}))
    assert one is not two
    assert one.mean_[0] != two.mean_[0]


def test_a_pickled_leaf_still_transforms():
    leaf = Transform.from_estimator(StandardScaler(), takes=("v",), returns=("v",))
    instance = leaf.fit(pa.table({"v": [1.0, 2.0, 3.0]}))
    revived = pickle.loads(pickle.dumps(leaf))  # noqa: S301 — our own objects, round-tripped
    out = revived.transform(instance, pa.table({"v": [2.0]}))
    assert out.column_names == ["v"]
    assert out["v"][0].as_py() == pytest.approx(0.0)
