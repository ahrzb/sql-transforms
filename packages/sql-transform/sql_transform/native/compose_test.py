"""Pipelines beyond the catalog's generated fits: every refusal, by name,
and a pipeline that only passes its features through."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
import sklearn
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, check, to_native


def _step(est, X: np.ndarray, types: list[pa.DataType]) -> PythonTransform:
    est.fit(X, np.arange(len(X)) % 2)
    width = np.asarray(est.transform(X[:1])).reshape(1, -1).shape[1]
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(width)]),
    )


def _numbers(n: int = 20) -> np.ndarray:
    return np.random.default_rng(0).uniform(1.0, 9.0, size=(n, 2))


def _numeric(est) -> PythonTransform:
    return _step(est, _numbers(), [pa.float64(), pa.float64()])


class _MyPipeline(Pipeline):
    pass


@pytest.mark.parametrize(
    "make, reason",
    [
        (
            lambda: make_pipeline(StandardScaler(), PCA(n_components=1)),
            "step 'pca': no translation for PCA",
        ),
        (
            lambda: make_pipeline(
                PowerTransformer("box-cox", standardize=False), StandardScaler()
            ),
            "step 'powertransformer': PowerTransformer is within 4 ulps",
        ),
        (
            lambda: make_pipeline(
                MaxAbsScaler(), PowerTransformer("box-cox", standardize=False)
            ),
            "step 'powertransformer': PowerTransformer is within 4 ulps",
        ),
        (
            lambda: make_pipeline(MissingIndicator(features="all"), StandardScaler()),
            "step 'missingindicator': MissingIndicator is not the last step"
            " and its output is boolean",
        ),
        (
            lambda: Pipeline(
                [("inner", make_pipeline(StandardScaler(), PCA(n_components=1)))]
            ),
            "step 'inner': Pipeline step 'pca': no translation for PCA",
        ),
        (
            lambda: Pipeline(
                [
                    (
                        "inner",
                        make_pipeline(
                            StandardScaler(), MissingIndicator(features="all")
                        ),
                    ),
                    ("s", StandardScaler()),
                ]
            ),
            "step 'inner': Pipeline is not the last step and its output is boolean",
        ),
        (
            lambda: _MyPipeline([("s", StandardScaler())]),
            "no translation for _MyPipeline",
        ),
    ],
    ids=[
        "unknown",
        "bounded-first",
        "bounded-last",
        "boolean",
        "nested",
        "nested-boolean",
        "subclass",
    ],
)
def test_a_pipeline_names_the_step_it_refuses(make, reason):
    step = _numeric(make())
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)


def _strings() -> tuple[np.ndarray, list[pa.DataType]]:
    X = np.empty((8, 2), dtype=object)
    X[:, 0] = ["a", "b", "a", "c", "b", "a", "c", "b"]
    X[:, 1] = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0]
    return X, [pa.string(), pa.float64()]


@pytest.mark.parametrize(
    "make, reason",
    [
        (
            lambda: make_pipeline(
                OneHotEncoder(sparse_output=False, dtype=np.int64), MaxAbsScaler()
            ),
            "step 'onehotencoder': OneHotEncoder is not the last step and its"
            " output is int64",
        ),
        (
            lambda: make_pipeline(OneHotEncoder(), MaxAbsScaler()),
            "step 'onehotencoder': OneHotEncoder.sparse_output=True.: the output"
            " is sparse",
        ),
        (
            lambda: Pipeline([("skip", "passthrough")]),
            "passthrough steps over a string feature",
        ),
    ],
    ids=["int-output", "inner-refusal", "passthrough-strings"],
)
def test_a_pipeline_over_strings_names_what_it_refuses(make, reason):
    X, types = _strings()
    est = make()
    est.fit(X)
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([("f0", pa.float64()), ("f1", pa.float64())]),
    )
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)


def test_a_pipeline_with_transform_input_stays_python():
    with sklearn.config_context(enable_metadata_routing=True):
        step = _numeric(Pipeline([("s", StandardScaler())], transform_input=["X_val"]))
        with pytest.raises(NotNative, match=r"transform_input=\['X_val'\]"):
            to_native(step, strict=True)


def test_a_pipeline_over_booleans_only_stays_python():
    from sklearn.feature_selection import VarianceThreshold

    X = np.array([[True, False], [False, False], [True, True]])
    step = _step(
        make_pipeline(VarianceThreshold(), StandardScaler()),
        X,
        [pa.bool_(), pa.bool_()],
    )
    with pytest.raises(NotNative, match="boolean features only"):
        to_native(step, strict=True)


def test_an_int_output_last_is_served():
    # The step reads the last step's output with float(): its dtype is moot.
    X, types = _strings()
    est = make_pipeline(
        SimpleImputer(), OneHotEncoder(sparse_output=False, dtype=np.int64)
    )
    X = X[:, 1:].astype(float)
    step = _step(est, X, types[1:])
    rows = pa.table(
        {
            "__iid": pa.array([0, 0, None, 0], pa.int64()),
            "x0": pa.array([1.0, 3.0, 2.0, 2.0]),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 4


def test_a_pipeline_that_only_passes_serves_its_features():
    # `transform` hands back the row it was given; the step reads it with
    # float(), a NULL as NaN, a boolean as 0/1, -0.0 as itself.
    est = Pipeline([("a", "passthrough"), ("b", None)]).fit(_numbers())
    types = [pa.float64(), pa.int64(), pa.bool_()]
    step = PythonTransform(
        "tf",
        {0: est, 1: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(3)]),
    )
    rows = pa.table(
        {
            "__iid": pa.array([0, 1, None, 0], pa.int64()),
            "x0": pa.array([-0.0, None, 1.5, float("nan")]),
            "x1": pa.array([3, None, -7, 2**62], pa.int64()),
            "x2": pa.array([True, False, None, None]),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 4
