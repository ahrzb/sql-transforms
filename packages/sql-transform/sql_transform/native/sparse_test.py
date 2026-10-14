"""The catalog serves a sparse output as the step reads it, dense
(loops/native/decisions/closed/sparse-outputs.md, the ruling). Each
configuration is checked against its twin under both of sklearn's sparse
containers, on rows the twin answers, so that `check` compares every row;
each sparse configuration that stays Python names why."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import SelectKBest
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    Normalizer,
    OneHotEncoder,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)

from sql_transform import _sparse_outputs_test
from sql_transform._udf import PythonTransform, _dense
from sql_transform.native import NotNative, check, to_native

# Both of sklearn's sparse containers (a fixture), and their classes.
interface = _sparse_outputs_test.interface
CONTAINERS = _sparse_outputs_test.CONTAINERS

# Two string features (the second of two categories) and two numbers,
# zeros of both signs among them.
CATS = ["a", "b", "c", "a", "b", "a", "c", "b", "a", "c", "b", "a"]
SIDES = ["l", "r", "r", "l", "l", "r", "l", "r", "r", "l", "l", "r"]


def _numbers() -> np.ndarray:
    X = np.random.default_rng(0).uniform(-5.0, 5.0, size=(len(CATS), 2))
    X[:3] = [[0.0, 1.0], [-0.0, -1.0], [2.5, 0.0]]
    return X


def _strings() -> np.ndarray:
    return np.array([CATS, SIDES], dtype=object).T


def _mixed() -> np.ndarray:
    X = np.empty((len(CATS), 4), dtype=object)
    X[:, :2] = _strings()
    X[:, 2:] = _numbers()
    return X


def _holes() -> np.ndarray:
    X = _numbers()
    X[[1, 4], 0] = np.nan
    return X


NUMBERS = [
    (0.0, -0.0),
    (-0.0, 0.0),
    (1.5, -2.25),
    (4.75, 3.0),
    (-4.9, 4.99),
    (7.5, -9.0),
    (-12.0, 30.0),
    (2.5, 1.0),
]
STRINGS = [("a", "l"), ("b", "r"), ("c", "l"), ("a", "r"), ("c", "r")]
UNSEEN = [*STRINGS, ("zz", "l"), ("a", ""), ("", "x")]
MIXED = [
    ("a", "l", 0.0, -0.0),
    ("b", "r", 1.5, 0.0),
    ("c", "l", -2.25, 4.0),
    ("a", "r", -0.0, 0.0),
    ("b", "l", 9.0, -7.5),
]
NULLS = [*NUMBERS, (None, 1.0), (None, -0.0)]

F64, STR = pa.float64(), pa.string()


def _step(est: Any, X: np.ndarray, types: list[pa.DataType]) -> PythonTransform:
    est.fit(X, np.arange(len(X)) % 2)
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        _returns(est, X),
    )


def _returns(est: Any, X: np.ndarray) -> pa.DataType:
    try:
        width = np.asarray(_dense(est.transform(X[:1]))).reshape(1, -1).shape[1]
    except ValueError:  # the twin raises (on every row, for some refused)
        width = len(est.get_feature_names_out())
    if width == 1:
        return pa.float64()
    return pa.struct([(f"f{j}", pa.float64()) for j in range(width)])


def _rows(step: PythonTransform, values: list[tuple]) -> pa.Table:
    cols = {"__iid": pa.array([0] * len(values), pa.int64())}
    for i, f in enumerate(step.takes):
        cols[f.name] = pa.array([v[i] for v in values], f.type)
    return pa.table(cols)


# (make, fit data, feature types, serving rows): every row answered.
SERVED: dict[str, tuple[Callable[[], Any], Callable[[], np.ndarray], list, list]] = {
    "OneHotEncoder": (OneHotEncoder, _strings, [STR, STR], STRINGS),
    "OneHotEncoder-unknown": (
        lambda: OneHotEncoder(handle_unknown="ignore"),
        _strings,
        [STR, STR],
        UNSEEN,
    ),
    "OneHotEncoder-one-field": (
        lambda: OneHotEncoder(drop="if_binary"),
        lambda: _strings()[:, 1:],
        [STR],
        [(s,) for _, s in STRINGS],
    ),
    "KBinsDiscretizer": (
        lambda: KBinsDiscretizer(n_bins=3),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "KBinsDiscretizer-float32": (
        lambda: KBinsDiscretizer(n_bins=4, strategy="uniform", dtype=np.float32),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "MissingIndicator": (
        lambda: MissingIndicator(sparse=True),
        _holes,
        [F64, F64],
        NULLS,
    ),
    "MissingIndicator-all": (
        lambda: MissingIndicator(features="all", sparse=True),
        _holes,
        [F64, F64],
        NULLS,
    ),
    "RandomTreesEmbedding": (
        lambda: RandomTreesEmbedding(n_estimators=3, max_depth=3, random_state=0),
        _holes,
        [F64, F64],
        NULLS,
    ),
    "SplineTransformer": (
        lambda: SplineTransformer(sparse_output=True),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "SplineTransformer-periodic": (
        lambda: SplineTransformer(
            degree=2, n_knots=4, extrapolation="periodic", sparse_output=True
        ),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "SplineTransformer-linear": (
        lambda: SplineTransformer(degree=1, extrapolation="linear", sparse_output=True),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    # Past 1e154 the dense twin's `0.0 * h` terms overflow to NaN; the
    # sparse one keeps the basis value.
    "SplineTransformer-continue": (
        lambda: SplineTransformer(extrapolation="continue", sparse_output=True),
        _numbers,
        [F64, F64],
        [*NUMBERS, (1e155, -1e155), (-1e300, 1e300)],
    ),
    # A knot gap under 1e-300 overflows `1 / gap`: the dense twin's
    # `0.0 * h` terms answer NaN there, and the sparse twin answers h.
    "SplineTransformer-tiny-gap": (
        lambda: SplineTransformer(
            knots=np.array([[0.0], [1e-310], [1.0], [2.0], [3.0]]),
            degree=2,
            extrapolation="continue",
            sparse_output=True,
        ),
        lambda: np.linspace(0.0, 3.0, 10)[:, None],
        [F64],
        [(v,) for v in (0.0, 5e-311, 1e-310, 2e-310, 1e-300, 0.5, 2.5, -1.0, 4.0)],
    ),
    "SplineTransformer-degree-0": (
        lambda: SplineTransformer(
            degree=0, n_knots=3, extrapolation="continue", sparse_output=True
        ),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "SplineTransformer-zeros": (
        lambda: SplineTransformer(handle_missing="zeros", sparse_output=True),
        lambda: _holes()[:, :1],
        [F64],
        [(v,) for v, _ in NULLS],
    ),
    "Pipeline-scaled": (
        lambda: make_pipeline(OneHotEncoder(), MaxAbsScaler()),
        _strings,
        [STR, STR],
        STRINGS,
    ),
    "Pipeline-selected": (
        lambda: make_pipeline(
            OneHotEncoder(handle_unknown="ignore"), SelectKBest(k=3), Normalizer()
        ),
        _strings,
        [STR, STR],
        UNSEEN,
    ),
    "Pipeline-binned": (
        lambda: make_pipeline(
            KBinsDiscretizer(n_bins=3),
            Binarizer(threshold=0.5),
            PolynomialFeatures(),
            RobustScaler(with_centering=False),
        ),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "Pipeline-standardized": (
        lambda: make_pipeline(
            OneHotEncoder(), FunctionTransformer(), StandardScaler(with_mean=False)
        ),
        _strings,
        [STR, STR],
        STRINGS,
    ),
    "Pipeline-embedded": (
        lambda: make_pipeline(
            OneHotEncoder(), RandomTreesEmbedding(n_estimators=2, random_state=0)
        ),
        _strings,
        [STR, STR],
        STRINGS,
    ),
    "Pipeline-indicated": (
        lambda: make_pipeline(
            OneHotEncoder(), MissingIndicator(missing_values=1.0, features="all")
        ),
        _strings,
        [STR, STR],
        STRINGS,
    ),
    # A stored -0.0, which the step reads as 0.0: a sparse pipeline's
    # output fields add 0.0.
    "Pipeline-negative-zero": (
        lambda: make_pipeline(
            OneHotEncoder(),
            SimpleImputer(
                missing_values=1.0,
                strategy="constant",
                fill_value=-0.0,
                keep_empty_features=True,
            ),
        ),
        _strings,
        [STR, STR],
        STRINGS,
    ),
    "Pipeline-spline-last": (
        lambda: make_pipeline(StandardScaler(), SplineTransformer(sparse_output=True)),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    # A sparse stack drops a -0.0 (`np.negative` of 0.0).
    "ColumnTransformer": (
        lambda: ColumnTransformer(
            [
                ("enc", OneHotEncoder(), [0, 1]),
                ("neg", FunctionTransformer(np.negative, validate=True), [2, 3]),
            ],
            sparse_threshold=1.0,
        ),
        _mixed,
        [STR, STR, F64, F64],
        MIXED,
    ),
    # A weighted sparse part scales its stored values alone: 0 * -2.0 is
    # 0.0 in the dense stack.
    "ColumnTransformer-weighted": (
        lambda: ColumnTransformer(
            [
                ("enc", OneHotEncoder(handle_unknown="ignore"), [0, 1]),
                ("num", StandardScaler(), [2, 3]),
            ],
            sparse_threshold=0.0,
            transformer_weights={"enc": -2.0},
        ),
        _mixed,
        [STR, STR, F64, F64],
        MIXED,
    ),
    "FeatureUnion": (
        lambda: FeatureUnion(
            [
                ("bins", KBinsDiscretizer(n_bins=3)),
                ("neg", FunctionTransformer(np.negative)),
            ]
        ),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
    "FeatureUnion-weighted": (
        lambda: FeatureUnion(
            [
                ("bins", KBinsDiscretizer(n_bins=3, strategy="uniform")),
                ("std", StandardScaler()),
            ],
            transformer_weights={"bins": -1.0},
        ),
        _numbers,
        [F64, F64],
        NUMBERS,
    ),
}


def _sparse_in(est: Any, X: np.ndarray, interface: str) -> bool:
    """`est`'s output is the container of `interface`, or, in a dense
    stack, a part's."""
    if isinstance(est.transform(X[:1]), CONTAINERS[interface]):
        return True
    return any(
        isinstance(part.transform(X[:1, cols]), CONTAINERS[interface])
        for _, part, cols in getattr(est, "transformers_", [])
        if hasattr(part, "transform")
    )


@pytest.mark.parametrize("name", list(SERVED))
def test_a_sparse_output_is_served_on_every_row(interface, name):
    make, data, types, values = SERVED[name]
    X = data()
    step = _step(make(), X, types)
    assert _sparse_in(step.instances[0], X, interface)
    rows = _rows(step, values)
    assert check(step, to_native(step, strict=True), rows) == rows.num_rows


def _two_features() -> np.ndarray:
    return _numbers()


def _nan_column() -> np.ndarray:
    X = _numbers()[:, :1].copy()
    X[:] = np.nan
    return X


REFUSED = {
    "spline-degree-0-constant": (
        lambda: SplineTransformer(
            degree=0, extrapolation="constant", sparse_output=True
        ),
        _numbers,
        [F64, F64],
        "above the knots the twin answers the basis at the last knot",
    ),
    "spline-degree-0-periodic": (
        lambda: SplineTransformer(
            degree=0, n_knots=3, extrapolation="periodic", sparse_output=True
        ),
        _numbers,
        [F64, F64],
        "extrapolation='periodic'.: the twin raises on every row",
    ),
    "spline-zeros-over-two": (
        lambda: SplineTransformer(
            extrapolation="continue", handle_missing="zeros", sparse_output=True
        ),
        _holes,
        [F64, F64],
        "answers 0.0 for a NaN in the first feature only",
    ),
    "spline-nan-knots": (
        lambda: SplineTransformer(
            knots="quantile", handle_missing="zeros", sparse_output=True
        ),
        _nan_column,
        [F64],
        "feature 0's knots are NaN, where scipy's design matrix raises",
    ),
    "pipeline-quantiles": (
        lambda: make_pipeline(OneHotEncoder(), QuantileTransformer(n_quantiles=5)),
        _strings,
        [STR, STR],
        "step 'quantiletransformer': QuantileTransformer is handed a sparse"
        " matrix, which it is not known to read as dense",
    ),
    "pipeline-function": (
        lambda: make_pipeline(OneHotEncoder(), FunctionTransformer(np.negative)),
        _strings,
        [STR, STR],
        "step 'functiontransformer': FunctionTransformer is handed a sparse"
        " matrix, to which it applies its func",
    ),
    "pipeline-negative-threshold": (
        lambda: make_pipeline(OneHotEncoder(), Binarizer(threshold=-1.0)),
        _strings,
        [STR, STR],
        "step 'binarizer': Binarizer is handed a sparse matrix, on which a"
        " negative threshold raises",
    ),
    "pipeline-spline-first": (
        lambda: make_pipeline(SplineTransformer(sparse_output=True), MaxAbsScaler()),
        _numbers,
        [F64, F64],
        "step 'splinetransformer': SplineTransformer is not the last step and"
        " hands on a sparse matrix of values other than 0 and 1",
    ),
    "pipeline-scaled-twice": (
        lambda: make_pipeline(
            OneHotEncoder(), MaxAbsScaler(), StandardScaler(with_mean=False)
        ),
        _strings,
        [STR, STR],
        "step 'maxabsscaler': MaxAbsScaler is not the last step and hands on",
    ),
    # Its fill of 49.0 then scaled parts by an ulp in 36 of 40 rows.
    "pipeline-filled": (
        lambda: make_pipeline(
            OneHotEncoder(),
            SimpleImputer(
                missing_values=1.0,
                strategy="constant",
                fill_value=49.0,
                keep_empty_features=True,
            ),
            StandardScaler(with_mean=False),
        ),
        _strings,
        [STR, STR],
        "step 'simpleimputer': SimpleImputer is not the last step and hands on",
    ),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_a_sparse_output_that_stays_python_says_why(name):
    make, data, types, reason = REFUSED[name]
    step = _step(make(), data(), types)
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)
