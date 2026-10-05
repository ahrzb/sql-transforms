"""Pipelines beyond the catalog's generated fits: every refusal, by name,
and a pipeline that only passes its features through."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pyarrow as pa
import pytest
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
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
            lambda: make_pipeline(FunctionTransformer(np.exp), StandardScaler()),
            "step 'functiontransformer': FunctionTransformer is within 1 ulps",
        ),
        (
            lambda: make_pipeline(StandardScaler(), FunctionTransformer(np.log2)),
            "step 'functiontransformer': FunctionTransformer is within 1 ulps",
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
        "bounded-function-first",
        "bounded-function-last",
        "boolean",
        "nested",
        "nested-boolean",
        "subclass",
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
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


# ------------------------------------------- ColumnTransformer, FeatureUnion


def _wide(n: int = 4) -> np.ndarray:
    return np.random.default_rng(1).uniform(1.0, 9.0, size=(20, n))


def _composed(est, X: np.ndarray | None = None, types=None) -> PythonTransform:
    X = _wide() if X is None else X
    return _step(est, X, types or [pa.float64()] * X.shape[1])


class _MyScaler(StandardScaler):
    pass


@pytest.mark.parametrize(
    "make, reason",
    [
        (
            lambda: ColumnTransformer([("p", PCA(n_components=1), [0, 1])]),
            "ColumnTransformer part 'p': no translation for PCA",
        ),
        (
            lambda: ColumnTransformer([("s", _MyScaler(), [0])]),
            "part 's': no translation for _MyScaler",
        ),
        (
            lambda: ColumnTransformer(
                [("pow", PowerTransformer("box-cox", standardize=False), [0])]
            ),
            "part 'pow': PowerTransformer is within 4 ulps",
        ),
        (
            lambda: ColumnTransformer(
                [("log", FunctionTransformer(np.log10, validate=True), [0])],
                remainder="passthrough",
            ),
            "part 'log': FunctionTransformer is within 2 ulps",
        ),
        (
            lambda: ColumnTransformer(
                [("s", StandardScaler(), [0])], remainder=PCA(n_components=1)
            ),
            "part 'remainder': no translation for PCA",
        ),
        (
            lambda: ColumnTransformer(
                [("f", FunctionTransformer(np.sqrt), [0])], remainder="passthrough"
            ),
            r"part 'f': a FunctionTransformer\(validate=False\) applies its func"
            " to objects",
        ),
        (
            lambda: ColumnTransformer(
                [
                    (
                        "f",
                        make_pipeline(FunctionTransformer(np.abs), StandardScaler()),
                        [0],
                    )
                ]
            ),
            r"part 'f': a FunctionTransformer\(validate=False\)",
        ),
        (
            lambda: ColumnTransformer(
                [("s", StandardScaler(), [0])], sparse_threshold=1.0
            ).set_output(transform="pandas"),
            r"set_output\(transform='pandas'\)",
        ),
        (
            lambda: ColumnTransformer(
                [("s", OneHotEncoder(), [0])], sparse_threshold=1.0
            ),
            r"the output is sparse \(sparse_output_\)",
        ),
        (
            lambda: ColumnTransformer(
                [("s", OneHotEncoder(dtype=np.float32, sparse_output=False), [0])],
                transformer_weights={"s": 0.5},
            ),
            "part 's': a weight on a float32 output",
        ),
        (
            lambda: ColumnTransformer(
                [("s", MissingIndicator(features="all"), [0])],
                transformer_weights={"s": -2},
            ),
            "part 's': an integer weight on an output that may not be float64"
            r" \(its output is boolean\)",
        ),
        (
            lambda: ColumnTransformer(
                [("s", StandardScaler(), [0])], transformer_weights={"s": 2**60 + 1}
            ),
            "part 's': weight 1152921504606846977 is not a double",
        ),
        (
            lambda: make_pipeline(
                ColumnTransformer(
                    [("s", StandardScaler(), [0])], remainder="passthrough"
                ),
                StandardScaler(),
            ),
            "step 'columntransformer': ColumnTransformer is not the last step and it"
            " passes columns through as objects",
        ),
        (
            lambda: FeatureUnion(
                [("pass", "passthrough"), ("tan", FunctionTransformer(np.tan))]
            ),
            "FeatureUnion part 'tan': FunctionTransformer is within 1 ulps",
        ),
        (
            lambda: FeatureUnion([("p", PCA(n_components=1))]),
            "FeatureUnion part 'p': no translation for PCA",
        ),
        (
            lambda: FeatureUnion(
                [("s", StandardScaler()), ("p", "passthrough")],
                transformer_weights={"p": 0.5},
            ),
            "FeatureUnion part 'p': a weighted passthrough",
        ),
        (
            lambda: FeatureUnion(
                [("s", StandardScaler())], transformer_weights={"s": Fraction(1, 3)}
            ),
            r"part 's': weight Fraction\(1, 3\) is not a double",
        ),
        (
            lambda: FeatureUnion(
                [("s", make_pipeline(StandardScaler(), PCA(n_components=1)))]
            ),
            "FeatureUnion part 's': Pipeline step 'pca': no translation for PCA",
        ),
        (
            lambda: FeatureUnion([("s", StandardScaler())]).set_output(
                transform="pandas"
            ),
            r"FeatureUnion.set_output\(transform='pandas'\)",
        ),
    ],
    ids=[
        "ct-unknown",
        "ct-subclass",
        "ct-bounded",
        "ct-bounded-function",
        "ct-remainder",
        "ct-object-func",
        "ct-object-func-nested",
        "ct-container",
        "ct-sparse",
        "ct-float32-weight",
        "ct-integer-weight",
        "ct-inexact-weight",
        "ct-not-last",
        "fu-bounded-function",
        "fu-unknown",
        "fu-weighted-passthrough",
        "fu-weight-type",
        "fu-nested",
        "fu-container",
    ],
)
def test_a_composition_names_the_part_it_refuses(make, reason):
    step = _composed(make())
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)


def test_a_column_transformer_fitted_on_names_stays_python():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(_wide(2), columns=["a", "b"])
    est = ColumnTransformer([("s", StandardScaler(), ["a"])]).fit(X)
    step = PythonTransform(
        "tf", {0: est}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())])
    )
    with pytest.raises(NotNative, match=r"part 's': columns \['a'\] are names"):
        to_native(step, strict=True)


@pytest.mark.parametrize(
    "make",
    [
        lambda: ColumnTransformer(
            [("num", StandardScaler(), [1])], remainder="passthrough"
        ),
        lambda: FeatureUnion([("p", "passthrough")]),
    ],
    ids=["ColumnTransformer", "FeatureUnion"],
)
def test_a_string_passed_through_stays_python(make):
    X, types = _strings()
    est = make().fit(X)
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(len(types))]),
    )
    with pytest.raises(NotNative, match="passes a string column through"):
        to_native(step, strict=True)


def test_a_column_transformer_reads_columns_as_its_twin_does():
    # Reordered indices, a negative one, a slice, a boolean mask, an empty
    # selection (skipped, as sklearn skips it), a weight of -0.0 and the
    # remainder through; served on signed zeros, NaN and a boolean.
    est = ColumnTransformer(
        [
            ("rev", "passthrough", [3, -4]),
            ("sl", StandardScaler(), slice(1, 3)),
            ("mask", MinMaxScaler(), np.array([False, True, False, True])),
            ("none", StandardScaler(), []),
        ],
        remainder="passthrough",
        transformer_weights={"sl": -0.0, "rev": 0.7},
    )
    step = _composed(est, types=[pa.float64(), pa.float64(), pa.int64(), pa.bool_()])
    rows = pa.table(
        {
            "__iid": pa.array([0, 0, None, 0], pa.int64()),
            "x0": pa.array([-0.0, None, 1.5, float("nan")]),
            "x1": pa.array([3.0, -1e300, 2.0, 0.0]),
            "x2": pa.array([3, None, -7, 2**40], pa.int64()),
            "x3": pa.array([True, False, None, True]),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 4


def test_an_exact_function_composes_where_a_bounded_one_does_not():
    # The part's own bound decides, not its class's ceiling.
    exact = FeatureUnion(
        [("id", FunctionTransformer()), ("sqrt", FunctionTransformer(np.sqrt))]
    )
    step = _numeric(make_pipeline(exact, StandardScaler()))
    rows = pa.table(
        {
            "__iid": pa.array([0, 0, None], pa.int64()),
            "x0": pa.array([4.0, -0.0, 1.5]),
            "x1": pa.array([2.0, float("nan"), 9.0]),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 3
    bounded = FeatureUnion(
        [("id", FunctionTransformer()), ("exp", FunctionTransformer(np.exp))]
    )
    with pytest.raises(
        NotNative, match="part 'exp': FunctionTransformer is within 1 ulps"
    ):
        to_native(_numeric(bounded), strict=True)
