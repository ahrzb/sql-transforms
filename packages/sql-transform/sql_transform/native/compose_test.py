"""Pipelines beyond the catalog's generated fits: every refusal, by name,
and a pipeline that only passes its features through."""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np
import pyarrow as pa
import pytest
import sklearn
from sklearn.cluster import FeatureAgglomeration, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, KernelPCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sql_transform._udf import PythonTransform, _dense
from sql_transform.native import (
    NotNative,
    ParityError,
    check,
    explain_native,
    to_native,
)


def _step(est, X: np.ndarray, types: list[pa.DataType]) -> PythonTransform:
    est.fit(X, np.arange(len(X)) % 2)
    width = np.asarray(_dense(est.transform(X[:1]))).reshape(1, -1).shape[1]
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


@pytest.fixture
def kernels_part(monkeypatch):
    """Each bounded function at its own bound from confit's kernel, as on
    x86-64 with AVX-512. Where this platform's kernel probe reads 0, the
    function is bit-exact and composes (function.py)."""
    from sql_transform.native import function

    monkeypatch.setattr(
        function, "kernel_distance", lambda f: function._BOUNDS.get(f, 0)
    )


class _MyPipeline(Pipeline):
    pass


@pytest.mark.parametrize(
    "make, reason",
    [
        (
            lambda: make_pipeline(StandardScaler(), KernelPCA(n_components=1)),
            "step 'kernelpca': no translation for KernelPCA",
        ),
        (
            lambda: make_pipeline(PCA(n_components=1), Binarizer()),
            r"step 'binarizer': Binarizer follows step 'pca', which is within"
            r" 5·eps·S \+ τ \(S its error scale\), and no bound is derived through"
            " it yet",
        ),
        (
            lambda: make_pipeline(
                PowerTransformer("box-cox", standardize=False),
                StandardScaler(),
                KBinsDiscretizer(n_bins=3, encode="ordinal"),
            ),
            "step 'kbinsdiscretizer': KBinsDiscretizer follows step"
            r" 'powertransformer', which is within 5·eps·S \+ τ \(S its error"
            r" scale\), and no bound",
        ),
        (
            lambda: make_pipeline(
                FunctionTransformer(np.exp), FunctionTransformer(np.log)
            ),
            "step 'functiontransformer-2': FunctionTransformer follows step"
            " 'functiontransformer-1', which is within 1 ulps, and no bound",
        ),
        (
            lambda: Pipeline(
                [
                    ("inner", make_pipeline(StandardScaler(), PCA(n_components=1))),
                    ("n", Normalizer()),
                ]
            ),
            "step 'n': Normalizer follows step 'inner', which is within the bound of"
            " its step 'pca', and no bound",
        ),
        (
            lambda: make_pipeline(
                FeatureUnion([("p", PCA(n_components=1)), ("s", StandardScaler())]),
                PowerTransformer(),
            ),
            "step 'powertransformer': PowerTransformer follows step 'featureunion',"
            " which is within the bound of its part 'p', and no bound",
        ),
        (
            lambda: make_pipeline(
                PCA(n_components=1),
                FeatureUnion([("s", StandardScaler()), ("b", Binarizer())]),
            ),
            "step 'featureunion': FeatureUnion part 'b': Binarizer follows step"
            " 'pca', which is within",
        ),
        (
            lambda: make_pipeline(
                PCA(n_components=2),
                make_pipeline(StandardScaler(), KMeans(n_clusters=2, n_init=1)),
            ),
            "step 'pipeline': Pipeline step 'kmeans': KMeans follows step 'pca',"
            " which is within",
        ),
        (
            lambda: make_pipeline(MissingIndicator(features="all"), StandardScaler()),
            "step 'missingindicator': MissingIndicator is not the last step"
            " and its output is boolean",
        ),
        (
            lambda: Pipeline(
                [("inner", make_pipeline(StandardScaler(), KernelPCA(n_components=1)))]
            ),
            "step 'inner': Pipeline step 'kernelpca': no translation for KernelPCA",
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
        "bounded-scale-first",
        "bounded-first",
        "bounded-function-first",
        "bounded-pipeline-first",
        "bounded-union-first",
        "bounded-then-union",
        "bounded-then-pipeline",
        "boolean",
        "nested",
        "nested-boolean",
        "subclass",
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.usefixtures("kernels_part")
def test_a_pipeline_names_the_step_it_refuses(make, reason):
    step = _numeric(make())
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True, allow_bound=True)


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
            lambda: Pipeline([("skip", "passthrough")]),
            "passthrough steps over a string feature",
        ),
    ],
    ids=["int-output", "passthrough-strings"],
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


_BITS = np.array([[True, False], [False, False], [True, True], [False, True]])
_BIT_ROWS = {
    "__iid": pa.array([0, 0, 0, 0, None], pa.int64()),
    "x0": pa.array([True, False, None, True, False]),
    "x1": pa.array([False, True, True, None, True]),
}


def _bits(est, beside: bool = False) -> tuple[PythonTransform, pa.Table]:
    """`est` over two boolean features (and a double, `beside`), and rows
    with booleans only, NULLs and a NULL id."""
    X, types, rows = _BITS, [pa.bool_(), pa.bool_()], _BIT_ROWS
    if beside:
        X = np.column_stack([_BITS.astype(float), [1.5, -2.0, 0.5, 3.0]])
        types = [*types, pa.float64()]
        rows = {**rows, "x2": pa.array([2.5, -0.0, None, 4.0, 1.0])}
    est.fit(X, np.arange(len(X)) % 2)
    # The width from a float row: a boolean one may raise (`np.negative`).
    width = np.asarray(_dense(est.transform(X[:1].astype(float)))).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", t) for i, t in enumerate(types)]),
        pa.struct([(f"f{j}", pa.float64()) for j in range(width)]),
    )
    return step, pa.table(rows)


@pytest.mark.parametrize("beside", [False, True], ids=["booleans", "beside-double"])
def test_a_pipeline_hands_booleans_on_as_its_twin_does(beside):
    # A selector hands a boolean row on as booleans, which the next step
    # reads as 0/1, as it reads its own boolean features.
    for est, compared in (
        (make_pipeline(VarianceThreshold(), StandardScaler()), 4),
        # Binarizer rejects NaN: the twin answers the rows without a NULL.
        (
            make_pipeline(
                VarianceThreshold(), Binarizer(threshold=0.5), MaxAbsScaler()
            ),
            2,
        ),
        (make_pipeline(VarianceThreshold(), FunctionTransformer(np.sqrt)), 4),
        (make_pipeline(Binarizer(), OneHotEncoder(sparse_output=False)), 2),
    ):
        step, rows = _bits(est, beside)
        assert check(step, to_native(step, strict=True), rows) == compared


@pytest.mark.parametrize(
    "make, reason",
    [
        (
            lambda: make_pipeline(
                VarianceThreshold(), FunctionTransformer(np.negative)
            ),
            r"step 'functiontransformer': FunctionTransformer\(func=np\.negative\)"
            " over boolean features only: numpy raises TypeError",
        ),
        (
            lambda: make_pipeline(
                VarianceThreshold(), FunctionTransformer(np.sqrt), StandardScaler()
            ),
            "step 'functiontransformer': FunctionTransformer hands on float16 over"
            " boolean features only",
        ),
    ],
    ids=["a-function-numpy-refuses", "a-narrow-dtype-handed-on"],
)
def test_a_pipeline_over_booleans_only_refuses(make, reason):
    step, _ = _bits(make())
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)
    # Beside a double the row is float64: served.
    step, rows = _bits(make(), beside=True)
    assert check(step, to_native(step, strict=True), rows) == 4


@pytest.mark.parametrize(
    "part",
    [
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        lambda: make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            StandardScaler(),
        ),
    ],
    ids=["encoder", "pipeline"],
)
def test_a_column_transformer_refuses_an_encoder_over_a_boolean(part):
    step, _ = _bits(ColumnTransformer([("enc", part(), [0, 1])]), beside=True)
    with pytest.raises(
        NotNative,
        match="part 'enc': an encoder over a boolean feature reads it as an object",
    ):
        to_native(step, strict=True)
    # Over the double only, it is served.
    step, rows = _bits(
        ColumnTransformer([("enc", part(), [2]), ("p", "passthrough", [0, 1])]),
        beside=True,
    )
    assert check(step, to_native(step, strict=True), rows) == 4


def test_an_integer_weight_over_booleans():
    # A FeatureUnion part that hands booleans back, times an integer, is an
    # integer (0 * -2 is 0, not -0.0): refused over booleans only, served
    # beside a double, where the row is float64.
    def make():
        return FeatureUnion(
            [("sel", VarianceThreshold()), ("std", StandardScaler())],
            transformer_weights={"sel": -2, "std": 3},
        )

    step, _ = _bits(make())
    with pytest.raises(
        NotNative,
        match="part 'sel': an integer weight on an output that may not be float64"
        r" \(bool over boolean features only\)",
    ):
        to_native(step, strict=True)
    step, rows = _bits(make(), beside=True)
    assert check(step, to_native(step, strict=True), rows) == 4
    # A part whose output is float64 on a boolean row takes it.
    step, rows = _bits(
        FeatureUnion([("std", StandardScaler())], transformer_weights={"std": -2})
    )
    assert check(step, to_native(step, strict=True), rows) == 4
    # A ColumnTransformer hands its parts objects, a boolean as a Python
    # bool, which a selector keeps: refused over any boolean feature.
    step, _ = _bits(
        ColumnTransformer(
            [("sel", VarianceThreshold(), [0, 2]), ("std", StandardScaler(), [1])],
            transformer_weights={"sel": -2},
        ),
        beside=True,
    )
    with pytest.raises(
        NotNative, match=r"part 'sel': an integer weight .* \(a boolean feature\)"
    ):
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
    assert check(step, to_native(step, strict=True), rows) == 3


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
    assert check(step, to_native(step, strict=True), rows) == 3


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
            lambda: ColumnTransformer([("p", KernelPCA(n_components=1), [0, 1])]),
            "ColumnTransformer part 'p': no translation for KernelPCA",
        ),
        (
            lambda: ColumnTransformer(
                [("p", PCA(n_components=1), [0, 1])],
                transformer_weights={"p": float("inf")},
            ),
            r"part 'p': PCA is within 5·eps·S \+ τ \(S its error scale\), and its"
            " weight inf is infinite",
        ),
        (
            lambda: ColumnTransformer([("s", _MyScaler(), [0])]),
            "part 's': no translation for _MyScaler",
        ),
        (
            lambda: ColumnTransformer(
                [
                    (
                        "pow",
                        make_pipeline(
                            PowerTransformer("box-cox", standardize=False),
                            StandardScaler(),
                            Binarizer(),
                        ),
                        [0],
                    )
                ]
            ),
            "part 'pow': Pipeline step 'binarizer': Binarizer follows step"
            r" 'powertransformer', which is within 5·eps·S \+ τ \(S its error"
            r" scale\), and no bound",
        ),
        (
            lambda: ColumnTransformer(
                [("log", FunctionTransformer(np.log10, validate=True), [0])],
                remainder="passthrough",
                transformer_weights={"log": -float("inf")},
            ),
            "part 'log': FunctionTransformer is within 2 ulps, and its weight -inf"
            " is infinite",
        ),
        (
            lambda: ColumnTransformer(
                [("s", StandardScaler(), [0])], remainder=KernelPCA(n_components=1)
            ),
            "part 'remainder': no translation for KernelPCA",
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
            lambda: make_pipeline(
                FunctionTransformer(np.tan),
                FeatureUnion(
                    [("pass", "passthrough"), ("s", StandardScaler())],
                    transformer_weights={"s": float("inf")},
                ),
            ),
            "step 'featureunion': FeatureUnion part 's': its weight inf multiplies"
            " an output that follows step 'functiontransformer', which is within 1"
            " ulps",
        ),
        (
            lambda: FeatureUnion([("p", KernelPCA(n_components=1))]),
            "FeatureUnion part 'p': no translation for KernelPCA",
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
                [("s", make_pipeline(StandardScaler(), KernelPCA(n_components=1)))]
            ),
            "FeatureUnion part 's': Pipeline step 'kernelpca': no translation for"
            " KernelPCA",
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
        "ct-infinite-weight",
        "ct-subclass",
        "ct-bounded-first",
        "ct-infinite-weight-function",
        "ct-remainder",
        "ct-object-func",
        "ct-object-func-nested",
        "ct-container",
        "ct-float32-weight",
        "ct-integer-weight",
        "ct-inexact-weight",
        "ct-not-last",
        "fu-infinite-weight-carried",
        "fu-unknown",
        "fu-weighted-passthrough",
        "fu-weight-type",
        "fu-nested",
        "fu-container",
    ],
)
@pytest.mark.usefixtures("kernels_part")
def test_a_composition_names_the_part_it_refuses(make, reason):
    step = _composed(make())
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True, allow_bound=True)


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
    assert check(step, to_native(step, strict=True), rows) == 3


@pytest.mark.usefixtures("kernels_part")
def test_an_exact_function_composes_anywhere_a_bounded_one_where_it_is_carried():
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
    assert check(step, to_native(step, strict=True), rows) == 2
    bounded = FeatureUnion(
        [("id", FunctionTransformer()), ("exp", FunctionTransformer(np.exp))]
    )
    step = _numeric(bounded)
    assert check(step, to_native(step, strict=True, allow_bound=True), rows) == 2
    # A scaler carries the bound of what it reads; a square root carries
    # none yet.
    step = _numeric(make_pipeline(bounded, StandardScaler()))
    assert check(step, to_native(step, strict=True, allow_bound=True), rows) == 2
    with pytest.raises(
        NotNative,
        match="step 'functiontransformer': FunctionTransformer follows step"
        " 'featureunion', which is within the bound of its part 'exp', and no"
        " bound is derived through it yet",
    ):
        to_native(
            _numeric(make_pipeline(bounded, FunctionTransformer(np.sqrt))),
            strict=True,
            allow_bound=True,
        )


@pytest.mark.parametrize(
    "make, within",
    [
        (
            lambda: make_pipeline(StandardScaler(), PCA(n_components=2)),
            "the bound of its step 'pca'",
        ),
        (
            lambda: make_pipeline(
                MaxAbsScaler(), PowerTransformer("box-cox", standardize=False)
            ),
            "the bound of its step 'powertransformer'",
        ),
        (
            lambda: make_pipeline(MinMaxScaler(), FunctionTransformer(np.exp)),
            "the bound of its step 'functiontransformer'",
        ),
        (
            lambda: ColumnTransformer(
                [("p", PCA(n_components=1), [0, 1]), ("s", StandardScaler(), [2])],
                remainder="passthrough",
            ),
            "the bound of its part 'p'",
        ),
        (
            lambda: ColumnTransformer(
                [
                    ("log", FunctionTransformer(np.log10, validate=True), [0]),
                    ("p", PCA(n_components=2), slice(1, 4)),
                ]
            ),
            "the bounds of its parts 'log' and 'p'",
        ),
        (
            lambda: FeatureUnion(
                [("pass", "passthrough"), ("tan", FunctionTransformer(np.tan))]
            ),
            "the bound of its part 'tan'",
        ),
        (
            lambda: FeatureUnion(
                [
                    ("p", make_pipeline(StandardScaler(), PCA(n_components=2))),
                    ("s", MaxAbsScaler()),
                ]
            ),
            "the bound of its part 'p'",
        ),
        (
            lambda: make_pipeline(
                StandardScaler(),
                FeatureUnion(
                    [("p", PCA(n_components=1)), ("id", FunctionTransformer())]
                ),
            ),
            "the bound of its step 'featureunion'",
        ),
    ],
    ids=[
        "pipeline-scale",
        "pipeline-ulps",
        "pipeline-function",
        "ct-scale",
        "ct-two-bounds",
        "fu-function",
        "fu-pipeline",
        "pipeline-fu",
    ],
)
@pytest.mark.usefixtures("kernels_part")
def test_a_bounded_step_serves_last_or_as_a_part_on_request(make, within):
    # Option 1 of decisions/closed/bounded-steps-in-compositions.md: what
    # the step reads is bit-exact, so each output field keeps the bound of
    # the step or part that gives it.
    est = make()
    step = _composed(est)
    with pytest.raises(
        NotNative,
        match=f"instance 0: {type(est).__name__} is within {within} of its twin,"
        " not bit-exact; to_native serves a bound above 0 only with"
        " allow_bound=True",
    ):
        to_native(step, strict=True)
    native = to_native(step, strict=True, allow_bound=True)
    assert explain_native(step, allow_bound=True).endswith(f"within {within}")
    x = np.random.default_rng(2).uniform(0.5, 12.0, size=(40, 4))
    rows = pa.table(
        {
            "__iid": pa.array([0] * 39 + [None], pa.int64()),
            **{f"x{i}": pa.array(x[:, i]) for i in range(4)},
        }
    )
    assert check(step, native, rows) == 39


@pytest.mark.parametrize(
    "make, within",
    [
        (
            lambda: make_pipeline(PCA(n_components=2), StandardScaler()),
            "steps 'pca' and 'standardscaler'",
        ),
        (
            lambda: make_pipeline(
                PowerTransformer("box-cox", standardize=False), StandardScaler()
            ),
            "steps 'powertransformer' and 'standardscaler'",
        ),
        (
            lambda: make_pipeline(
                FunctionTransformer(np.exp), MinMaxScaler(clip=True), MaxAbsScaler()
            ),
            "steps 'functiontransformer', 'minmaxscaler' and 'maxabsscaler'",
        ),
        (
            lambda: Pipeline(
                [
                    ("inner", make_pipeline(StandardScaler(), PCA(n_components=3))),
                    ("r", RobustScaler()),
                    ("v", VarianceThreshold()),
                ]
            ),
            "steps 'inner', 'r' and 'v'",
        ),
        (
            lambda: make_pipeline(
                StandardScaler(), PCA(n_components=3), TruncatedSVD(n_components=2)
            ),
            "steps 'pca' and 'truncatedsvd'",
        ),
        (
            lambda: make_pipeline(
                FastICA(n_components=3, random_state=0),
                FunctionTransformer(np.negative),
                FactorAnalysis(n_components=2, random_state=0),
            ),
            "steps 'fastica', 'functiontransformer' and 'factoranalysis'",
        ),
        (
            lambda: make_pipeline(
                PCA(n_components=3),
                FeatureAgglomeration(n_clusters=2),
                PLSRegression(n_components=1),
            ),
            "steps 'pca', 'featureagglomeration' and 'plsregression'",
        ),
        (
            lambda: make_pipeline(
                PCA(n_components=4),
                SparseRandomProjection(n_components=3, density=0.5, random_state=0),
            ),
            "steps 'pca' and 'sparserandomprojection'",
        ),
        (
            lambda: make_pipeline(
                PCA(n_components=2),
                FeatureUnion(
                    [("s", StandardScaler()), ("abs", FunctionTransformer(np.abs))],
                    transformer_weights={"s": -3.0},
                ),
            ),
            "steps 'pca' and 'featureunion'",
        ),
        (
            lambda: ColumnTransformer(
                [("p", PCA(n_components=1), [0, 1]), ("s", StandardScaler(), [2])],
                remainder="passthrough",
                transformer_weights={"p": 2.0, "s": 0.5},
            ),
            "part 'p' (weighted)",
        ),
        (
            lambda: ColumnTransformer(
                [
                    ("log", FunctionTransformer(np.log10, validate=True), [0]),
                    (
                        "p",
                        make_pipeline(
                            PowerTransformer("box-cox", standardize=False),
                            StandardScaler(),
                        ),
                        [1, 2],
                    ),
                ],
                transformer_weights={"log": 0.5},
            ),
            "parts 'log' (weighted) and 'p'",
        ),
        (
            lambda: FeatureUnion(
                [("pass", "passthrough"), ("tan", FunctionTransformer(np.tan))],
                transformer_weights={"tan": 0.5},
            ),
            "part 'tan' (weighted)",
        ),
        (
            lambda: make_pipeline(
                ColumnTransformer(
                    [
                        ("p", PCA(n_components=2), [0, 1, 2]),
                        ("s", StandardScaler(), [3]),
                    ]
                ),
                ColumnTransformer(
                    [("m", MinMaxScaler(), [0, 2]), ("q", PCA(n_components=1), [1, 2])],
                    transformer_weights={"q": 1e3},
                ),
            ),
            "steps 'columntransformer-1' and 'columntransformer-2'",
        ),
    ],
    ids=[
        "pca-scaler",
        "power-scaler",
        "function-scalers",
        "pipeline-scaler-selector",
        "two-projections",
        "ica-negative-factor",
        "pca-agglomeration-pls",
        "pca-sparse-projection",
        "pca-weighted-union",
        "ct-weighted-scale",
        "ct-weighted-function",
        "fu-weighted-function",
        "ct-ct",
    ],
)
@pytest.mark.usefixtures("kernels_part")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.DataDimensionalityWarning")
def test_a_bounded_step_is_carried_by_the_steps_after_it(make, within):
    # Option 2 of decisions/closed/bounded-steps-in-compositions.md: each
    # later step carries the bound on what it reads to its output fields,
    # and a weight carries the bound of its part.
    est = make()
    step = _composed(est)
    with pytest.raises(NotNative, match="to_native serves a bound above 0 only"):
        to_native(step, strict=True)
    native = to_native(step, strict=True, allow_bound=True)
    bound = "bounds" if within.startswith("parts") else "bound"
    assert explain_native(step, allow_bound=True).endswith(
        f"within the {bound} of its {within}"
    )
    x = np.random.default_rng(3).uniform(0.5, 12.0, size=(40, 4))
    rows = pa.table(
        {
            "__iid": pa.array([0] * 39 + [None], pa.int64()),
            **{f"x{i}": pa.array(x[:, i]) for i in range(4)},
        }
    )
    assert check(step, native, rows) == 39


def _flat(n: int = 40, seed: int = 4) -> np.ndarray:
    """Rows near a line: the second principal component spreads 1e-6 as
    far as the first."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.0, 9.0, size=(n, 1))
    return np.hstack([t, 2 * t]) + rng.normal(scale=1e-6, size=(n, 2))


def _another_hosts_pca(monkeypatch, pca: PCA) -> None:
    """Make the twin's PCA answer as another host's BLAS may: each output
    field moves by eps·S, inside PCA's bound of (n + 3)·eps·S. On a host
    whose BLAS adds as the entry does, the two sides agree to the bit, so a
    test that a bound is needed must make the difference itself."""
    from sql_transform.native import project

    transform = pca.transform
    eps = np.finfo(np.float64).eps

    def moved(X):
        y = np.asarray(transform(X), dtype=np.float64)
        x = np.asarray(X, dtype=np.float64)
        s = np.stack([project._s(pca, row) for row in x]).astype(np.float64)
        return y + eps * s

    monkeypatch.setattr(pca, "transform", moved)


def test_a_carried_bound_grows_as_the_step_that_carries_it(monkeypatch):
    # The scaler divides the second component by its spread, 1e-6 of the
    # first: what the two sides' PCA part by grows by as much, and so does
    # the bound the scaler carries. Carried as it is, it fails.
    from sql_transform.native import _registry

    est = make_pipeline(PCA(n_components=2), StandardScaler())
    step = _step(est, _flat(), [pa.float64()] * 2)
    native = to_native(step, strict=True, allow_bound=True)
    _another_hosts_pca(monkeypatch, est[0])
    x = _flat(400, seed=5)
    rows = pa.table(
        {
            "__iid": pa.array([0] * 400, pa.int64()),
            "x0": pa.array(x[:, 0]),
            "x1": pa.array(x[:, 1]),
        }
    )
    assert check(step, native, rows) == 400
    entry = _registry._CATALOG[StandardScaler]
    monkeypatch.setitem(
        _registry._CATALOG,
        StandardScaler,
        dataclasses.replace(entry, carry_of=lambda est: lambda x, e: e),
    )
    with pytest.raises(ParityError, match="past its part's bound"):
        check(step, native, rows)


def test_a_weight_carries_its_part_s_bound_times_the_weight(monkeypatch):
    from sql_transform.native import compose

    est = ColumnTransformer(
        [("p", PCA(n_components=2), [0, 1])], transformer_weights={"p": 1e6}
    )
    step = _step(est, _flat(), [pa.float64()] * 2)
    native = to_native(step, strict=True, allow_bound=True)
    _another_hosts_pca(monkeypatch, est.named_transformers_["p"])
    x = _flat(400, seed=5) * 1e3
    rows = pa.table(
        {
            "__iid": pa.array([0] * 400, pa.int64()),
            "x0": pa.array(x[:, 0]),
            "x1": pa.array(x[:, 1]),
        }
    )
    assert check(step, native, rows) == 400
    monkeypatch.setattr(compose, "weighted", lambda e, y, w: e)
    with pytest.raises(ParityError, match="past its part's bound"):
        check(step, native, rows)


def _unshared_fn(name: str):
    """A function whose let `v` calls `name` under a CASE that rules its
    trap out, and whose let `w` reads `v` three times."""
    import math

    from confit import SqlFunction
    from confit import sql as S

    from sql_transform.native._helpers import f64

    def body(x):
        v = S.case(x <= f64(0.0), f64(math.nan)).otherwise(S.fn(name, x))
        w = v * v + v
        return [w + w, w - w]

    return SqlFunction(
        "f", pa.schema([("x", pa.float64())]), pa.list_(pa.float64(), 2), body
    )


def test_unshared_writes_each_value_that_confit_cannot_share_at_each_read():
    # confit counts `ln` as able to trap, though a CASE rules its trap out,
    # and so each value that reads it: written at each read, they add what
    # the definition spelled out adds. It shares `exp`'s, which add none.
    from sql_transform.native._registry import unshared

    ln, exp = _unshared_fn("ln"), _unshared_fn("exp")
    assert len(ln.sql_lets) == len(exp.sql_lets) == 2
    let_form = sum(map(len, ln.sql_lets)) + len(ln.sql_let_body)
    assert unshared(ln) == len(ln.sql_body) - let_form > 0
    assert unshared(exp) == 0


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_a_composition_that_confit_cannot_share_stays_python():
    # After np.log2, under the CASE that rules its trap out, confit shares
    # no value of the steps after it: their build took 31 s and 3.8 GB
    # (T59). After np.exp, the same steps build in 0.05 s.
    def make(func):
        return make_pipeline(
            MinMaxScaler(feature_range=(1, 2)),
            FunctionTransformer(func),
            FactorAnalysis(8, random_state=0),
            GaussianRandomProjection(8, random_state=0),
            FeatureAgglomeration(4),
        )

    X = np.random.default_rng(0).normal(size=(300, 16)) * 3
    with pytest.raises(
        NotNative,
        match=r"confit does not build it: each value that confit cannot share,"
        r" written at each read, adds [\d,]+ characters to its definition, past"
        " 500,000",
    ):
        to_native(_composed(make(np.log2), X), strict=True, allow_bound=True)
    to_native(_composed(make(np.exp), X), strict=True, allow_bound=True)
