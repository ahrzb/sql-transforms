"""Every catalog entry is held to its declared bound against its Python twin,
on generated fits and generated serving rows (swap-the-entry, `check`).

Adding an entry means adding its fixtures here: `FIXTURES` maps each
registered class to estimator factories covering its configurations, and
`test_every_entry_has_fixtures` fails until it does.
"""

from __future__ import annotations

import os
import random
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_regression,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from threadpoolctl import threadpool_limits

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    NotNative,
    catalog,
    check,
    explain_native,
    to_native,
)

# An imputer whose fit saw a column only missing warns at every transform
# that it drops it, an encoder warns of an unseen category, and numpy warns
# when a twin overflows on an edge row; the fixtures make all three on
# purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Skipping features without any observed values:UserWarning",
    "ignore:Found unknown categories:UserWarning",
    "ignore:overflow encountered:RuntimeWarning",
    "ignore:invalid value encountered:RuntimeWarning",
    "ignore:Feature .* is constant:UserWarning",
    "ignore:Bins whose width are too small:UserWarning",
)


@pytest.fixture(autouse=True)
def _one_blas_thread():
    # The suite runs on xdist workers, and BLAS threads per worker
    # oversubscribe the cores: an RFECV fixture took 35 s instead of 0.4 s.
    with threadpool_limits(limits=1):
        yield


FIXTURES: dict[type, list[Callable[[], Any]]] = {
    StandardScaler: [
        StandardScaler,
        lambda: StandardScaler(with_mean=False),
        lambda: StandardScaler(with_std=False),
        lambda: StandardScaler(with_mean=False, with_std=False),
    ],
    MinMaxScaler: [
        MinMaxScaler,
        lambda: MinMaxScaler(feature_range=(-2.5, 7.0)),
        lambda: MinMaxScaler(clip=True),
        lambda: MinMaxScaler(feature_range=(-1.0, 0.0), clip=True),
    ],
    MaxAbsScaler: [MaxAbsScaler, lambda: MaxAbsScaler(clip=True)],
    RobustScaler: [
        RobustScaler,
        lambda: RobustScaler(with_centering=False),
        lambda: RobustScaler(with_scaling=False),
        lambda: RobustScaler(with_centering=False, with_scaling=False),
        lambda: RobustScaler(quantile_range=(10.0, 90.0), unit_variance=True),
    ],
    Normalizer: [
        lambda: Normalizer("l1"),
        lambda: Normalizer("l2"),
        lambda: Normalizer("max"),
    ],
    Binarizer: [
        Binarizer,
        lambda: Binarizer(threshold=2.5),
        lambda: Binarizer(threshold=-40.0),
    ],
    SimpleImputer: [
        SimpleImputer,
        lambda: SimpleImputer(strategy="median"),
        lambda: SimpleImputer(strategy="most_frequent"),
        lambda: SimpleImputer(strategy="constant", fill_value=-7.5),
        lambda: SimpleImputer(add_indicator=True),
        lambda: SimpleImputer(strategy="median", keep_empty_features=True),
        lambda: SimpleImputer(
            missing_values=-1.0, strategy="most_frequent", add_indicator=True
        ),
        lambda: SimpleImputer(
            missing_values=0.0,
            strategy="constant",
            fill_value=2.5,
            keep_empty_features=True,
            add_indicator=True,
        ),
    ],
    MissingIndicator: [
        MissingIndicator,
        lambda: MissingIndicator(features="all"),
        lambda: MissingIndicator(missing_values=-1.0, error_on_new=False),
    ],
    VarianceThreshold: [VarianceThreshold],
    SelectKBest: [lambda: SelectKBest(k=1), lambda: SelectKBest(f_regression, k=1)],
    SelectPercentile: [lambda: SelectPercentile(percentile=50)],
    SelectFpr: [lambda: SelectFpr(alpha=0.5)],
    SelectFdr: [lambda: SelectFdr(alpha=0.5)],
    SelectFwe: [lambda: SelectFwe(alpha=0.5)],
    GenericUnivariateSelect: [
        lambda: GenericUnivariateSelect(mode="percentile", param=50)
    ],
    SelectFromModel: [
        lambda: SelectFromModel(LogisticRegression()),
        lambda: SelectFromModel(
            LogisticRegression(), max_features=1, threshold=-np.inf
        ),
    ],
    RFE: [lambda: RFE(LogisticRegression(), n_features_to_select=1)],
    RFECV: [lambda: RFECV(LogisticRegression(), cv=2)],
    SequentialFeatureSelector: [
        lambda: SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=1, cv=2
        )
    ],
    PolynomialFeatures: [
        PolynomialFeatures,
        lambda: PolynomialFeatures(degree=3, include_bias=False),
        lambda: PolynomialFeatures(degree=3, interaction_only=True),
        lambda: PolynomialFeatures(degree=(2, 3)),
        lambda: PolynomialFeatures(degree=(2, 2), include_bias=False),
    ],
    OrdinalEncoder: [
        OrdinalEncoder,
        lambda: OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        ),
        lambda: OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=3
        ),
    ],
    OneHotEncoder: [
        lambda: OneHotEncoder(sparse_output=False),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        lambda: OneHotEncoder(
            sparse_output=False, drop="first", handle_unknown="ignore"
        ),
        lambda: OneHotEncoder(
            sparse_output=False, min_frequency=3, handle_unknown="infrequent_if_exist"
        ),
        lambda: OneHotEncoder(sparse_output=False, drop="if_binary"),
    ],
    TargetEncoder: [TargetEncoder, lambda: TargetEncoder(target_type="continuous")],
}

# KBinsDiscretizer: the strategies only shape the edges; constant features
# (edges [-inf, inf]) and dropped narrow bins come from the generator's
# zero-variance and few-valued columns.
FIXTURES[KBinsDiscretizer] = [
    lambda: KBinsDiscretizer(encode="ordinal", strategy="uniform"),
    lambda: KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="uniform"),
    lambda: KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile"),
    lambda: KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="quantile"),
    lambda: KBinsDiscretizer(
        n_bins=7, encode="ordinal", quantile_method="inverted_cdf"
    ),
    lambda: KBinsDiscretizer(n_bins=4, encode="onehot-dense", quantile_method="linear"),
    lambda: KBinsDiscretizer(
        n_bins=6, encode="ordinal", quantile_method="median_unbiased"
    ),
    lambda: KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="kmeans"),
    lambda: KBinsDiscretizer(n_bins=3, encode="onehot-dense", strategy="kmeans"),
    lambda: KBinsDiscretizer(
        n_bins=8, encode="onehot-dense", strategy="uniform", dtype=np.float64
    ),
]

# QuantileTransformer: the fits have 5 to 60 rows, so n_quantiles_ is the
# row count past it; few distinct values (kind 4) make runs of equal
# quantiles. quantile_test.py serves the default 1,000 quantiles.
FIXTURES[QuantileTransformer] = [
    QuantileTransformer,
    lambda: QuantileTransformer(n_quantiles=2),
    lambda: QuantileTransformer(n_quantiles=7),
    lambda: QuantileTransformer(n_quantiles=40),
    lambda: QuantileTransformer(n_quantiles=4, subsample=5, random_state=0),
]


def positive(factory: Callable[[], Any]) -> Callable[[], Any]:
    """`factory`'s fits and rows draw strictly positive numbers (an
    estimator that rejects the rest): absolute values, a fitted zero made
    one. A serving zero stays, for the twin to reject."""
    factory.positive = True  # type: ignore[attr-defined]
    return factory


# Fitted lambdas pinned on and around scipy's Box-Cox branches (`log`
# under |lambda| 1e-19; past `lambda * log(x)` 709.78, `exp`, which the
# rows' 1e300 and 1e-300 reach), one per feature in turn.
BOX_COX_LAMBDAS = [0.0, 1e-20, -1e-20, 1e-19, -1e-19, -3e-12, 3.0, -50.0, 2.0]


def _pinned_box_cox() -> PowerTransformer:
    """A Box-Cox PowerTransformer(standardize=False) whose fit then pins
    `lambdas_` to BOX_COX_LAMBDAS, where no fit lands."""
    est = PowerTransformer("box-cox", standardize=False)

    def fit(X, y=None):
        del est.fit  # the class's own again
        PowerTransformer.fit(est, X, y)
        n = len(est.lambdas_)
        est.lambdas_ = np.array(
            [BOX_COX_LAMBDAS[j % len(BOX_COX_LAMBDAS)] for j in range(n)]
        )
        return est

    est.fit = fit
    return est


# Box-Cox only, unstandardized: what the entry serves (power.py).
FIXTURES[PowerTransformer] = [
    positive(lambda: PowerTransformer("box-cox", standardize=False)),
    positive(_pinned_box_cox),
]

# A string feature's fitted values, and the unseen ones serving adds.
VOCAB = ["a", "b", "c", "d", "é", "日本"]
UNSEEN = ["zz", "", "A"]
# Serving values beyond the fit's range: signed zeros, extremes.
EDGES = [0.0, -0.0, 1e-300, -1e300, 1e300, 5e-324]
# A row of only these has a norm under sklearn's zero-scale threshold.
SMALL = [0.0, -0.0, 1e-300, -5e-324, 1e-17, -2.5e-16]
# The widest step drawn, in output lanes: a wider fixture checks the same
# translation, only slower. Set while confit's builds grew about as
# lanes^2.5 (a 1,351-lane PolynomialFeatures draw took 115 s); since #350
# they grow about as lanes^1.4, and PLANS "Next" raises it.
MAX_LANES = 300
# Seeds per configuration: 8 in the gate; a milestone report sweeps more
# (NATIVE_SEEDS=200, docs/native/reports.md).
SEEDS = int(os.environ.get("NATIVE_SEEDS", "8"))


def _fit_matrix(
    rng: np.random.Generator,
    kinds: list[int],
    types: list[pa.DataType],
    holes: list[str | None],
    marker: float,
    positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One instance's fit data, a column per feature of its kind (an integer
    one rounded), and a binary target. `kinds` and `holes` are the step's,
    the same for every instance, as a feature keeps its nature across
    fitted groups: `holes[j]` is None for nowhere missing (and never
    holding the marker), "some" for about a fifth of the rows and at least
    one, "all" for everywhere. `marker` spells missing."""
    n = int(rng.integers(5, 60))
    cols = []
    strings = [t == pa.string() for t in types]
    for kind, t, hole in zip(kinds, types, holes, strict=True):
        if t == pa.string():
            # The step's share of VOCAB for this feature: kind - 3 of them.
            c = rng.choice(np.array(VOCAB[: kind - 3], dtype=object), n)
            if hole == "all":
                c[:] = None
            elif hole == "some":
                c[rng.random(n) < 0.2] = None
                c[rng.integers(n)] = None
            cols.append(c)
            continue
        if kind == 0:
            c = rng.normal(rng.uniform(-100, 100), rng.uniform(0.01, 50), n)
        elif kind == 1:
            c = rng.integers(-1000, 1000, n).astype(float)
        elif kind == 2:
            c = np.full(n, rng.uniform(-5, 5))  # zero variance
        elif kind == 3:
            c = rng.exponential(rng.uniform(0.1, 1e6), n)
        else:
            c = rng.integers(-3, 4, n).astype(float)  # few distinct values
        if t == pa.int64():
            c = np.round(c)
        if positive:
            c = np.abs(c)
            c[c == 0] = 1.0
        if not np.isnan(marker):
            c[c == marker] = marker + 1.0
        if hole == "all":
            c[:] = marker
        elif hole == "some":
            c[rng.random(n) < 0.2] = marker
            c[rng.integers(n)] = marker
        cols.append(c)
    y = (rng.random(n) < 0.5).astype(int)
    y[:4] = (0, 1, 0, 1)  # two of each class, for a 2-fold split
    if any(strings):
        # A string beside numbers makes an object matrix, as the step's
        # rows reach `transform`.
        X = np.empty((n, len(cols)), dtype=object)
        for j, c in enumerate(cols):
            X[:, j] = c if strings[j] else [float(v) for v in c]
        return X, y
    return np.column_stack(cols), y


def _step(cls_factory, seed: int, variant: int = 0) -> PythonTransform:
    # Each of a class's configurations (`variant`) draws shapes of its own,
    # so a class's seeds are not the same few shapes for every one.
    rng = np.random.default_rng([seed, variant])
    # A shape the estimator cannot fit (a selector asked for more features
    # than the step has), that keeps no lane or that is wider than
    # MAX_LANES is drawn again.
    for _ in range(20):
        step = _draw(rng, cls_factory)
        if step is not None:
            return step
    raise AssertionError(f"no fixture of {cls_factory} fits in 20 draws")


def _draw(rng: np.random.Generator, cls_factory) -> PythonTransform | None:
    # Mostly narrow; sometimes wide enough for a row reduction's blocks.
    wide = rng.random() < 0.3
    n_features = int(rng.integers(5, 33) if wide else rng.integers(1, 5))
    types = [
        pa.float64() if rng.random() < 0.7 else pa.int64() for _ in range(n_features)
    ]
    kinds = [int(rng.integers(5)) for _ in range(n_features)]
    proto = cls_factory()
    positive = getattr(cls_factory, "positive", False)
    if proto.__sklearn_tags__().input_tags.categorical:
        # Categories: few distinct values per feature, about half of them
        # strings (kind 5..8: two to five of VOCAB).
        for j in range(n_features):
            if rng.random() < 0.5:
                types[j], kinds[j] = pa.string(), int(rng.integers(5, 9))
            else:
                kinds[j] = 4
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    # Missing values in the fit data, for an estimator that takes them: an
    # imputer's own `missing_values`, or NaN where sklearn says it allows it.
    marker = float(getattr(proto, "missing_values", np.nan))
    holes: list[str | None] = [None] * n_features
    if (
        hasattr(proto, "missing_values")
        or proto.__sklearn_tags__().input_tags.allow_nan
    ):
        holes = [
            ("some", "all", None)[int(np.searchsorted([0.4, 0.48], rng.random()))]
            for _ in range(n_features)
        ]
        if hasattr(proto, "missing_values") and "some" not in holes:
            holes[int(rng.integers(n_features))] = "some"
    # Every fit gets a target (an unsupervised one ignores it). A step's
    # instances share one width, as a fitted step's do: an instance that
    # fits to another width ends the step before it.
    instances: dict[int, Any] = {}
    width = 0
    with warnings.catch_warnings():  # all-missing columns, by design
        warnings.simplefilter("ignore")
        try:
            for k in range(int(rng.integers(1, 4))):
                X, y = _fit_matrix(rng, kinds, types, holes, marker, positive)
                est = cls_factory().fit(X, y)
                w = np.asarray(est.transform(X[:1])).reshape(1, -1).shape[1]
                if k and w != width:
                    break
                instances[k], width = est, w
        except ValueError:
            return None
    if width == 0 or width > MAX_LANES:
        return None
    if width == 1:
        returns = pa.float64()
    elif rng.random() < 0.3:  # unnamed lanes
        returns = pa.list_(pa.float64(), width)
    else:
        returns = pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
    return PythonTransform("tf", instances, takes, returns)


def _value(rng: random.Random, regime: str) -> float | None:
    r = rng.random()
    if regime == "nulls" and r < 0.3:
        return None
    if regime == "edges" and r < 0.3:
        return rng.choice(EDGES)
    if regime == "small":
        return rng.choice(SMALL)
    if regime == "ints":
        return float(rng.randint(-3, 3))
    return rng.uniform(-1e3, 1e3)


def _rows(step: PythonTransform, seed: int, positive: bool = False) -> pa.Table:
    rng = random.Random(seed)  # noqa: S311
    n = 40
    ids = [rng.choice([None, *step.instances]) for _ in range(n)]
    # A row's regime: plain values; some edges; some NULLs (where most
    # twins answer and a validating one raises); all small; or few distinct
    # integers (a category, or an imputer's numeric missing marker).
    regimes = rng.choices(
        ["plain", "edges", "nulls", "small", "ints"], [45, 15, 15, 10, 15], k=n
    )
    cols: dict[str, pa.Array] = {"__iid": pa.array(ids, pa.int64())}
    for f in step.takes:
        if f.type == pa.string():
            strs = [
                None
                if g == "nulls" and rng.random() < 0.3
                else rng.choice(VOCAB + UNSEEN)
                for g in regimes
            ]
            cols[f.name] = pa.array(strs, pa.string())
            continue
        vals = [_value(rng, g) for g in regimes]
        if positive:
            vals = [None if v is None else abs(v) for v in vals]
        if f.type == pa.int64():
            vals = [
                None if v is None else max(-(2**62), min(2**62, round(v))) for v in vals
            ]
        cols[f.name] = pa.array(vals, f.type)
    return pa.table(cols)


def test_every_entry_has_fixtures():
    assert set(catalog()) == set(FIXTURES), "add fixtures for every catalog entry"


@pytest.mark.parametrize(
    "cls, j, seed",
    [
        pytest.param(c, j, s, id=f"{c.__name__}-{j}-{s}")
        for c, fs in FIXTURES.items()
        for j in range(len(fs))
        for s in range(SEEDS)
    ],
)
def test_an_entry_matches_its_twin(cls, j, seed):
    make = FIXTURES[cls][j]
    step = _step(make, seed, j)
    try:
        native = to_native(step, strict=True)
    except NotNative as e:
        # A wide step may outgrow what confit builds; a narrow one may not.
        if len(step.takes) <= 4:
            raise
        pytest.skip(f"stays Python: {e}")
    rows = _rows(step, seed, getattr(make, "positive", False))
    assert check(step, native, rows) > 0


# ------------------------------------------------------------------ framework


def _scaler_step() -> PythonTransform:
    return _step(StandardScaler, 0)


def test_a_step_without_a_translation_comes_back_unchanged():
    step = _step(lambda: PCA(n_components=1), 0)
    assert to_native(step) is step
    assert "no translation for PCA" in explain_native(step)
    with pytest.raises(NotNative, match="PCA"):
        to_native(step, strict=True)


def test_to_native_is_idempotent():
    native = to_native(_scaler_step(), strict=True)
    assert to_native(native) is native


def test_a_mixed_step_stays_python():
    step = _scaler_step()
    X = np.random.default_rng(1).normal(size=(10, len(step.takes)))
    step.instances[len(step.instances)] = PCA(n_components=1).fit(X)
    assert to_native(step) is step


def test_an_unknown_id_raises_as_the_twin_does():
    from confit.oracle import Oracle

    from sql_transform.native._check import _serve
    from sql_transform.native._registry import query

    step = _scaler_step()
    native = to_native(step, strict=True)
    rows = _rows(step, 0)
    rows = rows.set_column(0, "__iid", pa.array([99] * rows.num_rows, pa.int64()))
    sql = query(step)
    assert isinstance(_serve(sql, rows, step), Exception)
    got = _serve(sql, rows, native)
    assert "not in the fitted instances" in str(got)
    with Oracle() as o:
        native.register(o)
        o.load("__THIS__", rows)
        assert "not in the fitted instances" in str(o.try_answer(sql))


@pytest.mark.parametrize(
    "make",
    [
        lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        lambda: OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        TargetEncoder,
    ],
    ids=["OrdinalEncoder", "OneHotEncoder", "TargetEncoder"],
)
def test_an_encoder_reads_a_string_feature_missing_at_fit(make):
    # Its categories are [None], which says nothing of its kind: the
    # declared type does (read as a number, it once answered wrongly).
    X = np.empty((8, 2), dtype=object)
    X[:, 0] = None
    X[:, 1] = ["a", "b", "a", "b", None, "a", "b", "a"]
    est = make().fit(X, np.array([0, 1, 0, 1, 0, 1, 1, 0]))
    width = np.asarray(est.transform(X[:1])).shape[1]
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.string()), ("x1", pa.string())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )
    rows = pa.table(
        {
            "__iid": pa.array([0] * 5, pa.int64()),
            "x0": pa.array([None, "a", "zz", None, "b"], pa.string()),
            "x1": pa.array(["a", None, "b", "zz", "a"], pa.string()),
        }
    )
    assert check(step, to_native(step, strict=True), rows) == 5


def test_a_null_id_is_a_null_struct():
    # The whole struct, as DuckDB reads it from each definition (confit
    # serves field reads, which are NULL either way).
    from confit.oracle import Oracle

    step = _step(StandardScaler, 1)
    assert pa.types.is_struct(step.returns)
    args = ", ".join(["__iid", *step.takes.names])
    rows = _rows(step, 1).slice(0, 2)
    rows = rows.set_column(0, "__iid", pa.array([None, 0], pa.int64()))
    answers = []
    for fn in (step, to_native(step, strict=True)):
        with Oracle() as o:
            fn.register(o)
            o.load("__THIS__", rows)
            answers.append(o.answer(f"SELECT tf({args}) AS s FROM __THIS__"))
    twin, native = (a.column("s").to_pylist() for a in answers)
    assert twin[0] is None and native[0] is None
    assert native[1] is not None and twin[1].keys() == native[1].keys()


def test_explain_names_the_kind():
    assert explain_native(_scaler_step()) == "'tf': StandardScaler -> SqlFunction"


def test_a_reordered_translation_is_caught(monkeypatch):
    # `x * (1/s)` is one rounding off `x / s` on some inputs: the bit-exact
    # bound must see it.
    from sql_transform.native import _registry
    from sql_transform.native._helpers import f64

    def reordered(est, x, types):
        terms = zip(x, est.mean_, est.scale_, strict=True)
        return [(xi - f64(m)) * f64(1.0 / s) for xi, m, s in terms]

    entry = _registry._CATALOG[StandardScaler]
    monkeypatch.setitem(
        _registry._CATALOG, StandardScaler, _registry.Entry(reordered, 0)
    )
    caught = 0
    for seed in range(8):
        step = _step(StandardScaler, seed)
        try:
            check(step, to_native(step, strict=True), _rows(step, seed))
        except AssertionError as e:
            assert "ulps, bound 0" in str(e)
            caught += 1
    assert caught, "a one-rounding reorder passed the bit-exact bound"
    assert entry.ulps == 0


# ------------------------------------------------------------ KBinsDiscretizer


def _kbins_step(est: Any) -> PythonTransform:
    width = np.asarray(est.transform(np.zeros((1, 2)))).shape[1]
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64()), ("x1", pa.float64())]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(width)]),
    )


@pytest.mark.parametrize("encode", ["ordinal", "onehot-dense"])
def test_kbins_counts_equal_edges_as_numpy_does(encode):
    # "uniform" keeps every edge of np.linspace, so a range a few doubles
    # wide repeats edges; served on the edges themselves and their
    # neighbours. The second feature's edges straddle zero, served with
    # both signed zeros.
    lo = 1.0
    hi = float(np.nextafter(np.nextafter(lo, 2.0), 2.0))
    X = np.array([[lo, -2.0], [hi, 2.0], [lo, 0.5]])
    est = KBinsDiscretizer(n_bins=8, encode=encode, strategy="uniform").fit(X)
    edges = est.bin_edges_[0][1:-1]
    assert len(set(edges)) < len(edges), "no repeated edge"
    assert 0.0 in est.bin_edges_[1]
    probes = sorted(
        {float(v) for e in edges for v in (np.nextafter(e, -9), e, np.nextafter(e, 9))}
    )
    zeros = [0.0, -0.0, 5e-324, -5e-324, 0.5, -0.5, 2.0, -2.0]
    n = max(len(probes), len(zeros))
    rows = pa.table(
        {
            "__iid": pa.array([0] * n, pa.int64()),
            "x0": pa.array([probes[i % len(probes)] for i in range(n)]),
            "x1": pa.array([zeros[i % len(zeros)] for i in range(n)]),
        }
    )
    step = _kbins_step(est)
    assert check(step, to_native(step, strict=True), rows) == n


@pytest.mark.parametrize(
    "make, why",
    [
        (lambda: KBinsDiscretizer(encode="onehot"), "sparse"),
        (lambda: KBinsDiscretizer(encode="ordinal", dtype=np.float32), "float32"),
    ],
    ids=["onehot", "float32"],
)
def test_kbins_refuses(make, why):
    X = np.random.default_rng(0).normal(size=(20, 2))
    est = make().fit(X)
    step = PythonTransform(
        "tf",
        {0: est},
        pa.schema([("x0", pa.float64()), ("x1", pa.float64())]),
        pa.float64(),
    )
    with pytest.raises(NotNative, match=why):
        to_native(step, strict=True)


@pytest.mark.parametrize(
    "make, reason",
    [
        (PowerTransformer, "method='yeo-johnson'"),
        (lambda: PowerTransformer(standardize=False), "method='yeo-johnson'"),
        (lambda: PowerTransformer("box-cox"), "standardize=True"),
    ],
    ids=["yeo-johnson", "yeo-johnson-unstandardized", "box-cox-standardized"],
)
def test_a_power_transform_without_a_small_bound_stays_python(make, reason):
    step = _step(positive(lambda: make()), 0)
    with pytest.raises(NotNative, match=reason):
        to_native(step, strict=True)
