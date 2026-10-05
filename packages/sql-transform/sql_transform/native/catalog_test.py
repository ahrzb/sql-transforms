"""Every catalog entry is held to its declared bound against its Python twin,
on generated fits and generated serving rows (swap-the-entry, `check`).

Adding an entry means adding its fixtures here: `FIXTURES` maps each
registered class to estimator factories covering its configurations, and
`test_every_entry_has_fixtures` fails until it does.
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    NotNative,
    catalog,
    check,
    explain_native,
    to_native,
)

# An imputer whose fit saw a column only missing warns at every transform
# that it drops it; the fixtures make such columns on purpose.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Skipping features without any observed values:UserWarning"
)

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
}

# Serving values beyond the fit's range: signed zeros, extremes.
EDGES = [0.0, -0.0, 1e-300, -1e300, 1e300, 5e-324]
# A row of only these has a norm under sklearn's zero-scale threshold.
SMALL = [0.0, -0.0, 1e-300, -5e-324, 1e-17, -2.5e-16]


def _fit_matrix(
    rng: np.random.Generator,
    types: list[pa.DataType],
    holes: list[str | None],
    marker: float,
) -> np.ndarray:
    """One instance's fit data, a column per feature type (an integer one
    rounded). `holes[j]` says where column j is missing, the same for every
    instance of a step (so fitted widths agree): None nowhere (and never
    holding the marker), "some" in about a fifth of its rows and at least
    one, "all" everywhere. `marker` spells missing."""
    n = int(rng.integers(5, 60))
    cols = []
    for t, hole in zip(types, holes, strict=True):
        kind = rng.integers(5)
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
        if not np.isnan(marker):
            c[c == marker] = marker + 1.0
        if hole == "all":
            c[:] = marker
        elif hole == "some":
            c[rng.random(n) < 0.2] = marker
            c[rng.integers(n)] = marker
        cols.append(c)
    return np.column_stack(cols)


def _step(cls_factory, seed: int) -> PythonTransform:
    rng = np.random.default_rng(seed)
    # Mostly narrow; sometimes wide enough for a row reduction's blocks.
    wide = rng.random() < 0.3
    n_features = int(rng.integers(5, 33) if wide else rng.integers(1, 5))
    types = [
        pa.float64() if rng.random() < 0.7 else pa.int64() for _ in range(n_features)
    ]
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    # Missing values in the fit data, for an estimator that takes them: an
    # imputer's own `missing_values`, or NaN where sklearn says it allows it.
    proto = cls_factory()
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
    instances = {}
    with warnings.catch_warnings():  # all-missing columns, by design
        warnings.simplefilter("ignore")
        for k in range(int(rng.integers(1, 4))):
            X = _fit_matrix(rng, types, holes, marker)
            instances[k] = cls_factory().fit(X)
        width = np.asarray(instances[0].transform(X[:1])).reshape(1, -1).shape[1]
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


def _rows(step: PythonTransform, seed: int) -> pa.Table:
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
        vals = [_value(rng, g) for g in regimes]
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
        for s in range(8)
    ],
)
def test_an_entry_matches_its_twin(cls, j, seed):
    step = _step(FIXTURES[cls][j], seed)
    try:
        native = to_native(step, strict=True)
    except NotNative as e:
        # A wide step may outgrow what confit builds; a narrow one may not.
        if len(step.takes) <= 4:
            raise
        pytest.skip(f"stays Python: {e}")
    assert check(step, native, _rows(step, seed)) > 0


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

    def reordered(est, x):
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
