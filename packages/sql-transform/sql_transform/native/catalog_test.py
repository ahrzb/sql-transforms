"""Every catalog entry is held to its declared bound against its Python twin,
on generated fits and generated serving rows (swap-the-entry, `check`).

Adding an entry means adding its fixtures here: `FIXTURES` maps each
registered class to estimator factories covering its configurations, and
`test_every_entry_has_fixtures` fails until it does.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
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
}

# Serving values beyond the fit's range: signed zeros, extremes.
EDGES = [0.0, -0.0, 1e-300, -1e300, 1e300, 5e-324]
# A row of only these has a norm under sklearn's zero-scale threshold.
SMALL = [0.0, -0.0, 1e-300, -5e-324, 1e-17, -2.5e-16]


def _fit_matrix(rng: np.random.Generator, n_features: int) -> np.ndarray:
    n = int(rng.integers(5, 60))
    cols = []
    for _ in range(n_features):
        kind = rng.integers(4)
        if kind == 0:
            cols.append(rng.normal(rng.uniform(-100, 100), rng.uniform(0.01, 50), n))
        elif kind == 1:
            cols.append(rng.integers(-1000, 1000, n).astype(float))
        elif kind == 2:
            cols.append(np.full(n, rng.uniform(-5, 5)))  # zero variance
        else:
            cols.append(rng.exponential(rng.uniform(0.1, 1e6), n))
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
    instances = {}
    for k in range(int(rng.integers(1, 4))):
        X = _fit_matrix(rng, n_features)
        for j, t in enumerate(types):
            if t == pa.int64():
                X[:, j] = np.round(X[:, j])
        instances[k] = cls_factory().fit(X)
    width = np.asarray(instances[0].transform(X[:1])).reshape(1, -1).shape[1]
    returns = (
        pa.float64()
        if width == 1
        else pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
    )
    return PythonTransform("tf", instances, takes, returns)


def _value(rng: random.Random, regime: str) -> float | None:
    r = rng.random()
    if regime == "nulls" and r < 0.3:
        return None
    if regime == "edges" and r < 0.3:
        return rng.choice(EDGES)
    if regime == "small":
        return rng.choice(SMALL)
    return rng.uniform(-1e3, 1e3)


def _rows(step: PythonTransform, seed: int) -> pa.Table:
    rng = random.Random(seed)  # noqa: S311
    n = 40
    ids = [rng.choice([None, *step.instances]) for _ in range(n)]
    # A row's regime: plain values; some edges; some NULLs (where most
    # twins answer and a validating one raises); or all small.
    regimes = rng.choices(["plain", "edges", "nulls", "small"], [55, 15, 15, 15], k=n)
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
