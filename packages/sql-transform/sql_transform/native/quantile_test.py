"""QuantileTransformer beyond the catalog's generated fits: the default
1,000 quantiles, serving values on and next to every quantile, and the
fits whose quantiles are degenerate."""

from __future__ import annotations

import warnings

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import QuantileTransformer

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, check, to_native
from sql_transform.native.quantile import interp_is_numpys


def _step(X: np.ndarray, **params) -> PythonTransform:
    with warnings.catch_warnings():  # n_quantiles past the rows, all-NaN
        warnings.simplefilter("ignore")
        est = QuantileTransformer(**params).fit(X)
    n = X.shape[1]
    takes = pa.schema([(f"x{i}", pa.float64()) for i in range(n)])
    returns = (
        pa.float64()
        if n == 1
        else pa.struct([(f"f{j}", pa.float64()) for j in range(n)])
    )
    return PythonTransform("tf", {0: est}, takes, returns)


def _rows(step: PythonTransform, seed: int) -> pa.Table:
    """Every quantile, its neighbouring doubles, edges and draws, per
    feature, each in a row of its own."""
    est = step.instances[0]
    rng = np.random.default_rng(seed)
    cols = []
    for q in est.quantiles_.T:
        finite = q[~np.isnan(q)]
        lo, hi = (finite.min() - 1, finite.max() + 1) if len(finite) else (-1, 1)
        v = np.concatenate(
            [
                q,
                np.nextafter(q, np.inf),
                np.nextafter(q, -np.inf),
                rng.uniform(lo, hi, 200),
                [0.0, -0.0, np.nan, 1e300, -1e300, 5e-324],
            ]
        )
        cols.append(v)
    n = max(len(c) for c in cols)
    cols = [np.resize(c, n) for c in cols]
    table = {"__iid": pa.array([0] * n, pa.int64())}
    for i, c in enumerate(cols):
        table[f"x{i}"] = pa.array(c, pa.float64())
    return pa.table(table)


def test_this_platform_interpolates_as_the_entry():
    assert interp_is_numpys()


def test_the_default_thousand_quantiles():
    rng = np.random.default_rng(0)
    n = 3000
    X = np.column_stack(
        [
            rng.normal(5, 3, n),
            rng.integers(-3, 4, n).astype(float),  # runs of equal quantiles
        ]
    )
    X[rng.random(n) < 0.05, 0] = np.nan
    step = _step(X)
    assert step.instances[0].n_quantiles_ == 1000
    assert check(step, to_native(step, strict=True), _rows(step, 0)) > 0


def test_subsampled_quantiles():
    rng = np.random.default_rng(1)
    X = rng.lognormal(0, 2, size=(500, 2))
    step = _step(X, n_quantiles=50, subsample=100, random_state=3)
    assert check(step, to_native(step, strict=True), _rows(step, 1)) > 0


@pytest.mark.parametrize(
    "column, params",
    [
        (np.full(20, 2.5), {}),  # one quantile value, repeated
        (np.full(20, np.nan), {}),  # missing everywhere
        (np.arange(20.0), {"n_quantiles": 1}),
        (np.array([0.0, -0.0] * 10), {}),
        (np.array([-1.0] * 9 + [0.0] + [3.0] * 10), {"n_quantiles": 5}),
        # Quantiles a subnormal apart: infinite slopes, NaN at their start.
        (np.arange(5.0) * 5e-324, {"n_quantiles": 5}),
    ],
    ids=["constant", "all-nan", "one-quantile", "signed-zeros", "runs", "subnormal"],
)
def test_degenerate_quantiles(column, params):
    step = _step(column[:, None], **params)
    assert check(step, to_native(step, strict=True), _rows(step, 2)) > 0


def test_past_the_quantile_budget_stays_python():
    step = _step(np.random.default_rng(4).normal(size=(1500, 3)))
    with pytest.raises(NotNative, match="3000 quantiles, past the 2000"):
        to_native(step, strict=True)


def test_a_normal_output_stays_python():
    step = _step(np.arange(20.0)[:, None], output_distribution="normal")
    with pytest.raises(NotNative, match="norm.ppf"):
        to_native(step, strict=True)
