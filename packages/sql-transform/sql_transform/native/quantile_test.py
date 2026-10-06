"""QuantileTransformer beyond the catalog's generated fits: the default
1,000 quantiles, serving values on and next to every quantile, the fits
whose quantiles are degenerate, and the one-tree derivation checked
against the twin over many fits without confit."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pyarrow as pa
import pytest
from sklearn.preprocessing import QuantileTransformer

from sql_transform._udf import PythonTransform
from sql_transform.native import NotNative, check, to_native
from sql_transform.native.quantile import _both, _leaves, interp_is_numpys


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


def _served(x: float, q: np.ndarray, r: np.ndarray) -> float:
    """What the entry answers, on a Python float: `_feature`'s arms, then
    the leaf whose start is the last at or below `x` (the tree's
    bisection on `x < start`)."""
    if math.isnan(x):
        return x
    if x <= q[0]:
        return 0.0
    if x >= q[-1]:
        return 1.0
    up, down, at = next(lf for lf in reversed(_leaves(q, r)) if lf[0][0] <= x)
    if at is not None and x == up[0]:
        return at
    return _both(x, up, down)


def _column(rng: np.random.Generator) -> np.ndarray:
    """A fit column with runs, signed zeros, subnormal spacings (slopes
    that overflow) or none of these."""
    n = int(rng.integers(3, 80))
    kind = rng.integers(5)
    if kind == 0:
        return rng.normal(size=n) * 10.0 ** rng.integers(-5, 5)
    if kind == 1:
        return rng.integers(-3, 4, n).astype(float)
    if kind == 2:
        return rng.integers(-3, 4, n) * 5e-324
    if kind == 3:
        return rng.choice([-1.0, -0.0, 0.0, 1.0], n)
    return np.concatenate([rng.normal(size=n), rng.integers(0, 3, n) * 5e-324])


def test_one_tree_answers_as_the_twin_on_and_between_breakpoints():
    """2,000 fits, each served at every quantile, its neighbouring doubles,
    both zeros and draws; a third with each zero's sign redrawn, so runs of
    zeros start and end on either sign. Bit for bit, from Python floats."""
    rng = np.random.default_rng(20261005)
    for _ in range(2000):
        col = _column(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = QuantileTransformer(
                n_quantiles=int(rng.integers(2, 2 * len(col)))
            ).fit(col[:, None])
        q = est.quantiles_[:, 0]
        if rng.random() < 0.3:
            zero = q == 0.0
            q[zero] = rng.choice([-0.0, 0.0], int(zero.sum()))
        x = np.concatenate(
            [
                q,
                np.nextafter(q, np.inf),
                np.nextafter(q, -np.inf),
                [0.0, -0.0, 5e-324, -5e-324, np.nan],
                rng.uniform(q[0] - 1, q[-1] + 1, 20),
            ]
        )
        want = est.transform(x[:, None])[:, 0]
        r = est.references_
        for v, w in zip(x.tolist(), want.tolist(), strict=True):
            assert repr(_served(v, q, r)) == repr(w), (q.tolist(), v)


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
        # Runs of zeros whose signs alternate inside the run.
        (np.array([-1.0] * 3 + [-0.0] * 10 + [1.0] * 3), {"n_quantiles": 16}),
        # Runs a subnormal apart: arms at every interior breakpoint.
        (np.repeat(np.arange(5.0) * 5e-324, 3), {"n_quantiles": 8}),
        (
            np.array([-5e-324] * 4 + [0.0, -0.0] * 4 + [5e-324] * 4 + [1e-300] * 3),
            {"n_quantiles": 9},
        ),
    ],
    ids=[
        "constant",
        "all-nan",
        "one-quantile",
        "signed-zeros",
        "runs",
        "subnormal",
        "signed-zero-runs",
        "subnormal-runs",
        "subnormal-across-zero",
    ],
)
def test_degenerate_quantiles(column, params):
    step = _step(column[:, None], **params)
    assert check(step, to_native(step, strict=True), _rows(step, 2)) > 0


@pytest.mark.parametrize(
    "zeros",
    [[-0.0, 0.0], [0.0, -0.0], [-0.0, -0.0], [-0.0, 0.0, -0.0], [0.0]],
    ids=["neg-pos", "pos-neg", "neg-neg", "neg-pos-neg", "pos"],
)
def test_runs_of_zeros_of_either_sign(zeros):
    """A run of zeros whose first and last quantile carry each sign (set
    on the fit; percentiles seldom land on a signed zero): the ascending
    search starts its piece at the last, the mirrored search ends its
    piece at the first."""
    q = np.array([-2.0, -1.0, *zeros, 1.0, 3.0])
    step = _step(q[:, None], n_quantiles=len(q))
    step.instances[0].quantiles_[:, 0] = q
    assert check(step, to_native(step, strict=True), _rows(step, 3)) > 0


def test_past_the_quantile_budget_stays_python():
    X = np.random.default_rng(4).normal(size=(2500, 9))
    step = _step(X, n_quantiles=1000)
    with pytest.raises(NotNative, match="9000 quantiles, past the 8000"):
        to_native(step, strict=True)


def test_a_normal_output_stays_python():
    step = _step(np.arange(20.0)[:, None], output_distribution="normal")
    with pytest.raises(NotNative, match="norm.ppf"):
        to_native(step, strict=True)
