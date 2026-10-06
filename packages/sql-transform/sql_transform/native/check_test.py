"""A parity bound in an error scale (`ErrorScale`), on a test-only family:
a dot product that the twin sums right to left and its entry left to
right, so that the two part by rounding alone. Each order is within
γ_n·S of the exact sum, S = Σ|x_i·c_i| and γ_n about n·eps/2, so the two
are within about n·eps·S: K = n + 1 covers it
(decisions/closed/matvec-parity-bound.md)."""

from __future__ import annotations

import math
import re
import sys

import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sql_transform._udf import PythonTransform
from sql_transform.native import (
    Entry,
    ErrorScale,
    NotNative,
    ParityError,
    bound_of,
    check,
    coverage,
    explain_native,
    to_native,
    translates,
)
from sql_transform.native._check import near
from sql_transform.native._helpers import f64

DBL_MAX = sys.float_info.max


class _Dot(BaseEstimator, TransformerMixin):
    """Output field k is Σ_i x_i·c[k][i], summed from the last feature."""

    def __init__(self, c=((1.0,),)):
        self.c = c

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.c[0])
        return self

    def transform(self, X):
        out = []
        for row in np.asarray(X, dtype=float):
            fields = []
            for ck in self.c:
                acc = 0.0
                for xi, ci in reversed(list(zip(row.tolist(), ck, strict=True))):
                    acc = acc + xi * ci
                fields.append(acc)
            out.append(fields)
        return np.array(out)


def _left_to_right(est, x, types):
    out = []
    for ck in est.c:
        e = x[0] * f64(ck[0])
        for xi, ci in zip(x[1:], ck[1:], strict=True):
            e = e + xi * f64(ci)
        out.append(e)
    return out


def _without_the_last_term(est, x, types):
    out = []
    for ck in est.c:
        e = x[0] * f64(ck[0])
        for xi, ci in zip(x[1:-1], ck[1:-1], strict=True):
            e = e + xi * f64(ci)
        out.append(e)
    return out


SCALE = ErrorScale(
    k=lambda est: est.n_features_in_ + 1,
    s=lambda est, x: np.abs(np.asarray(est.c)) @ np.abs(x),
    tau=lambda est: est.n_features_in_ * 2.0**-1074,
)


@pytest.fixture
def dot(monkeypatch):
    """`_Dot` in the catalog, within its error scale; `translate` swaps
    its translator."""
    from sql_transform.native import _registry

    def register(translate=_left_to_right):
        entry = Entry(translate, 0, None, SCALE)
        monkeypatch.setitem(_registry._CATALOG, _Dot, entry)

    register()
    return register


def _step(c) -> PythonTransform:
    est = _Dot(tuple(tuple(r) for r in c)).fit(None)
    n = len(c[0])
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", pa.float64()) for i in range(n)]),
        pa.struct([(f"f{k}", pa.float64()) for k in range(len(c))]),
    )


def _rows(x) -> pa.Table:
    x = np.asarray(x, dtype=float)
    cols = {f"x{i}": pa.array(x[:, i]) for i in range(x.shape[1])}
    return pa.table({"__iid": pa.array([0] * len(x), pa.int64()), **cols})


def _cancelling(n: int, rows: int, seed: int = 0) -> np.ndarray:
    """Rows whose terms span six decades and cancel: the two orders part."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, n)) * 10.0 ** rng.uniform(-3, 3, size=(rows, n))


def test_two_orders_part_within_the_error_scale(dot):
    c = np.random.default_rng(1).normal(size=(3, 8))
    step = _step(c)
    native = to_native(step, strict=True, allow_bound=True)
    rows = _rows(_cancelling(8, 64))
    with pytest.raises(ParityError, match="ulps, bound 0"):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == 64


def test_a_dropped_term_breaches_the_error_scale(dot):
    dot(_without_the_last_term)
    c = np.random.default_rng(1).normal(size=(3, 8))
    step = _step(c)
    native = to_native(step, strict=True, allow_bound=True)
    with pytest.raises(ParityError, match=r"lane 'f0'.*past its bound K\*eps\*S"):
        check(step, native, _rows(_cancelling(8, 8)))


def test_nan_and_null_rows_compare_as_nan(dot):
    step = _step([[1.0, -2.0, 0.5]])
    native = to_native(step, strict=True, allow_bound=True)
    rows = pa.table(
        {
            "__iid": pa.array([0, 0, None], pa.int64()),
            "x0": pa.array([None, math.nan, 1.0]),
            "x1": pa.array([1.0, 2.0, 2.0]),
            "x2": pa.array([3.0, 4.0, 3.0]),
        }
    )
    assert check(step, native, rows) == 3


def test_an_overflow_on_one_side_passes_at_dbl_max(dot):
    # Left to right, DBL_MAX + 2**970 is a tie that rounds to even, past
    # DBL_MAX: the entry answers inf. Right to left the twin adds 2**969
    # to DBL_MAX, a quarter of its ulp: it answers DBL_MAX.
    step = _step([[1.0, 1.0, 1.0]])
    native = to_native(step, strict=True, allow_bound=True)
    x = [DBL_MAX, 2.0**970, -(2.0**969)]
    assert step.instances[0].transform([x])[0, 0] == DBL_MAX
    rows = _rows([x])
    with pytest.raises(
        ParityError, match="'f0': step 1.7976931348623157e[+]308, native inf"
    ):
        check(step, native, rows, ulps=0)
    assert check(step, native, rows) == 1


@pytest.mark.skipif(
    np.finfo(np.longdouble).maxexp <= 1024, reason="a long double is a double here"
)
def test_an_error_scale_past_dbl_max_still_bounds(dot):
    # Left to right, 1e308 + 1e308 overflows and the entry answers inf;
    # right to left the twin answers 1e308, far from DBL_MAX. S, 3e308, is
    # past DBL_MAX: in long doubles the bound is 2.7e293, not inf.
    step = _step([[1.0, 1.0, 1.0]])
    native = to_native(step, strict=True, allow_bound=True)
    with pytest.raises(ParityError, match="native inf, past its bound"):
        check(step, native, _rows([[1e308, 1e308, -1e308]]))


@pytest.mark.parametrize(
    "a, b, bound, g, ok",
    [
        (math.nan, math.nan, 0.0, None, True),
        (math.nan, 1.0, math.inf, None, False),
        (math.inf, math.inf, 0.0, None, True),
        (math.inf, -math.inf, math.inf, None, False),
        # One side infinite: the other within the bound of DBL_MAX, same sign.
        (math.inf, DBL_MAX, 0.0, None, True),
        (-math.inf, -DBL_MAX, 0.0, None, True),
        (math.inf, -DBL_MAX, 1e300, None, False),
        (math.inf, 1e308, 1e300, None, False),
        (math.inf, 1e308, 8e307, None, True),
        # Signed zeros are one value.
        (0.0, -0.0, 0.0, None, True),
        # τ: two subnormals apart by more than an error scale of 0 allows.
        (5e-324, 1e-323, 0.0, None, False),
        (5e-324, 1e-323, 5e-324, None, True),
        # g: a distance compares its square.
        (3.0, 3.0000000000000004, 1e-15, None, True),
        (3.0, 3.0000000000000004, 1e-15, np.square, False),
        # Squares past DBL_MAX are long doubles: one side overflowed.
        (1.4e154, math.inf, 0.0, np.square, True),
        (1e150, math.inf, 1e300, np.square, False),
    ],
)
def test_near(a, b, bound, g, ok):
    assert near(a, b, bound, g) is ok
    assert near(b, a, bound, g) is ok


def test_an_error_scale_serves_only_on_request(dot):
    step = _step([[1.0, 2.0, 3.0]])
    est = step.instances[0]
    within = r"within 4·eps·S \+ τ \(S its error scale\)"
    with pytest.raises(NotNative, match=f"instance 0: _Dot is {within} of its twin"):
        to_native(step, strict=True)
    assert to_native(step) is step
    assert re.search(within + "$", explain_native(step, allow_bound=True))
    with pytest.raises(ValueError, match="its parity bound is an error scale"):
        bound_of(est)


def test_a_composition_refuses_an_error_scale(dot):
    pipe = Pipeline([("dot", _Dot(((1.0, 2.0),)))]).fit(np.zeros((2, 2)))
    step = PythonTransform(
        "p", {0: pipe}, pa.schema([("x0", pa.float64()), ("x1", pa.float64())])
    )
    with pytest.raises(
        NotNative,
        match=r"Pipeline step 'dot': _Dot is within 3·eps·S \+ τ \(S its error"
        r" scale\), which no later step keeps bounded",
    ):
        to_native(step, strict=True, allow_bound=True)


def test_an_error_scale_is_the_whole_bound():
    with pytest.raises(ValueError, match="an error scale is the whole bound"):
        translates(type("_Fake", (), {}), ulps=2, scale=SCALE)


def test_coverage_reads_an_error_scale():
    assert (
        coverage._exactness(Entry(_left_to_right, 0, None, SCALE))
        == "within K·eps·S + τ (S its error scale), served only with"
        " `allow_bound=True`"
    )
