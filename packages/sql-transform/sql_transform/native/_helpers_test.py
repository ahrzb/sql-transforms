"""The order helpers against numpy itself: each helper's SQL is run by the
oracle (DuckDB, which confit answers exactly) over rows of many widths, and
compared bit for bit with the numpy operation the twin performs on that row
(`_helpers.py` says which). The widths reach past numpy's pairwise blocks
(8, 128), where an order mistake would show."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from confit import sql as S
from confit.oracle import Oracle
from sklearn.externals.array_api_compat import numpy as np_compat
from sklearn.utils._array_api import _modify_in_place_if_numpy
from sklearn.utils.extmath import row_norms

from sql_transform.native._helpers import (
    SameTree,
    clip,
    f64,
    isnan,
    row_max,
    row_sum,
    row_sumsq,
    row_sumsq_is_numpys,
)

# The edge rows overflow and meet inf - inf on purpose.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

WIDTHS = [1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100, 127, 128, 129, 136, 257, 300]


def _same(a: float, b: float) -> bool:
    """Bit-equal, NaN payloads aside: -0.0 is not 0.0."""
    return (math.isnan(a) and math.isnan(b)) or repr(a) == repr(b)


def _matrix(seed: int, n: int, rows: int = 48) -> np.ndarray:
    """Rows of one scale (where summation order shows most) and rows of
    wildly mixed magnitude and sign, with exact ties, signed zeros,
    subnormals, values whose squares overflow, and some rows of only zeros
    or holding a NaN or an infinity."""
    rng = np.random.default_rng(seed * 1009 + n)
    x = rng.normal(size=(rows, n)) * 10.0 ** rng.integers(-12, 12, size=(rows, n))
    pick = rng.random((rows, n))
    pick[rows // 2 :] = 1.0  # the one-scale half takes no specials
    x[rows // 2 :] = rng.normal(size=(rows - rows // 2, n))
    x[pick < 0.05] = 0.0
    x[(pick >= 0.05) & (pick < 0.1)] = -0.0
    x[(pick >= 0.1) & (pick < 0.12)] = 5e-324
    x[(pick >= 0.12) & (pick < 0.14)] = 1e300
    x[(pick >= 0.14) & (pick < 0.2)] = np.round(x[(pick >= 0.14) & (pick < 0.2)])
    x[0] = -0.0
    x[1] = 0.0
    x[2, 0] = np.nan
    x[3, n - 1] = np.inf
    x[4, :] = 1e308
    x[4, n // 2] = -np.inf
    return x


def _oracle(expr, x: np.ndarray) -> list[float]:
    """DuckDB's reading of `expr(columns)` over each row of `x`."""
    names = [f"c{i}" for i in range(x.shape[1])]
    e = expr([S.col(c) for c in names])
    with Oracle() as o:
        o.load("t", pa.table({c: x[:, i] for i, c in enumerate(names)}))
        return o.answer(f"SELECT {e.sql()} AS r FROM t").column("r").to_pylist()


def _assert_rows(got: list[float], want: list[float], x: np.ndarray) -> None:
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert _same(g, w), f"row {i}: sql {g!r}, numpy {w!r}; row {x[i].tolist()}"


@pytest.mark.parametrize("n", WIDTHS)
def test_row_sum_is_numpys(n):
    x = _matrix(0, n)
    want = [float(np.sum(x[i : i + 1], axis=1)[0]) for i in range(len(x))]
    _assert_rows(_oracle(row_sum, x), want, x)
    # And on Python floats, the same order is numpy's arithmetic.
    _assert_rows([row_sum(r.tolist()) for r in x], want, x)


@pytest.mark.parametrize("n", WIDTHS)
def test_row_sumsq_is_row_norms(n):
    if not row_sumsq_is_numpys():
        pytest.skip("this platform's einsum kernel is not the modelled one")
    x = _matrix(1, n)
    want = [float(row_norms(x[i : i + 1], squared=True)[0]) for i in range(len(x))]
    _assert_rows(_oracle(row_sumsq, x), want, x)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 32, 48])
def test_row_max_is_numpys(n):
    # The twin validates its rows finite: no NaN here.
    x = np.nan_to_num(_matrix(2, n), nan=1.0, posinf=7.0, neginf=-7.0)
    want = [float(np.max(np.abs(x[i : i + 1]), axis=1)[0]) for i in range(len(x))]
    got = _oracle(lambda c: row_max([S.fn("abs", e) for e in c]), x)
    _assert_rows(got, want, x)


VALUES = [
    *(float(v) for v in (math.nan, -math.nan, math.inf, -math.inf, 0.0, -0.0)),
    *(1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1e-300, -5e-324, 1e300, 0.999999, -1.0000001),
]


@pytest.mark.parametrize(
    "lo, hi",
    [(0, 1), (-1.0, 1.0), (-1.0, 0.0), (0.0, 0.0), (-0.0, 0.0), (2.0, 1.0),
     (-math.inf, 0.5), (0.5, math.inf)],
)  # fmt: skip
def test_clip_is_sklearns(lo, hi):
    # As MinMaxScaler/MaxAbsScaler call it: sklearn's numpy namespace, in place.
    x = np.array([VALUES])
    y = x.copy()
    want = _modify_in_place_if_numpy(
        np_compat,
        np_compat.clip,
        y,
        np_compat.asarray(lo, dtype=x.dtype),
        np_compat.asarray(hi, dtype=x.dtype),
        out=y,
    )
    got = _oracle(lambda c: clip(c[0], lo, hi), x.T)
    _assert_rows(got, want[0].tolist(), x.T)


def test_isnan_is_numpys():
    x = np.array([VALUES]).T
    got = _oracle(lambda c: S.case(isnan(c[0]), 1.0).otherwise(0.0), x)
    assert got == [float(v) for v in np.isnan(x[:, 0])]


def _doubling(rounds: int, leaf: float) -> S.Expr:
    """A recurrence that reads each value twice: its text doubles per round
    (2**60 reads of `x` at 60 rounds), its nodes grow by one."""
    v = S.col("x") + f64(leaf)
    for _ in range(rounds):
        v = v * v
    return v


def test_same_tree_keys_a_tree_not_its_objects():
    same = SameTree()
    a = _doubling(60, 1.0)
    assert same.key(a) == same.key(_doubling(60, 1.0))
    assert same.key(a) != same.key(_doubling(60, 2.0))
    assert same.key(a) != same.key(_doubling(59, 1.0))


def test_same_tree_agrees_with_the_sql():
    x, y = S.col("x"), S.col("y")
    makers = [
        lambda: x,
        lambda: y,
        lambda: f64(0.0),
        lambda: f64(-0.0),
        lambda: f64(math.nan),
        lambda: S.lit(1),
        lambda: x + y,
        lambda: y + x,
        lambda: x + f64(0.0),
        lambda: x + f64(-0.0),
        lambda: x - y,
        lambda: S.case(x < f64(1.0), x).otherwise(y),
        lambda: S.case(x < f64(1.0), x),
        lambda: S.case(x < f64(1.0), x).when(y < f64(1.0), y),
        lambda: x.isin(1, 2),
        lambda: x.isin(2, 1),
        lambda: S.fn("least", x, y),
        lambda: S.fn("greatest", x, y),
        lambda: ~(x < y),
        lambda: x.isnull(),
    ]
    same = SameTree()
    # Each built twice, from new objects: one key per maker, as one text.
    keys = [{same.key(m()) for _ in range(2)} for m in makers]
    texts = [m().sql() for m in makers]
    assert len(set(texts)) == len(makers)
    assert all(len(k) == 1 for k in keys)
    assert len(set().union(*keys)) == len(makers)
