"""Quantile transforms: each feature through its fitted quantiles onto
[0, 1], in sklearn's own operation order (sklearn 1.9,
`QuantileTransformer._transform_col` in `sklearn/preprocessing/_data.py`).

The twin, per feature with quantiles `q` and references `r` (`linspace(0,
1, n)`), on the step's one-row column `x`:

    lower, upper = x == q[0], x == q[-1]
    x[~isnan(x)] = 0.5 * (np.interp(x, q, r) - np.interp(-x, -q[::-1], -r[::-1]))
    x[upper] = 1; x[lower] = 0

so NaN stays NaN, at or below `q[0]` is 0, at or above `q[-1]` is 1 (the
interps' fills, or the bounds), and the line is only met strictly inside
`(q[0], q[-1])`. Interpolating from
both ends and averaging is the point of the form: a run of equal quantiles
answers its last reference ascending and its first descending.

`np.interp` is numpy's C `arr_interp` (numpy 2.5,
`numpy/_core/src/multiarray/compiled_base.c`). On sorted `xp` its
`binary_search_with_guess` finds the last `j` with `xp[j] <= x`, whatever
its guess; an `x` strictly inside then answers `fp[j]` when `x == xp[j]`,
else `slope * (x - xp[j]) + fp[j]` with `slope = (fp[j+1] - fp[j]) /
(xp[j+1] - xp[j])` (precomputed when `len(xp) <= len(x)`, by the same
expression), retrying from `xp[j+1]` on a NaN. The entry spells that as a
balanced CASE over the intervals `xp[j] < xp[j+1]` (depth log2 of their
count), each leaf the interval's line, with the slopes computed here in
the same float64 arithmetic. The `x == xp[j]` arm is kept only where it
changes the answer (an infinite slope, whose line is NaN at `xp[j]`); the
NaN retry is never taken inside an interval whose width is finite, and a
fit with an infinite width stays Python. `interp_is_numpys` checks that
this platform's `np.interp` rounds as the leaves do (a compiler may fuse
the multiply-add, which SQL cannot spell).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import QuantileTransformer

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates

# The most quantiles served per estimator, over its features: their sum
# and the sum of their squares. A feature of q quantiles builds in about
# 0.57 ms * q + 0.87 us * q^2 (one feature: 1.4 s at 1,000, 4.4 s at
# 2,000, 16 s at 4,000; PLANS, "Needs from confit"), so the sum bounds the
# first term and the squares the second. The slowest steps served: the
# default 1,000 quantiles over four features build in 6.7 s, 500 over
# eight in 5.3 s, 100 over 40 in 3.6 s, and each serves 64 rows 18 to 35
# times faster than the twin (release build, master 49acad5, 2026-10-05).
MAX_QUANTILES = 4000
MAX_SQUARES = 4_000_000

# One interval of `np.interp`'s line: x in [start, next start) answers
# `slope * (x - start) + value`, or `value` at `start` when `exact`.
_Piece = tuple[float, float, float, bool]


def _pieces(xp: np.ndarray, fp: np.ndarray) -> list[_Piece]:
    """The intervals `np.interp(x, xp, fp)` answers an `x` strictly inside
    `(xp[0], xp[-1])` from, in order: one per `j` with `xp[j] < xp[j+1]`,
    since numpy's search lands on the last of a run of equal points."""
    out = []
    for j in range(len(xp) - 1):
        x0, x1, y0, y1 = float(xp[j]), float(xp[j + 1]), float(fp[j]), float(fp[j + 1])
        if not x0 < x1:
            continue
        if math.isinf(x1 - x0):
            raise NotNative(
                "QuantileTransformer: quantiles further apart than a double"
                " spans, where numpy's interp retries a NaN line"
            )
        slope = (y1 - y0) / (x1 - x0)
        out.append((x0, slope, y0, not math.isfinite(slope)))
    return out


def _line(x: Any, piece: _Piece) -> Any:
    start, slope, value, _ = piece
    if isinstance(x, S.Expr):
        return f64(slope) * (x - f64(start)) + f64(value)
    return slope * (x - start) + value


def _leaf(x: S.Expr, piece: _Piece) -> S.Expr:
    if piece[3]:
        return S.case(x == f64(piece[0]), f64(piece[2])).otherwise(_line(x, piece))
    return _line(x, piece)


def _tree(x: S.Expr, pieces: list[_Piece]) -> S.Expr:
    """The piece `x` falls in, by bisection on the starts."""
    if len(pieces) == 1:
        return _leaf(x, pieces[0])
    mid = len(pieces) // 2
    return S.case(x < f64(pieces[mid][0]), _tree(x, pieces[:mid])).otherwise(
        _tree(x, pieces[mid:])
    )


def _interp_py(x: float, pieces: list[_Piece]) -> float:
    """What `_tree` answers, on a Python float."""
    piece = next(p for p in reversed(pieces) if p[0] <= x)
    if piece[3] and x == piece[0]:
        return piece[2]
    return _line(x, piece)


@functools.cache
def interp_is_numpys() -> bool:
    """Whether this platform's `np.interp` answers as `_pieces` spells it,
    on probe fits with runs of equal points, one `x` at a time and many."""
    rng = np.random.default_rng(20261005)
    for n in (2, 3, 5, 8, 40, 300):
        for _ in range(8):
            xp = np.sort(rng.normal(size=n) * 10.0 ** rng.integers(-6, 6))
            xp[rng.random(n) < 0.3] = xp[0]
            xp = np.sort(xp)
            fp = np.linspace(0, 1, n)
            if not xp[0] < xp[-1]:
                continue
            x = rng.uniform(xp[0], xp[-1], size=2 * n)
            x = np.concatenate([x, xp[1:-1]])
            x = x[(x > xp[0]) & (x < xp[-1])]
            pieces = _pieces(xp, fp)
            want = [_interp_py(float(v), pieces) for v in x]
            if np.interp(x, xp, fp).tolist() != want:
                return False
            if [float(np.interp(v, xp, fp)) for v in x] != want:
                return False
    return True


def _feature(x: S.Expr, q: np.ndarray, r: np.ndarray) -> S.Expr:
    n = len(q)
    if n == 1:
        # numpy's one-point interp answers fp[0] for any number: 0.0 from
        # 0.5 * (0.0 - -0.0), the bounds answering 0.0 as well.
        return S.case(isnan(x), x).otherwise(f64(0.0))
    if np.isnan(q).all():
        # A feature missing everywhere at fit: every search lands on a NaN
        # line, and NaN is no bound.
        return f64(math.nan)
    if np.isnan(q).any() or not (q[:-1] <= q[1:]).all():
        raise NotNative(
            "QuantileTransformer: quantiles unsorted or partly NaN, where"
            " numpy's interp search answers by its guess"
        )
    up = _pieces(q, r)
    down = _pieces(-q[::-1], -r[::-1])
    # Below q[0] the two interps answer r[0] and -r[0], above q[-1] r[-1]
    # and -r[-1]: 0.0 and 1.0, as the bounds do at q[0] and q[-1] (the
    # lower bound last). NaN first: DuckDB orders it above every number.
    e = S.case(isnan(x), x).when(x <= f64(q[0]), f64(0.0))
    if not up:  # q[0] == q[-1]: nothing lies inside
        return e.otherwise(f64(1.0))
    e = e.when(x >= f64(q[-1]), f64(1.0))
    return e.otherwise(f64(0.5) * (_tree(x, up) - _tree(-x, down)))


@translates(QuantileTransformer)
def _quantile(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if est.output_distribution != "uniform":
        raise NotNative(
            f"QuantileTransformer(output_distribution={est.output_distribution!r}):"
            " scipy's norm.ppf has no SQL twin"
        )
    q, n = est.n_quantiles_, len(x)
    if q * n > MAX_QUANTILES or q * q * n > MAX_SQUARES:
        raise NotNative(
            f"QuantileTransformer with {q} quantiles over {n} features:"
            f" {q * n} quantiles and {q * q * n} squared, past the"
            f" {MAX_QUANTILES} and {MAX_SQUARES} confit builds in seconds"
        )
    if not interp_is_numpys():
        raise NotNative(
            "QuantileTransformer: this platform's np.interp does not round as"
            " numpy's unfused C loop, which the entry follows"
        )
    r = np.asarray(est.references_, dtype=np.float64)
    return [
        _feature(xi, np.asarray(est.quantiles_[:, i], dtype=np.float64), r)
        for i, xi in enumerate(x)
    ]
