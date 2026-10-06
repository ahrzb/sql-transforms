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
its guess (a linear scan `x >= xp[i]` below five points, else bisection
on `x >= xp[mid]`, both after the guess's checks narrow the same
invariant); an `x` strictly inside then answers `fp[j]` when `x ==
xp[j]`, else `slope * (x - xp[j]) + fp[j]` with `slope = (fp[j+1] -
fp[j]) / (xp[j+1] - xp[j])` (precomputed when `len(xp) <= len(x)`, by the
same expression), retrying from `xp[j+1]` on a NaN.

Both searches, one tree. Let `v_0 < ... < v_d` be the distinct values of
`q`, `first(v)` and `last(v)` the first and last index of `v`'s run (a run
of one: `first == last`; -0.0 and 0.0 are one value). For `x` strictly
inside `(v_0, v_d)`:

- Ascending, `np.interp(x, q, r)` lands on `j = last(v_i)` for `v_i <= x
  < v_{i+1}`, the piece `[q[j], q[j+1]]` with `q[j+1] = v_{i+1}`.
- Mirrored, `xp = -q[::-1]` puts `q[m]` at index `n-1-m`, and the last
  index with `-q[m] <= -x` is the smallest `m` with `q[m] >= x`. So the
  search lands on `m = first(v_{i+1})` for `v_i < x <= v_{i+1}`, the
  piece `[-q[m], -q[m-1]]` with `m-1 = last(v_i)`: the same pair of
  quantiles as the ascending piece, read from its other end.

So strictly between `v_i` and `v_{i+1}` both searches take the pair
`(last(v_i), first(v_{i+1}))`, and a leaf over `[v_i, v_{i+1})` answers
`0.5 * (up(x) - down(-x))` with each line in its own search's arithmetic
(`_pieces` of `q` and of `-q[::-1]`, the slopes as numpy computes them).
At `x == v_i` (`i >= 1`; `x <= v_0` is answered before the tree) the
ascending search takes its exact arm on that same leaf, `r[last(v_i)]`,
and the mirrored search its exact arm on the piece that ends at `v_i`,
`-r[first(v_i)]`; the twin answers `0.5 * (r[last(v_i)] -
-r[first(v_i)])`, computed here. The leaf keeps an `x == v_i` arm with
that constant only where its lines do not already answer it, bit for bit:
a run (the two arms read different references), an infinite slope (`inf
* 0` is NaN), or a mirrored line that rounds off at its far end. A fit of
continuous values needs none. The tree bisects on `x < v_i` (depth log2
of `d`), as one search would.

The NaN retry is never taken: inside an interval whose width is finite, a
line is NaN only at its start with an infinite slope, which the arm
answers, and a fit with an infinite width stays Python.
`interp_is_numpys` checks that this platform's `np.interp` rounds as the
leaves do (a compiler may fuse the multiply-add, which SQL cannot spell).
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

# The most quantiles served per estimator, summed over its features. One
# tree builds about linearly in its leaves (one feature, warm: 0.5-0.6 s
# at 1,000 quantiles, 1.3-1.4 s at 2,000, 2.7-3.2 s at 4,000, 6.4-6.9 s at
# 8,000, where both searches as two trees took 1.4-1.6, 4.7-4.8 and 23-24
# s up to 4,000), and so does an estimator in its total: 8,000 quantiles
# as 4,000 over two features build in 6.8-7.5 s, 2,000 over four in
# 6.6-7.1 s, 1,000 over eight in 6.9-7.1 s, 100 over 80 in 6.2-6.4 s, 10
# over 800 in 7.8-8.0 s (the first build in a process within 10% of
# these), against 7.0 s for the old caps' slowest, 1,000 over four. Fits
# with runs build faster: fewer leaves outweigh their arms. Serving 64
# rows is 25 times faster than the twin at 1,000 over one feature, 2 times
# at 8,000 over one, 3 at 1,000 over eight, 13 at 10 over 800 (release
# build, one container, 2026-10-05).
MAX_QUANTILES = 8000

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


def _interp_py(x: float, pieces: list[_Piece]) -> float:
    """What `np.interp` answers strictly inside `(xp[0], xp[-1])`, as
    `_pieces` reads it, on a Python float."""
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


# One interval `[v, w)` between neighbouring distinct quantiles, as both
# searches see it: the ascending piece, the mirrored piece over `(v, w]`,
# and what the twin answers at `x == v` where their lines do not.
_Leaf = tuple[_Piece, _Piece, float | None]


def _both(x: Any, up: _Piece, down: _Piece) -> Any:
    """The twin's `0.5 * (interp(x) - interp(-x))` off the breakpoints."""
    if isinstance(x, S.Expr):
        return f64(0.5) * (_line(x, up) - _line(-x, down))
    return 0.5 * (_line(x, up) - _line(-x, down))


def _same_double(a: float, b: float) -> bool:
    return repr(a) == repr(b)  # -0.0 is not 0.0, NaN is NaN


def _leaves(q: np.ndarray, r: np.ndarray) -> list[_Leaf]:
    """One leaf per interval strictly inside `(q[0], q[-1])`, in order
    (the module docstring derives which piece each search lands on)."""
    up = _pieces(q, r)
    down = _pieces(-q[::-1], -r[::-1])[::-1]
    out: list[_Leaf] = []
    for i, (u, d) in enumerate(zip(up, down, strict=True)):
        at = None
        if i:
            # At x == v both exact arms: the ascending one answers u's
            # value, the mirrored one the value of the piece ending at v.
            # When v is zero the other zero answers as u[0] does: both
            # values are nonzero past the first leaf, so a zero offset's
            # sign never shows.
            want = 0.5 * (u[2] - down[i - 1][2])
            if not _same_double(_both(u[0], u, d), want):
                at = want
        out.append((u, d, at))
    return out


def _leaf(x: S.Expr, leaf: _Leaf) -> S.Expr:
    up, down, at = leaf
    if at is None:
        return _both(x, up, down)
    return S.case(x == f64(up[0]), f64(at)).otherwise(_both(x, up, down))


def _tree(x: S.Expr, leaves: list[_Leaf]) -> S.Expr:
    """The leaf `x` falls in, by bisection on the interval starts."""
    if len(leaves) == 1:
        return _leaf(x, leaves[0])
    mid = len(leaves) // 2
    return S.case(x < f64(leaves[mid][0][0]), _tree(x, leaves[:mid])).otherwise(
        _tree(x, leaves[mid:])
    )


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
    leaves = _leaves(q, r)
    # Below q[0] the two interps answer r[0] and -r[0], above q[-1] r[-1]
    # and -r[-1]: 0.0 and 1.0, as the bounds do at q[0] and q[-1] (the
    # lower bound last). NaN first: DuckDB orders it above every number.
    e = S.case(isnan(x), x).when(x <= f64(q[0]), f64(0.0))
    if not leaves:  # q[0] == q[-1]: nothing lies inside
        return e.otherwise(f64(1.0))
    e = e.when(x >= f64(q[-1]), f64(1.0))
    return e.otherwise(_tree(x, leaves))


@translates(QuantileTransformer)
def _quantile(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if est.output_distribution != "uniform":
        raise NotNative(
            f"QuantileTransformer(output_distribution={est.output_distribution!r}):"
            " scipy's norm.ppf has no SQL twin"
        )
    q, n = est.n_quantiles_, len(x)
    if q * n > MAX_QUANTILES:
        raise NotNative(
            f"QuantileTransformer with {q} quantiles over {n} features:"
            f" {q * n} quantiles, past the {MAX_QUANTILES} confit builds in"
            " seconds"
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
