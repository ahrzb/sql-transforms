"""Isotonic regression as a transformer: the fitted step function,
interpolated linearly, in sklearn's and scipy's own operation order
(sklearn 1.9 `IsotonicRegression._transform` in `sklearn/isotonic.py`,
scipy 1.18 `interp1d` in `scipy/interpolate/_interpolate.py`).

The twin, on the step's one-row, one-feature `T`, with thresholds `xp =
X_thresholds_` (strictly increasing: the fit merges equal X) and `fp =
y_thresholds_`:

    T = check_array(T, dtype=xp.dtype)   # NaN and infinity raise
    if out_of_bounds == "clip": T = np.clip(T, X_min_, X_max_)
    f_(T)

where `f_` is `fp[0]` repeated when one threshold is left, whatever `T`
and `out_of_bounds` (no bounds check), and otherwise `interp1d(xp, fp,
kind="linear", bounds_error=out_of_bounds == "raise")`. For float64 `xp`
and `fp` (one-dimensional, no extrapolation) `interp1d.__init__` picks
`_call_linear_np`, which is `np.interp(T, xp, fp)`, not `_call_linear`'s
de Boor form; `_evaluate` then overwrites `T < xp[0]` and `T > xp[-1]`
with the fill value, NaN, or `_check_bounds` raises for "raise".

`np.interp` (numpy 2.5, `arr_interp` in `compiled_base.c`), for `x` in
`[xp[0], xp[-1]]`: with `j` the last index where `xp[j] <= x`, it answers
`fp[-1]` at `j = len(xp) - 1`, `fp[j]` when `x == xp[j]`, else `slope *
(x - xp[j]) + fp[j]` with `slope = (fp[j+1] - fp[j]) / (xp[j+1] -
xp[j])` (precomputed by the same expression when `len(xp) <= len(T)`),
retrying from `xp[j+1]` on a NaN. The entry spells that as a balanced CASE
over the intervals (quantile.py's tree shape), each leaf its line with the
slope computed here in the same float64 arithmetic. The exact arm `x ==
xp[j]` is kept only where it changes the answer: `fp[j]` a -0.0 (`+0.0 +
-0.0` is `+0.0`) or an infinite slope (`inf * 0` is NaN). The NaN retry is
never taken: with finite `fp` and finite widths a slope is never NaN and
`x - xp[j]` is never infinite, so the line is never NaN.

`out_of_bounds="clip"` needs no clip of its own: a clipped `T` equals
`xp[0]` or `xp[-1]` (as numbers: `X_min_` and `X_max_` are the thresholds'
ends), which `np.interp` answers `fp[0]` and `fp[-1]` by its exact arms,
so the entry answers those for `x <= xp[0]` and `x >= xp[-1]`. "raise"
answers NaN where the twin raises (loops/native/goal.md, "Tolerated
differences").
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, translates
from sql_transform.native.quantile import interp_is_numpys

# One interval of `np.interp`'s line: x in [start, next start) answers
# `slope * (x - start) + value`, or `value` at `start` when `exact`.
_Piece = tuple[float, float, float, bool]

# The most thresholds served. One CASE tree over the thresholds builds in
# about 0.7 ms a threshold, a little faster than linear: 1,000 in 0.5 s,
# 4,000 in 2.4 s, 8,000 in 5.8 s, 20,000 in 22 s. Serving 64 rows takes
# 37 us at the widest fixture (51 thresholds) and 366 us at 8,000,
# against the twin's 5,400 us (release build, master f2ef184,
# 2026-10-05).
MAX_THRESHOLDS = 8000


def _pieces(xp: np.ndarray, fp: np.ndarray) -> list[_Piece]:
    """The intervals `np.interp(x, xp, fp)` answers an `x` in `[xp[0],
    xp[-1])` from, one per `j < len(xp) - 1`."""
    out = []
    for j in range(len(xp) - 1):
        x0, x1, y0, y1 = float(xp[j]), float(xp[j + 1]), float(fp[j]), float(fp[j + 1])
        if math.isinf(x1 - x0):
            raise NotNative(
                "IsotonicRegression: thresholds further apart than a double"
                " spans, where numpy's interp retries a NaN line"
            )
        slope = (y1 - y0) / (x1 - x0)
        exact = not math.isfinite(slope) or (y0 == 0.0 and math.copysign(1.0, y0) < 0)
        out.append((x0, slope, y0, exact))
    return out


def _leaf(x: S.Expr, piece: _Piece) -> S.Expr:
    start, slope, value, exact = piece
    line = f64(slope) * (x - f64(start)) + f64(value)
    if exact:
        return S.case(x == f64(start), f64(value)).otherwise(line)
    return line


def _tree(x: S.Expr, pieces: list[_Piece]) -> S.Expr:
    """The piece `x` falls in, by bisection on the starts."""
    if len(pieces) == 1:
        return _leaf(x, pieces[0])
    mid = len(pieces) // 2
    return S.case(x < f64(pieces[mid][0]), _tree(x, pieces[:mid])).otherwise(
        _tree(x, pieces[mid:])
    )


@translates(IsotonicRegression)
def _isotonic(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    xp, fp = est.X_thresholds_, est.y_thresholds_
    if xp.dtype != np.float64 or fp.dtype != np.float64:
        raise NotNative(
            f"IsotonicRegression with {xp.dtype} thresholds: the twin casts"
            " its input to that dtype"
        )
    if len(x) != 1:
        raise NotNative(f"IsotonicRegression over {len(x)} features: it takes one")
    (v,) = x
    if len(xp) == 1:
        # `f_` repeats the one value, with no bounds check; the twin raises
        # on NaN.
        return [f64(fp[0])]
    if len(xp) > MAX_THRESHOLDS:
        raise NotNative(
            f"IsotonicRegression with {len(xp)} thresholds: past the"
            f" {MAX_THRESHOLDS} confit builds in about 6 s"
        )
    if getattr(est.f_, "_call", None) is not interp1d._call_linear_np:
        raise NotNative(
            "IsotonicRegression: scipy's interp1d does not delegate to"
            " np.interp here, which the entry follows"
        )
    if not np.isfinite(xp).all() or not np.isfinite(fp).all():
        raise NotNative("IsotonicRegression: thresholds not all finite")
    if not (xp[:-1] < xp[1:]).all():
        raise NotNative("IsotonicRegression: thresholds not strictly increasing")
    if not interp_is_numpys():
        raise NotNative(
            "IsotonicRegression: this platform's np.interp does not round as"
            " numpy's unfused C loop, which the entry follows"
        )
    lo, hi = f64(xp[0]), f64(xp[-1])
    inside = _tree(v, _pieces(xp, fp))
    if est.out_of_bounds == "clip":
        e = S.case(v <= lo, f64(fp[0])).when(v >= hi, f64(fp[-1]))
    else:
        # "nan", and "raise" where the twin raises. NaN reads above every
        # number in DuckDB, so it answers NaN (the twin raises on it).
        nan = f64(math.nan)
        e = S.case(v < lo, nan).when(v > hi, nan).when(v == hi, f64(fp[-1]))
    return [e.otherwise(inside)]
