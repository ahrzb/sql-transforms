"""Power transforms: Box-Cox per feature with its fitted lambda (sklearn
1.9, `sklearn/preprocessing/_data.py`; scipy 1.18, `xsf::boxcox`).

DuckDB 1.5.5 has `ln` and `exp` (glibc's, as confit's are) but neither
`log1p` nor `expm1`, and confit cannot add a function, as the definition
must run on DuckDB. So `expm1` is spelled from `exp` and `ln`, within a
few ulps of glibc's, and Box-Cox serves within 4 ulps of the twin: only
with `to_native(step, allow_bound=True)`. Yeo-Johnson and
`standardize=True` stay Python for now: their distance to the twin has no
small bound in ulps of the result, and the parity bound the owner ruled
for them (decisions/closed/power-parity-bound.md) is not in
`native.check` yet.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import PowerTransformer

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates

_ONE = f64(1.0)
# scipy's Box-Cox: `log(x)` below this |lambda|, and `expm1` below this
# `lambda * log(x)` (past it, an `exp` that does not overflow).
_LOG_LAMBDA = 1e-19
_EXPM1_BELOW = 709.78


def expm1(w: S.Expr) -> S.Expr:
    """`expm1(w)` for a non-NaN `w` whose `exp` does not overflow, as
    Kahan's `(u - 1) * (w / ln(u))`, `u = exp(w)`: `w` where `u` rounds to
    1, and -1 where `u - 1` does. The rounding of `exp(w)` cancels in the
    ratio (Higham, Accuracy and Stability, 1.14.1): within 2 ulps of
    glibc's `expm1` over 600,000 draws of `w` in [-40, 709] and of
    |w| in [1e-18, 3] (2026-10-05)."""
    u = S.fn("exp", w)
    # `u <= 0` is never taken (`u - 1 = -1` answers every such `u`): it is
    # the guard confit reads to know `ln(u)` cannot raise, so a struct field
    # read leaves the other lanes unevaluated (confit #362).
    return (
        S.case(u == _ONE, w)
        .when(u - _ONE == f64(-1.0), f64(-1.0))
        .when(u <= f64(0.0), f64(-1.0))
        .otherwise((u - _ONE) * (w / S.fn("ln", u)))
    )


def _box_cox(x: S.Expr, lam: float) -> S.Expr:
    # scipy.special.boxcox, C on glibc: `log(x)` when |lambda| < 1e-19;
    # else, with w = lambda * log(x), `expm1(w) / lambda` when w < 709.78,
    # else `copysign(1, lambda) * exp(w - log(|lambda|)) - 1 / lambda`.
    # Only `expm1` is not the twin's own rounding. The twin rejects x <= 0
    # (`np.nanmin(X) <= 0` raises), where the entry answers NaN rather than
    # trap in `ln`; NaN passes through the twin, and gets its own arm here,
    # as DuckDB orders NaN above every number.
    lx = S.fn("ln", x)
    if abs(lam) < _LOG_LAMBDA:
        e = lx
    else:
        w = f64(lam) * lx
        big = S.fn("exp", w - f64(math.log(abs(lam))))
        big = (big if lam > 0 else -big) - f64(1 / lam)
        e = S.case(w < f64(_EXPM1_BELOW), expm1(w) / f64(lam)).otherwise(big)
    return S.case(isnan(x) | (x <= f64(0.0)), f64(math.nan)).otherwise(e)


@translates(PowerTransformer, ulps=4)
def _power(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    """sklearn: per feature `boxcox(X[:, i], lambdas_[i])` in float64.

    Bound 4: the largest distance to the twin measured, 4 ulps over
    2,400,000 draws of (lambda, x) against scipy's `boxcox` (lambda in
    [-5, 5] or |lambda| log-uniform in [1e-19, 10]; x log-uniform in
    [1e-300, 1e300], uniform in (0, 1e3) or within 1e-6 of 1), and 2 ulps
    on the catalog's fixtures, seeds 0-199 of both configurations, steps
    of up to 32 features included since the width cap went (2026-10-05,
    x86-64 with AVX-512, glibc 2.39). `log` and `exp` are
    glibc's on both sides; `expm1` is within 2 ulps of glibc's, and the
    division by lambda rounds both."""
    if est.method != "box-cox":
        raise NotNative(
            f"PowerTransformer(method={est.method!r}): numpy's SIMD log1p and"
            " expm1 have no SQL spelling, and a rounding apart is amplified"
            " by lambda * log1p(x), past any small ulp bound; it waits on"
            " the parity bound (decisions/closed/power-parity-bound.md)"
        )
    if est.standardize:
        raise NotNative(
            "PowerTransformer(standardize=True): `x - mean_` cancels the"
            " rounding expm1 is apart by into any number of ulps; it waits"
            " on the parity bound (decisions/closed/power-parity-bound.md)"
        )
    return [_box_cox(xi, float(lam)) for xi, lam in zip(x, est.lambdas_, strict=True)]
