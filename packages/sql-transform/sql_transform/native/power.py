"""Power transforms: Box-Cox (scipy 1.18, `xsf::boxcox`) and Yeo-Johnson
(scipy's `_yeojohnson_transform`, `scipy/stats/_morestats.py`), per
feature with its fitted lambda (sklearn 1.9, `sklearn/preprocessing/_data.py`).

DuckDB 1.5.5 has `ln` and `exp` (the C library's, as confit's are) but
neither `log1p` nor `expm1`, and confit cannot add a function, as the
definition must run on DuckDB. So the entry spells `log1p` (Goldberg's) and
`expm1` (Kahan's) from `ln` and `exp`, a few roundings from the twin's,
and is held to the error scales of the ruling
(loops/native/decisions/closed/power-parity-bound.md): Box-Cox within
5·eps·|t| + tau and Yeo-Johnson within 10·eps·(1 + max(w, 0))·|t| + tau,
with t the twin's output and w the argument of its `expm1` (`_power`).
Both serve only with `to_native(step, allow_bound=True)`, only where the
kernel probe (`kernel_distances`) finds the functions the bounds read, and
only where the build is estimated within 7 s (`_build_estimate`).

`standardize=True` stays Python: the twin's scaler raises on a row whose t
is infinite, and near the overflow the entry and the twin can round to
either side of the largest double, so the entry cannot trap exactly where
the twin raises.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from confit import sql as S
from sklearn.preprocessing import PowerTransformer

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import ErrorScale, NotNative, rejects, translates
from sql_transform.native.function import doubles_apart, probe_values
from sql_transform.native.project import _TINY, MAX_BUILD_S

_ONE = f64(1.0)
_ZERO = f64(0.0)
_NAN = f64(math.nan)
# scipy's Box-Cox: `log(x)` below this |lambda|, and `expm1` below this
# `lambda * log(x)` (past it, an `exp` that does not overflow).
_LOG_LAMBDA = 1e-19
_EXPM1_BELOW = 709.78
# scipy's Yeo-Johnson tests lambda against the eps of the result's dtype,
# float64 over a row of doubles.
_EPS = np.finfo(np.float64).eps


def log1p(z: S.Expr) -> S.Expr:
    """`log1p(z)` for z >= 0 or NaN, as Goldberg's `z * (ln(u) / (u - 1))`,
    `u = 1 + z`: z where u rounds to 1. Within 4.04 u (u = 2**-53) of the
    exact value where `ln` is within 0.52 ulp, as glibc's is, and 5.04 u
    where 1 + z passes 2**53; measured within 2.01 ulps over 400,000 draws
    (loops/native/decisions/research/2026-10-06/power.md)."""
    u = z + _ONE
    # z >= 0, so `u <= 1` holds only where u rounds to 1. It is also the
    # guard confit reads to know `ln(u)` cannot raise (confit #362).
    return S.case(u <= _ONE, z).otherwise(z * (S.fn("ln", u) / (u - _ONE)))


def expm1(w: S.Expr) -> S.Expr:
    """`expm1(w)` for a non-NaN `w`, as Kahan's `(u - 1) * (w / ln(u))`,
    `u = exp(w)`: `w` where u rounds to 1, -1 where `u - 1` does, and inf
    where `exp(w)` overflows (the ratio would read inf * 0). The rounding
    of `exp(w)` cancels in the ratio (Higham, Accuracy and Stability,
    1.14.1): within 4.06 u of the exact value where `ln` and `exp` are
    within 0.52 and 0.51 ulp, as glibc's are, and 5.06 u where u is below
    1/2 or past 2**53; measured within 2.30 ulps over 400,000 draws
    (research power.md)."""
    u = S.fn("exp", w)
    # `u <= 0` is never taken (`u - 1 = -1` answers every such `u`): it is
    # the guard confit reads to know `ln(u)` cannot raise, so a struct field
    # read leaves the other output fields unevaluated (confit #362).
    return (
        S.case(u == _ONE, w)
        .when(u - _ONE == f64(-1.0), f64(-1.0))
        .when(u <= _ZERO, f64(-1.0))
        .when(u == f64(math.inf), f64(math.inf))
        .otherwise((u - _ONE) * (w / S.fn("ln", u)))
    )


def _box_cox(x: S.Expr, lam: float) -> S.Expr:
    # scipy.special.boxcox, C++ on the C library: `log(x)` when |lambda| <
    # 1e-19; else, with w = lambda * log(x), `expm1(w) / lambda` when w <
    # 709.78, else `copysign(1, lambda) * exp(w - log(|lambda|)) - 1 /
    # lambda`. Only `expm1` is not the twin's own rounding. The twin rejects
    # x <= 0 (`np.nanmin(X) <= 0` raises), where the input guard traps
    # (`_power`) and this arm answers NaN rather than trap in `ln`; NaN
    # passes through the twin, and gets its own arm here, as DuckDB orders
    # NaN above every number.
    lx = S.fn("ln", x)
    if abs(lam) < _LOG_LAMBDA:
        e = lx
    else:
        w = f64(lam) * lx
        big = S.fn("exp", w - f64(math.log(abs(lam))))
        big = (big if lam > 0 else -big) - f64(1 / lam)
        e = S.case(w < f64(_EXPM1_BELOW), expm1(w) / f64(lam)).otherwise(big)
    return S.case(isnan(x) | (x <= _ZERO), _NAN).otherwise(e)


def _yeo_johnson_t(lam: np.ndarray, v: np.ndarray, positive: bool) -> np.ndarray:
    """The twin's Yeo-Johnson of x = v where `positive`, else of x = -v
    (v >= 0), for the fitted lambdas `lam` broadcast against `v`, as scipy
    computes it (`_yeo_johnson`)."""
    with np.errstate(all="ignore"):
        lv = np.log1p(v)
        if positive:
            return np.where(np.abs(lam) < _EPS, lv, np.expm1(lam * lv) / lam)
        d = 2 - lam
        return np.where(np.abs(lam - 2) > _EPS, -np.expm1(d * lv) / d, -lv)


# Around the edge where the twin's transform overflows, the doubles on which
# it is checked to be finite below and infinite from the edge on.
_WINDOW = 1 << 13


def _edges(lam: np.ndarray, positive: bool) -> np.ndarray:
    """For each fitted lambda, the least |x| on its side (x >= 0 where
    `positive`, else x < 0) where the twin's Yeo-Johnson is infinite, by
    bisection on the doubles; NaN where it is finite up to the largest
    double. The twin computes `expm1(w)` before it divides by c, so it is
    infinite past c·DBL_MAX: with c > 1 the entry, which computes t
    without overflowing first, answers DBL_MAX / c there, out of the reach
    of `check`'s rule for one infinite side. So the entry answers ±inf from
    this edge on (`_yeo_johnson`). The twin's w is within about 2 ulps of
    the exact one at the edge (`_power`), so it can part from monotone only
    within about 4,300 doubles of it: `NotNative` where the twin is not
    finite below the edge and infinite from it on, over _WINDOW doubles
    either side."""
    top = np.full(lam.shape, np.finfo(np.float64).max)
    has = np.isinf(_yeo_johnson_t(lam, top, positive))
    lo, hi = np.zeros(lam.shape, np.int64), top.view(np.int64)
    while np.any(has & (hi - lo > 1)):
        mid = lo + (hi - lo) // 2
        inf = np.isinf(_yeo_johnson_t(lam, mid.view(np.float64), positive))
        hi, lo = np.where(inf, mid, hi), np.where(inf, lo, mid)
    j = np.flatnonzero(has)
    if j.size:
        k = np.arange(-_WINDOW, _WINDOW + 1)
        bits = np.clip(hi[j, None] + k, 0, top.view(np.int64)[0])
        inf = np.isinf(_yeo_johnson_t(lam[j, None], bits.view(np.float64), positive))
        bad = np.flatnonzero(np.any(inf != (k >= 0), axis=1))
        if bad.size:
            edge = float(hi.view(np.float64)[j[bad[0]]])
            raise NotNative(
                "PowerTransformer(method='yeo-johnson'): the twin's transform of"
                f" feature {j[bad[0]]} overflows on no single edge near"
                f" {'' if positive else '-'}{edge!r}"
            )
    return np.where(has, hi.view(np.float64), np.nan)


def _part(lx: S.Expr, c: Any) -> S.Expr:
    """`expm1(c * lx) / c`, for lx = log1p(|x|) and c the twin's lambda (or
    2 - lambda) as a numpy scalar: with c > 1, past w = c * lx of 709.78,
    `exp(w - ln(c)) - 1 / c`, as scipy's Box-Cox computes it, which does
    not overflow where t does not."""
    cf = float(c)
    w = f64(cf) * lx
    e = expm1(w) / f64(cf)
    if cf <= 1:
        return e
    big = S.fn("exp", w - f64(math.log(cf))) - f64(1 / cf)
    return S.case(w < f64(_EXPM1_BELOW), e).otherwise(big)


def _yeo_johnson(x: S.Expr, lam: Any, top: float, bottom: float) -> S.Expr:
    # scipy's `_yeojohnson_transform`, in float64 (the row's dtype): where
    # x >= 0, `log1p(x)` when |lambda| < eps, else `expm1(lambda *
    # log1p(x)) / lambda`; elsewhere `-log1p(-x)` unless |lambda - 2| > eps,
    # else `-expm1((2 - lambda) * log1p(-x)) / (2 - lambda)`. `lambda - 2`
    # and `2 - lambda` round in the dtype of the fit (NEP 50), so each test
    # and constant reads `lam` as the twin holds it, a numpy scalar of that
    # dtype. NaN takes the second branch in the twin, which answers NaN, and
    # gets its own arm here, as DuckDB orders NaN above every number. From
    # `top` up and from `-bottom` down the twin is infinite (`_edges`).
    pos = log1p(x) if abs(lam) < _EPS else _part(log1p(x), lam)
    nx = f64(-1.0) * x
    if abs(lam - 2) > _EPS:
        neg = f64(-1.0) * _part(log1p(nx), 2 - lam)
    else:
        neg = f64(-1.0) * log1p(nx)
    arms = S.case(isnan(x), _NAN)
    if not math.isnan(top):
        arms = arms.when(x >= f64(top), f64(math.inf))
    if not math.isnan(bottom):
        arms = arms.when(x <= f64(-bottom), f64(-math.inf))
    return arms.when(x >= _ZERO, pos).otherwise(neg)


def _power_s(est: Any, x: np.ndarray) -> np.ndarray:
    """S for each output field at the row `x` (long doubles): |t| for
    Box-Cox, (1 + max(w, 0))·|t| for Yeo-Johnson, with t the transform and
    w the argument of its `expm1` (0 on a `log1p` branch), each on the
    twin's branches and constants (`_box_cox`, `_yeo_johnson`)."""
    lam = np.asarray(est.lambdas_)
    c = lam.astype(np.longdouble)
    with np.errstate(all="ignore"):
        if est.method == "box-cox":
            lx = np.log(x)
            return np.abs(np.where(np.abs(c) < _LOG_LAMBDA, lx, np.expm1(c * lx) / c))
        pos = x >= 0
        z = np.log1p(np.abs(x))
        c = np.where(pos, c, (2 - lam).astype(np.longdouble))
        logs = np.where(pos, np.abs(lam) < _EPS, ~(np.abs(lam - 2) > _EPS))
        w = np.where(logs, 0, c * z)
        t = np.where(logs, z, np.expm1(w) / c)
        return (1 + np.maximum(w, 0)) * np.abs(t)


POWER = ErrorScale(
    k=lambda est: 5 if est.method == "box-cox" else 10,
    s=_power_s,
    tau=lambda est: _TINY,
)


def _c_library(f: Callable[[float], float], v: np.ndarray) -> np.ndarray:
    """`f` of Python's `math`, which calls the C library, on each double of
    `v`: inf where it overflows."""
    out = []
    for t in v.tolist():
        try:
            out.append(f(t))
        except OverflowError:
            out.append(math.inf)
    return np.array(out)


@functools.cache
def kernel_distances() -> dict[str, int] | None:
    """The kernel probe of the power transforms: the largest distance, in
    doubles, on the probe values of the function family
    (`function.probe_values`, their magnitudes for a logarithm), between
    confit's `ln` and `exp`, which the entry calls, and the C library's
    `log` and `exp`, which scipy's Box-Cox calls ("ln", "exp"); and between
    numpy's `log1p` and `expm1`, the Yeo-Johnson twin's, and the C
    library's ("log1p", "expm1"). None when the probe fails to run."""
    x = probe_values()
    z = np.abs(x)
    p = np.where(z == 0, 5e-324, z)
    rows = pa.table({"p": p, "x": x})
    ln, exp = S.fn("ln", S.col("p")).sql(), S.fn("exp", S.col("x")).sql()
    sql = f"SELECT {ln} AS l, {exp} AS e FROM __THIS__"  # noqa: S608 — fixed
    try:
        probe = DuckDBInferFn(
            sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[]
        )
        got = probe.infer_arrow(rows)
    except Exception:  # noqa: BLE001 — any failure leaves the entry out
        return None
    with np.errstate(all="ignore"):
        return {
            "ln": doubles_apart(_c_library(math.log, p), got.column("l").to_numpy()),
            "exp": doubles_apart(_c_library(math.exp, x), got.column("e").to_numpy()),
            "log1p": doubles_apart(_c_library(math.log1p, z), np.log1p(z)),
            "expm1": doubles_apart(_c_library(math.expm1, x), np.expm1(x)),
        }


def _probe(method: str) -> None:
    """Raise `NotNative` where the kernel probe does not find the functions
    the bound of `method` reads (`_power`)."""
    what = f"PowerTransformer(method={method!r})"
    d = kernel_distances()
    if d is None:
        raise NotNative(f"{what}: the kernel probe did not run")
    if d["ln"] or d["exp"]:
        raise NotNative(
            f"{what}: confit's ln and exp are {d['ln']} and {d['exp']} ulps from"
            " the C library's on the kernel probe, where the bound reads the C"
            " library's"
        )
    if method == "yeo-johnson" and max(d["log1p"], d["expm1"]) > 1:
        raise NotNative(
            f"{what}: this platform's numpy log1p and expm1 are {d['log1p']} and"
            f" {d['expm1']} ulps from the C library's on the kernel probe, past"
            " what K = 10 reads (1)"
        )


def _build_estimate(n: int, method: str) -> float:
    """The seconds of the whole warm `to_native` (release build) of a step
    of n features: the translation, the probe (`_registry._probed`) and
    confit's trial build (`_registry._builds`). The probe hands the twin a
    row of n features for each feature and each of its three special
    values, and a struct return reads its n fields one by one, so the build
    grows as n². Fitted on the slower return of each width (a list and a
    struct; a scalar for one feature) over 18 widths of Yeo-Johnson (48 to
    640 features; builds of 0.34 to 19.8 s) and 14 of Box-Cox (128 to 960
    features; builds of 0.40 to 14.4 s; master 1a14ce6 with T55's power.py,
    2026-10-17): each within 0.93 to 1.10 and 0.88 to 1.14 times the
    estimate. The cap took on none slower than 6.32 s and 5.65 s, and
    refused none faster than 7.04 s and 7.28 s. It takes about 347 features
    of Yeo-Johnson and 702 of Box-Cox. A Box-Cox of 960 features builds in
    4.8 s as a list and 14.4 s as a struct, and confit refuses one of 1,024
    features or more after 3.8 s or more, as an expression nested past
    Python's recursion limit."""
    if method == "box-cox":
        return 1.17e-3 * n + 1.25e-5 * n * n
    return 5.69e-3 * n + 4.15e-5 * n * n


@translates(PowerTransformer, scale=POWER)
def _power(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    """`PowerTransformer(standardize=False)` (`transform`): `validate_data`,
    where the infinities raise and NaN passes; Box-Cox raises where
    `np.nanmin(X) <= 0`; then per feature, in float64, scipy's `boxcox`
    or `_yeojohnson_transform` with the fitted lambda (`_box_cox`,
    `_yeo_johnson`).

    Bound (decisions/closed/power-parity-bound.md, the ruling): each output
    field is within K·eps·S + tau of the twin's, eps = 2**-52, tau =
    2**-1074, u = 2**-53 below. The bound takes the C library's `log` and
    `exp` within 0.52 and 0.51 ulp, as glibc's are, and the twin's
    `log1p` and `expm1` within 1 ulp (2 u): the research measured numpy's
    SIMD kernels of AVX-512 (Intel's SVML) within 0.60 ulp and glibc's
    within 0.78 ulp. So the step serves only where the kernel probe finds
    confit's `ln` and `exp` the C library's, and, for Yeo-Johnson, numpy's
    `log1p` and `expm1` within 1 ulp of it (`kernel_distances`).

    Box-Cox, K = 5, S = |t|: both sides take `log(x)` from the C library,
    so w = lambda·log(x) is one double, and the `log` and large-w branches
    are bit-exact. On the `expm1` branch, the twin's `expm1` is within 2 u,
    Kahan's within 5.06 u (`expm1`), and each division by lambda within u:
    the sides are within (2 + 5.06 + 2)·u·|t| = 4.53·eps·|t|.

    Yeo-Johnson, K = 10, S = (1 + max(w, 0))·|t|: with L = log1p(x) (or
    log1p(-x)) and c = lambda (or 2 - lambda), each side computes L within
    a relative error a, the product w = c·L within u, `expm1(w)` within b
    and the division by c within u. A relative error g in w moves expm1(w)
    by c(w)·g to first order, c(w) = w·e^w / (e^w - 1) <= 1 + max(w, 0).
    So the sides are within |t|·(c(w)·(a_entry + a_twin + 2 u) + b_entry +
    b_twin + 2 u), with a_entry <= 5.04 u (`log1p`), b_entry <= 5.06 u and
    a_twin, b_twin <= 2 u: within (9.04·c(w) + 9.06)·u·|t| <= 9.05·eps·S.
    The `log1p` branches are within (5.04 + 2)·u·|t|.

    Past w = 709.78 with c > 1, the entry computes t as exp(w - ln(c)) -
    1/c (`_part`), which does not overflow where t does not. Its argument
    is within (6.04·w + w + 1.04·ln(c))·u of the exact one, and ln(c) <= w
    there, so it is within (8.08·w + 1.02)·u·|t|, and the twin within
    (3·w + 6)·u·|t|: under K·eps·S.

    Where a value underflows, both sides compute the same double: a
    product c·L with |c| >= eps underflows only where L < 2**-970, where L
    is x (or -x) itself on both sides, and there each `expm1` answers its
    argument (numpy's does for each of 400,000 subnormal and tiny normal
    draws, 2026-10-17). Only a last division by a lambda past 4.5e307 can
    round into the subnormals, which tau covers.

    Where the twin's Yeo-Johnson is infinite, the entry is too: it answers
    ±inf from the twin's own edges on, found by bisection on the twin
    (`_edges`), as the twin overflows in `expm1(w)` before it divides by c.
    Below an edge, and in Box-Cox, a side can still overflow alone where t
    is about the largest double: `check` passes such a field if the other
    side is finite, of the same sign, and within K·eps·S of the largest
    double (the ruling, item 11).

    Measured on master 1a14ce6 with T55's power.py (release build,
    2026-10-17): over 153.6 million output fields of 32 seeds, 8 features
    of five kinds of lambdas (uniform within ±5, small, near the twin's
    branches, large, and pinned on the branches; a quarter of the fits in
    float32) over six kinds of rows (from the subnormals to the overflow
    edges), with numpy's `log1p` and `expm1` of AVX-512 and glibc's, no
    field is past K, and none is NaN or infinite on one side only. The
    largest (|entry - twin| - tau) / (eps·S) is 3.38 for Yeo-Johnson
    (K = 10) and 2.50 for Box-Cox (K = 5), and 22% and 12% of the fields
    part from the twin's. Serving 64 rows takes 32 us at one feature, 172
    to 197 us at 8 and 11.0 to 12.3 ms at 256 for Yeo-Johnson, 48 to 349
    times as fast as the twin, and 51 us, 159 to 226 us and 10.4 to 10.8
    ms for Box-Cox, 7.1 to 176 times as fast (a list or a struct, one
    thread, the median of 30 warm calls). A step builds in the time that
    `_build_estimate` estimates, which the cap reads."""
    if est.standardize:
        raise NotNative(
            "PowerTransformer(standardize=True): the twin's scaler raises on an"
            " infinite power transform, and near the overflow the entry and the"
            " twin can round to either side of the largest double"
        )
    method = est.method
    if method not in ("box-cox", "yeo-johnson"):
        raise NotNative(f"PowerTransformer(method={method!r})")
    if not np.isfinite(np.asarray(est.lambdas_, np.float64)).all():
        raise NotNative(f"PowerTransformer(method={method!r}): a lambda not finite")
    estimate = _build_estimate(len(x), method)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"PowerTransformer(method={method!r}): an estimated {estimate:.0f} s"
            f" build, past {MAX_BUILD_S:.0f} s ({len(x):,} features)"
        )
    _probe(method)
    if method == "box-cox":
        for xi in x:
            # `np.nanmin(X) <= 0` raises; NaN reads above every number, so it
            # meets no test, as `nanmin` skips it.
            rejects(xi <= _ZERO)
        return [
            _box_cox(xi, float(lam)) for xi, lam in zip(x, est.lambdas_, strict=True)
        ]
    lam = np.asarray(est.lambdas_)
    tops, bottoms = _edges(lam, True), _edges(lam, False)
    return [
        _yeo_johnson(xi, lam_i, top, bottom)
        for xi, lam_i, top, bottom in zip(x, lam, tops, bottoms, strict=True)
    ]
