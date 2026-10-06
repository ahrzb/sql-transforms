"""Function transformers: the identity, and numpy's elementwise functions
whose double DuckDB computes to the same bit (sklearn 1.9,
`sklearn/preprocessing/_function_transformer.py`; numpy 2.5, DuckDB 1.5.5).

The twin's `transform` validates `X` if `validate=True` (NaN and infinity
raise there), then answers `func(X, **(kw_args or {}))`, `func=None` being
the identity. The step hands it a row of doubles (NULL as NaN), so `func`
runs on float64 and the entry spells its value per feature.

Served, each checked against numpy on signed zeros, NaN, infinities and
subnormals: `abs`/`fabs`, `negative` (as `-1.0 * x`, the same double as
`-x`), `positive` and
`conjugate` (the identity on reals), `square` (`x * x`), `sqrt` (IEEE,
correctly rounded on both sides; DuckDB raises on a negative, so it is
guarded), `reciprocal` (`1.0 / x`), `floor`, `ceil`, `trunc`, `rint` (half
to even, spelled from `trunc`: confit has no `round_even`),
`sign`, and `sin`/`cos` where this platform's numpy answers as confit
does: all bit-exact. And within a bound of their own (`_BOUNDS`, the
entry's bound per estimator): `exp`, `log` (DuckDB's `ln`), `log2`,
`log10`, `tan` and `cbrt`, each DuckDB's function of that name, guarded
where DuckDB raises and numpy answers IEEE's value. numpy's float64
kernels for these are its own SIMD code on x86-64 with AVX-512, 1 to 3
ulps from glibc's, which DuckDB and confit call; numpy picks its kernels
by CPU, so each is served only where `kernel_distance` finds this
platform's numpy within the function's bound of confit, as `sin` and
`cos` within 0.

Refused: `log1p` and `expm1` (DuckDB has neither), and every other
function (confit has no inverse or hyperbolic trigonometry).
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
from sklearn.preprocessing import FunctionTransformer

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates

_NAN = f64(math.nan)
_ZERO = f64(0.0)


def _abs(x: S.Expr) -> S.Expr:
    # numpy's `abs`: `0.0 - x` at or below zero, which takes -0.0 to 0.0
    # (`-1.0 * x` would take 0.0 to -0.0). NaN is above every number in
    # DuckDB, so it keeps itself. The same double as DuckDB's `abs`.
    return S.case(x <= _ZERO, _ZERO - x).otherwise(x)


def _rint(x: S.Expr) -> S.Expr:
    # Half to even, from `trunc` and exact double arithmetic, as confit
    # counts both trap-free: `t = trunc(x)` and `f = x - t` (exact). Under
    # half, `t` (`trunc` keeps a zero's sign, as `rint(-0.3)` is -0.0);
    # over half, the next integer away from zero (`t +- 1` is exact, as a
    # fraction leaves |t| < 2**52); at an exact half, whichever of the two
    # is even (`t / 2` is exact). Infinity and NaN make `f` NaN, and keep
    # themselves; DuckDB orders NaN above every number, so it goes first.
    t = S.fn("trunc", x)
    f = x - t
    af = _abs(f)
    up = S.case(f > _ZERO, t + f64(1.0)).otherwise(t - f64(1.0))
    even = f64(0.5) * t == S.fn("trunc", f64(0.5) * t)
    return (
        S.case(isnan(f), x)
        .when(af < f64(0.5), t)
        .when((af > f64(0.5)) | ~even, up)
        .otherwise(t)
    )


def _sign(x: S.Expr) -> S.Expr:
    # numpy: 1 above zero, -1 below, 0.0 at either zero, NaN at NaN. DuckDB
    # orders NaN above every number, so NaN is tested first.
    return (
        S.case(isnan(x), _NAN)
        .when(x > _ZERO, f64(1.0))
        .when(x < _ZERO, f64(-1.0))
        .otherwise(_ZERO)
    )


def _sqrt(x: S.Expr) -> S.Expr:
    # DuckDB raises below zero where numpy answers NaN (-0.0 is not below
    # zero: its root is -0.0 on both sides).
    return S.case(x < _ZERO, _NAN).otherwise(S.fn("sqrt", x))


def _log(name: str) -> Callable[[S.Expr], S.Expr]:
    def spell(x: S.Expr) -> S.Expr:
        # numpy: -inf at either zero, NaN below zero (-inf included). DuckDB
        # raises on both. `x <= 0` passed is the guard confit reads as
        # ruling them out (#362), so a field read leaves the other lanes
        # unevaluated; NaN is above every number in DuckDB, so it reaches
        # `name`, which keeps it.
        return (
            S.case(x == _ZERO, f64(-math.inf))
            .when(x <= _ZERO, _NAN)
            .otherwise(S.fn(name, x))
        )

    return spell


def _trig(name: str) -> Callable[[S.Expr], S.Expr]:
    def spell(x: S.Expr) -> S.Expr:
        # DuckDB raises on an infinity where numpy answers NaN. The guard is
        # spelled with `abs`, the form confit reads as ruling the infinities
        # out (#375), so a field read leaves the other lanes unevaluated.
        inf = S.fn("abs", x) == f64(math.inf)
        return S.case(inf, _NAN).otherwise(S.fn(name, x))

    return spell


# `func` -> its value on one double feature.
_SERVED: dict[Any, Callable[[S.Expr], S.Expr]] = {
    np.absolute: _abs,
    np.fabs: _abs,
    np.negative: lambda x: f64(-1.0) * x,
    np.positive: lambda x: x,
    np.conjugate: lambda x: x,
    np.square: lambda x: x * x,
    np.sqrt: _sqrt,
    np.reciprocal: lambda x: f64(1.0) / x,
    np.floor: lambda x: S.fn("floor", x),
    np.ceil: lambda x: S.fn("ceil", x),
    np.trunc: lambda x: S.fn("trunc", x),
    np.rint: _rint,
    np.sign: _sign,
    np.sin: _trig("sin"),
    np.cos: _trig("cos"),
    # Total in DuckDB: exp overflows to inf and underflows to 0.0, as
    # numpy's does; cbrt keeps signed zeros, infinities and NaN.
    np.exp: lambda x: S.fn("exp", x),
    np.cbrt: lambda x: S.fn("cbrt", x),
    np.log: _log("ln"),
    np.log2: _log("log2"),
    np.log10: _log("log10"),
    np.tan: _trig("tan"),
}

# The functions served within a bound above 0: numpy's kernel is not
# glibc's, which DuckDB and confit call. Each bound is the largest distance
# measured, numpy 2.5.1 against DuckDB 1.5.5 and confit on x86-64 with
# AVX-512: over 1,600,000 draws (uniform in +-1e3 and +-50, normal, and
# +-exp(uniform(-700, 700))) exp, log, log2 and tan reach 1, log10 2 and
# cbrt 3; over NATIVE_SEEDS=200 of the catalog's fixtures (each function
# validated and not, 250,551 rows with the other functions') cbrt reaches
# 3 and the others 1 (2026-10-05; cbrt against confit after #390). The
# class's ceiling is the largest of them.
_BOUNDS = {
    np.exp: 1,
    np.log: 1,
    np.log2: 1,
    np.tan: 1,
    np.log10: 2,
    np.cbrt: 3,
}

# Served only where this platform's numpy kernel is within its bound of
# confit's (`kernel_distance`): sin and cos at 0, and the bounded ones.
_PROBED = {np.sin, np.cos, *_BOUNDS}

# Elementwise functions SQL does not spell.
_NO_SQL = {
    np.log1p: "DuckDB has no log1p",
    np.expm1: "DuckDB has no expm1",
}


def _name(func: Any) -> str:
    mod = getattr(func, "__module__", None)
    name = getattr(func, "__name__", None) or repr(func)
    if isinstance(func, np.ufunc) or mod == "numpy":
        return f"np.{name}"
    return name


def _ordered(x: np.ndarray) -> np.ndarray:
    """Each double's position on the line of all doubles (`_check`'s
    `ulp_distance`, over an array): -0.0 and 0.0 share one."""
    i = x.view(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFF_FFFF_FFFF_FFFF))


@functools.cache
def kernel_distance(func: Any) -> int | None:
    """The largest distance, in doubles, between numpy's float64 `func` and
    the entry's spelling of it as confit, the engine that serves the entry,
    computes it, on probe values: 40,000 draws over the magnitudes a double
    spans (huge arguments exercise the range reduction), and the signed
    zeros, subnormals, infinities and NaN. NaN against a number counts as
    2**64. None when the probe fails to run."""
    rng = np.random.default_rng(20261005)
    n = 10_000
    x = np.concatenate(
        [
            rng.uniform(-1e3, 1e3, n),
            rng.normal(0.0, 1.0, n),
            rng.uniform(-50.0, 50.0, n),
            np.exp(rng.uniform(-700.0, 700.0, n)) * rng.choice([-1.0, 1.0], n),
            [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, math.pi],
            [math.inf, -math.inf, math.nan, 1e308, -1e308, 709.78, 710.0],
        ]
    )
    rows = pa.table({"x": x})
    y = _SERVED[func](S.col("x"))
    sql = f"SELECT {y.sql()} AS y FROM __THIS__"  # noqa: S608 — a fixed spelling
    try:
        probe = DuckDBInferFn(
            sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[]
        )
        got = probe.infer_arrow(rows).column("y").to_numpy()
    except Exception:  # noqa: BLE001 — any failure leaves the entry out
        return None
    with np.errstate(all="ignore"):
        want = func(x)
    nan_w, nan_g = np.isnan(want), np.isnan(got)
    if np.any(nan_w != nan_g):
        return 1 << 64
    both = ~nan_w
    # Python ints: positions of opposite sign can differ past int64.
    a = _ordered(want[both]).tolist()
    b = _ordered(got[both]).tolist()
    return max((abs(p - q) for p, q in zip(a, b, strict=True)), default=0)


def _bound(est: Any) -> int:
    """The estimator's own bound: its function's, 0 for the rest."""
    try:
        return _BOUNDS.get(est.func, 0)
    except TypeError:  # an unhashable callable, which the entry refuses
        return 0


@translates(FunctionTransformer, ulps=max(_BOUNDS.values()), bound=_bound)
def _function(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: validate if `validate` (NaN and infinity raise), then
    # `func(X, **kw_args)`, elementwise on the float64 row.
    func = est.func
    what = f"FunctionTransformer(func={_name(func)})"
    if est.kw_args:
        raise NotNative(f"{what} with kw_args={est.kw_args!r}")
    if any(t == pa.string() for t in types):
        raise NotNative(f"{what} over a string feature: not a numeric row")
    if func is None:
        return list(x)
    if any(t == pa.bool_() for t in types):
        # numpy keeps a boolean row boolean (`negative` raises, `sqrt`
        # answers float16): not the float64 the entry spells.
        raise NotNative(f"{what} over a boolean feature")
    try:
        spell = _SERVED.get(func)
    except TypeError:  # an unhashable callable
        spell = None
    if spell is None:
        if isinstance(func, np.ufunc) and func in _NO_SQL:
            raise NotNative(f"{what}: {_NO_SQL[func]}")
        raise NotNative(f"{what}: not a function the entry serves")
    if func in _PROBED:
        d, b = kernel_distance(func), _bound(est)
        if d is None:
            raise NotNative(f"{what}: the kernel probe did not run")
        if d > b:
            raise NotNative(
                f"{what}: this platform's numpy kernel is {d} ulps from"
                f" confit's on the probe, past the entry's bound of {b}"
            )
    return [spell(xi) for xi in x]
