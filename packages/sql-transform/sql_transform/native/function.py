"""Function transformers: the identity, and numpy's elementwise functions
whose double DuckDB computes to the same bit (sklearn 1.9,
`sklearn/preprocessing/_function_transformer.py`; numpy 2.5, DuckDB 1.5.5).

The twin's `transform` validates `X` if `validate=True` (NaN and infinity
raise there), then answers `func(X, **(kw_args or {}))`, `func=None` being
the identity. The step hands it a row of doubles (NULL as NaN), so `func`
runs on float64 and the entry spells its value per feature.

Served, each checked against numpy on signed zeros, NaN, infinities and
subnormals: `abs`/`fabs`, `negative` (as `-1.0 * x`, the same double; confit
serves a product faster than a negation, PLANS), `positive` and
`conjugate` (the identity on reals), `square` (`x * x`), `sqrt` (IEEE,
correctly rounded on both sides; DuckDB raises on a negative, so it is
guarded), `reciprocal` (`1.0 / x`), `floor`, `ceil`, `trunc`, `rint` (half
to even, spelled from `round`, which DuckDB rounds half away from zero),
`sign`, and `sin`/`cos` where this platform's numpy answers as DuckDB does
(`kernel_is_duckdbs` probes it). A spelling that calls a function serves
up to `_MAX_CALL_WIDTH` features.

Refused: every other transcendental. numpy's float64 `exp`, `log`, `log2`,
`log10`, `tan` and `cbrt` are its own SIMD kernels on x86-64 with AVX-512,
1 to 3 ulps from glibc's, which DuckDB calls (`_NOT_EXACT`); an entry's
bound is its class's, and this class is bit-exact. DuckDB has no `log1p`
nor `expm1`, and confit no inverse or hyperbolic trigonometry.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import FunctionTransformer

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates

_NAN = f64(math.nan)
_ZERO = f64(0.0)
# The widest step served by a spelling that calls a function (`sqrt`,
# `floor`, `round`, `sin`, ...). confit counts every call as one that may
# trap, so a struct field read also evaluates the other lanes' calls and
# serving grows as the square of the width: per row, three instances,
# 1,024-row batches, against the Python step, `rint` 1.5 vs 31 us at 4
# features, 35 vs 44 at 12, 54 vs 69 at 16, 130 vs 80 at 24; `sqrt` 81 vs
# 79 at 24; at 128 confit refuses past Cranelift's size limit (2026-10-05),
# until confit knows a call that cannot trap (PLANS, "Needs from confit").
# The call-free spellings serve 128 features (the identity: 24 vs 334 us).
_MAX_CALL_WIDTH = 12


def _abs(x: S.Expr) -> S.Expr:
    # `abs` without a call (above): `0.0 - x` at or below zero, which takes
    # -0.0 to 0.0 as `abs` does (`-1.0 * x` would take 0.0 to -0.0). NaN
    # is above every number in DuckDB, so it keeps itself.
    return S.case(x <= _ZERO, _ZERO - x).otherwise(x)


def _rint(x: S.Expr) -> S.Expr:
    # `round` is half away from zero; at an exact half (`x - trunc(x)` is
    # exact) the even neighbour is `2 * round(x / 2)`, `x / 2` exact as
    # |x| >= 0.5. Infinity and NaN miss the half test (NaN), and `round`
    # keeps them, and a zero's sign, as `rint` does.
    half = _abs(x - S.fn("trunc", x)) == f64(0.5)
    return S.case(half, f64(2.0) * S.fn("round", f64(0.5) * x)).otherwise(
        S.fn("round", x)
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


def _trig(name: str) -> Callable[[S.Expr], S.Expr]:
    def spell(x: S.Expr) -> S.Expr:
        # DuckDB raises on an infinity where numpy answers NaN.
        inf = _abs(x) == f64(math.inf)
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
}

# The spellings that call a function, capped at _MAX_CALL_WIDTH features.
_CALLS = {np.sqrt, np.floor, np.ceil, np.trunc, np.rint, np.sin, np.cos}

# Served only where this platform's numpy kernel is DuckDB's, to the bit.
_PROBED = {np.sin: "sin", np.cos: "cos"}

# Elementwise functions SQL spells, but not to numpy's bit: the largest
# distance measured, numpy 2.5.1 against DuckDB 1.5.5 on x86-64 with
# AVX-512, 1,600,000 draws (uniform in +-1e3 and +-50, normal, and
# +-exp(uniform(-700, 700))), 2026-10-05.
_NOT_EXACT = {
    np.exp: 1,
    np.log: 1,
    np.log2: 1,
    np.tan: 1,
    np.log10: 2,
    np.cbrt: 3,
}

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


@functools.cache
def kernel_is_duckdbs(name: str) -> bool:
    """Whether numpy's float64 `name` answers as DuckDB's `name` on probe
    values: 40,000 draws over the magnitudes a double spans (huge arguments
    exercise the range reduction), and the signed zeros and subnormals."""
    from confit.oracle import Oracle

    rng = np.random.default_rng(20261005)
    n = 10_000
    x = np.concatenate(
        [
            rng.uniform(-1e3, 1e3, n),
            rng.normal(0.0, 1.0, n),
            rng.uniform(-50.0, 50.0, n),
            np.exp(rng.uniform(-700.0, 700.0, n)) * rng.choice([-1.0, 1.0], n),
            [0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, math.pi],
        ]
    )
    with Oracle() as o:
        o.load("p", pa.table({"x": x}))
        sql = f"SELECT {name}(x) AS y FROM p"  # noqa: S608 — a fixed name
        got = o.answer(sql).column("y").to_numpy()
    want = getattr(np, name)(x)
    return bool(np.array_equal(want.view(np.int64), got.view(np.int64)))


@translates(FunctionTransformer)
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
        if isinstance(func, np.ufunc) and func in _NOT_EXACT:
            raise NotNative(
                f"{what}: numpy's kernel is up to {_NOT_EXACT[func]} ulp(s) from"
                " DuckDB's, and the entry is bit-exact"
            )
        if isinstance(func, np.ufunc) and func in _NO_SQL:
            raise NotNative(f"{what}: {_NO_SQL[func]}")
        raise NotNative(f"{what}: not a function the entry serves")
    if func in _CALLS and len(x) > _MAX_CALL_WIDTH:
        raise NotNative(
            f"{what} over {len(x)} features: every lane's call is evaluated"
            f" at each field read, past {_MAX_CALL_WIDTH} (PLANS, 'A call"
            " confit knows cannot trap')"
        )
    if func in _PROBED and not kernel_is_duckdbs(_PROBED[func]):
        raise NotNative(f"{what}: this platform's numpy kernel is not DuckDB's")
    return [spell(xi) for xi in x]
