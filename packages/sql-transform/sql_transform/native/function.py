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
does (`kernel_is_confits` probes it), up to `_MAX_TRIG_WIDTH` features.

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
from confit import DuckDBInferFn
from confit import sql as S
from sklearn.preprocessing import FunctionTransformer

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates

_NAN = f64(math.nan)
_ZERO = f64(0.0)
# The widest step served by `sin` or `cos`. They raise on an infinity in
# DuckDB, and confit counts them as able to trap even under the guard that
# rules it out, so a struct field read also evaluates every other lane's
# call and serving grows as the square of the width: per row, three
# instances, 1,024-row batches, against the Python step, 13.7 vs 20.8 us
# at 8 features, 26.8 vs 25.8 at 10, 36.6 vs 29.1 at 12, 178 vs 49 at 24
# (release build, master with confit #362, 2026-10-05), until confit knows
# a guarded `sin` and `cos` (PLANS, "Needs from confit"). The other
# spellings are trap-free and serve 128 features.
_MAX_TRIG_WIDTH = 8


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

# The spellings capped at _MAX_TRIG_WIDTH features.
_TRIG = {np.sin, np.cos}

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
def kernel_is_confits(name: str) -> bool:
    """Whether numpy's float64 `name` answers as confit's `name`, the engine
    that serves the entry, on probe values: 40,000 draws over the
    magnitudes a double spans (huge arguments exercise the range
    reduction), and the signed zeros and subnormals. A probe that fails to
    run answers no."""
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
    rows = pa.table({"x": x})
    sql = f"SELECT {name}(x) AS y FROM __THIS__"  # noqa: S608 — a fixed name
    try:
        probe = DuckDBInferFn(
            sql, row_tables={"__THIS__": rows.schema}, static_tables={}, udfs=[]
        )
        got = probe.infer_arrow(rows).column("y").to_numpy()
    except Exception:  # noqa: BLE001 — any failure leaves the entry out
        return False
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
    if func in _TRIG and len(x) > _MAX_TRIG_WIDTH:
        raise NotNative(
            f"{what} over {len(x)} features: every lane's call is evaluated"
            f" at each field read, past {_MAX_TRIG_WIDTH} (PLANS, 'sin and"
            " cos under a guard')"
        )
    if func in _PROBED and not kernel_is_confits(_PROBED[func]):
        raise NotNative(f"{what}: this platform's numpy kernel is not confit's")
    return [spell(xi) for xi in x]
