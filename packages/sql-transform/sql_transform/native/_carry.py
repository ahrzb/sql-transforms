"""How a step carries a bound on its input to its output fields: the parity
bound of a composition where a step reads the output of a step whose bound
is above 0 (option 2 of the owner's ruling,
proposals/closed/bounded-steps-in-compositions.md).

There, the entry reads a value v_e and the twin reads a value v_t, and
the bound e says |v_e - v_t| <= e, field by field, for the twin of any
call and any host, as each family's bound does: where e is 0, the two are
bit-exact. A step carries e to its output fields where its derivation
bounds |y_e - y_t| for each such pair, with y_e its entry's answer at v_e
and y_t its twin's answer at v_t. With f the exact map of the step,

    |y_e - y_t| <= |y_e - f(v_e)| + |f(v_e) - f(v_t)| + |f(v_t) - y_t|.

For an affine step, f(v) = A·v + b, the middle term is at most |A|·e. Each
outer term is the rounding of one side, which the family bounds at that
side's input. `check` reads the twin's value v_r of one call of one row
(the reference): the entry's v_e is within e of it, and each twin's v_t
within 2e. So each rounding term is bounded at the corner |v_r| + 2e of
these boxes, and the bound of y holds again for the twin of any call, with
the twin's y_r at v_r as its reference. One operation rounds its exact
value z by at most u·|z| + 2**-1075, u = 2**-53 (Higham, *Accuracy and
Stability of Numerical Algorithms*, §2.2, with gradual underflow), and a
sum or difference whose result is subnormal is exact.

The bound of a field is 0 only where the field is bit-exact on both
sides: a step that is bit-exact carries 0 to each output field that reads
only bit-exact input. As each family's bound, a carried bound assumes that
no sum overflows on one side only. Where one does, at values of about
DBL_MAX/n, an output field can be infinite or NaN on one side alone, a
later step's validation can raise on one side alone, either side, and the
twin itself answers such a row one way in one call and another way in
another. The bound does not cover that row, and
questions/open/where-a-sum-overflows.md asks the owner what
holds there.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from sql_transform.native._registry import FieldBounds

LD = np.longdouble
EPS = LD(2.0**-52)
U = LD(2.0**-53)
TINY = LD(2.0**-1074)
_DBL_MAX = LD(sys.float_info.max)
# Each formula below takes a few operations in long doubles a field, each
# rounding by at most a long double's eps: its answer times this covers
# them.
SLACK = 1 + 16 * np.finfo(LD).eps


def gamma(m: Any) -> Any:
    """γ_m = m·u/(1 - m·u) (Higham, §3.1), as long doubles."""
    mu = np.asarray(m, LD) * U
    return mu / (1 - mu)


def carried(out: np.ndarray, live: np.ndarray) -> np.ndarray:
    """`out` where `live` (an output field that reads an input field whose
    bound is above 0), 0 elsewhere; NaN, which an infinite bound times 0
    gives, as inf."""
    with np.errstate(invalid="ignore"):
        out = np.where(np.isnan(out), LD(np.inf), out)
    return np.where(live, out, LD(0))


def absolute(fields: FieldBounds, y: np.ndarray) -> np.ndarray:
    """Each output field's bound `fields` (an ulp bound where `near` is
    NaN) as a bound on |entry - twin|, at the reference `y`: 0 where the
    field is bit-exact or the twin's value NaN (both sides are then NaN),
    inf where the bound reaches past DBL_MAX. An entry within k doubles of
    the twin's of any call is within 2k of the reference, so within
    2k·spacing(y) of the twin."""
    y = np.asarray(y, LD)
    ulps = np.asarray(fields.ulps, LD)
    near = np.asarray(fields.near, LD)
    with np.errstate(all="ignore"):
        ay = np.abs(y)
        spacing = np.spacing(ay.astype(np.float64)).astype(LD)
        e = np.where(np.isnan(near), 2 * ulps * spacing, np.maximum(near, TINY))
        past = np.isnan(e) | ~np.isfinite(y) | (ay + 4 * e >= _DBL_MAX)
        e = np.where(past, LD(np.inf), e)
    exact = np.isnan(near) & (ulps == 0)
    return np.where(exact | np.isnan(y), LD(0), e)


def as_fields(e: np.ndarray) -> FieldBounds:
    """A carried bound as `check` reads it: bit-exact where it is 0, else
    |entry - twin| <= e."""
    e = np.asarray(e, LD)
    with np.errstate(invalid="ignore"):
        e = np.where(np.isnan(e), LD(np.inf), e)
    return FieldBounds(np.zeros(len(e), np.int64), np.where(e > 0, e, LD(np.nan)))


def shifted(e: np.ndarray, x: np.ndarray, m: Any = None, s: Any = None) -> np.ndarray:
    """The bound that `fl(fl(v - m) / s)` carries, field by field, from the
    reference `x` and its bound `e`; `m` or `s` is None where the twin
    skips that operation (a scaler, or a projection's centering). With
    a = |x - m|, one side is within γ_2·a/|s| + 2**-1075 of (v - m)/s (u
    for one operation, and the difference rounds to no subnormal), so

        |y_e - y_t| <= (e + γ_2·((a + e) + (a + 2e)))/|s| + 2**-1074."""
    e = np.asarray(e, LD)
    if m is None and s is None:
        return e.copy()
    x = np.asarray(x, LD)
    with np.errstate(all="ignore"):
        a = np.abs(x if m is None else x - np.asarray(m, LD))
        g = gamma(2) if m is not None and s is not None else U
        out = e + g * (2 * a + 3 * e)
        if s is not None:
            out = out / np.abs(np.asarray(s, LD)) + TINY
        return carried(out * SLACK, e > 0)


def scaled(e: np.ndarray, x: np.ndarray, s: Any, m: Any) -> np.ndarray:
    """The bound that `fl(fl(v·s) + m)` carries (`MinMaxScaler`): one side
    is within γ_2·(|v·s| + |m|) + (1 + u)·2**-1075 of v·s + m, so with
    a = |x|,

        |y_e - y_t| <= |s|·e + γ_2·(|s|·(2a + 3e) + 2|m|) + 2·2**-1074."""
    e = np.asarray(e, LD)
    with np.errstate(all="ignore"):
        a = np.abs(np.asarray(x, LD))
        sa, ma = np.abs(np.asarray(s, LD)), np.abs(np.asarray(m, LD))
        out = sa * e + gamma(2) * (sa * (2 * a + 3 * e) + 2 * ma) + 2 * TINY
        return carried(out * SLACK, e > 0)


def weighted(e: np.ndarray, y: np.ndarray, w: float) -> np.ndarray:
    """The bound that a part's weight carries, `fl(y·w)` (compose.py): from
    the reference `y`, the part's twin answer, and its bound `e`,

        |y_e·w - y_t·w| <= |w|·e + u·|w|·((|y| + e) + (|y| + 2e)) + 2**-1074."""
    e = np.asarray(e, LD)
    with np.errstate(all="ignore"):
        a = np.abs(np.asarray(y, LD))
        aw = np.abs(LD(w))
        out = aw * e + U * aw * (2 * a + 3 * e) + TINY
        return carried(out * SLACK, e > 0)
