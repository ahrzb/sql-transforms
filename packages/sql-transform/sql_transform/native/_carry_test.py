"""The carries of _carry.py against the operations they bound, on random
doubles of every magnitude: from a reference x and a bound e, the entry
reads v_e within e of x, and a twin v_t within e of v_e. Each side's
answer must lie within the carried bound of the other's."""

from __future__ import annotations

import numpy as np
import pytest

from sql_transform.native._carry import (
    absolute,
    as_fields,
    scaled,
    shifted,
    weighted,
)
from sql_transform.native._registry import FieldBounds

LD = np.longdouble


def _doubles(rng: np.random.Generator, n: int) -> np.ndarray:
    """Doubles of random sign and binary exponent, subnormals and zeros
    among them."""
    x = rng.choice([-1.0, 1.0], n) * np.ldexp(
        rng.uniform(1.0, 2.0, n), rng.integers(-1080, 1000, n)
    )
    x[rng.random(n) < 0.02] = 0.0
    return x


def _sides(rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
    """References x, bounds e (from 2**-1074 to 2**-20 of |x|), and doubles
    v_e and v_t with |v_e - x| <= e and |v_t - v_e| <= e."""
    x = _doubles(rng, n)
    e = np.maximum(np.abs(x) * np.ldexp(1.0, rng.integers(-60, -20, n)), 2.0**-1074)
    e = np.where(rng.random(n) < 0.05, 2.0**-1074 * rng.integers(1, 8, n), e)
    # A quarter of each at a corner of its box.
    a = np.where(
        rng.random(n) < 0.25, rng.choice([-1.0, 1.0], n), rng.uniform(-1, 1, n)
    )
    b = np.where(
        rng.random(n) < 0.25, rng.choice([-1.0, 1.0], n), rng.uniform(-1, 1, n)
    )
    v_e = x + a * e
    v_t = v_e + b * e
    ok = (np.abs(LD(v_e) - LD(x)) <= LD(e)) & (np.abs(LD(v_t) - LD(v_e)) <= LD(e))
    return x[ok], e[ok], v_e[ok], v_t[ok]


def _within(y_e: np.ndarray, y_t: np.ndarray, bound: np.ndarray) -> None:
    both = np.isfinite(y_e) & np.isfinite(y_t)
    gap = np.abs(LD(y_e[both]) - LD(y_t[both]))
    worst = np.max(gap / bound[both], initial=0)
    assert worst <= 1, f"a pair {float(worst):.3g} times its carried bound apart"


def _op(op: str, v: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The double operation `op` on the fields v, with shifts m and scales
    s."""
    if op == "shift-divide":
        return (v - m) / s
    if op == "shift":
        return v - m
    if op == "divide":
        return v / s
    if op == "multiply-add":
        return v * s + m
    return v * s


@pytest.mark.parametrize(
    "op, seed",
    [
        ("shift-divide", 0),
        ("shift", 1),
        ("divide", 2),
        ("multiply-add", 3),
        ("weight", 4),
    ],
)
def test_a_carry_bounds_the_two_sides(op, seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        x, e, v_e, v_t = _sides(rng, 20_000)
        m = _doubles(rng, len(x)) * (rng.random(len(x)) < 0.8)
        s = np.where(rng.random(len(x)) < 0.5, _doubles(rng, len(x)), x)
        s = np.where(s == 0.0, 1.0, s)
        with np.errstate(all="ignore"):
            if op == "shift-divide":
                bound = shifted(e, x, m, s)
            elif op == "shift":
                bound = shifted(e, x, m)
            elif op == "divide":
                bound = shifted(e, x, s=s)
            elif op == "multiply-add":
                bound = scaled(e, x, s, m)
            else:
                bound = weighted(e, x, s)
            _within(_op(op, v_e, m, s), _op(op, v_t, m, s), bound)


def test_a_bit_exact_field_carries_0():
    x = np.array([1.5, -2.0, np.inf, np.nan])
    e = np.zeros(4, LD)
    for bound in (
        shifted(e, x, np.ones(4), np.full(4, 3.0)),
        scaled(e, x, np.full(4, 3.0), np.ones(4)),
        weighted(e, x, 7.0),
    ):
        assert (bound == 0).all()


def test_an_ulp_bound_is_twice_its_doubles_in_spacing():
    y = np.array([1.0, -3.5, 0.0, 5e-324, np.inf, np.nan, 1.7976931348623157e308, 2.0])
    fields = FieldBounds(np.array([2, 1, 3, 1, 1, 1, 1, 0]), np.full(8, np.nan, LD))
    e = absolute(fields, y)
    assert e[0] == 4 * 2.0**-52
    assert e[1] == 2 * 2.0**-51
    assert e[2] == 6 * 2.0**-1074
    assert e[3] == 2 * 2.0**-1074
    assert np.isinf(e[4]) and e[5] == 0 and np.isinf(e[6]) and e[7] == 0
    # Each double within 2k positions of y is within 2k·spacing(y) of it.
    for k, v in ((1, 1.0), (3, 2.0), (2, 0.75), (5, 2.0**-1070)):
        near = np.array([v])
        for _ in range(2 * k):
            near = np.append(near, [np.nextafter(near[-1], np.inf)])
        far = absolute(
            FieldBounds(np.array([k]), np.array([np.nan], LD)), np.array([v])
        )
        assert np.abs(LD(near[-1]) - LD(v)) <= 2 * far[0]


def test_an_error_scale_is_its_own_bound():
    y = np.array([1.0, np.nan, np.inf, 1.7e308])
    fields = FieldBounds(
        np.zeros(4, np.int64), np.array([1e-15, 1e-15, 1e-15, 1e307], LD)
    )
    e = absolute(fields, y)
    assert e[0] == LD(1e-15) and e[1] == 0 and np.isinf(e[2]) and np.isinf(e[3])
    back = as_fields(np.array([0.0, 2.5, np.inf], LD))
    assert np.isnan(back.near[0]) and back.near[1] == 2.5 and np.isinf(back.near[2])
    assert (back.ulps == 0).all()
