"""Pieces shared by catalog entries, on `confit.sql`.

Each spells one numpy operation in the order numpy performs it, so an entry
built from them is bit-exact with its twin wherever the twin performs that
same sequence (loops/native/goal.md, "Parity"). The order functions are
generic over `+` and `*`: run on Python floats they ARE numpy's arithmetic,
which `_helpers_test.py` checks against numpy itself.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from confit import sql as S

# The threshold under which sklearn's `_handle_zeros_in_scale` reads a scale
# as zero and replaces it by 1.0, computed as sklearn computes it.
TINY_SCALE = float(10 * np.finfo(np.float64).eps)


def f64(v: Any) -> S.Const:
    """A fitted number as a DOUBLE constant (a numpy scalar included)."""
    return S.lit(float(v))


def isnan(e: S.Expr) -> S.Expr:
    """`e` is NaN. DuckDB orders NaN above every number and equal to itself,
    so `e = NaN` is the test (`e <> e` is false there)."""
    return e == f64(math.nan)


def clip(e: S.Expr, lo: Any, hi: Any) -> S.Expr:
    """`xp.clip(e, lo, hi)` in sklearn's numpy namespace, which is
    array_api_compat's clip, not `np.clip`: below `lo` becomes `lo`, then
    above `hi` becomes `hi`, by IEEE comparisons. So NaN stays NaN (DuckDB's
    `NaN > hi` is true, hence the guard), and a signed zero equal to a zero
    bound keeps its sign. When `lo > hi`, the low arm also meets the high
    test, as in the twin's second pass."""
    lo, hi = float(lo), float(hi)
    if math.isnan(lo) or math.isnan(hi):
        raise ValueError("a NaN clip bound")
    low = hi if lo > hi else lo
    return (
        S.case(e < f64(lo), f64(low))
        .when((e > f64(hi)) & ~isnan(e), f64(hi))
        .otherwise(e)
    )


def row_max(xs: list[S.Expr]) -> S.Expr:
    """The largest of `xs`, none of them NaN (`np.max` over a row the twin
    validated finite): one `greatest` over all of them, exact whichever of
    two equal operands wins. A tournament of two-way CASEs mentions each
    half twice, so its expression is quadratic in `len(xs)` and confit
    refuses to expand it past 32 features of a `Normalizer`; nested
    two-way `greatest` calls are worse, as confit repeats their arguments
    (PLANS, "Needs from confit")."""
    return xs[0] if len(xs) == 1 else S.fn("greatest", *xs)


def dot(xs: list[S.Expr], ws: list[Any], bias: Any | None = None) -> S.Expr:
    """`sum(x_i * w_i) + bias`, accumulated left to right. NumPy/BLAS may
    reassociate a matvec, so an entry built on this declares an ulp bound."""
    acc: S.Expr | None = None
    for x, w in zip(xs, ws, strict=True):
        term = x * f64(w)
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError("an empty dot product")
    return acc if bias is None else acc + f64(bias)


def row_sum(xs: list[Any]) -> Any:
    """`np.sum(X, axis=1)` of a one-row `X` (float64): `0.0 + pairwise(row)`.

    The reduction starts from the identity 0.0, so a row of -0.0 sums to
    0.0, and adds numpy's pairwise sum of the row (`pairwise_sum` in numpy's
    `loops_utils.h.src`): fewer than 8 terms in order; up to 128 in eight
    interleaved accumulators combined as `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`,
    then the rest in order; more than 128 split at a multiple of 8 below
    the half, each side summed the same way."""
    if not xs:
        raise ValueError("an empty sum")
    return _zero(xs[0]) + _pairwise(xs)


def _pairwise(xs: list[Any]) -> Any:
    n = len(xs)
    if n < 8:
        # numpy starts this run from a zero, whose sign the outer 0.0 + ...
        # erases; starting from the first term is the same sum.
        acc = xs[0]
        for v in xs[1:]:
            acc = acc + v
        return acc
    if n <= 128:
        r = list(xs[:8])
        i = 8
        while i < n - n % 8:
            r = [r[j] + xs[i + j] for j in range(8)]
            i += 8
        acc = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for v in xs[i:]:
            acc = acc + v
        return acc
    half = n // 2
    half -= half % 8
    return _pairwise(xs[:half]) + _pairwise(xs[half:])


def _zero(like: Any) -> Any:
    return f64(0.0) if isinstance(like, S.Expr) else 0.0


def row_sumsq(xs: list[Any]) -> Any:
    """sklearn's `row_norms(X, squared=True)` of a one-row `X`, which is
    `np.einsum("ij,ij->i", X, X)`, in the order numpy's x86-64 baseline
    kernel runs it (`sum_of_products_contig_contig_outstride0_two`: SSE2,
    two lanes, multiply then add, unfused).

    Lane `l` takes the row's elements `l, l+2, ...`: per block of eight it
    adds its four products last-first, `x0*x0 + (x2*x2 + (x4*x4 + (x6*x6 +
    acc)))`; past the blocks it adds one product per pair; then `lane0 +
    lane1`. Starting a lane from 0.0 is exact to skip: a square is never
    -0.0. Other kernels (a fused multiply-add, wider lanes) give other
    sums; `row_sumsq_is_numpys` says whether this platform's is this one."""
    if not xs:
        raise ValueError("an empty sum")
    lanes: list[Any] = [None, None]

    def mac(v: Any, acc: Any) -> Any:
        sq = v * v
        return sq if acc is None else sq + acc

    n, i = len(xs), 0
    while n - i >= 8:
        for lane in (0, 1):
            for k in (3, 2, 1, 0):
                lanes[lane] = mac(xs[i + 2 * k + lane], lanes[lane])
        i += 8
    while i < n:
        for lane in (0, 1):
            if i + lane < n:
                lanes[lane] = mac(xs[i + lane], lanes[lane])
        i += 2
    a, b = lanes
    return a if b is None else a + b


@functools.cache
def row_sumsq_is_numpys() -> bool:
    """Whether this platform's `row_norms` accumulates as `row_sumsq` does,
    on probe rows of every width up to 40 (an aarch64 build fuses the
    multiply-add, and the entry cannot)."""
    from sklearn.utils.extmath import row_norms

    rng = np.random.default_rng(20261005)
    for n in range(1, 41):
        for _ in range(4):
            x = rng.normal(size=(1, n)) * 10.0 ** rng.integers(-8, 8, size=(1, n))
            got = row_sumsq([float(v) for v in x[0]])
            if got != float(row_norms(x, squared=True)[0]):
                return False
    return True
