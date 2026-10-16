"""Kernel approximations (sklearn 1.9, `sklearn/kernel_approximation.py`):
`AdditiveChi2Sampler`, which maps each feature on its own, and
`RBFSampler`, the cosine of a random projection of the row (`_rbf`).

For each feature x, with s the sample interval, the twin
(`_transform_dense`) answers 0 where x is 0, else `sqrt(x * s)`; then, for
each j from 1 to `sample_steps - 1`, two more fields, `f * cos(j * (s *
log(x)))` and `f * sin(...)`, with `f = sqrt(2 * x * s / cosh(pi * j * s))`.
The first field is a product and a square root, each correctly rounded on
both sides, so `sample_steps=1` is bit-exact.

The other fields read numpy's `log`, which on x86-64 with AVX-512 is SVML's
kernel, up to 1 ulp from glibc's (DuckDB's and confit's `ln`), and a
rounding apart there moves `cos` and `sin` by any number of ulps near their
zeros. So they serve within the error scale the owner ruled
(decisions/closed/additive-chi2-parity-bound.md): for the cos or sin field
of feature x and step j,

    |entry - twin| <= K·eps·S + tau,   S = f·(1 + |j·s·ln x|),

with K = 4. The entry spells the twin's operations in its order, with the
twin's `cosh(pi * j * s)` as a constant computed by numpy, so `f` is the
same double on both sides, and so are `cos` and `sin` of the same double
(numpy's are glibc's). Only `ln` differs, by d = 1 ulp, and the roundings
of `s * ln x` and `j * t` amplify that by A < 3 (A = 2/σ(s), with σ(s) the
significand of s, or 1 where s is a power of two). The rounding of the
field itself adds κ <= 3, and the record bounds K by max(d·A, κ), plus 1
where the constant is not the twin's own: K = 4 holds wherever numpy's
`cosh` runs. A product rounded below the least normal double errs by up to
2**-1075 on each side, which `tau = 2**-1074` covers. Where the kernel
probes of `log`, `sin` and `cos` read 0 and numpy's `cosh` is glibc's for
the step's constants, as on a host without AVX-512, the entry is bit-exact
and serves by default (`_exact`).

Measured over 303 million cos and sin fields (8 configurations of 2 to
18 steps and 7 intervals; x uniform in (0, 1e3), of random significands,
near 1, log-uniform in (1e-300, 1e300) and small integers; 10 seeds;
master 5edd450 with this module, 2026-10-16): 0.13% part from the twin,
and the largest |entry - twin| / (eps·S) is 1.96, at 5 steps of s = 0.3,
against K = 4. The first field equals the twin's in every row. The
measured value is evidence for K, not K.

Each output field reads its own feature, so a step builds about as fast as
a `StandardScaler` of its width: at 1,024 features, in 1.1 s as a list and
13.2 s as a struct, against 1.0 s and 12.6 s (warm `to_native`, release
build, master 5edd450, 2026-10-16).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.kernel_approximation import AdditiveChi2Sampler, RBFSampler

from sql_transform.native._helpers import f64
from sql_transform.native._registry import ErrorScale, NotNative, rejects, translates
from sql_transform.native.function import kernel_distance
from sql_transform.native.project import MAX_BUILD_S, _refuse_past_cap, _tree

# The sample interval sklearn takes where `sample_interval` is None, for
# `sample_steps` 1 to 3 (Vedaldi and Zisserman, 2011, figure 2 c).
_INTERVAL = {1: 0.8, 2: 0.5, 3: 0.4}
_NAN = f64(math.nan)
_ZERO = f64(0.0)


def _given(est: Any) -> Any:
    """The sample interval as the twin reads it: `sample_interval`, or
    sklearn's default for `sample_steps`."""
    given, steps = est.sample_interval, est.sample_steps
    if given is None and steps not in _INTERVAL:
        raise NotNative(
            f"AdditiveChi2Sampler(sample_steps={steps}) without a sample_interval"
        )
    return _INTERVAL[steps] if given is None else given


def _interval(est: Any) -> float:
    """The sample interval as a double. NotNative where the twin computes
    in another dtype, or where it is not a finite number of 0 or more,
    which the fit rejects and `transform` does not read again (at inf, the
    entry's `0 * s` is NaN where the twin answers 0)."""
    given = _given(est)
    dtype = (np.ones(1) * given).dtype
    if dtype != np.float64:
        # A long double interval makes the twin compute `X * s` and its root
        # in long doubles, then round the root to a double: 13% of 200,000
        # draws part from the double's root at s = 0.3.
        raise NotNative(
            f"AdditiveChi2Sampler(sample_interval={given!r}): the twin computes"
            f" in {dtype}"
        )
    s = float(given)
    if not (math.isfinite(s) and s >= 0.0):
        raise NotNative(f"AdditiveChi2Sampler(sample_interval={given!r})")
    return s


def _cosh(est: Any) -> list[float]:
    """The twin's `np.cosh(np.pi * j * s)` for each step j past the first,
    as numpy computes it. NotNative where numpy computes it narrower than
    a double (an interval of float32 or float16): the constant would then
    differ by up to an ulp of that dtype between hosts."""
    given = _given(est)
    out = []
    for j in range(1, est.sample_steps):
        c = np.cosh(np.pi * j * given)
        if np.asarray(c).dtype != np.float64:
            raise NotNative(
                f"AdditiveChi2Sampler(sample_interval={given!r}): the twin"
                f" computes its cosh in {np.asarray(c).dtype}"
            )
        out.append(float(c))
    return out


def _exact(est: Any) -> bool:
    """The entry is bit-exact for `est`: one sample step, or numpy's `log`,
    `sin` and `cos` are confit's (their kernel probes read 0), and numpy's
    `cosh` is glibc's for the step's constants, so that a twin on any such
    host computes what the entry spells."""
    if est.sample_steps == 1:
        return True
    try:
        s, consts = _interval(est), _cosh(est)
    except NotNative:
        return True  # it stays Python: no bound to read
    if any(kernel_distance(f) != 0 for f in (np.log, np.sin, np.cos)):
        return False
    return all(c == _glibc_cosh(math.pi * j * s) for j, c in enumerate(consts, start=1))


def _glibc_cosh(v: float) -> float:
    """glibc's `cosh(v)`, which `math.cosh` calls: inf where it overflows,
    where `math.cosh` raises instead (at an interval of 1e306)."""
    try:
        return math.cosh(v)
    except OverflowError:
        return math.inf


def _scale(est: Any, x: np.ndarray) -> np.ndarray:
    """S for each output field at the row `x` (long doubles): 0 in the first
    block, which is bit-exact, and `f·(1 + |j·s·ln x|)` in the cos and sin
    blocks of step j; 0 where x is 0, whose fields are 0 on both sides."""
    s = np.longdouble(_interval(est))
    x = np.asarray(x, np.longdouble)
    pos = x > 0
    safe = np.where(pos, x, np.longdouble(1.0))
    blocks = [np.zeros_like(x)]
    for j, c in enumerate(_cosh(est), start=1):
        f = np.sqrt(np.longdouble(2.0) * safe * s / np.longdouble(c))
        sj = np.where(pos, f * (1 + np.abs(np.longdouble(j) * s * np.log(safe))), 0)
        blocks += [sj, sj]
    return np.concatenate(blocks)


_SCALE = ErrorScale(
    k=lambda est: 4,
    s=_scale,
    tau=lambda est: np.longdouble(2.0**-1074),
)


def _trig(name: str, u: S.Expr) -> S.Expr:
    """`cos` or `sin` (`name`) of `u` as numpy answers it: NaN at an
    infinity, where DuckDB raises. The test reads `abs(u)`, the form confit
    reads as ruling the infinities out (#375)."""
    return S.case(S.fn("abs", u) == f64(math.inf), _NAN).otherwise(S.fn(name, u))


def _root(w: S.Expr) -> S.Expr:
    """`sqrt(w)` for a `w` that is not negative, or NaN: the test of `w`
    itself is how confit knows that the root cannot raise."""
    return S.case(w < _ZERO, _NAN).otherwise(S.fn("sqrt", w))


@translates(AdditiveChi2Sampler, scale=_SCALE, scaled=lambda est: not _exact(est))
def _additive_chi2(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: `validate_data(..., ensure_non_negative=True)`, where NaN,
    # infinity and any x < 0 raise (-0.0 passes: `X.min() < 0` reads it as
    # 0), then, over the x that are not 0, in X's dtype, `np.sqrt(X * s)`,
    # `t = s * np.log(X)`, `v = 2 * X * s`, and for each j,
    # `np.sqrt(v / np.cosh(np.pi * j * s))` times `np.cos(j * t)` and
    # `np.sin(j * t)`; 0 elsewhere.
    steps = est.sample_steps
    s = _interval(est)
    root = math.sqrt(s)
    if all(t == pa.bool_() for t in types) and (steps > 1 or root not in (0.0, 1.0)):
        # A row of boolean features none NULL reaches the twin as a boolean
        # array, which `np.zeros_like(X)` keeps boolean: True answers 1.0,
        # not sqrt(s) (nor a cos or sin field). Any number or NULL in the
        # row makes it float64.
        raise NotNative(
            "AdditiveChi2Sampler over boolean features only: the twin answers"
            f" True as 1.0, not sqrt({s!r})"
        )
    # NaN, above every number in DuckDB, is the least only where every
    # feature is NaN, which the probe's test traps.
    rejects((x[0] if len(x) == 1 else S.fn("least", *x)) < _ZERO)
    first = []
    for xi in x:
        p = xi * f64(s)
        # `p > 0` is `x != 0` on what the guard passes, but where `p`
        # underflows to 0, whose root is 0 (x = -0.0 makes `p` -0.0, and the
        # twin's `zeros_like` answers 0.0). It keeps the root off the
        # negatives, which the guard traps: confit then knows that the root
        # cannot raise, as the test reads the root's own argument, so a read
        # of one field of a struct return leaves the others unevaluated
        # (confit #362).
        first.append(S.case(p > _ZERO, S.fn("sqrt", p)).otherwise(_ZERO))
    if steps == 1:
        return first
    # K = 4 reads numpy's `log` within d = 1 ulp of glibc's, and its `sin`
    # and `cos` equal to glibc's: on a platform whose probes say otherwise,
    # no K is derived.
    d = {f.__name__: kernel_distance(f) for f in (np.log, np.sin, np.cos)}
    if d["log"] is None or d["log"] > 1 or d["sin"] != 0 or d["cos"] != 0:
        raise NotNative(
            f"AdditiveChi2Sampler(sample_steps={steps}): this platform's numpy"
            f" log, sin and cos are {d['log']}, {d['sin']} and {d['cos']} ulps"
            " from confit's on the kernel probe, past what K = 4 reads (1, 0, 0)"
        )
    blocks: list[list[S.Expr]] = [first]
    t = [f64(s) * S.fn("ln", xi) for xi in x]
    v = [(f64(2.0) * xi) * f64(s) for xi in x]
    for j, c in enumerate(_cosh(est), start=1):
        cos, sin = [], []
        for xi, ti, vi in zip(x, t, v, strict=True):
            f = _root(vi / f64(c))
            u = f64(j) * ti
            # `x > 0` keeps `ln` off 0 and the negatives (the guard traps
            # these), which confit reads as ruling out its raise.
            cos.append(S.case(xi > _ZERO, f * _trig("cos", u)).otherwise(_ZERO))
            sin.append(S.case(xi > _ZERO, f * _trig("sin", u)).otherwise(_ZERO))
        blocks += [cos, sin]
    return [e for b in blocks for e in b]


def _factor(est: Any) -> float:
    """The twin's output factor, `(2.0 / n_components) ** 0.5`."""
    return (2.0 / est.n_components) ** 0.5


def _rbf_s(est: Any, x: np.ndarray) -> np.ndarray:
    """S for each output field at the row `x` (long doubles): c·(Σ|x_i|·|W_ij|
    + |b_j| + 1), with c the output factor (`_rbf`)."""
    w = np.abs(np.asarray(est.random_weights_, dtype=np.longdouble))
    b = np.abs(np.asarray(est.random_offset_, dtype=np.longdouble))
    return np.longdouble(_factor(est)) * (np.abs(x) @ w + b + 1)


RBF = ErrorScale(
    k=lambda est: est.n_features_in_ + 3,
    s=_rbf_s,
    tau=lambda est: 0,
)


def _rbf_build_estimate(n: int, m: int, apart: bool) -> float:
    """The seconds confit takes to build the entry (warm, release build)
    from its n features and m components, where each product reads `x -
    0.0` (`apart`) or `coalesce(x, NaN)`. Each is about linear in the
    products n·m, with a term in n²·m. Over 1 s, a struct return builds in
    0.62 to 1.44 times the time of a list read `x - 0.0` (the list is the
    slower at 3,072 features or more), and 1.15 to 2.1 times read `coalesce(x,
    NaN)`. Fitted on the slower return of each shape over 61
    shapes of each reading (2 to 4,096 features, 1 to 2,000 components;
    builds of 0.01 to 18.7 s; master 7f96c17, 2026-10-16): over 0.3 s, each
    within 0.77 to 1.33 times the estimate read `x - 0.0`, and 0.77 to
    1.28 read `coalesce(x, NaN)`. The cap took on none slower than 6.78 s
    and 7.54 s, and refused none faster than 7.03 s and 7.01 s. The
    projections' estimates (project.py) took a struct of 512 features of
    100 components read `x - 0.0` at 6.9 s, which built in 9.2 s, and one
    of 1,024 features of 50 read `coalesce(x, NaN)` at 6.8 s, which built
    in 15.4 s."""
    if apart:
        return (
            7.61e-4 * n
            + 5.21e-4 * m
            + 1.01e-4 * n * m
            + 1.23e-7 * n * n * m
            + (2.20e-7 * n * n if m > 1 else 0.0)
        )
    return 6.48e-4 * n + 5.80e-4 * m + 2.15e-4 * n * m + 6.17e-8 * n * n * m


def _rbf_reads_apart(est: Any) -> bool:
    """The entry reads each feature as `x - 0.0`, which confit computes
    once a row, where the build so read stays within the cap, as a
    projection does (`project._reads_apart`); past the cap, it reads
    `coalesce(x, NaN)`, which builds faster over 2,048 features or more of
    few components."""
    n, m = np.shape(est.random_weights_)
    return _rbf_build_estimate(n, m, True) <= MAX_BUILD_S


@translates(RBFSampler, scale=RBF, apart=_rbf_reads_apart)
def _rbf(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    """`RBFSampler` (`transform`): `validate_data`, where NaN and the
    infinities raise, then `X @ random_weights_` (BLAS), `+=
    random_offset_`, `np.cos`, and `*= (2.0 / n_components) ** 0.5`.

    BLAS sums the products in an order that SQL cannot spell (project.py),
    so the entry sums them as a balanced tree, and is held to the error
    scale of the ruling (decisions/closed/matvec-parity-bound.md, item 4):
    for output field j,

        |entry - twin| <= (n + 3)·eps·S_j,   S_j = c·(T_j + 1),
        T_j = sum_i |x_i·W_ij| + |b_j|,

    with W the random weights, b the random offsets, c the output factor
    and n the number of features. On each side the angle, the sum of the
    n products and then the offset, is within γ_{n+1}·T_j of the exact
    one, in any order, fused or not (Higham, *Accuracy and Stability of
    Numerical Algorithms*, §3.1; γ_k = k·u/(1 - k·u), u = eps/2), so the
    two angles are within (n + 1)·eps·T_j to first order. `cos` is
    1-Lipschitz and the same function on both sides, where numpy's `cos`
    is confit's (its kernel probe reads 0; elsewhere the step stays
    Python): each side's rounding of it adds up to eps, the products by c
    eps·c between them, and a host whose `pow` rounds c apart eps·c. So
    the two fields are within c·eps·((n + 1)·T_j + 4), which (n + 3)·eps·S_j
    covers at every n of 1 or more, with room for the rounding of S in
    `check`. A product that underflows errs by at most 2**-1075, far below
    eps·c, so tau is 0. As for the projections, the bound assumes that no
    partial sum overflows; where the angle is infinite, numpy's `cos`
    answers NaN, as the entry does.

    Measured over 20 seeds of 1, 2, 3, 5, 8, 17, 32 and 64 features of 64
    components (`gamma` of 0.01, 1, 10 and "scale"; rows of scales 1e-2
    to 1e3, all near 300, of magnitudes 1e-5 to 1e5 with either sign, and
    small whole numbers; master 7f96c17, 2026-10-16): 328 million output
    fields against the twin of the whole call, 39% of them apart, and 33
    million against the twin of each row alone, 33% apart (the two twins
    part in 35%). The largest |entry - twin| / (eps·S) is 6.79, at 64
    features against K = 67; the least room is at 2 features, 1.96 against
    K = 5. At one feature each field equals the twin's. The measured value
    is evidence for K, not K.

    A call of 64 rows takes 71 to 93 µs at 2 features of 10 components,
    against 8.0 to 8.5 ms for the twin; 1.4 to 1.5 ms at 64 of 100 (15 to
    25 ms); 3.5 to 5.0 ms at 256 of 64 (34 to 41 ms) and 11 ms at 1,024 of
    16 (102 ms): a list or a struct, one thread, the median of 30 warm
    calls, release build. A step builds by `_rbf_build_estimate`, which
    the cap reads.

    A fit in float32 serves: numpy widens the weights and offsets to
    float64 before the product, over a row of doubles. Over boolean
    features only, the twin multiplies a boolean array, which keeps the
    product in float32: the step stays Python there."""
    w = np.asarray(est.random_weights_)
    b = np.asarray(est.random_offset_)
    n, m = w.shape
    if all(t == pa.bool_() for t in types) and np.result_type(w, b) != np.float64:
        raise NotNative(
            "RBFSampler over boolean features only, fitted in"
            f" {np.result_type(w, b)}: the twin computes in {np.result_type(w, b)}"
        )
    apart = _rbf_reads_apart(est)
    _refuse_past_cap("RBFSampler", n, m, lambda n, m: _rbf_build_estimate(n, m, apart))
    if not (np.isfinite(w).all() and np.isfinite(b).all()):
        raise NotNative("RBFSampler: a random weight or offset not finite")
    d = kernel_distance(np.cos)
    if d != 0:
        raise NotNative(
            f"RBFSampler: this platform's numpy cos is {d} ulps from confit's on"
            " the kernel probe, where K = n + 3 reads the same function"
        )
    c = f64(_factor(est))
    w64, b64 = w.astype(np.float64), b.astype(np.float64)
    out = []
    for j in range(m):
        angle = _tree([xi * f64(wi) for xi, wi in zip(x, w64[:, j], strict=True)])
        out.append(_trig("cos", angle + f64(b64[j])) * c)
    return out
