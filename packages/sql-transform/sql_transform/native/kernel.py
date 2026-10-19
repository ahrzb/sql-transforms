"""Kernel approximations (sklearn 1.9, `sklearn/kernel_approximation.py`):
`AdditiveChi2Sampler`, which maps each feature on its own; `RBFSampler`,
the cosine of a random projection of the row (`_rbf`); `SkewedChi2Sampler`,
the same over the logarithm of each feature shifted by the skewedness
(`_skewed_chi2`); and `Nystroem` of the rbf kernel, the kernel of the row
to each fitted component, times a fitted matrix (`_nystroem`).

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

Each output field reads its own feature. With one sample step, a list
builds about linearly in the features, and a struct, which `query` reads
field by field, about as their square: at 1,024 features, in 0.85 to 0.92 s
and 8.3 to 9.6 s (warm `to_native`, release build, master 6b7e7f1,
2026-10-17). With more steps, both grow faster: at 3 steps over 256
features, in 19.5 s and 26.5 s. The entry refuses a step whose build
`_additive_chi2_build_estimate` estimates past MAX_BUILD_S.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from scipy import sparse
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.metrics.pairwise import check_pairwise_arrays, rbf_kernel
from sklearn.utils.extmath import row_norms

from sql_transform.native._helpers import f64
from sql_transform.native._registry import (
    EPS,
    MAX_BUILD_S,
    ErrorScale,
    NotNative,
    build_estimate,
    rejects,
    translates,
)
from sql_transform.native.distance import _squared_products
from sql_transform.native.function import kernel_distance
from sql_transform.native.project import _TINY, _refuse_past_cap, _tree

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


@build_estimate(
    fields=9.06e-4,
    n_fields=3.80e-6,
    n_pairs=6.92e-5,
    n2_fields=4.28e-9,
    n_pairs2=2.16e-7,
)
def _additive_chi2_build_estimate(n: int, steps: int) -> dict[str, tuple[float, ...]]:
    """The seconds of the whole warm `to_native` of the entry, as
    `_rbf_build_estimate`, from its n features and s sample steps: its F =
    n·(2s - 1) output fields and its p = n·(s - 1) pairs of a cos and a sin
    field, by terms in F, n·F, n·p, n²·F and n·p². The struct return is the
    slower: over 1 s, it builds in 1.2 to 2.3 times the time of the list
    with 2 steps or more, and 4.2 to 10.7 times with one step, where the
    list grows about linearly in n. Fitted on the slower return of each
    shape over 117 shapes (1 to 1,024 features, 1 to 400 steps; builds of
    0.003 to 51 s; the median of three builds at one step over 896, 960 and
    1,024 features; master 6b7e7f1, 2026-10-17): over 0.3 s, each within
    0.84 to 1.22 times the estimate. The cap took on none slower than 7.39 s
    and refused none faster than 7.07 s. It takes about 900 features of one
    step, 220 of two, 148 of three, 68 of eight, 32 of 24 or 12 of 100.
    confit refused the struct of one step over 1,536 features after 2.8 s,
    past 4,000,000 tokens, and the cap refuses it at an estimated 26 s."""
    f = n * (2 * steps - 1)
    p = n * (steps - 1)
    return {
        "fields": (f,),
        "n_fields": (n, f),
        "n_pairs": (n, p),
        "n2_fields": (n, n, f),
        "n_pairs2": (n, p, p),
    }


@translates(
    AdditiveChi2Sampler,
    scale=_SCALE,
    scaled=lambda est: not _exact(est),
    builds=(_additive_chi2_build_estimate,),
)
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
    n = len(x)
    estimate = _additive_chi2_build_estimate(n, steps)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"AdditiveChi2Sampler: an estimated {estimate:.0f} s build, past"
            f" {MAX_BUILD_S:.0f} s ({steps:,} sample step{'s' * (steps != 1)}"
            f" over {n:,} features)"
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


@build_estimate(
    apart_n=7.61e-4,
    apart_m=5.21e-4,
    apart_nm=1.01e-4,
    apart_n2m=1.23e-7,
    apart_list_n2=2.20e-7,
    plain_n=6.48e-4,
    plain_m=5.80e-4,
    plain_nm=2.15e-4,
    plain_n2m=6.17e-8,
)
def _rbf_build_estimate(n: int, m: int, apart: bool) -> dict[str, tuple[float, ...]]:
    """The seconds of the whole warm `to_native` of the entry, release
    build: the translation, the probe (`_registry._probed`) and confit's
    trial build (`_registry._builds`). It grows with the n features and m
    components, where each product reads `x - 0.0` (`apart`) or
    `coalesce(x, NaN)`. Each is about linear in the products n·m, with a
    term in n²·m. Over 1 s, a struct return builds in 0.62 to 1.44 times the
    time of a list read `x - 0.0` (the list is the slower at 3,072 features
    or more), and 1.15 to 2.1 times read `coalesce(x, NaN)`. Fitted on the
    slower return of each shape over 61 shapes of each reading (2 to 4,096
    features, 1 to 2,000 components; builds of 0.01 to 18.7 s; master
    7f96c17, 2026-10-16): over 0.3 s, each within 0.77 to 1.33 times the
    estimate read `x - 0.0`, and 0.77 to 1.28 read `coalesce(x, NaN)`. The
    cap took on none slower than 6.78 s and 7.54 s, and refused none faster
    than 7.03 s and 7.01 s. The projections' estimates (project.py) took a
    struct of 512 features of 100 components read `x - 0.0` at 6.9 s, which
    built in 9.2 s, and one of 1,024 features of 50 read `coalesce(x, NaN)`
    at 6.8 s, which built in 15.4 s."""
    prefix = "apart" if apart else "plain"
    terms = {
        f"{prefix}_n": (n,),
        f"{prefix}_m": (m,),
        f"{prefix}_nm": (n, m),
        f"{prefix}_n2m": (n, n, m),
    }
    if apart:
        terms["apart_list_n2"] = (n, n) if m > 1 else (0.0,)
    return terms


def _rbf_reads_apart(est: Any) -> bool:
    """The entry reads each feature as `x - 0.0`, which confit computes
    once a row, where the build so read stays within the cap, as a
    projection does (`project._reads_apart`); past the cap, it reads
    `coalesce(x, NaN)`, which builds faster over 2,048 features or more of
    few components."""
    n, m = np.shape(est.random_weights_)
    return _rbf_build_estimate(n, m, True) <= MAX_BUILD_S


@translates(
    RBFSampler,
    scale=RBF,
    apart=_rbf_reads_apart,
    builds=(_rbf_build_estimate,),
    sum_products=lambda est, x: ((x, np.asarray(est.random_weights_).T),),
)
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
    return _projection_cos(est, x, _factor(est))


def _projection_cos(est: Any, ins: list[S.Expr], c: float) -> list[S.Expr]:
    """`c * cos(sum_i ins_i * W_ij + b_j)` for each component j, with W the
    random weights and b the random offsets as doubles (numpy widens a
    float32 fit's over a row of doubles): the products summed as a
    balanced tree, then the offset, then `cos`, NaN at an infinite angle
    as numpy answers (`_trig`), then the product by c."""
    w = np.asarray(est.random_weights_, dtype=np.float64)
    b = np.asarray(est.random_offset_, dtype=np.float64)
    out = []
    for j in range(w.shape[1]):
        angle = _tree([v * f64(wi) for v, wi in zip(ins, w[:, j], strict=True)])
        out.append(_trig("cos", angle + f64(b[j])) * f64(c))
    return out


def _skewedness(est: Any) -> float:
    """The skewedness s as a double, as the twin adds it to a row of
    doubles. NotNative where the twin adds it in another dtype (a long
    double rounds `x + s` twice), or where it is not finite."""
    given = est.skewedness
    try:
        dtype = (np.ones(1) + given).dtype
        s = float(given)
    except (TypeError, ValueError, OverflowError) as e:
        raise NotNative(f"SkewedChi2Sampler(skewedness={given!r})") from e
    if dtype != np.float64:
        raise NotNative(
            f"SkewedChi2Sampler(skewedness={given!r}): the twin adds it in {dtype}"
        )
    if not math.isfinite(s):
        raise NotNative(f"SkewedChi2Sampler(skewedness={given!r})")
    return s


def _skewed_factor(est: Any) -> float:
    """The twin's output factor, `np.sqrt(2.0) / np.sqrt(n_components)`:
    two square roots and a quotient, each correctly rounded, so the same
    double on every host."""
    return float(np.sqrt(2.0) / np.sqrt(est.n_components))


def _skewed_s(est: Any, x: np.ndarray) -> np.ndarray:
    """S for each output field at the row `x` (long doubles):
    c·(Σ|L_i|·|W_ij| + |b_j| + 1), with L_i = ln(x_i + s) and c the output
    factor. L_i reads 0 where x_i + s is not above 0, which the twin
    rejects."""
    p = np.asarray(x, np.longdouble) + np.longdouble(_skewedness(est))
    ln = np.abs(np.log(np.where(p > 0, p, np.longdouble(1.0))))
    w = np.abs(np.asarray(est.random_weights_, dtype=np.longdouble))
    b = np.abs(np.asarray(est.random_offset_, dtype=np.longdouble))
    return np.longdouble(_skewed_factor(est)) * (ln @ w + b + 1)


SKEWED_CHI2 = ErrorScale(
    k=lambda est: est.n_features_in_ + 4,
    s=_skewed_s,
    tau=lambda est: 0,
)


def _skewed_products(
    est: Any, x: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    shifted = np.asarray(x, np.float64) + _skewedness(est)
    return ((np.log(shifted), np.asarray(est.random_weights_).T),)


def _above_skew(est: Any, types: list[pa.DataType]) -> list[Any]:
    """The input guard's probe row: each feature above -s, which the twin
    takes. A number reads 1.0, or -2·s where s < 0; a boolean False where
    s > 0, else True (where s <= -1, the twin rejects every boolean); a
    string, which the twin rejects, an empty string (`_registry._base`)."""
    s = _skewedness(est)
    number = 1.0 if s >= 0 else -2.0 * s
    return [
        "" if t == pa.string() else s <= 0 if t == pa.bool_() else number for t in types
    ]


# The most products x_i·W_ij the entry takes (`_skewed_chi2`): confit
# computes the logarithm in each, as it shares no `ln`, so past this many
# a struct serves a call about as fast as the twin, or more slowly.
_SKEWED_PRODUCTS = 2048


@build_estimate(
    n=7.40e-4,
    nm=3.57e-4,
    n2m=7.48e-7,
    nm2=8.76e-6,
    list_n2=6.13e-6,
)
def _skewed_build_estimate(n: int, m: int) -> dict[str, tuple[float, ...]]:
    """The seconds of the whole warm `to_native` of the entry, as
    `_rbf_build_estimate`, from its n features and m components, where n·m
    is at most `_SKEWED_PRODUCTS`. confit computes the logarithm in each
    product, and a struct read field by field carries every field's
    logarithms, so a struct grows about as n·m²: one of 1,024 components
    over one feature builds in 12.8 s, against 0.61 s as a list, and one of
    2,048 ran out of 16 GB of memory. Fitted on the slower return of each
    shape over 53 shapes (1 to 2,048 features, 1 to 1,024 components; builds
    of 0.004 to 26.1 s; master 220a5e8, 2026-10-16): over 0.3 s, each within
    0.75 to 1.41 times the estimate. The cap took none slower than 5.18 s
    and refused none faster than 8.91 s, and it refuses the struct that ran
    out of memory, at an estimated 37 s."""
    return {
        "n": (n,),
        "nm": (n, m),
        "n2m": (n, n, m),
        "nm2": (n, m, m),
        "list_n2": (n, n) if m > 1 else (0.0,),
    }


@translates(
    SkewedChi2Sampler,
    scale=SKEWED_CHI2,
    base=_above_skew,
    builds=(_skewed_build_estimate,),
    sum_products=_skewed_products,
)
def _skewed_chi2(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    """`SkewedChi2Sampler` (`transform`): `validate_data` into doubles (a
    boolean reads 0 or 1), where NaN and the infinities raise; a raise
    where any x <= -s; then `X += s`, `np.log`, `X @ random_weights_`
    (BLAS), `+= random_offset_`, `np.cos`, and `*= np.sqrt(2.0) /
    np.sqrt(n_components)`.

    The entry is `_rbf`'s over L_i = ln(x_i + s), held to the error scale
    of the ruling (decisions/closed/matvec-parity-bound.md, item 4): for
    output field j,

        |entry - twin| <= (n + 4)·eps·S_j,   S_j = c·(T_j + 1),
        T_j = sum_i |L_i·W_ij| + |b_j|.

    `x + s` is correctly rounded on both sides, and above 0 where the twin
    takes x. numpy's `log` is within d = 1 ulp of confit's `ln` (SVML's on
    x86-64 with AVX-512, glibc's elsewhere; where the kernel probe reads
    more, the step stays Python), and an ulp of L_i is at most eps·|L_i|,
    so the two sides' L differ by eps·|L_i| at most, and the exact sums of
    their products and the offset by eps·T_j. With `_rbf`'s γ_{n+1}·T_j
    on each side, the two angles are within (n + 2)·eps·T_j to first
    order. `cos` is 1-Lipschitz and the same function on both sides (its
    probe reads 0): each side's rounding of it adds up to eps, and the
    products by c eps·c between them. c is the same double on every host.
    So the fields are within c·eps·((n + 2)·T_j + 3), which (n + 4)·eps·S_j
    covers at every n, with room for the rounding of S in `check`; tau is
    0, as for `_rbf`. `check` reads L_i as the logarithm of the exact sum
    x_i + s, which parts from each side's by eps/2 at most, a change far
    within that room.

    Measured over 6 seeds of 1, 2, 3, 5, 8, 17, 32 and 64 features of 64
    components (s of 1, 0.01, 10, -1.5 and 0; x + s near 1, of magnitudes
    1e-5 to 1e5, uniform in (0, 1e3), 1e-300 to 1e-5, and small whole
    numbers; master 220a5e8, 2026-10-16): 154 million output fields
    against the twin of the whole call, 34% of them apart, and 15 million
    against the twin of each row alone, 29% apart (the two twins part in
    31%). 102 million read numpy's `log` of AVX-512, 1 ulp from confit's
    on its probe, and 51 million glibc's (`NPY_DISABLE_CPU_FEATURES`). The
    largest |entry - twin| / (eps·S) is 2.91, at 64 features against
    K = 68; the least room is at 2 features, 1.86 against K = 6. At one
    feature, each field equals the twin's where numpy's `log` is glibc's,
    and is within 1.00·eps·S where it is AVX-512's. The measured value is
    evidence for K, not K.

    confit computes the logarithm in each of the n·m products, as it
    shares no `ln` (a value that can raise), so the entry serves more
    slowly than `_rbf`, the more so as n·m grows. A call of 64 rows takes
    119 to 123 µs at 2 features of 10 components, against 8.2 to 9.0 ms for
    the twin; 1.6 to 2.9 ms at 4 of 100, 6.3 times as fast as the twin; and
    4.9 to 41 ms at 1,600 to 2,048 products, 1.5 to 5.2 times as fast (a
    list or a struct, one thread, the median of 30 warm calls, release
    build, master 220a5e8). Past 2,048 products a call can take longer
    than the twin's: a struct serves 0.9 times as fast at 3,200 products
    (64 features of 50 components), and a list and a struct 0.9 and 0.7
    times as fast at 6,400 (64 of 100). So the step stays Python past
    2,048 (`_SKEWED_PRODUCTS`). Within them, a step builds by
    `_skewed_build_estimate`, which the cap reads.

    A fit in float32 serves: the twin's validation makes every row a
    double, booleans included, and numpy widens the weights and offsets
    to float64 over it."""
    w = np.asarray(est.random_weights_)
    b = np.asarray(est.random_offset_)
    n, m = w.shape
    s = _skewedness(est)
    if n * m > _SKEWED_PRODUCTS:
        raise NotNative(
            f"SkewedChi2Sampler: {n} features of {m} components, {n * m:,}"
            f" products past {_SKEWED_PRODUCTS:,}: confit computes the logarithm"
            " in each, so the entry would serve about as slowly as the twin"
        )
    _refuse_past_cap("SkewedChi2Sampler", n, m, _skewed_build_estimate)
    if not (np.isfinite(w).all() and np.isfinite(b).all()):
        raise NotNative("SkewedChi2Sampler: a random weight or offset not finite")
    d = {f.__name__: kernel_distance(f) for f in (np.log, np.cos)}
    if d["log"] is None or d["log"] > 1 or d["cos"] != 0:
        raise NotNative(
            "SkewedChi2Sampler: this platform's numpy log and cos are"
            f" {d['log']} and {d['cos']} ulps from confit's on the kernel probe,"
            " past what K = n + 4 reads (1 and 0)"
        )
    # NaN, above every number in DuckDB, is the least only where every
    # feature is NaN, which the probe's test traps.
    rejects((x[0] if n == 1 else S.fn("least", *x)) <= f64(-s))
    # `x + s` is above 0 where the guard passes x, so `ln` raises only
    # where the guard traps. Bare, `ln` builds and serves faster than under
    # a CASE that rules its raise out: confit shares neither (each product
    # computes its own), and the CASE splits a block in each.
    logs = [S.fn("ln", xi + f64(s)) for xi in x]
    return _projection_cos(est, logs, _skewed_factor(est))


def _nystroem_factor(est: Any) -> float:
    """-γ, the double the twin's `rbf_kernel` multiplies each squared
    distance by (`K *= -gamma`): of `gamma`, else of the gamma of
    `kernel_params`, else of `1.0 / n_features` (`_get_kernel_params`,
    whose parameters `pairwise_kernels` filters to the kernel's own).
    NotNative where the twin multiplies in another dtype than a double (a
    long double γ rounds the product twice), or where -γ is not a finite
    double of 0 or less."""
    params = est.kernel_params
    if params is not None and not isinstance(params, dict):
        raise NotNative(f"Nystroem(kernel_params={params!r})")
    given = est.gamma
    if given is None and params is not None:
        given = params.get("gamma")
    if given is None:
        return -(1.0 / est.n_features_in_)
    if isinstance(given, np.ndarray) or not isinstance(given, numbers.Real):
        raise NotNative(f"Nystroem(gamma={given!r})")
    with np.errstate(all="ignore"):
        neg = -given
        dtype = (np.ones(1) * neg).dtype
    if dtype != np.float64:
        raise NotNative(f"Nystroem(gamma={given!r}): the twin multiplies in {dtype}")
    try:
        factor = float(neg)
    except OverflowError as e:
        raise NotNative(f"Nystroem(gamma={given!r})") from e
    if not (math.isfinite(factor) and factor <= 0.0):
        raise NotNative(f"Nystroem(gamma={given!r})")
    return factor


def _nystroem_components(est: Any) -> tuple[np.ndarray, np.ndarray]:
    """The components as the twin reads them, float64, one per row, and
    each one's ‖c_j‖² as the twin computes it, a constant: through
    `check_pairwise_arrays`, then `row_norms` (sparse where the fit was,
    as distance.py). NotNative where a component is not finite, which
    `check_pairwise_arrays` rejects."""
    c = est.components_
    if not np.isfinite(c.data if sparse.issparse(c) else np.asarray(c)).all():
        raise NotNative("Nystroem: a component not finite")
    _, y = check_pairwise_arrays(np.zeros((1, est.n_features_in_)), c)
    dense = y.toarray() if sparse.issparse(y) else y
    return np.asarray(dense, np.float64), row_norms(y, squared=True)


def _nystroem_s(est: Any, x: np.ndarray) -> np.ndarray:
    """S_l = Σ_j |N_lj|·k̃_j·(1 + γ·S2_j) for each output field l at the row
    `x` (long doubles), with S2_j = ‖x‖² + ‖c_j‖² + 2·Σ_i |x_i·c_ji| and
    k̃_j = exp(-γ·max(q_j - (n + 4)/2·eps·S2_j, 0)), q_j = ‖x - c_j‖² as the
    sum of the squares of the differences (`_nystroem`)."""
    c = est.components_
    c = np.asarray(c.toarray() if sparse.issparse(c) else c, np.longdouble)
    g = -np.longdouble(_nystroem_factor(est))
    s2 = np.sum(x * x) + np.sum(c * c, axis=1) + 2 * (np.abs(c) @ np.abs(x))
    q = np.sum((x - c) ** 2, axis=1)
    near = np.maximum(q - (c.shape[1] + 4) / 2 * EPS * s2, 0)
    nrm = np.abs(np.asarray(est.normalization_, np.longdouble))
    return nrm @ (np.exp(-g * near) * (1 + g * s2))


def _nystroem_tau(est: Any) -> np.ndarray:
    """tau_l = 2**-1074·(m + (4nγ + 4)·Σ_j |N_lj|) for each output field l
    (`_nystroem`)."""
    nrm = np.abs(np.asarray(est.normalization_, np.longdouble))
    g = -np.longdouble(_nystroem_factor(est))
    n = est.n_features_in_
    return np.longdouble(_TINY) * (len(nrm) + (4 * n * g + 4) * nrm.sum(axis=1))


NYSTROEM = ErrorScale(
    k=lambda est: max(est.n_features_in_ + 5, len(est.normalization_) + 3),
    s=_nystroem_s,
    tau=_nystroem_tau,
)


def _nystroem_products(
    est: Any, x: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    c, _ = _nystroem_components(est)
    yield from _squared_products(x, c)
    k = rbf_kernel(
        np.asarray(x, np.float64).reshape(1, -1),
        c,
        gamma=-_nystroem_factor(est),
    )[0]
    yield k, np.asarray(est.normalization_)


@build_estimate(n=4.36e-4, nm=7.05e-5, m2=5.40e-5, m3=1.35e-7, n2=5.55e-7)
def _nystroem_build_estimate(n: int, m: int) -> dict[str, tuple[float, ...]]:
    """The seconds of the whole warm `to_native` of the entry, as
    `_rbf_build_estimate`, from its n features and m components: the n·m
    products of the squared distances, the m² products of the output
    fields' sums, a term in m³, and terms in n and n² that the build grows
    by at wide rows. Over 1 s, a struct return builds in 0.87 to 1.23 times
    the time of a list. Fitted on the slower return of each shape over 110
    shapes (1 to 4,096 features, 1 to 400 components; builds of 0.3 to
    23.6 s, one each; master 3e22aa6, 2026-10-17): each within 0.86 to 1.13
    times the estimate. The cap took on none slower than 7.44 s and refused
    none faster than 7.86 s. It takes about 276 components over 1 to 8
    features, 256 over 64, 200 over 256, 77 over 1,024 or 25 over 2,048,
    and one component up to 3,100 features. confit refuses a step past
    4,000,000 tokens after 6.3 to 23 s: in the grid, from 500 components
    over 1 to 32 features, 400 over 64 and 128, 350 over 256, 250 over 512,
    150 over 1,024 (a struct) and 100 over 2,048. The cap refuses each of
    them at an estimated 13.5 s or more."""
    return {
        "n": (n,),
        "nm": (n, m),
        "m2": (m, m),
        "m3": (m**3,),
        "n2": (n, n),
    }


@translates(
    Nystroem,
    scale=NYSTROEM,
    builds=(_nystroem_build_estimate,),
    sum_products=_nystroem_products,
)
def _nystroem(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    """`Nystroem(kernel="rbf")` (`transform`): `validate_data`, where NaN
    and the infinities raise; `pairwise_kernels(X, components_)`, which for
    the rbf kernel is `rbf_kernel` (`sklearn/metrics/pairwise.py`): the
    squared distance to each component c_j as distance.py spells it,
    `max(((-2·(x @ c_j)) + ‖x‖²) + ‖c_j‖², 0)`, then `*= -gamma` and
    `np.exp`, the kernel k_j; then `@ normalization_.T` (BLAS): output
    field l is the sum over j of k_j·N_lj, with N the normalization.

    The entry spells the same expression: ‖x‖², each x @ c_j and each
    output field's sum as a balanced tree, ‖c_j‖² as numpy computes it (a
    constant, `_nystroem_components`), and confit's `exp`. It is held to the
    error scale of the ruling (decisions/closed/matvec-parity-bound.md,
    item 4): for output field l,

        |entry - twin| <= max(n + 5, m + 3)·eps·S_l + tau_l,
        S_l = sum_j |N_lj|·k̃_j·(1 + γ·S2_j),
        S2_j = ‖x‖² + ‖c_j‖² + 2·sum_i |x_i·c_ji|,

    with n the number of features, m of components and γ the kernel's
    gamma. k̃_j = exp(-γ·max(q_j - (n + 4)/2·eps·S2_j, 0)) is the kernel at
    the least squared distance within the rounding of q_j = ‖x - c_j‖²: k_j
    itself to first order.

    On each side, the clipped square is within (γ_n + 2u)·S2_j of q_j
    (distance.py; Higham, *Accuracy and Stability of Numerical
    Algorithms*, §3.1; γ_n = n·u/(1 - n·u), u = eps/2), and its product by
    -γ within γ·(n + 3)·u·S2_j of -γ·q_j, to first order. So the two
    sides' arguments of `exp` are within γ·(n + 3)·eps·S2_j of each other,
    and both are at most min(0, -γ·q_j + γ·(n + 3)·u·S2_j): by the mean
    value theorem, the exact exps of the two part by at most k̃_j times
    that difference. confit's `exp` is Rust's, which calls the C library's
    (`duck_exp` in confit's `kernels.rs`): on Linux glibc's, within 0.511
    ulp of the exact value (glibc, `sysdeps/ieee754/dbl-64/e_exp_data.c`),
    and the kernel probe reads 0 where numpy's `exp` is glibc's, as without
    AVX-512. numpy's `exp` is within d = 1 ulp of confit's on the probe
    (elsewhere the step stays Python), so the two roundings add 2·0.511 + d
    = 2.022 ulps of the kernel at most, and the two kernels are within
    k̃_j·eps·((n + 3)·γ·S2_j + 2.022). Each side sums the m products
    k_j·N_lj within γ_m·sum_j |k_j·N_lj| of the exact sum, in any order,
    fused or not. So the two fields are within
    eps·sum_j |N_lj|·k̃_j·((n + 3)·γ·S2_j + m + 2.022), which
    max(n + 3, m + 2.022)·eps·S_l covers, and K = max(n + 5, m + 3) covers
    with room for the rounding of S in `check`.

    A product or a square that underflows errs by up to 2**-1075 more on
    each side, and so does each side's product by -γ; an `exp` below the
    least normal double errs by up to 2**-1074 on each side, and the two
    `exp`s part by up to d·2**-1074 more there.
    tau_l = 2**-1074·(m + (4nγ + 4)·sum_j |N_lj|) covers them. As
    distance.py's, the bound assumes that no partial sum overflows.

    The research's S reads k_j in place of k̃_j (research/2026-10-06/
    distances.md, §2.2). k̃_j/k_j is at most exp((n + 4)/2·γ·eps·S2_j), so
    the two part by more than a factor of e only where γ·eps·S2_j exceeds
    2/(n + 4): where the rounding of q_j can move the argument of `exp` by
    1 or more, and the twin's kernel is itself rounding noise. There, k_j
    can be far below what both sides compute, and the research's S fails.

    Measured over 16 seeds of 1, 2, 3, 5, 8, 17, 32 and 64 features of 8
    and 64 components, at γ = 1/n and at 1, 0.1 and 10 over n times the
    mean variance of the fit (rows normal; offset from 0 by 300 times their
    spread; of magnitudes 1e-5 to 1e5; small whole numbers; each component
    and rows near one; and with means of 1e6 times their spread, or of 1e8
    to 1e9, as the fit's; master 3e22aa6, 2026-10-17): 295 million output
    fields against the twin of the whole call, 88% of them apart, and 59
    million against the twin of each row alone, 84% apart (the two twins
    part in 87%). 265 million read numpy's `exp` of AVX-512, 1 ulp from
    confit's on its probe, and 88 million glibc's
    (`NPY_DISABLE_CPU_FEATURES`). No field is past its bound, and none
    parts from the twin in NaN. The largest |entry - twin| / (eps·S) is
    3.19, at 32 features of 64 components against K = 67; the least room
    is at 1 feature of 8 components, 1.99 against K = 11. With the
    research's S, 11.1 million fields are past K, in 685 of the 3,072
    configurations of rows far from 0, at up to 9.5e286·eps·S. The measured
    value is evidence for K, not K.

    A call of 64 rows takes 61 to 77 µs at 2 features of 10 components,
    against 33 ms for the twin; 1.7 to 3.5 ms at 8 to 256 features of 50
    to 100 components, 12.5 to 25.7 times as fast as the twin; and 6.6 to
    9.5 ms at 64 features of 200 components or 8 of 250, 4.9 to 10.3 times
    as fast (a list or a struct, one thread, the median of 30 warm calls,
    release build, master 3e22aa6). A row computes n·m products for the
    distances and m² for the output fields. A step builds in the time that
    `_nystroem_build_estimate` estimates, which the cap reads: 8 features
    of 250 components build in 5.6 s as a list and 6.1 s as a struct.

    `check` runs the definition that DuckDB registers, which spells each
    kernel out in every output field, so its SQL grows as m²·n: 25 MB at 8
    features of 100 components, and DuckDB ran out of 16 GB at 64 of 100.
    The tests and catalog fixtures hold few components.

    Only the rbf kernel serves. A fit in float32, on booleans or on a
    sparse matrix serves: `check_pairwise_arrays` widens the components and
    the row to float64, numpy widens a float32 normalization in the
    product, and sparse components read the same sums in scipy's order."""
    kernel = est.kernel
    if not (isinstance(kernel, str) and kernel == "rbf"):
        raise NotNative(f"Nystroem(kernel={kernel!r}): only the rbf kernel serves")
    factor = _nystroem_factor(est)
    c, cc = _nystroem_components(est)
    m, n = c.shape
    nrm = np.asarray(est.normalization_)
    if nrm.shape != (m, m) or not np.isfinite(nrm).all():
        raise NotNative("Nystroem: a normalization not finite")
    _refuse_past_cap("Nystroem", n, m, _nystroem_build_estimate)
    d = kernel_distance(np.exp)
    if d is None or d > 1:
        raise NotNative(
            f"Nystroem: this platform's numpy exp is {d} ulps from confit's on"
            " the kernel probe, past what K = max(n + 5, m + 3) reads (1)"
        )
    xx = _tree([xi * xi for xi in x])
    k = []
    for cj, ccj in zip(c, cc, strict=True):
        dot = _tree([xi * f64(ci) for xi, ci in zip(x, cj, strict=True)])
        q = ((f64(-2.0) * dot) + xx) + f64(ccj)
        k.append(S.fn("exp", S.fn("greatest", q, _ZERO) * f64(factor)))
    w = np.asarray(nrm, np.float64)
    return [_tree([kj * f64(v) for kj, v in zip(k, row, strict=True)]) for row in w]
