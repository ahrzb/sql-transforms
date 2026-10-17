"""Linear projections: the row times a fitted matrix (sklearn 1.9).

`PCA` and `IncrementalPCA` (`_BasePCA._transform`): `X @ components_.T`,
minus the projected mean `mean_ @ components_.T`, then under `whiten`
divided by `sqrt(explained_variance_)`, which sklearn clips below at eps.
`TruncatedSVD` and `GaussianRandomProjection`: `X @ components_.T`. The
centered projections, `FastICA`, `LinearDiscriminantAnalysis` and the PLS
family, subtract a fitted mean from the row before the product, and may
divide it by a fitted scale: `_project_centered` gives their bound.
`SparseRandomProjection` multiplies by the CSR matrix that its fit makes,
which scipy sums in stored order, and its entry is bit-exact
(`_sparse_project`); of density 1, its components are dense, and it is
`GaussianRandomProjection`'s product. `FactorAnalysis` multiplies the
centered row by `components_ / noise_variance_`, then by an inverse that
its twin computes with LAPACK on each call: `_factor_analysis` gives its
bound, which holds for the inverse of another host's LAPACK too.

The twin hands each product to BLAS: `ddot` for one component, `gemv` for
more. OpenBLAS picks its kernel for the CPU at run time and sums with fused
multiply-adds, in an order that SQL cannot spell, and the twin itself
answers differently for one row and for a batch
(decisions/research/2026-10-06/matvec.md). So the entry sums the products
as a balanced tree and is held to the matvec parity bound
(decisions/closed/matvec-parity-bound.md, the ruling): for output field k,

    |entry - twin| <= (n + 3)·eps·S_k + tau_k,
    S_k = (sum_i |x_i·c_ki| + sum_i |m_i·c_ki|) / s_k,

with n the number of features, c_k the component, m the mean (none in
`TruncatedSVD` and the random projection) and s_k the whitening scale (1
without `whiten`). Each side's sum of n products is within γ_n·sum_i
|x_i·c_ki| of the exact one, in any order, fused or not (Higham, *Accuracy
and Stability of Numerical Algorithms*, §3.1), and the projected mean
within γ_n·sum_i |m_i·c_ki| on any host, where γ_n = n·u/(1 - n·u) and
u = eps/2. With the subtraction and the division, the two sides are within
(n + 2)·eps·S_k to first order; n + 3 also covers the second order and the
rounding of S in `check`. `tau_k = (2n/s_k + 1)·2**-1074` covers the
products and the quotient that underflow.
The bound assumes that no partial sum overflows: one order can overflow
where another does not only past |x_i·c_ki| of about DBL_MAX/n. A step
within the bound serves only with `to_native(step, allow_bound=True)`.

The bound allows either sign of a zero, but the entry gives the twin's
sign where it can (finding 77). Where every product of an output field is an
exact zero, numpy's `X @ C.T` answered 0.0 in each of 855 shapes (1 to
1,024 features, 1 to 64 components, 1 to 64 rows), and the balanced tree
answers -0.0 where every product is -0.0. So the entry adds 0.0 to its
sum, which makes a -0.0 0.0 and leaves every other value as it is. Where
every product underflows to -0.0, the sign of the twin's zero depends on
the BLAS kernel: -0.0 in 213 of the same shapes, 0.0 in the others. Over
seeds 0 to 199 of the parity run, the entry answered -0.0 for the twin's
0.0 in 403 output fields without the addition, and in none with it.

The translation computes the projected mean with numpy, in the twin's own
spelling, so that on one host both sides share it. A `PCA` or
`IncrementalPCA` fitted in float32 projects its mean in float32, which the
bound does not cover: the step stays Python. A float32 `TruncatedSVD` or
random projection serves, since numpy widens its components to float64
before the product. A whitened component whose scale sklearn clipped to eps,
or whose variance is at most eps times the largest, is rounding noise of
size O(1) for the twin too: on rank-deficient data the twin flips labels
against itself about as often as an entry does (the ruling, item 7). The
step stays Python there.

Measured over 200 seeds of each class at 1, 2, 3, 5, 8, 17, 32 and 64
features (features of scales 1e-2 to 1e4 with large means, `whiten` on or
off, rows near the mean and away from it): 6,196 steps, 5.6 million output
fields, 68% of them apart from the twin. The largest |entry - twin| /
(eps·S) is 3.90, for `TruncatedSVD` at 17 features, against K = 20; the
least room is at 5 features, 1.97 against K = 8. Four steps of 64 features
stayed Python, each for a whitened component of rounding noise. The
measured value is evidence for K, not K.

A step builds in about the time of its n·k products, with terms in the
features and their square (`_build_estimate`). Read `coalesce(x, NaN)`, 64
features of 64 components built in 0.34 to 0.48 s, 256 of 64 in
1.5 to 1.8 s, 256 of 256 in 6.4 to 8.2 s, and one component over 6,144
features in 8.7 to 9.5 s (release build, master af89dae, 2026-10-16). The
entry refuses a step estimated past MAX_BUILD_S, before confit is asked:
about 233 components over 256 features, 116 over 512, 51 over 1,024, 3 over
4,096 or one over 5,000. Serving 64 rows takes 43 to 138 us up to 16
features of 4 components, against the twin's 6.9 to 8.8 ms; 0.26 to 0.38 ms
at 32 of 16 (twin 10.6 to 13.9 ms); 0.9 to 1.0 ms at 64 of 64 (twin
16.9 to 21.8 ms); 3.7 to 4.3 ms at 256 of 64 (twin 29.7 to 34.0 ms);
12.7 to 13.6 ms at 256 of 256 (twin 47 to 53 ms); and 17.7 to 19.0 ms at
512 of 128 (twin 56 to 62 ms). Release build, master bea1372 with confit
#459 and #461, one container, 2026-10-15. Before confit #459, the entry
served 256 of 64 in 77 to 83 ms, twice the twin's time: confit read a value
that the body builds once, such as a feature's `coalesce(x, NaN)`, in about
60 ns a read.

Since T44, where `_centered_build_estimate` keeps the build within the cap,
each product reads the feature as `x - 0.0`, a value confit computes once a
row (`_reads_apart`), and the build grows with the features as a centered
projection's does: up to about 220 components over 256 features, 101 over
512, 39 over 1,024, 9 over 2,048 or one over 4,700. A list of few
components then builds about as the square of the features (T48, finding
87). At 256 features of 16 and 64 components, 64 rows then served in
0.56 to 0.67 times the time (the five classes, one process; 2.7 to 4.5 ms
at 64 components, against 4.1 to 6.7 ms), and the query of 256 of 64 built
in 1.17 to 1.20 s, against 1.02 to 1.07 s (release build, master 9f74bc0,
2026-10-16). A wider step reads `coalesce(x, NaN)`, which confit computes
again at each read, and builds as before."""

from __future__ import annotations

import functools
import warnings
import weakref
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
from confit import sql as S
from scipy import linalg, sparse
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import (
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    TruncatedSVD,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sql_transform.native._helpers import f64
from sql_transform.native._registry import ErrorScale, NotNative, translates

_EPS = float(np.finfo(np.float64).eps)
_TINY = 2.0**-1074


def _mean(est: Any) -> np.ndarray | None:
    """`mean_`, which the twin subtracts after the product, or None."""
    return est.mean_ if isinstance(est, PCA | IncrementalPCA) else None


def _scale(est: Any) -> np.ndarray:
    """The whitening scale of each output field: `sqrt(explained_variance_)`
    clipped below at eps, as sklearn computes it; 1.0 without `whiten`."""
    if not getattr(est, "whiten", False):
        return np.ones(len(est.components_))
    s = np.sqrt(est.explained_variance_)
    s[s < np.finfo(s.dtype).eps] = np.finfo(s.dtype).eps
    return s


def _k(est: Any) -> int:
    return est.n_features_in_ + 3


def _s(est: Any, x: np.ndarray) -> np.ndarray:
    c = np.abs(np.asarray(est.components_, dtype=np.longdouble))
    s = c @ np.abs(x)
    m = _mean(est)
    if m is not None:
        s = s + c @ np.abs(np.asarray(m, dtype=np.longdouble))
    return s / np.asarray(_scale(est), dtype=np.longdouble)


def _tau(est: Any) -> np.ndarray:
    return (2.0 * est.n_features_in_ / _scale(est) + 1.0) * _TINY


MATVEC = ErrorScale(k=_k, s=_s, tau=_tau)

# The longest build the entry takes on for one estimator, in seconds, by
# `_build_estimate` (as quantile.py's, spline.py's and trees.py's caps).
MAX_BUILD_S = 7.0


def _build_estimate(n: int, k: int) -> float:
    """The seconds of the whole warm `to_native` (release build) of a step
    whose products read `coalesce(x, NaN)`, past the reach of
    `_centered_build_estimate` (`_reads_apart`), from its n features and k
    components: the translation, the probe (`_registry._probed`) and
    confit's trial build (`_registry._builds`). The build is about linear in
    the products n·k, with terms in n²·k and n·k² past that, and terms in
    the features and their square. Fitted on the slower return of each shape
    (a list and a struct; of one component, a scalar and a struct of one
    field) over 41 shapes of `PCA` (64 to 6,144 features, 1 to 256
    components; builds of 0.34 to 9.5 s; master af89dae, 2026-10-16): each
    within 0.90 to 1.18 times the estimate. The cap took on none slower than
    7.11 s and refused none faster than 7.12 s. A Gaussian random projection
    of more components than features (2 to 128 features, 125 to 2,000
    components) took 0.76 to 1.09 times it over 1 s. Before T48 the estimate
    had no term in n² (master bea1372): these builds took 1.09 to 1.89 times
    it, and one component over 6,144 features built in 9.5 s against an
    estimate of 5.0 s (finding 87)."""
    p = n * k
    return (
        6.10e-4 * n + 9.28e-5 * p + 2.14e-8 * p * n + 6.82e-8 * p * k + 1.14e-7 * n * n
    )


def _centered_build_estimate(n: int, k: int) -> float:
    """The seconds of the whole warm `to_native`, as `_build_estimate`,
    where each product reads a value that confit shares: the feature's
    difference from the mean of a centered projection that subtracts one
    (`_project_centered`), or, since T44, the feature as `x - 0.0`
    (`_reads_apart`). It is `_build_estimate`, a term in n²·k, and a term in
    n² where the return can be a list, of two components or more. A list
    over shared values builds about as their number squared, where a struct,
    which `query` reads field by field, and a single component do not: `PCA`
    of 2 components over 4,096 features, read `x - 0.0`, built in 11.3 s as
    a list and 5.9 s as a struct (PLANS.md, "Needs from confit"). Fitted on
    the slower return of each shape over 177 shapes (64 to 6,144 features, 1
    to 256 components; builds of 0.29 to 17 s; `PCA` read `x - 0.0`,
    `FastICA` in both readings, `PLSRegression` and
    `LinearDiscriminantAnalysis` by "svd"; master af89dae, 2026-10-16): each
    within 0.74 to 1.27 times the estimate, and 0.86 to 1.21 over 1 s. The
    cap took on none slower than 6.85 s and refused none faster than 6.88 s.
    It takes about 220 components over 256 features, 101 over 512, 39 over
    1,024, 9 over 2,048, or one over 4,700, against 233, 116, 51, 18 and
    5,000 for `_build_estimate`. A Gaussian random projection of more
    components than features, read `x - 0.0`, took 0.70 to 0.93 times it
    over 1 s. On the builds of T41 (master 9186b3d), each centered
    projection that subtracts a mean took 0.75 to 1.06 times it. Before T48
    the estimate had no term in n²: these builds took 0.96 to 2.04 times it,
    and `FastICA` of 2 components over 4,096 features built in 11.7 s as a
    list against an estimate of 5.9 s (finding 87)."""
    return (
        _build_estimate(n, k)
        + 3.01e-8 * n * n * k
        + (2.99e-7 * n * n if k > 1 else 0.0)
    )


def _refuse_past_cap(
    name: str, n: int, k: int, by: Callable[[int, int], float] = _build_estimate
) -> None:
    estimate = by(n, k)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"{name}: an estimated {estimate:.0f} s build, past {MAX_BUILD_S:.0f} s"
            f" ({k:,} component{'s' * (k != 1)} over {n:,} features)"
        )


# The classes of `_project_centered`.
_CENTERED = (
    FastICA,
    LinearDiscriminantAnalysis,
    PLSRegression,
    PLSCanonical,
    CCA,
    PLSSVD,
)


def _reads_apart(est: Any) -> bool:
    """The entry reads each feature as `x - 0.0` (`_registry._apart`), a
    value confit computes once a row, where the build stays within the cap
    so: each product then reads a value that confit shares, as the
    centered projections' differences are, and the build grows with the
    features as theirs does (`_centered_build_estimate`). Past that, it
    reads `coalesce(x, NaN)`, which confit computes again at each read: it
    serves more slowly, and builds by `_build_estimate`."""
    if isinstance(est, LinearDiscriminantAnalysis) and est.solver == "lsqr":
        return False  # no transform: `_project_centered` refuses it
    c = _centered(est)[2] if isinstance(est, _CENTERED) else est.components_
    k, n = np.shape(c)
    return _centered_build_estimate(n, k) <= MAX_BUILD_S


def _tree(terms: list[S.Expr]) -> S.Expr:
    """The sum of `terms` as a balanced tree: its depth is log2 of their
    number, and its error bound γ_{⌈log2 n⌉+1}, half the provable one of a
    left-to-right sum (matvec.md, the verifier's item 5)."""
    if len(terms) == 1:
        return terms[0]
    h = len(terms) // 2
    return _tree(terms[:h]) + _tree(terms[h:])


@translates(
    PCA,
    IncrementalPCA,
    TruncatedSVD,
    GaussianRandomProjection,
    scale=MATVEC,
    apart=_reads_apart,
)
def _project(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    name = type(est).__name__
    c = np.asarray(est.components_)
    m = _mean(est)
    if m is not None and np.result_type(m, c) != np.float64:
        raise NotNative(f"{name}: a float32 fit projects its mean in float32")
    k, n = c.shape
    _refuse_past_cap(name, n, k)
    whiten = bool(getattr(est, "whiten", False))
    s = _scale(est)
    if whiten:
        ev = np.asarray(est.explained_variance_)
        noise = (np.sqrt(ev) < _EPS) | (ev <= _EPS * ev.max())
        if noise.any():
            raise NotNative(
                f"{name}(whiten=True): component {int(np.argmax(noise))} is a"
                " whitened component of rounding noise"
            )
    # The twin's own spelling of the projected mean (`_BasePCA._transform`).
    mc = None if m is None else (np.reshape(m, (1, -1)) @ c.T)[0]
    out = []
    for j, cj in enumerate(c):
        # `+ 0.0`: where every product is a zero, the twin answers 0.0.
        e = _tree([xi * f64(ci) for xi, ci in zip(x, cj, strict=True)]) + f64(0.0)
        if mc is not None:
            e = e - f64(mc[j])
        if whiten:
            e = e / f64(s[j])
        out.append(e)
    return out


def _csr_sum(c: Any, x: list[Any], const: Callable[[float], Any]) -> list[Any]:
    """`x @ c.T` as scipy's `csr_matvec` sums it, for a CSR matrix `c`: each
    output field from 0.0, adding each stored product in stored order, each
    product and sum rounded on its own. Over doubles (`const=float`), the
    twin's answer; over expressions (`const=f64`), the entry's."""
    data = np.asarray(c.data, dtype=np.float64)
    out = []
    for i in range(c.shape[0]):
        e = const(0.0)
        for jj in range(c.indptr[i], c.indptr[i + 1]):
            e = e + x[c.indices[jj]] * const(data[jj])
        out.append(e)
    return out


@functools.cache
def csr_sum_is_scipys() -> bool:
    """Whether this platform's scipy sums the product of rows and a CSR
    matrix's transpose as `_csr_sum` does, on probe matrices with unsorted
    indices, one row at a time and three: a build that fuses the
    multiply-add (aarch64 may) answers otherwise, and SQL cannot fuse it.
    The first probe tells them apart on every build: a·x - a·x is 0 unfused
    and the rounding error of a·x fused."""
    a = 1.0 + 2.0**-30
    probes = [
        (sparse.csr_matrix(([a, a], [0, 1], [0, 2]), shape=(1, 2)), np.array([[a, -a]]))
    ]
    rng = np.random.default_rng(20261017)
    for n, k in ((3, 2), (8, 5), (40, 7), (300, 4)):
        nnz = rng.integers(1, n + 1, size=k)
        indices = np.concatenate([rng.permutation(n)[:m] for m in nnz])
        data = rng.choice([-1.0, 1.0], size=len(indices)) * 10.0 ** rng.uniform(-3, 3)
        c = sparse.csr_matrix((data, indices, np.r_[0, np.cumsum(nnz)]), shape=(k, n))
        probes.append(
            (c, rng.normal(size=(3, n)) * 10.0 ** rng.integers(-8, 8, size=(3, n)))
        )
    for c, xs in probes:
        want = np.array([_csr_sum(c, [float(v) for v in row], float) for row in xs])
        if not (
            np.array_equal(xs @ c.T, want)
            and np.array_equal(np.vstack([r[None, :] @ c.T for r in xs]), want)
        ):
            return False
    return True


# The most entries that `_sparse_project` sums in one output field: a sum
# in scipy's order nests two levels of SQL an entry, and confit's parser
# refuses an expression nested past 1,000 levels. A component of 493 stored
# entries builds as a scalar, a list and a struct, and one of 494 does not
# as a struct (release build, master daf0e4c with T57's project.py,
# 2026-10-17); 480 leaves room for an input nested deeper than a column.
_MAX_STORED = 480


def _repeated_sums(c: Any) -> int:
    """The distinct sums that two output fields or more of the CSR matrix
    `c` spell alike: the same entries stored in the same order, of the same
    values. A field of no stored entry is the constant 0.0, which no field
    shares."""
    reads: dict[tuple[bytes, bytes], int] = {}
    for i in range(c.shape[0]):
        lo, hi = c.indptr[i], c.indptr[i + 1]
        if lo < hi:
            key = (c.indices[lo:hi].tobytes(), c.data[lo:hi].tobytes())
            reads[key] = reads.get(key, 0) + 1
    return sum(1 for v in reads.values() if v >= 2)


def _sparse_build_estimate(c: Any) -> float:
    """The seconds of the whole warm `to_native` (release build) of a
    `SparseRandomProjection` whose CSR `components_` is `c`: the
    translation, the probe (`_registry._probed`) and confit's trial build
    (`_registry._builds`), of the slower return. The entry cannot tell the
    returns apart, and a struct, which `query` reads field by field, took
    0.96 to 64 times as long as a list where it took 1 s or more (PLANS.md,
    "Needs from confit").

    It reads the n features, the k components, the p stored entries, the
    sum q of each component's stored entries squared, and the r sums that
    two output fields or more repeat (`_repeated_sums`). The terms in n and
    n² are the probe's and the input guard's, as in `_build_estimate`. The
    terms in p and q are the sums': a long sum builds about as its entries
    squared. The terms in n·k, n²·k and k² are a struct's field reads,
    each of which passes the n features and carries the k fields. The term
    in k·r is a struct's too: confit computes a sum that two fields repeat
    once a row (`share.rs`), and a struct read field by field then builds
    in about 17 µs a field and repeated sum.

    Fitted on the slower return of each shape over 116 shapes that built
    in 0.30 to 42.7 s: 88 fits of 16 to 16,384 features and 1 to 4,096
    components at the "auto" density, 1/3 and 0.1, and 28 projections of
    16 and 64 features whose 512 to 4,096 fields repeat sums (master
    daf0e4c with T57's project.py, 2026-10-17). Each took 0.66 to 2.57
    times the estimate, and 0.66 to 1.54 over 1 s. The cap took on none
    slower than 6.75 s and refused none faster than 7.20 s. Over 50 random
    fits not fitted on (22 to 5,261 features, 2 to 3,904 components, the
    "auto" density or 0.02 to 0.5; builds of 0.29 to 12.3 s), each took
    0.80 to 1.26 times the estimate, and the cap took on none slower than
    6.76 s and refused none faster than 8.39 s. At the "auto" density the
    cap takes about 2,900 components over 16 features, 4,500 over 64,
    2,100 over 256, 560 over 1,024, 210 over 2,048 or 45 over 4,096, and
    one component over 6,000 features. Each shape that confit refused, past
    4,000,000 tokens (a struct of about 105,000 stored entries or more) or
    past the code generator's size limit, is estimated at 17 s or more."""
    k, n = c.shape
    m = np.diff(c.indptr).astype(np.float64)
    p, q = float(m.sum()), float(m @ m)
    r = _repeated_sums(c)
    return (
        4.99e-4 * n
        + 1.08e-7 * n * n
        + 6.43e-5 * p
        + 2.72e-7 * q
        + 6.55e-6 * n * k
        + 2.22e-9 * n * n * k
        + 1.30e-7 * k * k
        + 1.68e-5 * k * r
    )


@translates(
    SparseRandomProjection,
    scale=MATVEC,
    scaled=lambda est: not sparse.issparse(est.components_),
    apart=lambda est: sparse.issparse(est.components_) or _reads_apart(est),
)
def _sparse_project(
    est: Any, x: list[S.Expr], types: list[pa.DataType]
) -> list[S.Expr]:
    """`SparseRandomProjection` (`transform`, sklearn 1.9): `validate_data`,
    which raises on NaN and the infinities, then `safe_sparse_dot(X,
    components_.T)`. The fit makes `components_` a CSR matrix, or, of
    density 1 (and of one feature at the "auto" density), a dense one of
    ±1/sqrt(n_components), which the twin hands to BLAS as
    `GaussianRandomProjection`'s: that one is held to the matvec bound
    (`_project`).

    For a dense X and a CSR matrix, scipy computes `components_ @ X.T`
    (`csr_matvecs`): each output field from 0.0, adding the stored
    products in stored order (the fit stores the indices unsorted), each
    product and sum rounded on its own. The entry spells the same sum, so
    it is bit-exact (`_csr_sum`), where the kernel probe finds scipy's sums
    so (`csr_sum_is_scipys`). A fit in float32 stores float32 components,
    which scipy widens to float64, exactly, before the product. The step
    stays Python where `components_` is not the CSR matrix the fit makes
    or not all finite, where a component stores more than `_MAX_STORED`
    entries, or past an estimated 7 s build (`_sparse_build_estimate`).

    Measured over 36 seeds of random fits (1 to 512 features, 1 to 128
    components, the "auto" density, 1/3, 0.1, 0.01 to 0.99 and 1, a
    quarter of them in float32) and rows of seven kinds (normal values,
    magnitudes of 1e-300 to 1e300, subnormals and tiny normals, small
    whole numbers, about the largest double, signed zeros, and NaN, an
    infinity or NULL in one feature), 4 of the seeds with numpy's AVX-512
    code turned off, as on CI's runner (release build, master daf0e4c with
    T57's project.py, 2026-10-17). Over sparse components, each of 567
    million output fields equals the twin's bit for bit, NaN as NaN, in
    calls of 20,000 rows, and so does each of 5.7 million in calls of one
    row; 3.6% of them are infinite, 0.16% NaN and 1.7% subnormal. The
    entry traps on each of the 30,045 rows where the twin raises, and on
    no other row. Of density 1, `check` held 116,972 rows within the
    matvec bound, and it failed 23 of the 40 fits over rows of about the
    largest double, where a partial sum overflows (the module's bound
    assumes that none does).

    Serving 64 rows at the "auto" density takes 29 us at 2 features of one
    component, against the twin's 11.2 ms; 0.13 to 0.14 ms at 16 of 4
    (12.6 to 14.5 ms); 0.46 to 0.73 ms at 64 of 16 (17.8 to 18.4 ms); 2.1
    to 2.2 ms at 256 of 64 (33 to 38 ms); 3.0 to 3.1 ms at 256 of 256 (43
    to 63 ms); 9.4 to 11.0 ms at 1,024 of 64 and of 256 (102 to 125 ms);
    and 37 to 39 ms at 4,096 of 16 (363 to 383 ms): 10 to 386 times as
    fast (release build, one thread, the same master)."""
    c = est.components_
    if not sparse.issparse(c):
        return _project(est, x, types)
    name = type(est).__name__
    if c.format != "csr":
        raise NotNative(f"{name}: components_ in {c.format}, not the csr the fit makes")
    if not np.isfinite(c.data).all():
        raise NotNative(f"{name}: components_ not all finite")
    k, n = c.shape
    longest = int(np.diff(c.indptr).max(initial=0))
    if longest > _MAX_STORED:
        raise NotNative(
            f"{name}: a component of {longest:,} stored entries, past the"
            f" {_MAX_STORED} whose sum in scipy's order confit parses"
        )
    estimate = _sparse_build_estimate(c)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"{name}: an estimated {estimate:.0f} s build, past {MAX_BUILD_S:.0f} s"
            f" ({c.nnz:,} stored entries of {k:,} components over {n:,} features)"
        )
    if not csr_sum_is_scipys():
        raise NotNative(
            f"{name}: this platform's scipy does not sum a CSR product as the entry"
            " does (the kernel probe)"
        )
    return _csr_sum(c, x, f64)


def _centered(est: Any) -> tuple[Any, Any, np.ndarray]:
    """What the twin subtracts from each feature before the product, what it
    then divides each difference by (None where it does neither), and the
    matrix of the product, one row per output field."""
    if isinstance(est, FastICA):
        return (est.mean_ if est.whiten else None), None, np.asarray(est.components_)
    if isinstance(est, LinearDiscriminantAnalysis):
        c = np.asarray(est.scalings_)[:, : est._max_components].T
        return (est.xbar_ if est.solver == "svd" else None), None, c
    w = est.x_weights_ if isinstance(est, PLSSVD) else est.x_rotations_
    return est._x_mean, est._x_std, np.asarray(w).T


def _d(est: Any, x: np.ndarray) -> np.ndarray:
    """The row the twin multiplies, in its own float64 arithmetic."""
    m, sd, _ = _centered(est)
    d = np.asarray(x, dtype=np.float64)
    if m is not None:
        d = d - m
    if sd is not None:
        d = d / sd
    return d


def _centered_s(est: Any, x: np.ndarray) -> np.ndarray:
    c = np.abs(np.asarray(_centered(est)[2], dtype=np.longdouble))
    return c @ np.abs(np.asarray(_d(est, x), dtype=np.longdouble))


CENTERED = ErrorScale(
    k=lambda est: est.n_features_in_ + 1,
    s=_centered_s,
    tau=lambda est: 2.0 * est.n_features_in_ * _TINY,
)


def _refuse_a_degenerate_ica(est: Any, name: str) -> None:
    """A `FastICA` fit whose output fields mix in rounding noise of size
    O(1), or that broke down, stays Python, as a whitened `PCA` component
    of rounding noise does (the ruling, item 7)."""
    if not est.whiten:
        # The twin takes the data as white: the unmixing matrix is
        # orthogonal, each row a unit vector, unless the fit broke down
        # (its symmetric decorrelation divided by an eigenvalue that it
        # raised to the smallest normal double, to rows of norm 1e147).
        norms = np.linalg.norm(np.asarray(est.components_, dtype=np.float64), axis=1)
        off = ~(np.abs(norms - 1.0) <= 1e-6)
        if off.any():
            j = int(np.argmax(off))
            raise NotNative(
                f"{name}(whiten=False): row {j} of the unmixing matrix has a norm"
                f" of {norms[j]:.9g}, not 1: its fit broke down"
            )
        return
    # Row j of `whitening_` is a unit vector over the j-th singular value
    # d_j of the centered fit data: the smaller d_j, the larger the norm (a
    # row of zeros reads nothing, where sklearn's sign rule zeroed it). A
    # d_j of 0 (a norm not finite), at most sqrt(eps) times the largest, or
    # that the "eigh" solver raised to its floor sqrt(10·eps) is noise.
    norms = np.linalg.norm(np.asarray(est.whitening_, dtype=np.float64), axis=1)
    top = np.min(norms, initial=np.inf, where=np.isfinite(norms) & (norms > 0))
    noise = ~(top > np.sqrt(_EPS) * norms)
    if est.whiten_solver == "eigh":
        floor = np.sqrt(10 * np.finfo(est.whitening_.dtype).eps)
        noise |= norms * floor >= 1 - 1e-9
    if noise.any():
        raise NotNative(
            f"{name}(whiten={est.whiten!r}): the whitening's direction"
            f" {int(np.argmax(noise))} is a whitened component of rounding noise"
        )
    # Under "unit-variance", sklearn divides each row by the spread of its
    # source over the fit data: a source of no spread, such as a column
    # constant but for the rounding of its mean, makes the row infinite.
    off = ~np.isfinite(np.asarray(est.components_)).all(axis=1)
    if off.any():
        raise NotNative(
            f"{name}(whiten={est.whiten!r}): row {int(np.argmax(off))} of"
            " components_ is not finite: its fit broke down"
        )


@translates(*_CENTERED, scale=CENTERED, apart=_reads_apart)
def _project_centered(
    est: Any, x: list[S.Expr], types: list[pa.DataType]
) -> list[S.Expr]:
    """The centered projections (sklearn 1.9). The twin subtracts a fitted
    mean from the row, and may divide each difference by a fitted scale,
    before the product (`_centered`):

    - `FastICA` (`transform`): under `whiten`, `X -= mean_`; then
      `np.dot(X, components_.T)`.
    - `LinearDiscriminantAnalysis` (`transform`): `(X - xbar_) @ scalings_`
      under the "svd" solver and `X @ scalings_` under "eigen", then the
      first `_max_components` columns. Under "lsqr" the twin raises.
    - `PLSRegression`, `PLSCanonical` and `CCA` (`_PLS.transform`):
      `X -= _x_mean`, `X /= _x_std`, then `np.dot(X, x_rotations_)`.
      `PLSSVD`: `(X - _x_mean) / _x_std`, then `np.dot(Xr, x_weights_)`.

    Each subtraction and division is one IEEE operation on each side, so
    the entry multiplies the twin's own row d, and only the sums differ.
    For output field k,

        |entry - twin| <= (n + 1)·eps·S_k + tau_k,   S_k = sum_i |d_i·c_ki|,

    with c_k the row of the matrix that gives field k. Each side is within
    γ_n·S_k of the exact sum (the module's derivation), so the two are
    within n·eps·S_k to first order; n + 1 also covers the second order and
    the rounding of S in `check`. `tau_k = 2n·2**-1074` covers the products
    that underflow. As for the projections above, the bound assumes that no
    partial sum overflows. A fit in float32 serves: numpy widens the mean,
    the scale and the matrix to float64 before it subtracts, divides and
    multiplies.

    `FastICA` stays Python where its output fields mix in rounding noise of
    size O(1), as a whitened `PCA` component of rounding noise does, or
    where its fit broke down (`_refuse_a_degenerate_ica`). Where the twin
    subtracts a mean, the entry builds in 0.93 to 2.1 times the time of a
    `PCA` of the same shape that reads `coalesce(x, NaN)`, and in 0.81 to
    1.35 times that of one that reads `x - 0.0` (T48), and the cap
    estimates it so (`_centered_build_estimate`).

    Measured over 200 seeds of each class at 1, 2, 3, 5, 8, 17, 32 and 64
    features (correlated features of scales 1e-2 to 1e4 and means up to
    100 times as large, a random configuration of each class, rows near
    the mean and rows drawn like the fit; master 9186b3d, 2026-10-16):
    9,599 steps, 4.3 million output fields, 64% of them apart from the
    twin. The largest |entry - twin| / (eps·S) is 4.09, for `PLSCanonical`
    at 64 features, against K = 65; the least room is at 3 features, 1.68
    for `LinearDiscriminantAnalysis` against K = 4. At one feature each
    field equals the twin's. One `FastICA` step of 32 features stayed
    Python, for a whitened component of rounding noise.

    Each field gives a zero the twin's sign. Where every product is a
    zero, the twin's sum of more than one product is 0.0, so the entry
    adds 0.0 to each sum. At one feature and one output field, the twin's
    `np.dot` of the 1×1 row and the 1×1 matrix answers the product
    itself, so the entry of `FastICA` and of the PLS classes answers the
    product too: -0.0 where it is -0.0. Before T43, the entry added 0.0
    there as well, and in reading 7's parity run 54 fields of `FastICA` at
    one feature answered 0.0 where the twin answered -0.0 (finding 81).
    The `@` of `LinearDiscriminantAnalysis` adds the product to 0.0."""
    name = type(est).__name__
    if isinstance(est, LinearDiscriminantAnalysis) and est.solver == "lsqr":
        raise NotNative(f"{name}(solver='lsqr') has no transform")
    m, sd, c = _centered(est)
    k, n = c.shape
    if k == 0:
        raise NotNative(f"{name}: no output fields")
    _refuse_past_cap(
        name, n, k, _build_estimate if m is None else _centered_build_estimate
    )
    if isinstance(est, FastICA):
        _refuse_a_degenerate_ica(est, name)
    if not all(np.isfinite(a).all() for a in (m, sd, c) if a is not None):
        raise NotNative(f"{name}: a fitted mean, scale or matrix not all finite")
    d = list(x)
    if m is not None:
        d = [di - f64(mi) for di, mi in zip(d, m, strict=True)]
    if sd is not None:
        # Dividing by 1.0 (a scale of 0 that sklearn replaced, or
        # `scale=False`) leaves every double as it is.
        d = [di if si == 1.0 else di / f64(si) for di, si in zip(d, sd, strict=True)]
    sums = [_tree([di * f64(ci) for di, ci in zip(d, cj, strict=True)]) for cj in c]
    # `+ 0.0`: where every product is a zero, the twin answers 0.0. But
    # `np.dot` of a 1×1 row and a 1×1 matrix answers the product itself,
    # -0.0 included (finding 81), where `@` adds it to 0.0.
    if n == k == 1 and not isinstance(est, LinearDiscriminantAnalysis):
        return sums
    return [s + f64(0.0) for s in sums]


def _cov_z(w: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`Wpsi` and `cov_z` in the twin's own spelling
    (`FactorAnalysis.transform`): `w / psi`, and scipy's inverse of
    `I + Wpsi @ w.T`, with BLAS's product."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy warns of an ill-conditioned matrix
        ih = np.eye(len(w))
        wpsi = w / psi
        return wpsi, linalg.inv(ih + np.dot(wpsi, w.T))


class _Cholesky(NamedTuple):
    """The steps of `_factor_inverse`: the matrix that it forms,
    `I + v @ w.T`; its Cholesky factor L (`low`, lower triangular); L's
    inverse Z (`z`); and the inverse `c = Z.T @ Z`."""

    formed: np.ndarray
    low: np.ndarray
    z: np.ndarray
    c: np.ndarray


def _cholesky(v: np.ndarray, w: np.ndarray) -> _Cholesky | None:
    """`inv(I + v @ w.T)` in numpy's operations on whole arrays alone, each
    one IEEE operation (add, multiply, divide, sqrt) rounded on its own, in
    an order fixed here: no BLAS, no LAPACK and no reduction, so that every
    host computes the same bits. The product is the sum of the features'
    outer products, in order; then the Cholesky factor L of the sum's lower
    triangle, L's inverse by forward substitution, and the inverse as the
    sum of the outer products of the inverse's rows (exactly symmetric).
    None where a pivot is not positive or a value is not finite."""
    k, n = v.shape
    g = np.zeros((k, k))
    for f in range(n):
        g = g + np.multiply.outer(v[:, f], w[:, f])
    formed = np.eye(k) + g
    a = formed.copy()
    low = np.zeros((k, k))
    for p in range(k):
        if not a[p, p] > 0.0:
            return None
        r = np.sqrt(a[p, p])
        low[p, p] = r
        col = a[p + 1 :, p] / r
        low[p + 1 :, p] = col
        a[p + 1 :, p + 1 :] = a[p + 1 :, p + 1 :] - np.multiply.outer(col, col)
    rest = np.eye(k)
    z = np.zeros((k, k))
    for j in range(k):
        z[j] = rest[j] / low[j, j]
        rest[j + 1 :] = rest[j + 1 :] - np.multiply.outer(low[j + 1 :, j], z[j])
    c = np.zeros((k, k))
    for p in range(k):
        c = c + np.multiply.outer(z[p], z[p])
    return _Cholesky(formed, low, z, c) if np.isfinite(c).all() else None


def _factor_inverse(v: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """The entry's `cov_z`, `inv(I + v @ w.T)` (`_cholesky`), or None."""
    parts = _cholesky(v, w)
    return None if parts is None else parts.c


def _pivoted_lu(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The LU factors of `a` with partial pivoting, in numpy's operations on
    whole arrays: each pivot is the first largest candidate of its column,
    as LAPACK's getrf picks it (idamax). Answers P^T·L and U, whose product
    is about `a`, and the least ratio of a pivot to the next largest
    candidate of its column (inf where no column has two)."""
    a = a.copy()
    k = len(a)
    perm = np.arange(k)
    low = np.eye(k)
    margin = np.inf
    for p in range(k):
        col = np.abs(a[p:, p])
        r = int(np.argmax(col))
        rest = np.delete(col, r)
        if rest.size and rest.max() > 0.0:
            margin = min(margin, float(col[r] / rest.max()))
        r += p
        if r != p:
            a[[p, r]] = a[[r, p]]
            low[[p, r], :p] = low[[r, p], :p]
            perm[[p, r]] = perm[[r, p]]
        f = a[p + 1 :, p] / a[p, p]
        low[p + 1 :, p] = f
        a[p + 1 :, p:] = a[p + 1 :, p:] - np.multiply.outer(f, a[p, p:])
        a[p + 1 :, p] = 0.0
    ptl = np.empty_like(low)
    ptl[perm] = low
    return ptl, np.triu(a), margin


def _gamma(m: int) -> np.longdouble:
    """γ_m = m·u/(1 - m·u), u = eps/2 (Higham, §3.1), as a long double."""
    mu = m * np.longdouble(_EPS) / 2
    return mu / (1 - mu)


def _norm(a: np.ndarray) -> np.longdouble:
    """max(|a|_1, |a|_inf) of a matrix of no negative entry: it bounds each
    entry, and the norm of a product by the product of the norms."""
    return max(a.sum(axis=0).max(), a.sum(axis=1).max())


# FactorAnalysis serves where theta (`_factor_prior`) is at most THETA_MAX,
# and where each pivot of the LU of its I + Wpsi @ components_.T is past
# the next largest candidate of its column by a ratio of at least
# MARGIN_MIN (`_factor_analysis`).
THETA_MAX = 2.0**-10
MARGIN_MIN = 1.0 + 2.0**-8


def _factor_prior(
    v: np.ndarray, w: np.ndarray, parts: _Cholesky
) -> tuple[np.ndarray, float, float]:
    """D, which bounds |cov_z - inv(A)| entry by entry for A = I + v @ w.T,
    for the entry's `cov_z` and for the twin's on any host (the derivation
    in `_factor_analysis`), as long doubles; theta, the bound on the
    second-order terms; and the least margin of a pivot of the LU of A."""
    k, n = v.shape
    ld = np.longdouble
    ptl, up, margin = _pivoted_lu(parts.formed)
    ac = np.abs(parts.c).astype(ld)
    al, az = np.abs(parts.low).astype(ld), np.abs(parts.z).astype(ld)
    mg = np.abs(ptl).astype(ld) @ np.abs(up).astype(ld)
    mc = al @ al.T
    b = np.eye(k, dtype=ld) + np.abs(v).astype(ld) @ np.abs(w).astype(ld).T
    nl, nr = az.T @ al.T, al @ az
    fg = ac @ mg @ ac
    fc = nl @ ac + ac @ nr + ac @ mc @ ac + az.T @ az
    gg = ac @ b @ ac
    g3, gc, gf = _gamma(3 * k + 2), _gamma(k + 2), _gamma(n + 3)
    d0 = np.maximum(g3 * fg, gc * fc) + gc * fc + 2 * gf * gg
    e = 2 * _norm(d0)
    # What each term of d0 gains where |cov_z| stands for the twin's and
    # the exact inverse's magnitudes, a factor for each error in them.
    lu = _norm(mg @ ac) + _norm(ac @ mg) + _norm(mg) * e
    chol = _norm(nl) + _norm(nr) + _norm(mc @ ac) + _norm(ac @ mc) + _norm(mc) * e
    form = _norm(b @ ac) + _norm(ac @ b) + _norm(b) * e
    theta = g3 * lu + 2 * gc * chol + 2 * gf * form
    return d0 + 2 * theta * _norm(d0), float(theta), margin


class _Factor(NamedTuple):
    """What `FactorAnalysis`'s entry and error scale read: the twin's
    `Wpsi`, the entry's `cov_z` (`_factor_inverse`), and D
    (`_factor_prior`), with `Wpsi` and the magnitudes as long doubles."""

    v: np.ndarray
    c: np.ndarray
    v_ld: np.ndarray
    abs_v: np.ndarray
    abs_c: np.ndarray
    d: np.ndarray


# `_factor`'s answers: an estimator's id -> (a weak reference to it, the
# fitted arrays they read, the answer). Holding the arrays keeps their ids.
_FACTORS: dict[int, tuple[Any, np.ndarray, np.ndarray, _Factor]] = {}


def _factor(est: Any) -> _Factor:
    """`_Factor` of `est`, computed once for its fitted arrays: `check`
    reads it at each row. Raises `NotNative` where the entry's `cov_z` is
    not finite, where theta is past THETA_MAX, or where a pivot's margin
    is below MARGIN_MIN."""
    w, psi = est.components_, est.noise_variance_
    hit = _FACTORS.get(id(est))
    if hit is not None and hit[0]() is est and hit[1] is w and hit[2] is psi:
        return hit[3]
    name = type(est).__name__
    with np.errstate(all="ignore"):
        v = np.asarray(np.asarray(w) / np.asarray(psi), np.float64)
        wd = np.asarray(w, np.float64)
        parts = _cholesky(v, wd)
        if parts is None:
            raise NotNative(
                f"{name}: I + Wpsi @ components_.T has no finite inverse"
                " (noise_variance_ of the fit too small)"
            )
        d, theta, margin = _factor_prior(v, wd, parts)
    if not theta <= THETA_MAX:
        raise NotNative(
            f"{name}: I + Wpsi @ components_.T is too ill-conditioned for a"
            f" bound that holds on any host (theta {theta:.3g}, past 2**-10)"
        )
    if not margin >= MARGIN_MIN:
        raise NotNative(
            f"{name}: two pivot candidates of the LU of I + Wpsi @ components_.T"
            f" are within a ratio of {margin:.9g}: another host's LAPACK may"
            " pick the other one"
        )
    ld = np.longdouble
    out = _Factor(
        v=v,
        c=parts.c,
        v_ld=np.asarray(v, ld),
        abs_v=np.abs(np.asarray(v, ld)),
        abs_c=np.abs(np.asarray(parts.c, ld)),
        d=d,
    )
    if len(_FACTORS) >= 16:
        _FACTORS.clear()
    _FACTORS[id(est)] = (weakref.ref(est), w, psi, out)
    return out


def _factor_k(est: Any) -> int:
    k, n = np.shape(est.components_)
    return n + k + 1


def _factor_s(est: Any, x: np.ndarray) -> np.ndarray:
    f = _factor(est)
    k, n = np.shape(est.components_)
    d = np.asarray(np.asarray(x, dtype=np.float64) - est.mean_, dtype=np.longdouble)
    s1 = f.abs_v @ np.abs(d)
    t = np.abs(f.v_ld @ d)
    keps = np.longdouble(_factor_k(est)) * np.longdouble(_EPS)
    return s1 @ f.abs_c + ((t + _gamma(n + k) * s1) @ f.d) / keps


def _factor_tau(est: Any) -> np.ndarray:
    f = _factor(est)
    k, n = np.shape(est.components_)
    return (n * (f.abs_c.sum(axis=0) + f.d.sum(axis=0)) + 2 * k + 1) * _TINY


FACTOR = ErrorScale(k=_factor_k, s=_factor_s, tau=_factor_tau)


def _factor_build_estimate(n: int, k: int) -> float:
    """The seconds of the whole warm `to_native` (release build) of a
    `FactorAnalysis` of n features and k components: the translation,
    which computes D (`_factor_prior`), the probe (`_registry._probed`) and
    confit's trial build (`_registry._builds`), of the slower return. The
    build is about linear in the products n·k, with terms in n²·k and
    n·k², in the features, in their square where the return can be a list
    (as `_centered_build_estimate`: each product reads a difference
    x - mean_ that confit shares), and in k² for the second product.

    Fitted on the slower return of each shape over 71 shapes that built in
    0.31 to 19.3 s (64 to 4,096 features, 1 to 256 components; T58's draft
    of the entry, branch commit 0e4cf11, 2026-10-17): each took 0.88 to
    1.53 times the estimate, and 0.88 to 1.14 over 1 s. The cap took on
    none slower than 5.88 s and refused none faster than 6.85 s. Over 40
    random shapes not fitted on, with the entry as it is (67 to 5,201
    features, 1 to 160 components; builds of 0.29 to 13.3 s), each took
    0.89 to 1.31 times the estimate, and 0.93 to 1.31 over 1 s. The cap
    took on none slower than 8.01 s and refused none faster than 7.62 s.
    It takes each k up to 128 features, about 144 components over 256, 74
    over 512, 25 over 1,024, 5 over 2,048 or 1 over 4,096, and one
    component over 4,700 features."""
    p = n * k
    return (
        5.97e-4 * n
        + 4.70e-5 * p
        + 1.78e-7 * p * n
        + 3.07e-7 * p * k
        + (1.89e-7 * n * n if k > 1 else 0.0)
        + 8.48e-5 * k * k
    )


@translates(FactorAnalysis, scale=FACTOR)
def _factor_analysis(
    est: Any, x: list[S.Expr], types: list[pa.DataType]
) -> list[S.Expr]:
    """`FactorAnalysis` (`transform`, sklearn 1.9): with
    `Wpsi = components_ / noise_variance_` and
    `cov_z = linalg.inv(I + Wpsi @ components_.T)`, the twin answers
    `((X - mean_) @ Wpsi.T) @ cov_z`. It hands each product to BLAS and
    computes `cov_z` on each call with LAPACK: by Cholesky (potrf, potri)
    where `I + Wpsi @ components_.T` is exactly symmetric, which scipy 1.18
    detects, and else by LU with partial pivoting (getrf, getri). So the
    bits of `cov_z` depend on the host's BLAS and LAPACK kernels and on
    their thread count. Over 80 fits on 8 BLAS setups of one machine (T58),
    2,966 of the 4,480 ordered pairs of setups answered `cov_z` apart, by
    up to 4.1e-6 of its largest entry. A bound that reads the twin's own
    `cov_z` in the process failed against the twin of another setup in 46
    of the 80 fits, by up to 2.9e8 times.

    The entry multiplies the twin's own differences d = x - mean_ and its
    own `Wpsi`, each one IEEE operation on either side, then its own
    `cov_z`, C (`_cholesky`: no BLAS and no LAPACK, the same bits on every
    host), each product summed as a balanced tree. With A =
    I + Wpsi·components_^T, S1_j = sum_i |d_i·Wpsi_ji| and t = d·Wpsi^T,
    for output field l,

        |entry - twin| <= 2γ_{n+k}·(S1·|C|)_l + ((|t| + γ_{n+k}·S1)·D)_l + tau_l,

    where D bounds, entry by entry, how far C and the twin's `cov_z` on
    any host are from the exact inverse of A (`_factor_prior`). On either
    side the two products, of n terms and then of k, are within
    γ_{n+k}·S1·|cov_z| of the exact product with that side's `cov_z`, in
    any order, fused or not (the module's derivation); and the twin's
    `cov_z` is within D of C. So K = n + k + 1 and
    S = S1·|C| + (|t| + γ_{n+k}·S1)·D/(K·eps): K·eps covers 2γ_{n+k} and
    the rounding of S in `check`. tau = (n·colsum(|C| + D) + 2k + 1)·2**-1074
    covers the products that underflow. As for the projections above, the
    bound assumes that no partial sum overflows.

    D adds first-order error bounds (Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., Theorems 9.3 and 10.3, §14.2 and
    §14.3), with B = I + |Wpsi|·|components_|^T:

    - each side forms A within γ_{n+3}·B, which moves the inverse by up
      to γ_{n+3}·G, G = |C|·B·|C|: γ_{n+1} for the product and the
      identity, and 2u where the entry reads one triangle of A, whose
      products round apart from the other's;
    - the twin's LU path leaves a left residual |X·A - I| of at most
      γ_{3k+2}·|X|·|P^T·L|·|U| (getrf, then getri: U's inverse, then
      the solve by L), so its inverse X is within γ_{3k+2}·Fg,
      Fg = |C|·|P^T·L|·|U|·|C|, with the factors of the entry's own
      elimination of A (`_pivoted_lu`);
    - a Cholesky path, the twin's (potrf, potri) or the entry's, is within
      γ_{k+2}·Fc, Fc = |Z^T|·|L^T|·|C| + |C|·|L|·|Z| + |C|·|L|·|L^T|·|C|
      + |Z^T|·|Z|, with L the Cholesky factor and Z its inverse.

    So D0 = max(γ_{3k+2}·Fg, γ_{k+2}·Fc) + γ_{k+2}·Fc + 2γ_{n+3}·G. Each
    term reads magnitudes of the exact inverse, of the twin's `cov_z` and
    of the twin's factors, where D0 reads C and the entry's factors. In the
    norm max(|.|_1, |.|_inf), theta bounds what that costs relative to an
    error E of these magnitudes: the terms gain at most theta·|E|, and
    |E| <= 2·|D0| where theta <= 1/2, so D = D0 + 2·theta·|D0| at each
    entry. The step serves only where theta <= THETA_MAX = 2**-10, and
    stays Python elsewhere: features of almost no noise meet sklearn's
    floor of noise variance, 1e-12, and then A can be so ill-conditioned
    that no inverse in doubles has a correct digit. Over the 80 fits, 11
    have a theta past 2**-10, each an A of a condition number past 1e14.

    The bound assumes that the twin's LAPACK picks the pivots of the
    entry's elimination: the step stays Python where a pivot is less than
    MARGIN_MIN = 1 + 2**-8 times the next largest candidate of its column.
    On each of the 8 setups, getrf picked the entry's pivots for each of
    the 68 fits that it factors (the other 12 are exactly symmetric).

    Over the 69 fits that the entry serves on the 8 setups, each twin is
    within 0.129 of the bound, in calls of a batch and of one row, and the
    twins of two setups differ by up to 0.084 of it. In T58's parity run
    (44 seeds of 10 fits, 1 to 128 features and components; seeds 32 to
    43 without numpy's AVX-512 code or on OpenBLAS's Haswell and
    Sandybridge kernels), the entry served 415 fits and stayed Python for
    25 (24 past THETA_MAX, one within MARGIN_MIN). Of 178.9 million fields
    of 8 kinds of rows, 130.1 million differ from the twin's and none
    passes the bound, in calls of 5,000 rows and of one row: the largest
    difference is 0.148 of the bound. The 369,221 fields where one side
    is not finite and the two differ are in rows where a partial sum can
    overflow.

    Each field gives a zero the twin's sign in a call of one row. Where
    every product is a zero, the twin's sums of more than one product
    answer 0.0, and the entry adds 0.0 to its sums. At one feature and one
    output field, the twin's two `np.dot`s of a one-row call multiply 1×1
    arrays and answer the products themselves, as the entry does (finding
    81). In a call of more rows, they add each product to 0.0: there the
    twin answers 0.0 where the entry answers -0.0, which the bound
    allows."""
    name = type(est).__name__
    w = np.asarray(est.components_)
    psi = np.asarray(est.noise_variance_)
    m = np.asarray(est.mean_)
    k, n = w.shape
    if k == 0:
        raise NotNative(f"{name}: no output fields")
    if not (
        np.isfinite(w).all()
        and np.isfinite(m).all()
        and (psi > 0).all()
        and np.isfinite(psi).all()
    ):
        raise NotNative(
            f"{name}: a fitted mean, component or noise variance not all"
            " finite, or a noise variance not positive"
        )
    _refuse_past_cap(name, n, k, _factor_build_estimate)
    f = _factor(est)
    d = [xi - f64(mi) for xi, mi in zip(x, m, strict=True)]
    t = [_tree([di * f64(vi) for di, vi in zip(d, vj, strict=True)]) for vj in f.v]
    out = [
        _tree([ti * f64(ci) for ti, ci in zip(t, f.c[:, j], strict=True)])
        for j in range(k)
    ]
    if n == k == 1:
        return out
    return [o + f64(0.0) for o in out]
