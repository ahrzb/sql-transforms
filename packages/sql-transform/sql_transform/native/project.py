"""Linear projections: the row times a fitted matrix (sklearn 1.9).

`PCA` and `IncrementalPCA` (`_BasePCA._transform`): `X @ components_.T`,
minus the projected mean `mean_ @ components_.T`, then under `whiten`
divided by `sqrt(explained_variance_)`, which sklearn clips below at eps.
`TruncatedSVD` and `GaussianRandomProjection`: `X @ components_.T`. The
centered projections, `FastICA`, `LinearDiscriminantAnalysis` and the PLS
family, subtract a fitted mean from the row before the product, and may
divide it by a fitted scale: `_project_centered` gives their bound.

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
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from scipy import sparse
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, TruncatedSVD
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


def _sparse_build_estimate(n: int, p: int) -> float:
    """The seconds of the whole warm `to_native` (release build) of a
    `SparseRandomProjection` of n features whose components store p entries
    in all."""
    return 6.10e-4 * n + 9.28e-5 * p


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

    For a dense row and a CSR matrix, scipy computes `components_ @ X.T`
    by `csr_matvec`: each output field from 0.0, adding the stored
    products in stored order (the fit stores the indices unsorted), each
    product and sum rounded on its own. The entry spells the same sum, so
    it is bit-exact (`_csr_sum`), where the kernel probe finds scipy's sums
    so (`csr_sum_is_scipys`). A fit in float32 stores float32 components,
    which scipy widens to float64, exactly, before the product."""
    c = est.components_
    if not sparse.issparse(c):
        return _project(est, x, types)
    name = type(est).__name__
    if c.format != "csr":
        raise NotNative(f"{name}: components_ in {c.format}, not the csr the fit makes")
    if not np.isfinite(c.data).all():
        raise NotNative(f"{name}: components_ not all finite")
    k, n = c.shape
    estimate = _sparse_build_estimate(n, c.nnz)
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
