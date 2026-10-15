"""Linear projections: the row times a fitted matrix (sklearn 1.9).

`PCA` and `IncrementalPCA` (`_BasePCA._transform`): `X @ components_.T`,
minus the projected mean `mean_ @ components_.T`, then under `whiten`
divided by `sqrt(explained_variance_)`, which sklearn clips below at eps.
`TruncatedSVD` and `GaussianRandomProjection`: `X @ components_.T`.

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

A step builds in about the time of its n·k products (`_build_estimate`):
64 features of 64 components in 0.25 to 0.37 s, 256 of 64 in 1.1 to 1.5 s,
256 of 256 in 5.0 to 6.8 s. The entry refuses a step estimated past
MAX_BUILD_S, before confit is asked: about 280 components over 256
features, 140 over 512, 63 over 1,024 or 6 over 4,096. Serving 64 rows
takes 43 to 138 us up to 16 features of 4 components, against the twin's
6.9 to 8.8 ms; 0.26 to 0.38 ms at 32 of 16 (twin 10.6 to 13.9 ms); 0.9 to
1.0 ms at 64 of 64 (twin 16.9 to 21.8 ms); 3.7 to 4.3 ms at 256 of 64
(twin 29.7 to 34.0 ms); 12.7 to 13.6 ms at 256 of 256 (twin 47 to 53 ms);
and 17.7 to 19.0 ms at 512 of 128 (twin 56 to 62 ms). Release build,
master bea1372 with confit #459 and #461, one container, 2026-10-15.
Before confit #459, the entry served 256 of 64 in 77 to 83 ms, twice the
twin's time: confit read a value that the body builds once, such as a
feature's `coalesce(x, NaN)`, in about 60 ns a read."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection

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
    """The seconds confit takes to build the entry (warm, release build),
    from its n features and k components. The build is about linear in the
    products n·k, with terms in n²·k and n·k² past that and a term in the
    features. Fitted on the slower of the two returns, a struct and a list,
    over 93 builds of 0.3 to 22 s (64 to 4,096 features, 1 to 256
    components; `PCA` with and without `whiten`, and the Gaussian random
    projection; master bea1372 with confit #459 and #461, 2026-10-15): each
    within 0.86 to 1.41 times the estimate, and 0.86 to 1.31 over 1 s. The
    cap took on none slower than 6.9 s and refused none faster than 6.5 s.
    Before confit #459 and #461, the same builds took 1.3 to 9 times as
    long, a median of 3.2 (master bf42757)."""
    p = n * k
    return 5.86e-4 * n + 6.54e-5 * p + 2.72e-8 * p * n + 8.63e-8 * p * k


def _tree(terms: list[S.Expr]) -> S.Expr:
    """The sum of `terms` as a balanced tree: its depth is log2 of their
    number, and its error bound γ_{⌈log2 n⌉+1}, half the provable one of a
    left-to-right sum (matvec.md, the verifier's item 5)."""
    if len(terms) == 1:
        return terms[0]
    h = len(terms) // 2
    return _tree(terms[:h]) + _tree(terms[h:])


@translates(PCA, IncrementalPCA, TruncatedSVD, GaussianRandomProjection, scale=MATVEC)
def _project(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    name = type(est).__name__
    c = np.asarray(est.components_)
    m = _mean(est)
    if m is not None and np.result_type(m, c) != np.float64:
        raise NotNative(f"{name}: a float32 fit projects its mean in float32")
    k, n = c.shape
    estimate = _build_estimate(n, k)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"{name}: an estimated {estimate:.0f} s build, past {MAX_BUILD_S:.0f} s"
            f" ({k:,} component{'s' * (k != 1)} over {n:,} features)"
        )
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
