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
within γ_n·sum_i |m_i·c_ki| on any host. With the subtraction and the
division, the two sides are within (n + 2)·eps·S_k to first order; n + 3
also covers the second order and the rounding of S in `check`. `tau_k =
(2n/s_k + 1)·2**-1074` covers the products and the quotient that underflow.
The bound assumes that no partial sum overflows: one order can overflow
where another does not only past |x_i·c_ki| of about DBL_MAX/n. A step
within the bound serves only with `to_native(step, allow_bound=True)`.

The translation computes the projected mean with numpy, in the twin's own
spelling, so that on one host both sides share it. A fitted state in
float32 projects the mean in float32, which the bound does not cover: the
step stays Python. A whitened component whose scale sklearn clipped to eps,
or whose variance is at most eps times the largest, is rounding noise of
size O(1) for the twin too: on rank-deficient data the twin flips labels
against itself about as often as an entry does (the ruling, item 7). The
step stays Python there."""

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
    fitted = [c] + ([m] if m is not None else [])
    if any(a.dtype != np.float64 for a in fitted):
        raise NotNative(f"{name}: a float32 fit projects its mean in float32")
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
    for k, ck in enumerate(c):
        e = _tree([xi * f64(ci) for xi, ci in zip(x, ck, strict=True)])
        if mc is not None:
            e = e - f64(mc[k])
        if whiten:
            e = e / f64(s[k])
        out.append(e)
    return out
