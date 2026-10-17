"""Distances to fitted centres (sklearn 1.9): `KMeans`, `MiniBatchKMeans`
and `BisectingKMeans` (`_BaseKMeans.transform`), and `Birch` (`transform`),
each `euclidean_distances(X, centres)`, one output field per centre
(`cluster_centers_`; Birch's `subcluster_centers_`).

The twin (`sklearn/metrics/pairwise.py`, `_euclidean_distances`, float64)
answers, for centre c_k,

    d_k = sqrt(max(((-2·(x @ c_k)) + ‖x‖²) + ‖c_k‖², 0)),

with ‖x‖² and ‖c_k‖² from `row_norms` (`np.einsum`) on each call, and the
products summed by BLAS (`gemv`), which uses fused multiply-adds in an
order that SQL cannot spell (project.py). A centre fitted in float32 is
widened to float64 first, as the row is a double. The entry spells the same
expression: ‖x‖² and the products each as a balanced tree, and ‖c_k‖² as
numpy computes it, a constant. So on this host ‖x‖² and the sum of the
products differ from the twin's; on another, ‖c_k‖² may differ too.
numpy's own order for ‖x‖² (`_helpers.row_sumsq`, which `Normalizer`
follows) is a chain n/2 additions deep, which confit does not build at
1,024 features; a balanced tree is log2(n) deep.

The distance itself has no small bound: near a centre the twin's square
cancels to its rounding, whose square root is about sqrt(eps·S) apart
from the entry's. So the bound compares squares
(decisions/closed/matvec-parity-bound.md, the ruling, item 3): for output
field k,

    |entry_k² - twin_k²| <= (n + 5)·eps·S_k + tau,
    S_k = ‖x‖² + ‖c_k‖² + 2·sum_i |x_i·c_ki|,

with n the number of features. On each side, the sum of the n products is
within γ_n·sum_i |x_i·c_ki| of the exact one, in any order, fused or not,
and ‖x‖² and ‖c_k‖² within γ_n of theirs (Higham, *Accuracy and Stability
of Numerical Algorithms*, §3.1; γ_n = n·u/(1 - n·u), u = eps/2). The two
additions round by u·S_k each, and taking the maximum with 0 moves no
value apart. So the two squares before the root are within (n + 2)·eps·S_k
to first order. The root and the square that `check` takes of it add
eps·S_k on each side, and K = n + 5 also covers the rounding of S and of
the squares in `check`. Where a product or a square underflows, it errs by
at most 2**-1075 more, and the sums of such values are exact: `tau =
4n·2**-1074` covers the products, ‖x‖² and ‖c_k‖² on both sides. The bound
assumes that no partial sum overflows, as project.py's does. A step within
it serves only with `to_native(step, allow_bound=True)`; inside a
composition it stays Python, since `check` compares a composition's fields
without the map (compose.py).

Measured over 200 seeds of each class at 1, 2, 3, 5, 8, 17, 32 and 64
features (correlated features of scales 1e-2 to 1e4 and means up to 100
times as large, in 1 to 6 blobs, a random configuration of each class;
rows at a centre, near one, drawn like the fit and far from it; master
c06a15a, 2026-10-16): 6,386 steps, 13.6 million output fields, 53% of them
apart from the twin. The largest |entry_k² - twin_k²| / (eps·S_k) is 7.39,
for `Birch` at 64 features, against K = 69; the least room is at 2
features, 2.79 for `MiniBatchKMeans` against K = 7. At one feature each
field equals the twin's. In 42,622 fields, at a centre or near one, one
side answered 0 and the other did not, within the bound. The measured
value is evidence for K, not K.

A call of 64 rows takes 42 µs at 2 features of 3 centres, against 28 ms
for the twin, 537 µs at 64 of 16 (36 ms), 2.9 ms at 256 of 64 (65 ms) and
9.2 ms at 1,024 of 16 (126 ms): one thread, the median of 30 warm calls,
release build. A step builds by `_build_estimate`, which the cap reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.cluster import Birch, BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.extmath import row_norms

from sql_transform.native._helpers import f64
from sql_transform.native._registry import ErrorScale, NotNative, translates
from sql_transform.native.project import _TINY, MAX_BUILD_S, _tree


def _centres(est: Any) -> np.ndarray:
    """The fitted centres as the twin reads them: float64, one per row."""
    c = est.subcluster_centers_ if isinstance(est, Birch) else est.cluster_centers_
    return np.asarray(c, dtype=np.float64)


def _s(est: Any, x: np.ndarray) -> np.ndarray:
    c = np.asarray(_centres(est), dtype=np.longdouble)
    return np.sum(x * x) + np.sum(c * c, axis=1) + 2 * (np.abs(c) @ np.abs(x))


def _square(v: Any) -> Any:
    return v * v


DISTANCE = ErrorScale(
    k=lambda est: est.n_features_in_ + 5,
    s=_s,
    tau=lambda est: 4.0 * est.n_features_in_ * _TINY,
    g=_square,
    on="on the squares",
)


def _squared_products(
    x: np.ndarray, c: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The products and the final three-term sum of a squared distance."""
    yield x, x
    yield c, c
    yield x, c
    x = np.asarray(x, np.float64)
    xx = row_norms(x.reshape(1, -1), squared=True)[0]
    cc = row_norms(c, squared=True)
    terms = np.column_stack((-2.0 * (c @ x), np.full(len(c), xx), cc))
    yield terms, np.ones(3)


def _build_estimate(n: int, k: int) -> float:
    """The seconds of the whole warm `to_native` of the entry, release
    build: the translation, the probe (`_registry._probed`) and confit's
    trial build (`_registry._builds`). It grows with the n features and k
    centres: the products (n·k), terms in n², n²·k and n·k² that the build
    grows by at wide rows, where a list or a scalar return builds more
    slowly than a struct, and terms in k and k³ that a struct return takes,
    as `query` reads each of its fields on its own (PLANS.md, "Needs from
    confit"). Fitted on the slower of the two returns over 66 shapes (1 to
    4,096 features, 1 to 2,000 centres; builds of 0.1 to 75 s; `KMeans`, and
    `Birch` of one subcluster a row; master c06a15a, 2026-10-16): each
    within 0.85 to 1.27 times the estimate. The cap took on none slower than
    7.7 s and refused none faster than 6.6 s. It takes about 920 centres
    over 2 features, 600 over 64, 230 over 256, 100 over 512, 36 over 1,024
    or 7 over 2,048, and one centre up to 2,830 features. A struct of 2,000
    centres over 2 features built in 73 s, against 1.1 s as a list, and
    confit refused one of 4,000 after 475 s, as past its code generator's
    size limit. The estimate of a centered projection
    (`project._centered_build_estimate`), whose products also read shared
    values, takes about 6,500 centres over 2 features, and one centre over
    4,096 features (14.8 s)."""
    return (
        1.14e-3 * n
        + 8.11e-4 * k
        + 7.47e-5 * n * k
        + 3.76e-7 * n * n
        + 6.64e-8 * n * n * k
        + 7.68e-8 * n * k * k
        + 7.75e-9 * k**3
    )


@translates(
    KMeans,
    MiniBatchKMeans,
    BisectingKMeans,
    Birch,
    scale=DISTANCE,
    sum_products=lambda est, x: _squared_products(x, _centres(est)),
)
def _distances(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    name = type(est).__name__
    c = _centres(est)
    k, n = c.shape
    if k == 0:
        raise NotNative(f"{name}: no centres")
    if not np.isfinite(c).all():
        raise NotNative(f"{name}: a fitted centre not all finite")
    estimate = _build_estimate(n, k)
    if estimate > MAX_BUILD_S:
        raise NotNative(
            f"{name}: an estimated {estimate:.0f} s build, past {MAX_BUILD_S:.0f} s"
            f" ({k:,} centre{'s' * (k != 1)} over {n:,} features)"
        )
    # The twin's own spelling of each centre's ‖c_k‖²: its centres through
    # `check_pairwise_arrays`, then `row_norms`.
    _, y = check_pairwise_arrays(np.zeros((1, n)), c)
    cc = row_norms(y, squared=True)
    xx = _tree([xi * xi for xi in x])
    out = []
    for ck, cck in zip(c, cc, strict=True):
        dot = _tree([xi * f64(ci) for xi, ci in zip(x, ck, strict=True)])
        q = ((f64(-2.0) * dot) + xx) + f64(cck)
        out.append(S.fn("sqrt", S.fn("greatest", q, f64(0.0))))
    return out
