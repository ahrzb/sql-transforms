"""Discretizers: each feature's value becomes the index of its fitted bin,
as a number or a one-hot block (sklearn 1.9,
`sklearn/preprocessing/_discretization.py`).

`KBinsDiscretizer.transform` validates X as float64 (NaN and infinity
raise), then per feature `np.searchsorted(bin_edges_[j][1:-1], x_j,
side="right")`: the number of inner edges at or below `x_j`, stored as a
float64. With `encode="onehot-dense"` that bin goes through the fitted
`_encoder`, a `OneHotEncoder` over `np.arange(n_bins_[j])`, so lane `k` of
feature `j` is 1.0 when `x_j` falls in bin `k`, else 0.0. The strategy
(`uniform`, `quantile` and its `quantile_method`, `kmeans`) and `n_bins`
only shape `bin_edges_`, so the entry reads the edges and nothing else.

Comparisons and constants only, so the entry is bit-exact: `x < e` is
IEEE's on both sides, and -0.0 equals 0.0 in numpy as in SQL.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import KBinsDiscretizer

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, translates


def _inner(est: Any, j: int) -> list[float]:
    """Feature `j`'s inner edges, `bin_edges_[j][1:-1]`, as doubles (a
    float32 fit's edges widen exactly, as searchsorted's float64 compare
    widens them). searchsorted assumes them sorted; a run of equal edges
    (`uniform` over a range a few doubles wide) is sorted, and counts as
    numpy counts it."""
    edges = np.asarray(est.bin_edges_[j], dtype=np.float64)[1:-1]
    if np.isnan(edges).any() or (np.diff(edges) < 0).any():
        raise NotNative(
            f"KBinsDiscretizer: feature {j}'s bin edges are not sorted numbers"
        )
    return [float(e) for e in edges]


def _bin(x: S.Expr, edges: list[float]) -> S.Expr:
    """The ordinal lane, `searchsorted(edges, x, side="right")` over sorted
    edges: the first `k` with `x < edges[k]`, else `len(edges)`. A
    repeated edge's arm is never reached, as numpy never lands between
    equal edges."""
    if not edges:
        return f64(0.0)  # a constant feature: edges [-inf, inf], one bin
    e = S.case(x < f64(edges[0]), f64(0.0))
    for k, v in enumerate(edges[1:], 1):
        e = e.when(x < f64(v), f64(float(k)))
    return e.otherwise(f64(float(len(edges))))


def _onehot(x: S.Expr, edges: list[float]) -> list[S.Expr]:
    """The one-hot block: lane `k` is 1.0 when `edges[k-1] <= x <
    edges[k]` (no lower test at `k = 0`, no upper at the last), else 0.0.
    Between two equal edges the bin is empty, and so is its lane."""
    m = len(edges)
    if not m:
        return [f64(1.0)]
    lanes = [S.case(x < f64(edges[0]), f64(1.0)).otherwise(f64(0.0))]
    for k in range(1, m):
        lanes.append(
            S.case(x < f64(edges[k - 1]), f64(0.0))
            .when(x < f64(edges[k]), f64(1.0))
            .otherwise(f64(0.0))
        )
    lanes.append(S.case(x < f64(edges[m - 1]), f64(0.0)).otherwise(f64(1.0)))
    return lanes


@translates(KBinsDiscretizer)
def _kbins(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if est.encode not in ("ordinal", "onehot-dense"):
        raise NotNative(
            f"KBinsDiscretizer(encode={est.encode!r}): the output is sparse"
        )
    if est.dtype is not None and np.dtype(est.dtype) != np.float64:
        # The twin rounds x to float32 before it bins it.
        raise NotNative(f"KBinsDiscretizer(dtype={np.dtype(est.dtype).name})")
    out: list[S.Expr] = []
    for j, xj in enumerate(x):
        edges = _inner(est, j)
        if est.encode == "ordinal":
            out.append(_bin(xj, edges))
        else:
            if int(est.n_bins_[j]) != len(edges) + 1:
                raise NotNative(
                    f"KBinsDiscretizer: feature {j} has {est.n_bins_[j]} bins"
                    f" over {len(edges) + 2} edges"
                )
            out += _onehot(xj, edges)
    return out
