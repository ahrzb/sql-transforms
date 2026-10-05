"""Feature agglomeration: each output lane pools the features of one fitted
cluster, in sklearn's own operation order (sklearn 1.9,
`AgglomerationTransform.transform` in
`sklearn/cluster/_feature_agglomeration.py`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.cluster import FeatureAgglomeration

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, translates


def _name(func: Any) -> str:
    name = getattr(func, "__name__", None) or repr(func)
    return f"np.{name}" if getattr(np, str(name), None) is func else name


@translates(FeatureAgglomeration)
def _agglomeration(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: validate (NaN and infinity raise), then, for
    # `pooling_func == np.mean` over dense X, each row is
    # `np.bincount(labels_, X[i, :]) / np.bincount(labels_)`. numpy's
    # weighted bincount (`arr_bincount`, `compiled_base.c`) zeroes the bins
    # and adds `weights[i]` to bin `labels_[i]` for i in feature order, so
    # a lane is `((0.0 + x_a) + x_b) + ...` over its cluster's features in
    # order, divided by the cluster's size (an exact double). Checked by
    # probe too: 20,000 random rows and labelings, numpy 2.5.1, bit-equal.
    # Starting from 0.0, a cluster of -0.0 pools to 0.0. Every lane reads
    # its own features, so nothing repeats: 128 features in 128 clusters
    # build in 0.11 s and serve a row in 6.8 us against the twin's 366
    # (master f2ef184, 2026-10-05).
    #
    # Any other `pooling_func` runs `pooling_func(X[:, labels_ == l],
    # axis=1)` per label. `np.max` and `np.min` are not served: on a tie of
    # signed zeros numpy's SIMD reduction answers the zero its lane order
    # reaches, neither the first nor the last of the tied operands (e.g.
    # the max of [0.0, -1.0, 0.0, -0.0, -1.0, 0.0, -0.0, -1.0, -1.0, 0.0,
    # -0.0, -0.0, -0.0, -0.0, -1.0, 0.0, 0.0] is neither zero's pick;
    # x86-64 with AVX-512, numpy 2.5.1, 2026-10-05), while
    # `greatest`/`least` keep the first.
    func = est.pooling_func
    if not func == np.mean:
        raise NotNative(f"FeatureAgglomeration(pooling_func={_name(func)})")
    labels = np.asarray(est.labels_)
    size = np.bincount(labels)
    if not size.all():
        # Never fitted (labels are 0..n_clusters-1); the twin's lane would
        # be 0 / 0.
        raise NotNative("FeatureAgglomeration: labels_ skip a cluster")
    out = []
    for lane, count in enumerate(size):
        acc: S.Expr = f64(0.0)
        for i in np.flatnonzero(labels == lane):
            acc = acc + x[i]
        out.append(acc / f64(count))
    return out
