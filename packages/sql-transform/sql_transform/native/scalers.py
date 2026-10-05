"""Scalers: one affine map per feature, in sklearn's own operation order."""

from __future__ import annotations

from typing import Any

from confit import sql as S
from sklearn.preprocessing import StandardScaler

from sql_transform.native._helpers import f64
from sql_transform.native._registry import translates


@translates(StandardScaler)
def _standard(est: Any, x: list[S.Expr]) -> list[S.Expr]:
    # sklearn: `X -= mean_` if with_mean, then `X /= scale_` if with_std, in
    # float64. `scale_` already has zero variances replaced by 1.0.
    out = []
    for i, xi in enumerate(x):
        e = xi
        if est.with_mean:
            e = e - f64(est.mean_[i])
        if est.with_std:
            e = e / f64(est.scale_[i])
        out.append(e)
    return out
