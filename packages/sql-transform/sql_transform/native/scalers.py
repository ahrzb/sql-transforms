"""Scalers: one affine map per feature, or one row norm, in sklearn's own
operation order (sklearn 1.9, `sklearn/preprocessing/_data.py`)."""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

from sql_transform.native._helpers import (
    TINY_SCALE,
    clip,
    f64,
    row_max,
    row_sum,
    row_sumsq,
    row_sumsq_is_numpys,
)
from sql_transform.native._registry import NotNative, translates


@translates(StandardScaler)
def _standard(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
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


@translates(MinMaxScaler)
def _minmax(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: `X *= scale_`, `X += min_`, then with `clip` the array-API
    # clip to `feature_range` (`_helpers.clip`). `scale_` already has zero
    # ranges replaced by 1.0.
    lo, hi = est.feature_range
    if est.clip and (math.isnan(lo) or math.isnan(hi)):
        raise NotNative(f"MinMaxScaler(feature_range={est.feature_range!r}, clip=True)")
    out = []
    for i, xi in enumerate(x):
        e = xi * f64(est.scale_[i]) + f64(est.min_[i])
        out.append(clip(e, lo, hi) if est.clip else e)
    return out


@translates(MaxAbsScaler)
def _maxabs(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: `X /= scale_`, then with `clip` the array-API clip to [-1, 1].
    # `scale_` already has zero maxima replaced by 1.0.
    out = []
    for i, xi in enumerate(x):
        e = xi / f64(est.scale_[i])
        out.append(clip(e, -1.0, 1.0) if getattr(est, "clip", False) else e)
    return out


@translates(RobustScaler)
def _robust(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: `X -= center_` if with_centering, then `X /= scale_` if
    # with_scaling. `quantile_range` and `unit_variance` only shape the
    # fitted `scale_`.
    out = []
    for i, xi in enumerate(x):
        e = xi
        if est.with_centering:
            e = e - f64(est.center_[i])
        if est.with_scaling:
            e = e / f64(est.scale_[i])
        out.append(e)
    return out


# The widest max norm served. confit computes the norm every lane repeats
# once per row (#363), but its `greatest` builds in time that grows faster
# than the square of its arity: one instance builds in 0.20 s at 16
# features, 1.9 s at 32, 6.4 s at 48 and 16.6 s at 64, and serves a row
# in 1 to 18 us against the twin's 230 to 320 (release build, master
# f0fa925, 2026-10-05; PLANS, "Needs from confit").
_MAX_NORM_WIDTH = 48


@translates(Normalizer)
def _normalizer(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn `normalize`: the row's norm (l1: `np.sum(abs(X), axis=1)`,
    # numpy's pairwise sum; l2: `sqrt(row_norms(X, squared=True))`, an
    # einsum; max: `np.max(abs(X), axis=1)`, exact), a norm under
    # 10 * eps read as 1.0, then `X /= norm`. NaN and infinity raise in the
    # twin, so the row is finite wherever the twin answers. Every lane
    # repeats the norm, which confit computes once per row (#363): l1 and
    # l2 build in 1.8 s and 5.1 s at 128 features.
    if est.norm == "l1":
        norm = row_sum([S.fn("abs", xi) for xi in x])
    elif est.norm == "l2":
        if not row_sumsq_is_numpys():
            raise NotNative(
                "Normalizer(norm='l2'): this platform's row_norms does not"
                " accumulate as numpy's x86-64 kernel, which the entry follows"
            )
        norm = S.fn("sqrt", row_sumsq(x))
    elif est.norm == "max":
        if len(x) > _MAX_NORM_WIDTH:
            raise NotNative(
                f"Normalizer(norm='max') over {len(x)} features: every lane"
                f" repeats a row maximum whose build grows faster than the square"
                f" of the row, past {_MAX_NORM_WIDTH} (PLANS, 'A greatest that"
                " builds linearly')"
            )
        norm = row_max([S.fn("abs", xi) for xi in x])
    else:
        raise NotNative(f"Normalizer(norm={est.norm!r})")
    norm = S.case(norm < f64(TINY_SCALE), f64(1.0)).otherwise(norm)
    return [xi / norm for xi in x]


@translates(Binarizer)
def _binarizer(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn `binarize`: 1 where `X > threshold`, else 0, in X's float64.
    # NaN raises in the twin.
    t = f64(est.threshold)
    return [S.case(xi > t, f64(1.0)).otherwise(f64(0.0)) for xi in x]
