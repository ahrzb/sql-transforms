"""Imputers: a missing value becomes its feature's fitted statistic, and
missing-value indicators, in sklearn's own operation order (sklearn 1.9,
`sklearn/impute/_base.py`)."""

from __future__ import annotations

from typing import Any

import numpy as np
from confit import sql as S
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.utils._mask import _get_mask
from sklearn.utils._missing import is_scalar_nan

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates


def _missing(est: Any, x: list[S.Expr]) -> list[S.Expr]:
    """sklearn's `_get_mask(X, missing_values)` per feature: `isnan` for a
    NaN marker, `x == marker` for a number (NaN is never equal to one, and
    the twin's validation rejects it then anyway)."""
    mv = est.missing_values
    if is_scalar_nan(mv):
        return [isnan(xi) for xi in x]
    if isinstance(mv, bool) or not isinstance(mv, int | float | np.number):
        raise NotNative(f"{type(est).__name__}(missing_values={mv!r})")
    return [xi == f64(mv) for xi in x]


def _indicate(miss: list[S.Expr], features: Any) -> list[S.Expr]:
    # The mask's columns as 1.0 / 0.0: a boolean array, which the step reads
    # as floats.
    return [S.case(miss[int(f)], f64(1.0)).otherwise(f64(0.0)) for f in features]


@translates(SimpleImputer)
def _simple(est: Any, x: list[S.Expr]) -> list[S.Expr]:
    # sklearn: the mask is computed first; features whose statistic is NaN
    # (no observed value at fit) are dropped unless keep_empty_features;
    # each missing value of a kept feature becomes its statistic, cast to
    # the fit dtype; then the indicator's lanes over the same mask.
    if est._fit_dtype.kind not in "fiu":
        raise NotNative(f"SimpleImputer over {est._fit_dtype} data")
    miss = _missing(est, x)
    stats = est.statistics_
    if est.keep_empty_features:
        keep = np.arange(len(x))
        valid = stats.astype(est._fill_dtype, copy=False)
    else:
        mask = np.logical_not(_get_mask(stats, np.nan))
        keep = np.flatnonzero(mask)
        valid = stats[mask].astype(est._fill_dtype, copy=False)
    out = [
        S.case(miss[i], f64(v)).otherwise(x[i])
        for i, v in zip(keep, valid, strict=True)
    ]
    if est.add_indicator:
        out += _indicate(miss, est.indicator_.features_)
    return out


@translates(MissingIndicator)
def _indicator(est: Any, x: list[S.Expr]) -> list[S.Expr]:
    # sklearn: the mask's columns that had a missing value at fit
    # (features="missing-only"; with error_on_new a new one raises in the
    # twin), or all of them, as booleans.
    if est.sparse is True:
        raise NotNative("MissingIndicator(sparse=True): the output is sparse")
    if est._precomputed:
        raise NotNative("a MissingIndicator fitted on a mask transforms masks")
    return _indicate(_missing(est, x), est.features_)
