"""Feature selectors: `transform` keeps a fitted subset of the columns,
unchanged (sklearn 1.9, `SelectorMixin.transform` in
`sklearn/feature_selection/_base.py`, which no selector overrides)."""

from __future__ import annotations

from typing import Any

import numpy as np
from confit import sql as S
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
)

from sql_transform.native._registry import NotNative, translates


@translates(
    GenericUnivariateSelect,
    RFE,
    RFECV,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
)
def _select(est: Any, x: list[S.Expr]) -> list[S.Expr]:
    # sklearn: validate (NaN raises unless the estimator's tags allow it),
    # then `X[:, get_support()]`. The mask is fitted state, read once here.
    try:
        keep = np.flatnonzero(est.get_support())
    except Exception as e:  # noqa: BLE001 — the twin's transform raises it too
        raise NotNative(f"{type(est).__name__}.get_support() raises: {e}") from None
    if not len(keep):
        raise NotNative(f"{type(est).__name__} keeps no feature")
    return [x[i] for i in keep]
