"""Feature selectors: `transform` keeps a fitted subset of the columns,
unchanged (sklearn 1.9, `SelectorMixin.transform` in
`sklearn/feature_selection/_base.py`, which no selector overrides). So a
selector carries the bound of each column it keeps as it is (_carry.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
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

from sql_transform.native._registry import Carry, NotNative, translates


def _carry(est: Any) -> Carry | None:
    try:
        keep = np.flatnonzero(est.get_support())
    except Exception:  # noqa: BLE001 — the translation refuses it
        return None
    return lambda x, e: np.asarray(e, dtype=np.longdouble)[keep]


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
    uncapped="The fitted support mask selects at most the input width (_select).",
    carry=_carry,
)
def _select(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: validate (NaN raises unless the estimator's tags allow it),
    # then `X[:, get_support()]`. The mask is fitted state, read once here.
    try:
        keep = np.flatnonzero(est.get_support())
    except Exception as e:  # noqa: BLE001 — the twin's transform raises it too
        raise NotNative(f"{type(est).__name__}.get_support() raises: {e}") from None
    if not len(keep):
        raise NotNative(f"{type(est).__name__} keeps no feature")
    return [x[i] for i in keep]
