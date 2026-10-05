"""Polynomial features: the products of the features up to a degree, in
sklearn's own generation order (sklearn 1.9, the dense branch of
`PolynomialFeatures.transform` in `sklearn/preprocessing/_polynomial.py`)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import PolynomialFeatures

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, translates


@translates(PolynomialFeatures)
def _polynomial(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn fills its output column by column: a bias of 1, the features,
    # then for each degree d >= 2 and each feature f, the degree d-1 terms
    # from f's block on (with interaction_only, from the block after f's),
    # each times x_f. So every term is its predecessor times one feature,
    # and following the same loop multiplies in the same nesting. With a
    # minimum degree above 1, the low-degree columns (all but the bias) are
    # cut afterwards.
    n = len(x)
    out: list[S.Expr] = [f64(1.0)] if est.include_bias else []
    if est._max_degree == 0:
        return out
    current = len(out)
    out.extend(x)
    index = [*range(current, current + n), current + n]
    current += n
    for _ in range(2, est._max_degree + 1):
        new_index = []
        end = index[-1]
        for f in range(n):
            start = index[f]
            new_index.append(current)
            if est.interaction_only:
                start += index[f + 1] - index[f]
            if end - start <= 0:
                break
            out.extend([out[c] * x[f] for c in range(start, end)])
            current += end - start
        new_index.append(current)
        index = new_index
    if len(out) != est._n_out_full:
        raise NotNative(
            f"PolynomialFeatures: {len(out)} terms where sklearn counts"
            f" {est._n_out_full}"
        )
    if est._min_degree > 1:
        kept = est.n_output_features_
        if est.include_bias:
            out = [out[0], *out[len(out) - kept + 1 :]]
        else:
            out = out[len(out) - kept :]
    return out
