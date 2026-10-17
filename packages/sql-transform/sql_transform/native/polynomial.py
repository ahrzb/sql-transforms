"""Polynomial features: the products of the features up to a degree, in
sklearn's own generation order (sklearn 1.9, the dense branch of
`PolynomialFeatures.transform` in `sklearn/preprocessing/_polynomial.py`).

The structural limit is 13,041 generated terms, before cutting low degrees.
Whole warm `to_native`, one instance, degree 2, a fixed-size list return,
release confit on Windows x64, master 40f6ab17, 2026-10-17: 144 features
(10,585 terms) took a median 5.91 s; 160 (13,041) 6.76 s; 176 (15,753)
8.03 s (three builds each). The limit keeps the measured shape near the
7 s budget and avoids allocating millions of terms before a refusal.
One separate fresh-process run was much faster (160/176 features at
1.35/1.21 s); this is a conservative structural limit, not a time estimate.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from confit import sql as S
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, translates

MAX_TERMS = 13_041


def refuse_large_polynomial(est: Any) -> None:
    """Refuse before output metadata, including a composition's, is allocated."""
    from sql_transform.native.compose import _parts

    pending, seen = [est], set()
    while pending:
        part = pending.pop()
        if id(part) in seen:
            continue
        seen.add(id(part))
        if type(part) is PolynomialFeatures and part._n_out_full > MAX_TERMS:
            raise NotNative(
                f"PolynomialFeatures(degree={part.degree!r}): {part._n_out_full}"
                f" generated terms, past the structural limit of {MAX_TERMS}"
            )
        if isinstance(part, Pipeline):
            pending.extend(child for _, _, child in part._iter())
        elif isinstance(part, (ColumnTransformer, FeatureUnion)):
            pending.extend(child for _, child, _ in _parts(part))


@translates(PolynomialFeatures)
def _polynomial(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    refuse_large_polynomial(est)
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
