"""Encoders: each feature's value becomes the output block of its fitted
category (sklearn 1.9, `sklearn/preprocessing/_encoders.py` and
`_target_encoder.py`).

An encoder's `transform` treats features independently: a row's output is
one block per feature, and a block depends only on that feature's value,
through the class the value falls in: one of the fitted categories,
missing, or unknown. So rather than restating sklearn's rules (infrequent
categories, `drop`, `encoded_missing_value`, `handle_unknown`, target
means), the translation asks the fitted estimator: it transforms one probe
row per class, reads the feature's block, and spells each lane as a CASE
over the classes. The values are the twin's own, so the entry is bit-exact;
a class the twin raises on (an unknown category under
`handle_unknown="error"`) is NULL, where the twin answers nothing.

The classes follow the feature's declared type, which fixes what the step
hands `transform`: a string or None, a float or NaN, or a boolean or NaN
(`_classes`).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, translates


def _widths(est: Any, n: int) -> list[int]:
    """Each feature's block width, in output order."""
    if isinstance(est, OneHotEncoder):
        return [int(w) for w in est._n_features_outs]
    if isinstance(est, TargetEncoder) and est.target_type_ == "multiclass":
        return [len(est.classes_)] * n
    return [1] * n


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _classes(cats: Any, t: pa.DataType) -> list[Any]:
    """The values `transform` can be handed for a feature of declared type
    `t` that fall in a fitted category, then missing: a string feature's
    fitted strings and None (NULL), a numeric feature's fitted numbers, as
    the floats the step hands, and NaN (NULL or NaN); a boolean feature's
    False, True and NaN, fitted or not, the fitted first. A category of the
    other kind is never met (sklearn matches by equality, and "1" is not
    1.0), and missing is probed whether it was fitted or not."""
    if t == pa.string():
        return [*(str(c) for c in cats if isinstance(c, str)), None]
    if t == pa.bool_():
        # Whatever was fitted: the step hands False, True or NaN, which
        # the twin matches against its categories by equality (False is
        # 0.0), its own way for each of their dtypes (bool, float64,
        # object). The fitted ones first, for the probe row.
        def fitted(v: Any) -> bool:
            if _missing(v):
                return any(_missing(c) for c in cats)
            return any(not _missing(c) and c == v for c in cats)

        return sorted([False, True, math.nan], key=lambda v: not fitted(v))
    numbers = (bool, int, float, np.number, np.bool_)
    present = [float(c) for c in cats if isinstance(c, numbers)]
    return [*(c for c in present if not math.isnan(c)), math.nan]


def _unknown(classes: list[Any], string: bool) -> Any:
    """A value of no fitted category: finite, since the twin's validation
    rejects an infinity."""
    i = 0
    while True:
        v: Any = f"\uffffunseen{i}" if string else 0.5 + i * 1.000001
        if not any(not _missing(c) and c == v for c in classes):
            return v
        i += 1


@translates(OneHotEncoder, OrdinalEncoder, TargetEncoder)
def _encode(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if getattr(est, "sparse_output", False):
        raise NotNative("OneHotEncoder(sparse_output=True): the output is sparse")
    classes = [_classes(est.categories_[j], t) for j, t in enumerate(types)]
    # The probe row: every feature at its first class, a fitted category
    # where it has one, so a probe raises only for the feature it sets.
    base = [c[0] for c in classes]
    widths = _widths(est, len(x))
    # Over boolean features only, the row is a boolean array when none is
    # NULL and a float64 one otherwise, and sklearn's matching is not the
    # same on both: under `handle_unknown="use_encoded_value"`, categories
    # [nan] answer True as unknown in a float64 row and raise in a boolean
    # one. So each lane is probed in both, and reads the float64 answer
    # where any feature is NaN.
    booleans = all(t == pa.bool_() for t in types)
    dtype = np.float64 if booleans else None
    if booleans:
        bits = [next((c for c in cs if not _missing(c)), False) for cs in classes]
        null = isnan(x[0])
        for xk in x[1:]:
            null = null | isnan(xk)
    out: list[S.Expr] = []
    offset = 0
    for j, (xj, t) in enumerate(zip(x, types, strict=True)):
        string = t == pa.string()
        rows = [(c, _probe(est, base, j, c, dtype)) for c in classes[j]]
        # A boolean feature's classes are all it can be handed: no value
        # is unknown, and the ELSE is never met.
        unknown = (
            None
            if t == pa.bool_()
            else _probe(est, base, j, _unknown(classes[j], string))
        )
        if booleans:
            known = [c for c in classes[j] if not _missing(c)]
            rows_b = [(c, _probe(est, bits, j, c, np.bool_)) for c in known]
        for lane in range(offset, offset + widths[j]):
            e = _lane(xj, string, rows, unknown, lane)
            if booleans:
                b = _lane(xj, string, rows_b, None, lane)
                if b.sql() != e.sql():
                    e = S.case(null, e).otherwise(b)
            out.append(e)
        offset += widths[j]
    if offset != len(_probe(est, base, 0, base[0]) or []):
        raise NotNative(f"{type(est).__name__}: blocks of {offset} lanes in all")
    return out


def _probe(
    est: Any, base: list[Any], j: int, v: Any, dtype: Any = None
) -> list[float] | None:
    """The twin's whole output row for `base` with feature `j` set to `v`,
    handed as the step's list or as an array of `dtype`, or None where the
    twin raises."""
    row = list(base)
    row[j] = v
    arg = [row] if dtype is None else np.array([row], dtype=dtype)
    try:
        with warnings.catch_warnings():  # an unknown category, by design
            warnings.simplefilter("ignore")
            got = np.asarray(est.transform(arg), dtype=np.float64)
    except ValueError:
        return None
    return [float(v) for v in got.reshape(-1)]


def _lane(
    xj: S.Expr,
    string: bool,
    rows: list[tuple[Any, list[float] | None]],
    unknown: list[float] | None,
    lane: int,
) -> S.Expr:
    """One output lane: the CASE over the feature's classes, the values
    grouped, unknown values (and classes the twin raises on) in the ELSE."""
    groups: dict[str, tuple[float, list[Any]]] = {}
    for c, got in rows:
        if got is None:
            continue
        v = got[lane]
        groups.setdefault(repr(v), (v, []))[1].append(c)
    other = None if unknown is None else unknown[lane]
    arms = []
    for _, (v, members) in groups.items():
        if other is not None and repr(v) == repr(other):
            continue  # the ELSE answers it
        arms.append((_member(xj, string, members), f64(v)))
    tail = S.lit(None, "DOUBLE") if other is None else f64(other)
    if not arms:
        return tail
    (c0, v0), *rest = arms
    e = S.case(c0, v0)
    for c, v in rest:
        e = e.when(c, v)
    return e.otherwise(tail)


def _member(xj: S.Expr, string: bool, members: list[Any]) -> S.Expr:
    """`xj` falls in one of `members`: a missing member by IS NULL or isnan,
    the rest by equality (so -0.0 is 0.0, as sklearn's matching has it)."""
    present = [c for c in members if not _missing(c)]
    conds = []
    if present:
        lits = [S.lit(c) if string else f64(c) for c in present]
        conds.append(xj == lits[0] if len(lits) == 1 else xj.isin(*lits))
    if len(present) < len(members):
        conds.append(xj.isnull() if string else isnan(xj))
    e = conds[0]
    for c in conds[1:]:
        e = e | c
    return e
