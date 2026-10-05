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
hands `transform`: a string or None, or a float or NaN (`_classes`).
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
    the floats the step hands, and NaN (NULL or NaN). A category of the
    other kind is never met (sklearn matches by equality, and "1" is not
    1.0), and missing is probed whether it was fitted or not."""
    if t == pa.string():
        return [*(str(c) for c in cats if isinstance(c, str)), None]
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
    for j, t in enumerate(types):
        if t == pa.bool_():
            raise NotNative(
                f"{type(est).__name__}: feature {j} is boolean, which the"
                " catalog's fixtures do not make yet"
            )
    classes = [_classes(est.categories_[j], t) for j, t in enumerate(types)]
    # The probe row: every feature at its first class, a fitted category
    # where it has one, so a probe raises only for the feature it sets.
    base = [c[0] for c in classes]
    widths = _widths(est, len(x))
    out: list[S.Expr] = []
    offset = 0
    for j, (xj, t) in enumerate(zip(x, types, strict=True)):
        string = t == pa.string()
        rows = [(c, _probe(est, base, j, c)) for c in classes[j]]
        unknown = _probe(est, base, j, _unknown(classes[j], string))
        for lane in range(offset, offset + widths[j]):
            out.append(_lane(xj, string, rows, unknown, lane))
        offset += widths[j]
    if offset != len(_probe(est, base, 0, base[0]) or []):
        raise NotNative(f"{type(est).__name__}: blocks of {offset} lanes in all")
    return out


def _probe(est: Any, base: list[Any], j: int, v: Any) -> list[float] | None:
    """The twin's whole output row for `base` with feature `j` set to `v`,
    or None where the twin raises."""
    row = list(base)
    row[j] = v
    try:
        with warnings.catch_warnings():  # an unknown category, by design
            warnings.simplefilter("ignore")
            got = np.asarray(est.transform([row]), dtype=np.float64)
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
