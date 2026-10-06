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
over the classes. The values are the twin's own, so the entry is bit-exact.

Where the twin raises (an unknown category under `handle_unknown="error"`,
or a missing value it did not fit), the input guard traps: one test a
feature, a value in none of the classes the twin answers (`_reject`).
Every other value a lane's ELSE then meets traps too, so the ELSE answers
the lane's largest group, and a one-hot lane is one comparison.

confit builds and runs the guard again at each field read of a struct
output (loops/native/PLANS.md, "Needs from confit"), and it spells an IN
list as one comparison a value, so the guard sets the cost of a wide
encoder. Past SEARCH_PAST strings, an encoder with one output field a
feature (an ordinal one) tests a string by one substring search instead
(`_among`), which confit builds in a size that does not grow with the
strings. A row costs more by the search than by a short IN list: with the
search past 4 strings, a one-hot encoder over 8 features served a row 15
times as slowly at 10 categories and 2.8 times at 50, and an ordinal one
over 32 features of 125 categories 1.1 times as slowly. A one-hot encoder
keeps the IN list: each of its fields runs the guard, and over 8 features
of 125 categories the search served a row in 11.9 ms, against the twin's
2.8 ms.
Warm builds of `to_native`'s trial build and us a row over 2,000-row
batches, string features unless the row says, struct returns (release
build, master e1f15b9; the twin over 1,000 rows):

    encoder, features x categories   fields   build              serve       twin
    OneHotEncoder, 8 x 10                80   0.21 s             18 us   1,028 us
    OneHotEncoder, 8 x 50               400   5.3 s             920 us   1,695 us
    OneHotEncoder, 8 x 50 numbers       400   2.3 s             180 us   1,836 us
    OneHotEncoder, 8 x 125            1,000   (refused after 8.2 s)
    OrdinalEncoder, 8 x 125               8   0.15 s             89 us   1,613 us
    OrdinalEncoder, 32 x 125             32   1.2 s           1,127 us   5,447 us
    OrdinalEncoder, 32 x 500             32   3.3 s           3,208 us  14,458 us
    OrdinalEncoder, 32 x 125 numbers     32   16.5 s            580 us   4,272 us

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
from sql_transform.native._registry import NotNative, rejects, translates

# Past this many strings, the input guard of an encoder with one output
# field a feature tests a string by one substring search rather than an
# equality with each (`_among`, module docstring).
SEARCH_PAST = 100
# Separators for that search: the first that no string holds.
_SEPARATORS = ("\x1f", "\x1e", "\x1d", "\x1c")


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


def _first_classes(est: Any, types: list[pa.DataType]) -> list[Any]:
    """The probe row: every feature at its first class, a fitted category
    where it has one, so a probe raises only for the feature it sets
    (the input guard's too)."""
    return [_classes(est.categories_[j], t)[0] for j, t in enumerate(types)]


@translates(OneHotEncoder, OrdinalEncoder, TargetEncoder, base=_first_classes)
def _encode(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    if getattr(est, "sparse_output", False):
        raise NotNative("OneHotEncoder(sparse_output=True): the output is sparse")
    classes = [_classes(est.categories_[j], t) for j, t in enumerate(types)]
    base = _first_classes(est, types)
    widths = _widths(est, len(x))
    # One output field a feature: the guard runs at each field read, so a
    # search costs a row about one search a feature a field (module
    # docstring).
    search = all(w == 1 for w in widths)
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
        raises = t != pa.bool_() and unknown is None
        if raises and all(got is None for _, got in rows):
            raise NotNative(
                f"{type(est).__name__}: the twin raises on every value of feature {j}"
            )
        _reject(xj, string, rows, raises, null if booleans else None, search)
        if booleans:
            _reject(xj, string, rows_b, False, ~null)
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


def _reject(
    xj: S.Expr,
    string: bool,
    rows: list[tuple[Any, list[float] | None]],
    unknown: bool,
    where: S.Expr | None,
    search: bool = False,
) -> None:
    """The input guard's test for one feature (`_registry.rejects`), on
    rows where `where` holds (the form, float64 or boolean, the probes ran
    in): where the twin raises on an `unknown` value
    (`handle_unknown="error"`), a value in none of the classes it answers,
    by a `search` past SEARCH_PAST strings, else one of the classes it
    raises on (`rows`)."""
    if unknown:
        kept = [c for c, got in rows if got is not None]
        test = ~_member(xj, string, kept, search)
        if string and not any(_missing(c) for c in kept):
            test = test | xj.isnull()  # NOT on NULL is NULL, not true
    else:
        raised = [c for c, got in rows if got is None]
        if not raised:
            return
        test = _member(xj, string, raised)
    rejects(test if where is None else where & test)


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
    grouped. The ELSE answers unknown values or, where the twin raises on
    them (`unknown` is None), the largest group: every other value it then
    meets is one the twin raises on, where the input guard traps."""
    groups: dict[str, tuple[float, list[Any]]] = {}
    for c, got in rows:
        if got is None:
            continue
        v = got[lane]
        groups.setdefault(repr(v), (v, []))[1].append(c)
    if unknown is not None:
        other = unknown[lane]
    else:
        other = max(groups.values(), key=lambda g: len(g[1]), default=(None,))[0]
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


def _member(
    xj: S.Expr, string: bool, members: list[Any], search: bool = False
) -> S.Expr:
    """`xj` falls in one of `members`: a missing member by IS NULL or isnan,
    the rest by `_among`."""
    present = [c for c in members if not _missing(c)]
    conds = []
    if present:
        conds.append(_among(xj, string, present, search))
    if len(present) < len(members):
        conds.append(xj.isnull() if string else isnan(xj))
    e = conds[0]
    for c in conds[1:]:
        e = e | c
    return e


def _among(xj: S.Expr, string: bool, present: list[Any], search: bool) -> S.Expr:
    """`xj` equals one of `present`, none of them missing (on a NULL `xj`,
    NULL or false): by equality, so -0.0 is 0.0 as sklearn's matching has
    it, or, with `search` past SEARCH_PAST strings, by one substring search.
    Join the strings with a separator that none holds, and wrap them in it.
    An `xj` that holds no separator, wrapped in it, is found there exactly
    when it is one of the strings: the match starts at a separator, where a
    string starts, and its last character is the next separator, where that
    string ends. An `xj` that holds one is none of them."""
    if search and string and len(present) > SEARCH_PAST:
        sep = next((s for s in _SEPARATORS if not any(s in c for c in present)), None)
        if sep is not None:
            joined = S.lit(sep + sep.join(present) + sep)
            wrapped = S.fn("concat", S.lit(sep), xj, S.lit(sep))
            return S.fn("contains", joined, wrapped) & ~S.fn("contains", xj, S.lit(sep))
    lits = [S.lit(c) if string else f64(c) for c in present]
    return xj == lits[0] if len(lits) == 1 else xj.isin(*lits)
