"""Encoders: each feature's value becomes the output block of its fitted
category (sklearn 1.9, `sklearn/preprocessing/_encoders.py` and
`_target_encoder.py`).

An encoder's `transform` treats features independently: a row's output is
one block per feature, and a block depends only on that feature's value,
through the class the value falls in: one of the fitted categories, missing,
or unknown. So rather than restating sklearn's rules (infrequent categories,
`drop`, `encoded_missing_value`, `handle_unknown`, target means), the
translation asks the fitted estimator: it transforms one probe row per
class, reads the feature's block, and spells each output field as a CASE
over the classes. The values are the twin's own, so the entry is bit-exact.
`OneHotEncoder(sparse_output=True)`, the default, returns the matrix that
`sparse_output=False` makes dense with `.toarray()`; the step and the probes
make it dense too (`_udf._dense`), so the entry serves both.
A feature's probe rows go to `transform` in one call (`_probes`). Each call
builds every feature's table of categories again
(`sklearn/utils/_encode.py`, `_map_to_integer`), so a call a row cost about
(features x categories)^2 checks: `to_native` of `OrdinalEncoder` over 32
string features of 125 categories took 40.8 s in 4,098 calls on master
44052cf, and takes 1.3 s in 130 calls, about four a feature (first in the
process, release build, 2026-10-14). Over 32 of 500 it takes 5.1 s, of which
the trial build takes 2.5 s.

Where the twin raises (an unknown category under `handle_unknown="error"`,
or a missing value it did not fit), the input guard traps: one test a
feature, a value in none of the classes the twin answers (`_reject`). Every
other value an output field's ELSE then meets traps too, so the ELSE answers
the output field's largest group, and a one-hot output field is one
comparison.

confit spells the guard's IN list as one comparison a value. Past
SEARCH_PAST strings, an encoder with one output field a feature (an
ordinal one) tests a string by one substring search instead (`_among`),
which confit builds in a size that does not grow with the strings:
`OrdinalEncoder` over 32 string features of 125 categories builds in
0.66 s, against 3.2 s with the IN lists. A one-hot encoder keeps the
IN list. Since confit #430 its guard runs once a row, not at each field
read of a struct output. Over 8 to 32 features, the search took 1.3 to
2.4 times as long as the IN list to build, and 0.8 to 1.4 times as long
to serve a row.

Every encoder finds a number among the categories in a run of RUN_FROM
or more consecutive integers by a range: `a <= x <= b` and
`floor(x) = x`, four operations whatever the run (`_runs`). Over 32
features of the integers 0 to 124, `OrdinalEncoder` built in 4.6 s with
the IN lists and served a row in 482 us. With the ranges it builds in
0.74 s, about as fast as without a guard (0.70 to 0.80 s under
`handle_unknown="use_encoded_value"`), and serves a row in 163 us.

Warm builds of `to_native`'s trial build and us a row over 2,000-row
batches, string features unless the row says (the numbers are integers),
struct returns (release build, the dev shell's CPython 3.14.7, master
de5b226 with the ranges, 2026-10-14; the twin over 1,000 rows):

    encoder, features x categories   fields   build     serve        twin
    OneHotEncoder, 8 x 10                80   0.04 s    10 us    1,523 us
    OneHotEncoder, 8 x 50               400   0.19 s    88 us    2,552 us
    OneHotEncoder, 8 x 50 numbers       400   0.14 s    42 us    2,547 us
    OneHotEncoder, 8 x 125            1,000   0.51 s   256 us    4,338 us
    OneHotEncoder, 16 x 125           2,000   1.3 s    516 us    8,189 us
    OrdinalEncoder, 8 x 125               8   0.15 s    65 us    2,430 us
    OrdinalEncoder, 32 x 125             32   0.59 s   266 us    9,089 us
    OrdinalEncoder, 32 x 500             32   2.6 s  1,016 us   25,820 us
    OrdinalEncoder, 32 x 125 numbers     32   0.74 s   163 us    5,895 us

The classes follow the feature's declared type, which fixes what the step
hands `transform`: a string or None, a float or NaN, or a boolean or NaN
(`_classes`).
"""

from __future__ import annotations

import functools
import math
import operator
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from sql_transform._udf import _dense
from sql_transform.native._helpers import f64, isnan
from sql_transform.native._registry import NotNative, rejects, translates

# Past this many strings, the input guard of an encoder with one output
# field a feature tests a string by one substring search rather than an
# equality with each (`_among`, module docstring).
SEARCH_PAST = 100
# Separators for that search: the first that no string holds.
_SEPARATORS = ("\x1f", "\x1e", "\x1d", "\x1c")
# From this many consecutive integers on, the numbers a test finds `xj`
# among are a range (`_runs`), not an equality with each.
RUN_FROM = 4


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
    classes = [_classes(est.categories_[j], t) for j, t in enumerate(types)]
    base = _first_classes(est, types)
    widths = _widths(est, len(x))
    # One output field a feature: past SEARCH_PAST strings, the search
    # builds smaller than the IN list. A one-hot encoder builds faster
    # with the IN list (module docstring).
    search = all(w == 1 for w in widths)
    # Over boolean features only, the row is a boolean array when none is
    # NULL and a float64 one otherwise, and sklearn's matching is not the
    # same on both: under `handle_unknown="use_encoded_value"`, categories
    # [nan] answer True as unknown in a float64 row and raise in a boolean
    # one. So each output field is probed in both, and reads the float64
    # answer where any feature is NaN.
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
        rows = list(
            zip(classes[j], _probes(est, base, j, classes[j], dtype), strict=True)
        )
        # A boolean feature's classes are all it can be handed: no value
        # is unknown, and the ELSE is never met.
        unknown = (
            None
            if t == pa.bool_()
            else _probe(est, base, j, _unknown(classes[j], string))
        )
        if booleans:
            known = [c for c in classes[j] if not _missing(c)]
            rows_b = list(
                zip(known, _probes(est, bits, j, known, np.bool_), strict=True)
            )
        raises = t != pa.bool_() and unknown is None
        if raises and all(got is None for _, got in rows):
            raise NotNative(
                f"{type(est).__name__}: the twin raises on every value of feature {j}"
            )
        _reject(xj, string, rows, raises, null if booleans else None, search)
        if booleans:
            _reject(xj, string, rows_b, False, ~null)
        present = [c for c in classes[j] if not _missing(c)]
        ends = (
            (min(present), max(present))
            if present and t not in (pa.string(), pa.bool_())
            else None
        )
        for field in range(offset, offset + widths[j]):
            e = _field(xj, string, rows, unknown, field, ends)
            if booleans:
                b = _field(xj, string, rows_b, None, field)
                if b.sql() != e.sql():
                    e = S.case(null, e).otherwise(b)
            out.append(e)
        offset += widths[j]
    if offset != len(_probe(est, base, 0, base[0]) or []):
        raise NotNative(
            f"{type(est).__name__}: blocks of {offset} output fields in all"
        )
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
    return _probes(est, base, j, [v], dtype)[0]


def _probes(
    est: Any, base: list[Any], j: int, values: list[Any], dtype: Any = None
) -> list[list[float] | None]:
    """`_probe` for each of `values`, in few calls of `transform`: the
    values that are not missing in one call, and a missing one alone (the
    twin raises on it where it did not fit it). Where a call raises, each
    half of its rows goes to a call of its own, down to the rows that
    raise. The twin answers each row on its own: its block depends only on
    the row's values, and the rows hand `transform` the same types."""
    rows = []
    for v in values:
        row = list(base)
        row[j] = v
        rows.append(row)
    out: list[list[float] | None] = [None] * len(rows)

    def run(idx: list[int]) -> None:
        if not idx:
            return
        arg = [rows[i] for i in idx]
        try:
            with warnings.catch_warnings():  # an unknown category, by design
                warnings.simplefilter("ignore")
                got = est.transform(arg if dtype is None else np.array(arg, dtype))
        except ValueError:
            if len(idx) > 1:
                run(idx[: len(idx) // 2])
                run(idx[len(idx) // 2 :])
            return
        got = np.asarray(_dense(got), dtype=np.float64).reshape(len(idx), -1)
        for i, r in zip(idx, got, strict=True):
            out[i] = [float(v) for v in r]

    run([i for i, v in enumerate(values) if not _missing(v)])
    for i, v in enumerate(values):
        if _missing(v):
            run([i])
    return out


def _field(
    xj: S.Expr,
    string: bool,
    rows: list[tuple[Any, list[float] | None]],
    unknown: list[float] | None,
    field: int,
    ends: tuple[float, float] | None = None,
) -> S.Expr:
    """One output field: the CASE over the feature's classes, the values
    grouped. The ELSE answers unknown values or, where the twin raises on
    them (`unknown` is None), the largest group: every other value it then
    meets is one the twin raises on, where the input guard traps.

    Over a number feature whose classes run from `ends[0]` to `ends[1]`, a
    CASE of more than one arm first answers a number outside them as the
    ELSE does. That test reads the feature twice in one arm, so that confit
    computes it once a row, where each arm of the chain read it again
    (`_registry._apart`). At 64 features of 64 classes, a row of a
    1,024-row call took 37 µs, against 84 µs without the test, and the
    build 0.33 s, against 0.48 s (release build, master 12aca98, 2026-10-16)."""
    groups: dict[str, tuple[float, list[Any]]] = {}
    for c, got in rows:
        if got is None:
            continue
        v = got[field]
        groups.setdefault(repr(v), (v, []))[1].append(c)
    if unknown is not None:
        other = unknown[field]
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
    if ends is not None and len(arms) > 1:
        lo, hi = f64(ends[0]), f64(ends[1])
        arms = [((xj < lo) | ((xj > hi) & ~isnan(xj)), tail), *arms]
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
    string ends. An `xj` that holds one is none of them. Numbers in a run
    of RUN_FROM or more consecutive integers are found by a range
    (`_runs`), the others by equality."""
    if search and string and len(present) > SEARCH_PAST:
        sep = next((s for s in _SEPARATORS if not any(s in c for c in present)), None)
        if sep is not None:
            joined = S.lit(sep + sep.join(present) + sep)
            wrapped = S.fn("concat", S.lit(sep), xj, S.lit(sep))
            return S.fn("contains", joined, wrapped) & ~S.fn("contains", xj, S.lit(sep))
    tests = []
    if not string:
        runs, present = _runs(present)
        if runs:
            inside = [(xj >= f64(a)) & (xj <= f64(b)) for a, b in runs]
            tests.append(
                (S.fn("floor", xj) == xj) & functools.reduce(operator.or_, inside)
            )
    if present:
        lits = [S.lit(c) if string else f64(c) for c in present]
        tests.append(xj == lits[0] if len(lits) == 1 else xj.isin(*lits))
    return functools.reduce(operator.or_, tests)


def _runs(present: list[Any]) -> tuple[list[tuple[float, float]], list[Any]]:
    """The runs of RUN_FROM or more consecutive integers among the numbers
    `present`, each as its least and its greatest, and the other numbers in
    their order. Every integer within 2**53 of zero is a double, so a double
    `x` is one of a run's exactly when `a <= x <= b` and `floor(x) = x`.
    NaN fails `x <= b`, as DuckDB orders NaN above every number, ±inf fails
    the range, and -0.0 is 0.0, as in sklearn's matching."""
    ints = sorted(
        {float(v) + 0.0 for v in present if _integral(v)}  # -0.0 is 0.0
    )
    runs: list[tuple[float, float]] = []
    start = 0
    for i in range(1, len(ints) + 1):
        if i == len(ints) or ints[i] != ints[i - 1] + 1.0:
            if i - start >= RUN_FROM:
                runs.append((ints[start], ints[i - 1]))
            start = i
    rest = [
        v
        for v in present
        if not (_integral(v) and any(a <= float(v) <= b for a, b in runs))
    ]
    return runs, rest


def _integral(v: Any) -> bool:
    """`v` is an integer within 2**53 of zero."""
    return float(v).is_integer() and abs(float(v)) <= 2.0**53
