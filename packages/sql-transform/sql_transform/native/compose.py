"""Compositions of catalog entries: a fitted `Pipeline` is the composition of
its steps' translations (sklearn 1.9, `Pipeline.transform` in
`sklearn/pipeline.py`); a `ColumnTransformer` or `FeatureUnion` the
concatenation of its parts' (below, `_column_transformer`, `_union`).

The twin: `Xt = X`, then for each step of `_iter()` (every step, the final
one included, skipping `"passthrough"` and `None`),
`Xt = step.transform(Xt)`. The first step that runs is handed the row as the
step hands it; every later one the previous step's output, a float64 array,
or over boolean features only a boolean one where a step hands booleans
back. So the translation hands the first step the features and their
declared types, and each later step the previous step's output fields typed
DOUBLE, or boolean then. A nested `Pipeline` is one more catalog entry, and
composes the same way.

A step whose parity bound is above 0, in ulps or in an error scale,
serves in a composition only with `to_native(step, allow_bound=True)` (the
owner's ruling, proposals/closed/bounded-steps-in-compositions.md).
Where it reads bit-exact input, the input its twin reads, its own bound
holds (option 1): as a part of a `ColumnTransformer` or `FeatureUnion`,
times its weight where it has one, or as the first such step of a
pipeline. A step that reads the output of such a step carries the bound to
its own output fields (option 2, _carry.py). An output field within k ulps
of its twin is not within k ulps after `x - mean_` where `x` is near
`mean_`, but it is within 2k·spacing(x) plus the subtraction's rounding.
The affine steps carry a bound, each by its module's derivation: the
scalers but `Normalizer`, the selectors, the mean of `FeatureAgglomeration`
and the linear projections; and so do the identity and the functions of
`FunctionTransformer` that move two values no further apart (function.py),
and a weight. `check` holds each output
field to the bound of the step or part that gives it, carried by the steps
after it, and the rest bit-exact (`_pipeline_bounds`, `_parts_bounds`). A
step that tests a threshold, such as a discretizer, can answer otherwise
where the bound moves a value across it (option 3), and a step that is not
affine has no derivation yet: where either reads the output of a step whose
bound is above 0, the composition stays Python, as it does where an
infinite weight multiplies such an output. The bound read is the step's,
not its class's: `FunctionTransformer()` composes anywhere, and
`FunctionTransformer(np.exp)` where only steps that carry its bound read
its output (anywhere where this platform's numpy kernel reads 0,
`function.py`).
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _check_X, _is_empty_column_selection
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils._indexing import _determine_key_type, _safe_indexing

from sql_transform._udf import _dense
from sql_transform.native._carry import absolute, as_fields, weighted
from sql_transform.native._helpers import f64
from sql_transform.native._registry import (
    FieldBounds,
    NotNative,
    PartBounds,
    catalog,
    handed,
    handed_on,
    translate_part,
    translates,
)


def _float64_out(est: Any) -> str | None:
    """Why `est`'s output is not the float64 array the next step is
    assumed to read, or None when it is. Every catalog entry returns
    float64 on a float64 or mixed row but these: a mask is boolean, an
    encoder or discretizer with `dtype` returns that dtype, and a pipeline
    returns what its last step that runs returns (its input, when none
    runs)."""
    if isinstance(est, Pipeline):
        runs = list(est._iter(with_final=True, filter_passthrough=True))
        return _float64_out(runs[-1][2]) if runs else None
    if isinstance(est, ColumnTransformer | FeatureUnion):
        # The hstack of the parts' outputs. A ColumnTransformer handed the
        # step's row passes columns through as an object array.
        for _, part, _ in _parts(est):
            if isinstance(est, ColumnTransformer) and _passes(part):
                return "it passes columns through as objects"
            why = _float64_out(part)
            if why:
                return why
        return None
    if isinstance(est, MissingIndicator):
        return "its output is boolean"
    dtype = getattr(est, "dtype", None)
    if dtype is None:
        return None
    if np.dtype(dtype) != np.float64:
        return f"its output is {np.dtype(dtype).name}"
    return None


# Sparse outputs (decisions/closed/sparse-outputs.md, the ruling). The
# step reads a sparse output dense (`_udf._dense`), but inside a `Pipeline`
# a sparse step hands the next step the sparse matrix, and the next step
# then runs its own sparse branch, where the entries follow the dense one.
# Measured on each catalog entry that fits on a sparse matrix, one row a
# call (sklearn 1.9, 2026-10-14): handed the 0/1 matrix of `OneHotEncoder`,
# `KBinsDiscretizer` or `RandomTreesEmbedding`, each of `_SPARSE_READERS`
# answers as on its dense array, but `QuantileTransformer` maps the stored
# values only (under `output_distribution="normal"` a 0 keeps 0.0, where
# its dense map answers -5.2), a `FunctionTransformer` applies its `func`
# to the matrix (`np.negative` leaves a 0 at 0.0, not -0.0, and `np.exp`
# raises), and `Binarizer` under a negative threshold raises. Handed real
# values (a sparse `SplineTransformer`'s, or an imputer's fill of a stored
# 1), the scalers, `Normalizer` and `PolynomialFeatures` part from their
# dense answers by an ulp or more, so a sparse step before the last hands
# on 0s and 1s only (`_ones`). The step reads a stored -0.0 (an imputer's
# `fill_value=-0.0`) as 0.0, so each output field of a sparse `Pipeline`
# is `x + 0.0`.
_SPARSE_READERS = (
    SelectorMixin,
    Binarizer,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    PolynomialFeatures,
    SimpleImputer,
    MissingIndicator,
    RandomTreesEmbedding,
    FunctionTransformer,
)


def _sparse(est: Any, sparse_in: bool = False) -> bool:
    """`est.transform` returns a scipy sparse matrix, handed a dense row
    (or a sparse matrix: `sparse_in`). An encoder, discretizer, embedding,
    spline or indicator says so by its parameter, a `ColumnTransformer` by
    `sparse_output_` (set at fit), a `FeatureUnion` by a part's, a
    `Pipeline` by its last step's; any other step keeps a sparse input
    sparse (`_reads_sparse` refuses the rest)."""
    if isinstance(est, Pipeline):
        for _, _, step in est._iter(with_final=True, filter_passthrough=True):
            sparse_in = _sparse(step, sparse_in)
        return sparse_in
    if isinstance(est, ColumnTransformer):
        return bool(est.sparse_output_)
    if isinstance(est, FeatureUnion):
        return any(_sparse(part) for _, part, _ in _parts(est))
    if isinstance(est, OneHotEncoder | RandomTreesEmbedding | SplineTransformer):
        return bool(est.sparse_output)
    if isinstance(est, KBinsDiscretizer):
        return est.encode == "onehot"
    if isinstance(est, MissingIndicator):
        return est.sparse is True or (est.sparse == "auto" and sparse_in)
    return sparse_in


def _reads_sparse(step: Any) -> str | None:
    """Why `step` does not answer a sparse matrix of 0s and 1s as its
    dense array, or None when it does (above)."""
    if not isinstance(step, _SPARSE_READERS):
        return "is handed a sparse matrix, which it is not known to read as dense"
    if isinstance(step, FunctionTransformer) and step.func is not None:
        return "is handed a sparse matrix, to which it applies its func"
    if isinstance(step, Binarizer) and step.threshold < 0:
        return "is handed a sparse matrix, on which a negative threshold raises"
    return None


def _ones(step: Any) -> bool:
    """`step`'s output, where sparse, holds only 0s and 1s: an encoder's,
    a discretizer's or an embedding's from any row, and a reader's from
    the sparse 0/1 matrix it is handed (the only sparse matrix a step
    before the last hands on)."""
    if isinstance(step, OneHotEncoder | KBinsDiscretizer | RandomTreesEmbedding):
        return True
    if isinstance(step, SimpleImputer):
        # A sparse fit refuses `missing_values=0`: on 0s and 1s the imputer
        # fills a stored 1 alone, with the feature's statistic.
        m = step.missing_values
        fills = np.asarray(step.statistics_, dtype=np.float64)
        fills = fills[~np.isnan(fills)]
        return not (isinstance(m, numbers.Number) and m == 1) or bool(
            np.isin(fills, (0.0, 1.0)).all()
        )
    return isinstance(step, SelectorMixin | Binarizer | PolynomialFeatures) or (
        isinstance(step, FunctionTransformer) and step.func is None
    )


def _numbers(X: Any) -> np.ndarray:
    """The one row of `X`, as the twin hands it to a part, as long doubles:
    NaN for None, a boolean as 0 or 1 (as `check` reads a step's features
    for its error scale)."""
    row = np.asarray(_dense(X), dtype=object).reshape(1, -1)[0]
    return np.array(
        [math.nan if v is None else float(v) for v in row], dtype=np.longdouble
    )


def _width(part: Any, X: Any) -> int:
    """How many output fields `part` gives at its one-row input `X`."""
    return np.asarray(_dense(part.transform(X)), dtype=object).reshape(1, -1).shape[1]


def _exact_at(part: Any, X: Any) -> FieldBounds:
    """The bound of a bit-exact part: 0 ulps in each output field."""
    w = _width(part, X)
    return FieldBounds(np.zeros(w, np.int64), np.full(w, np.nan, np.longdouble))


def _bounds_of(where: str, part: Any) -> PartBounds | None:
    """`part`'s bound, output field by output field, as a composition
    holds it; None where it is bit-exact (or has no entry, which its
    translation refuses)."""
    entry = catalog().get(type(part))
    if entry is None:
        return None
    try:
        bounds = entry.part_bounds(part)
    except NotNative as e:
        raise NotNative(f"{where}: {e}") from None
    if bounds is not None:
        return bounds
    scale = entry.scale_for(part)
    if scale is None:
        # A zero ceiling is already exact. Calling bound() would discover
        # part_bounds again, doubling the work at every exact nesting level.
        if entry.ulps == 0:
            return None
        u = entry.bound(part)
        if u == 0:
            return None

        def ulps(X: Any) -> FieldBounds:
            w = _width(part, X)
            return FieldBounds(
                np.full(w, u, np.int64), np.full(w, np.nan, np.longdouble)
            )

        return PartBounds(ulps, entry.within(part))
    if scale.g is not None:
        raise NotNative(
            f"{where}: {type(part).__name__} compares through a map, which a"
            " composition does not pass on"
        )

    def near(X: Any) -> FieldBounds:
        w = _width(part, X)
        return FieldBounds(np.zeros(w, np.int64), scale.at(part, _numbers(X), w))

    return PartBounds(near, entry.within(part))


def _named(names: list[str]) -> str:
    """`names` quoted, as a message lists them: 'a', 'b' and 'c'."""
    *rest, last = names
    return f"{', '.join(rest)} and {last}" if rest else last


def _pipeline_bounds(est: Any) -> PartBounds | None:
    """A pipeline's bound, from its first step whose bound is above 0: that
    step's own, at the input the steps before it hand on, which are
    bit-exact, so that the twin's step reads what the entry's reads
    (option 1 of the ruling, decisions/closed/bounded-steps-in-compositions.md).
    Each later step carries it to its output fields (option 2, `_carry_of`),
    or refuses where no bound is derived through it: `NotNative`."""
    steps = list(est._iter(with_final=True, filter_passthrough=True))
    for first, (_, name, step) in enumerate(steps):  # noqa: B007 — read below
        b = _bounds_of(f"Pipeline step {name!r}", step)
        if b is not None:
            break
    else:
        return None
    source = f"step {name!r}, which is {b.within}"
    later = []
    for _, name_after, after in steps[first + 1 :]:
        try:
            later.append((after, _carry_of(after, source)))
        except NotNative as e:
            raise NotNative(f"Pipeline step {name_after!r}: {e}") from None
    before = [prior for _, _, prior in steps[:first]]

    def at(X: Any) -> FieldBounds:
        for prior in before:
            X = prior.transform(X)
        fields = b.at(X)
        if not later:
            return fields
        X = step.transform(X)
        e = absolute(fields, _numbers(X))
        for after, carry in later:
            e = carry(X, e)
            X = after.transform(X)
        return as_fields(e)

    names = [repr(n) for _, n, _ in steps[first:]]
    s = "s" * (len(names) > 1)
    return PartBounds(at, f"within the bound of its step{s} {_named(names)}")


def _carry_of(est: Any, source: str) -> Callable[[Any, np.ndarray], np.ndarray]:
    """How `est` carries a bound on its input to its output fields, where
    it reads the output of `source`, a step whose bound is above 0: a
    function of the one-row input `X` as the twin hands it and of each
    input field's bound `e` (`_registry.Carry`, at the twin's `X`), which
    answers each output field's bound. `NotNative` where no bound is
    derived through `est` or one of its steps or parts: a step that is not
    affine, or that tests a threshold."""
    if isinstance(est, Pipeline):
        return _pipeline_carry(est, source)
    if isinstance(est, ColumnTransformer | FeatureUnion):
        return _parts_carry(est, source)
    entry = catalog().get(type(est))
    carry = None if entry is None else entry.carry(est)
    if carry is None:
        raise NotNative(
            f"{type(est).__name__} follows {source}, and no bound is derived"
            " through it yet"
        )
    return lambda X, e: carry(_numbers(X), e)


def _pipeline_carry(est: Any, source: str) -> Callable[[Any, np.ndarray], np.ndarray]:
    """A pipeline's carry: each step's in turn, at the input the twin's
    steps before it hand on."""
    steps = []
    for _, name, step in est._iter(with_final=True, filter_passthrough=True):
        try:
            steps.append((step, _carry_of(step, source)))
        except NotNative as e:
            raise NotNative(f"Pipeline step {name!r}: {e}") from None

    def carry(X: Any, e: np.ndarray) -> np.ndarray:
        for step, c in steps:
            e = c(X, e)
            X = step.transform(X)
        return e

    return carry


def _parts_carry(est: Any, source: str) -> Callable[[Any, np.ndarray], np.ndarray]:
    """A `ColumnTransformer`'s or `FeatureUnion`'s carry: each part's, from
    the bounds of the columns the twin hands it, times its weight
    (`_carry.weighted`), in the parts' order."""
    owner = type(est).__name__
    weights = est.transformer_weights or {}
    parts = []
    for name, part, columns in _parts(est):
        try:
            c = _carry_of(part, source)
            w = weights.get(name)
            if isinstance(w, numbers.Real) and math.isinf(w):
                raise NotNative(
                    f"its weight {w!r} multiplies an output that follows {source}"
                )
        except NotNative as e:
            raise NotNative(f"{owner} part {name!r}: {e}") from None
        parts.append((part, columns, w, c))
    ct = isinstance(est, ColumnTransformer)
    n = est.n_features_in_ if ct else 0

    def carry(X: Any, e: np.ndarray) -> np.ndarray:
        X = _check_X(X) if ct else X
        out = []
        for part, columns, w, c in parts:
            if ct:
                cols = _columns(columns, n)
                if _is_empty_column_selection(cols):
                    continue  # as `_column_transformer` skips it
                Xp, ep = _safe_indexing(X, columns, axis=1), e[cols]
            else:
                Xp, ep = X, e
            o = c(Xp, ep)
            if w is not None:
                o = weighted(o, _numbers(part.transform(Xp)), float(w))
            out.append(o)
        return np.concatenate(out) if out else np.zeros(0, np.longdouble)

    return carry


def _parts_bounds(est: Any) -> PartBounds | None:
    """A `ColumnTransformer`'s or `FeatureUnion`'s bound: each part's own,
    at the input the twin hands that part, which is the entry's, times the
    part's weight where it has one (`_carry.weighted`)."""
    owner = type(est).__name__
    parts = _parts(est)
    weights = est.transformer_weights or {}
    bounds = [_bounds_of(f"{owner} part {name!r}", part) for name, part, _ in parts]
    names = [
        repr(name) + " (weighted)" * (weights.get(name) is not None)
        for (name, _, _), b in zip(parts, bounds, strict=True)
        if b is not None
    ]
    if not names:
        return None
    ct = isinstance(est, ColumnTransformer)
    n = est.n_features_in_ if ct else 0

    def at(X: Any) -> FieldBounds:
        X = _check_X(X) if ct else X
        ulps, near = [], []
        for (name, part, columns), b in zip(parts, bounds, strict=True):
            if ct:
                if _is_empty_column_selection(_columns(columns, n)):
                    continue  # as `_column_transformer` skips it
                Xp = _safe_indexing(X, columns, axis=1)
            else:
                Xp = X
            if b is None:
                f = _exact_at(part, Xp)
            else:
                f, w = b.at(Xp), weights.get(name)
                if w is not None:
                    y = _numbers(part.transform(Xp))
                    f = as_fields(weighted(absolute(f, y), y, float(w)))
            ulps.append(f.ulps)
            near.append(f.near)
        return FieldBounds(np.concatenate(ulps), np.concatenate(near))

    s = "s" * (len(names) > 1)
    return PartBounds(at, f"within the bound{s} of its part{s} {_named(names)}")


@translates(
    Pipeline,
    composes=True,
    part_bounds=_pipeline_bounds,
    uncapped=(
        "Parts enforce their own time caps; MAX_UNSHARED bounds only "
        "unshared-value growth, not total build time."
    ),
)
def _pipeline(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # `transform_input` transforms fit metadata only; `memory` and
    # `verbose` act in fit. Refused all the same: the composition is
    # checked without them.
    if est.transform_input is not None:
        raise NotNative(f"Pipeline(transform_input={est.transform_input!r})")
    entries = catalog()
    steps = list(est._iter(with_final=True, filter_passthrough=True))
    sparse = False
    _pipeline_bounds(est)  # refuses a step whose bound is above 0 before the last
    for k, (_, name, step) in enumerate(steps):
        entry = entries.get(type(step))
        if entry is None:
            raise NotNative(
                f"Pipeline step {name!r}: no translation for {type(step).__name__}"
            )
        why = _float64_out(step) if k < len(steps) - 1 else None
        if why:
            raise NotNative(
                f"Pipeline step {name!r}: {type(step).__name__} is not the last"
                f" step and {why}, not float64"
            )
        why = _reads_sparse(step) if sparse else None
        if why:
            raise NotNative(f"Pipeline step {name!r}: {type(step).__name__} {why}")
        sparse = _sparse(step, sparse)
        if sparse and k < len(steps) - 1 and not _ones(step):
            raise NotNative(
                f"Pipeline step {name!r}: {type(step).__name__} is not the last"
                " step and hands on a sparse matrix of values other than 0 and 1"
            )
    if not steps and pa.string() in types:
        # Every step passes: the step's float() of a string raises.
        raise NotNative("Pipeline of passthrough steps over a string feature")
    # Over boolean features only, a row none NULL is a boolean array (where
    # the pipeline is handed the step's list, or such an array), and a step
    # that hands booleans back (a selector, `Binarizer`, the identity) hands
    # the next one a boolean array too: its output fields stay typed
    # boolean, for the next step's own rule (`FunctionTransformer` has one).
    hand = handed()
    booleans = (
        hand in ("list", "booleans")
        and bool(types)
        and all(t == pa.bool_() for t in types)
    )
    for k, (_, name, step) in enumerate(steps):
        try:
            x = translate_part(step, x, types, hand)
        except NotNative as e:
            raise NotNative(f"Pipeline step {name!r}: {e}") from None
        if booleans:
            dtype = on_booleans(step, len(types))
            booleans = dtype == np.bool_
            if dtype not in (None, np.bool_, np.float64) and k < len(steps) - 1:
                raise NotNative(
                    f"Pipeline step {name!r}: {type(step).__name__} hands on"
                    f" {dtype} over boolean features only"
                )
        types = [pa.bool_() if booleans else pa.float64()] * len(x)
        if k < len(steps) - 1:
            hand = _next_hand(name, step, hand, booleans)
    return [o + f64(0.0) for o in x] if sparse else x


def _next_hand(name: str, step: Any, hand: str, booleans: bool) -> str:
    """How the twin hands the step after `step` its row (`HANDS`): a
    boolean array where the row has no NULL while booleans flow, else
    `step`'s output, float64 (`_float64_out`), or, from a step handed
    objects that keeps its input's dtype (a selector,
    `SimpleImputer(strategy="most_frequent")`), objects."""
    if booleans:
        return "booleans"
    if hand != "objects":
        return "doubles"
    if catalog()[type(step)].composes:
        raise NotNative(
            f"Pipeline step {name!r}: a {type(step).__name__} handed objects"
            " (a ColumnTransformer's part), whose output the input guard"
            " does not probe"
        )
    return handed_on()


def on_booleans(est: Any, n: int) -> np.dtype | None:
    """The dtype of `est`'s output on a row of `n` booleans, as the step
    hands a row of boolean features none NULL; None where `est` raises on
    it (as `SimpleImputer(strategy="most_frequent")` does: the twin then
    answers only rows with a NULL, which are float64)."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return np.asarray(_dense(est.transform(np.zeros((1, n), dtype=bool)))).dtype
    except Exception:  # noqa: BLE001 — the twin raises it too
        return None


# ColumnTransformer and FeatureUnion (sklearn 1.9). Both run each fitted
# part's `transform` through `_transform_one`, which returns `res * weight`
# when the part has a weight in `transformer_weights` and `res` otherwise,
# and `_hstack` the results in the parts' order. The step hands `transform`
# its row as a one-row list (`PythonTransform.__call__`); then:
#
# - `ColumnTransformer.transform`: `_check_X` turns the list into an object
#   array (`check_array(dtype=object)`); the parts are `_iter(fitted=True,
#   skip_drop=True, skip_empty_columns=True)` over `transformers_`, in its
#   order, the remainder last; each is handed `_safe_indexing(X, columns,
#   axis=1)`, its columns as resolved at fit (a callable's result, a
#   slice, indices, a boolean mask; names need a DataFrame). A
#   "passthrough" is a fitted identity `FunctionTransformer` there, so it
#   hands back its object columns as they are. `_hstack` is `np.hstack`
#   over the parts, a sparse part made dense (`.toarray()`), or, when
#   `sparse_output_` (set at fit), `sparse.hstack` over the parts as
#   `check_array` makes them numbers.
# - `FeatureUnion.transform`: each part of `_iter()` (skipping "drop"; a
#   "passthrough" is an identity `FunctionTransformer`, which hands back
#   the list itself) is handed the whole row, the list; `_hstack` is
#   `xp.concat(axis=1)` (numpy's concatenate), or `sparse.hstack` when a
#   part's output is sparse.
#
# The step reads the stacked row with `float()`, so a part's output fields
# keep their values whatever dtype the stack takes (float64, or object
# beside a passthrough), and a weight is `x * w` in doubles: the twin's
# product of a float64 (or Python float) by the weight. A sparse matrix
# stores no zero, -0.0 included, and the step makes it dense by adding each
# stored value to 0.0 (`_udf._dense`): where the stack is sparse, and where
# a sparse part is weighted (`res * w` scales the stored values only), each
# output field is `x + 0.0`, which is `x` but for a -0.0, which it makes
# 0.0 (in DuckDB and confit alike).


def _passes(part: Any) -> bool:
    """`part` is an identity: "passthrough", or a `FunctionTransformer`
    without `func`, which returns its input unchecked or validated."""
    return isinstance(part, FunctionTransformer) and part.func is None


def _parts(est: Any) -> list[tuple[str, Any, Any]]:
    """`(name, part, columns)` for each part `transform` runs, in order;
    `columns` is None for a `FeatureUnion` part, which reads the row."""
    if isinstance(est, ColumnTransformer):
        return [
            (name, part, columns)
            for name, part, columns, _ in est._iter(
                fitted=True,
                column_as_labels=False,
                skip_drop=True,
                skip_empty_columns=True,
            )
        ]
    return [(name, part, None) for name, part, _ in est._iter()]


def _reads_objects(part: Any) -> str | None:
    """Why `part` does not read an object array as it reads the step's
    row, or None when it does. Every catalog entry validates its input to
    numbers (or, an encoder, per column) except `FunctionTransformer`
    without validation, whose ufunc then calls each element's method
    (`np.sqrt` on a float object raises)."""
    if isinstance(part, FunctionTransformer):
        if part.func is not None and not part.validate:
            return "a FunctionTransformer(validate=False) applies its func to objects"
        return None
    if isinstance(part, Pipeline):
        runs = list(part._iter(with_final=True, filter_passthrough=True))
        return _reads_objects(runs[0][2]) if runs else None
    if isinstance(part, FeatureUnion):
        for _, p, _ in _parts(part):
            why = _reads_objects(p)
            if why:
                return why
    return None


def _encodes(part: Any) -> bool:
    """An encoder reads `part`'s columns: it is one, a pipeline's first
    step that runs, or a part of a union."""
    if isinstance(part, Pipeline):
        runs = list(part._iter(with_final=True, filter_passthrough=True))
        return bool(runs) and _encodes(runs[0][2])
    if isinstance(part, ColumnTransformer | FeatureUnion):
        return any(_encodes(p) for _, p, _ in _parts(part))
    return isinstance(part, OneHotEncoder | OrdinalEncoder | TargetEncoder)


def _narrow_float(est: Any) -> bool:
    """`est`'s output may be float32 or narrower, which a weight then
    multiplies in that precision."""
    if isinstance(est, Pipeline):
        runs = list(est._iter(with_final=True, filter_passthrough=True))
        return bool(runs) and _narrow_float(runs[-1][2])
    if isinstance(est, ColumnTransformer | FeatureUnion):
        return any(_narrow_float(p) for _, p, _ in _parts(est))
    dtype = getattr(est, "dtype", None)
    if dtype is None:
        return False
    dt = np.dtype(dtype)
    return dt.kind == "f" and dt.itemsize < 8


def _weight(
    owner: str, name: str, part: Any, w: Any, types: list[pa.DataType]
) -> float:
    """The weight as the double the twin multiplies by, or `NotNative`."""
    where = f"{owner} part {name!r}"
    if not isinstance(w, numbers.Real) or not (float(w) == w or math.isnan(w)):
        raise NotNative(f"{where}: weight {w!r} is not a double")
    if _narrow_float(part):
        raise NotNative(f"{where}: a weight on a float32 output multiplies in float32")
    if isinstance(w, numbers.Integral):
        # An integer output (or a passed boolean) times an integer is an
        # integer, whose zero has no sign: 0 * -2 is 0, not -0.0.
        why = _float64_out(part) or _passes_booleans(owner, part, types)
        if why:
            raise NotNative(
                f"{where}: an integer weight on an output that may not be float64"
                f" ({why}) multiplies in integers"
            )
    return float(w)


def _passes_booleans(owner: str, part: Any, types: list[pa.DataType]) -> str | None:
    """Why `part` may hand on booleans, or None when it cannot. A
    `ColumnTransformer` hands its parts objects, a boolean feature as a
    Python bool, which a selector keeps. A `FeatureUnion` hands each part
    the row, a boolean array only over boolean features none NULL (a
    number or a NULL makes it float64): the part's output on one says."""
    if pa.bool_() not in types:
        return None
    if owner == "ColumnTransformer":
        return "a boolean feature"
    if not all(t == pa.bool_() for t in types):
        return None
    dtype = on_booleans(part, len(types))
    if dtype in (None, np.float64):
        return None
    return f"{dtype} over boolean features only"


def _part(
    owner: str,
    name: str,
    part: Any,
    weight: Any,
    x: list[S.Expr],
    types: list[pa.DataType],
    hand: str,
    stack: bool,
) -> list[S.Expr]:
    """One part's output fields over its own features, weighted; the twin
    hands it its row as `hand`, and stacks the parts sparse (`stack`) or
    dense."""
    where = f"{owner} part {name!r}"
    entry = catalog().get(type(part))
    if entry is None:
        raise NotNative(f"{where}: no translation for {type(part).__name__}")
    if _passes(part) and pa.string() in types:
        raise NotNative(
            f"{where} passes a string column through; the step reads output"
            " fields with float()"
        )
    w = None if weight is None else _weight(owner, name, part, weight, types)
    b = None if w is None or not math.isinf(w) else _bounds_of(where, part)
    if b is not None:
        # inf times the two sides' zeros, or values of either sign about
        # zero, is NaN or an infinity of either sign.
        raise NotNative(
            f"{where}: {type(part).__name__} is {b.within}, and its weight"
            f" {weight!r} is infinite"
        )
    try:
        out = translate_part(part, x, types, hand)
    except NotNative as e:
        raise NotNative(f"{where}: {e}") from None
    if w is not None:
        out = [o * f64(w) for o in out]
    if stack or (w is not None and _sparse(part)):
        out = [o + f64(0.0) for o in out]
    return out


def _refuse_container(est: Any) -> None:
    config = getattr(est, "_sklearn_output_config", {}).get("transform")
    if config not in (None, "default"):
        # A DataFrame row, which the step's `transform(...)[0]` misreads.
        raise NotNative(f"{type(est).__name__}.set_output(transform={config!r})")


def _columns(columns: Any, n: int) -> list[int]:
    """The indices of the columns `columns` selects of `n`: the twin's own
    selection, run on the column indices."""
    picked = _safe_indexing(np.arange(n).reshape(1, n), columns, axis=1)
    if np.ndim(picked) != 2:
        raise NotNative(f"column {columns!r} selects a 1-D array")
    return [int(i) for i in picked[0]]


@translates(
    ColumnTransformer,
    composes=True,
    part_bounds=_parts_bounds,
    uncapped=(
        "Parts enforce their own time caps; MAX_UNSHARED bounds only "
        "unshared-value growth, not total build time."
    ),
)
def _column_transformer(
    est: Any, x: list[S.Expr], types: list[pa.DataType]
) -> list[S.Expr]:
    _refuse_container(est)
    n = len(x)
    if est.n_features_in_ != n:
        raise NotNative(
            f"ColumnTransformer fitted on {est.n_features_in_} features, the step"
            f" has {n}"
        )
    weights = est.transformer_weights or {}
    # `_check_X` makes the step's list an object array, and hands an array
    # on as it is: each part is handed its columns of it.
    hand = handed()
    hand = "objects" if hand == "list" else hand
    out: list[S.Expr] = []
    for name, part, columns in _parts(est):
        where = f"ColumnTransformer part {name!r}"
        if _determine_key_type(columns) == "str":
            raise NotNative(
                f"{where}: columns {columns!r} are names, which need a DataFrame"
            )
        try:
            cols = _columns(columns, n)
        except NotNative as e:
            raise NotNative(f"{where}: {e}") from None
        if _is_empty_column_selection(cols):
            continue  # an empty slice, which `_iter` keeps
        why = _reads_objects(part)
        if why:
            raise NotNative(f"{where}: {why}, as the twin hands it")
        if any(types[i] == pa.bool_() for i in cols) and _encodes(part):
            # An object array of Python bools, which sklearn matches its
            # own way (categories [nan]: [True, False] answers as objects
            # and raises as booleans); the entry probes the step's rows.
            raise NotNative(
                f"{where}: an encoder over a boolean feature reads it as an"
                " object, which the entry does not probe"
            )
        out += _part(
            "ColumnTransformer",
            name,
            part,
            weights.get(name),
            [x[i] for i in cols],
            [types[i] for i in cols],
            hand,
            bool(est.sparse_output_),
        )
    return out


@translates(
    FeatureUnion,
    composes=True,
    part_bounds=_parts_bounds,
    uncapped=(
        "Parts enforce their own time caps; MAX_UNSHARED bounds only "
        "unshared-value growth, not total build time."
    ),
)
def _union(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    _refuse_container(est)
    weights = est.transformer_weights or {}
    parts = _parts(est)
    stack = any(_sparse(part) for _, part, _ in parts)
    out: list[S.Expr] = []
    for name, part, _ in parts:
        if _passes(part) and weights.get(name) is not None:
            # It hands back the step's list, and `list * w` raises.
            raise NotNative(f"FeatureUnion part {name!r}: a weighted passthrough")
        w = weights.get(name)
        out += _part("FeatureUnion", name, part, w, x, types, handed(), stack)
    return out
