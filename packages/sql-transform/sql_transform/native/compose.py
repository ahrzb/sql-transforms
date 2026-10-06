"""Compositions of catalog entries: a fitted `Pipeline` is the composition of
its steps' translations (sklearn 1.9, `Pipeline.transform` in
`sklearn/pipeline.py`); a `ColumnTransformer` or `FeatureUnion` the
concatenation of its parts' (below, `_column_transformer`, `_union`).

The twin: `Xt = X`, then for each step of `_iter()` (every step, the final
one included, skipping `"passthrough"` and `None`), `Xt = step.transform(Xt)`.
The first step that runs is handed the row as the step hands it; every later
one the previous step's output, a float64 array, or over boolean features
only a boolean one where a step hands booleans back. So the translation
hands the first step the features and their declared types, and each later
step the previous step's lanes typed DOUBLE, or boolean then. A nested
`Pipeline` is one more catalog entry, and composes the same way.

Only bit-exact steps compose: a lane within k ulps of its twin, read by a
later step, is not within k ulps after it (`x - mean_` near `mean_` turns
a 4-ulp difference into any number of ulps), so a step whose own bound is
not 0 refuses the pipeline. The bound read is the step's, not its class's:
`FunctionTransformer()` composes, `FunctionTransformer(np.exp)` does not.
"""

from __future__ import annotations

import math
import numbers
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _is_empty_column_selection
from sklearn.impute import MissingIndicator
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from sklearn.utils._indexing import _determine_key_type, _safe_indexing

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, catalog, translates


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


@translates(Pipeline)
def _pipeline(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # `transform_input` transforms fit metadata only; `memory` and
    # `verbose` act in fit. Refused all the same: the composition is
    # checked without them.
    if est.transform_input is not None:
        raise NotNative(f"Pipeline(transform_input={est.transform_input!r})")
    entries = catalog()
    steps = list(est._iter(with_final=True, filter_passthrough=True))
    for k, (_, name, step) in enumerate(steps):
        entry = entries.get(type(step))
        if entry is None:
            raise NotNative(
                f"Pipeline step {name!r}: no translation for {type(step).__name__}"
            )
        ulps = entry.bound(step)
        if ulps:
            raise NotNative(
                f"Pipeline step {name!r}: {type(step).__name__} is within"
                f" {ulps} ulps, which no later step keeps bounded"
            )
        why = _float64_out(step) if k < len(steps) - 1 else None
        if why:
            raise NotNative(
                f"Pipeline step {name!r}: {type(step).__name__} is not the last"
                f" step and {why}, not float64"
            )
    if not steps and pa.string() in types:
        # Every step passes: the step's float() of a string raises.
        raise NotNative("Pipeline of passthrough steps over a string feature")
    # Over boolean features only, a row none NULL is a boolean array, and a
    # step that hands booleans back (a selector, `Binarizer`, the identity)
    # hands the next one a boolean array too: its lanes stay typed boolean,
    # for the next step's own rule (`FunctionTransformer` has one).
    booleans = bool(types) and all(t == pa.bool_() for t in types)
    for k, (_, name, step) in enumerate(steps):
        try:
            x = list(entries[type(step)].translate(step, x, types))
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
    return x


def on_booleans(est: Any, n: int) -> np.dtype | None:
    """The dtype of `est`'s output on a row of `n` booleans, as the step
    hands a row of boolean features none NULL; None where `est` raises on
    it (as `SimpleImputer(strategy="most_frequent")` does: the twin then
    answers only rows with a NULL, which are float64)."""
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return np.asarray(est.transform(np.zeros((1, n), dtype=bool))).dtype
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
#   hands back its object columns as they are. `_hstack` is `np.hstack`,
#   or a sparse matrix when `sparse_output_` (set at fit).
# - `FeatureUnion.transform`: each part of `_iter()` (skipping "drop"; a
#   "passthrough" is an identity `FunctionTransformer`, which hands back
#   the list itself) is handed the whole row, the list; `_hstack` is
#   `xp.concat(axis=1)` (numpy's concatenate), or sparse when a part's
#   output is.
#
# The step reads the stacked row with `float()`, so a part's lanes keep
# their values whatever dtype the stack takes (float64, or object beside a
# passthrough), and a weight is `x * w` in doubles: the twin's product of a
# float64 (or Python float) by the weight.


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
) -> list[S.Expr]:
    """One part's lanes over its own features, weighted."""
    where = f"{owner} part {name!r}"
    entry = catalog().get(type(part))
    if entry is None:
        raise NotNative(f"{where}: no translation for {type(part).__name__}")
    ulps = entry.bound(part)
    if ulps:
        raise NotNative(
            f"{where}: {type(part).__name__} is within {ulps} ulps;"
            " a composition serves bit-exact parts only"
        )
    if _passes(part) and pa.string() in types:
        raise NotNative(
            f"{where} passes a string column through; the step reads lanes with float()"
        )
    w = None if weight is None else _weight(owner, name, part, weight, types)
    try:
        out = list(entry.translate(part, x, types))
    except NotNative as e:
        raise NotNative(f"{where}: {e}") from None
    return out if w is None else [o * f64(w) for o in out]


def _refuse_container(est: Any) -> None:
    config = getattr(est, "_sklearn_output_config", {}).get("transform")
    if config not in (None, "default"):
        # A DataFrame row, which the step's `transform(...)[0]` misreads.
        raise NotNative(f"{type(est).__name__}.set_output(transform={config!r})")


@translates(ColumnTransformer)
def _column_transformer(
    est: Any, x: list[S.Expr], types: list[pa.DataType]
) -> list[S.Expr]:
    if est.sparse_output_:
        raise NotNative("ColumnTransformer: the output is sparse (sparse_output_)")
    _refuse_container(est)
    n = len(x)
    if est.n_features_in_ != n:
        raise NotNative(
            f"ColumnTransformer fitted on {est.n_features_in_} features, the step"
            f" has {n}"
        )
    weights = est.transformer_weights or {}
    out: list[S.Expr] = []
    for name, part, columns in _parts(est):
        where = f"ColumnTransformer part {name!r}"
        if _determine_key_type(columns) == "str":
            raise NotNative(
                f"{where}: columns {columns!r} are names, which need a DataFrame"
            )
        # The twin's own selection, run on the column indices.
        picked = _safe_indexing(np.arange(n).reshape(1, n), columns, axis=1)
        if np.ndim(picked) != 2:
            raise NotNative(f"{where}: column {columns!r} selects a 1-D array")
        cols = [int(i) for i in picked[0]]
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
        )
    return out


@translates(FeatureUnion)
def _union(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    _refuse_container(est)
    weights = est.transformer_weights or {}
    out: list[S.Expr] = []
    for name, part, _ in _parts(est):
        if _passes(part) and weights.get(name) is not None:
            # It hands back the step's list, and `list * w` raises.
            raise NotNative(f"FeatureUnion part {name!r}: a weighted passthrough")
        out += _part("FeatureUnion", name, part, weights.get(name), x, types)
    return out
