"""Compositions of catalog entries: a fitted `Pipeline` is the composition of
its steps' translations (sklearn 1.9, `Pipeline.transform` in
`sklearn/pipeline.py`).

The twin: `Xt = X`, then for each step of `_iter()` (every step, the final
one included, skipping `"passthrough"` and `None`), `Xt = step.transform(Xt)`.
The first step that runs is handed the row as the step hands it; every later
one the previous step's output, a float64 array. So the translation hands
the first step the features and their declared types, and each later step
the previous step's lanes typed DOUBLE. A nested `Pipeline` is one more
catalog entry, and composes the same way.

Only bit-exact steps compose: a lane within k ulps of its twin, read by a
later step, is not within k ulps after it (`x - mean_` near `mean_` turns
a 4-ulp difference into any number of ulps), so a step registered with a
bound refuses the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from confit import sql as S
from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline

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
        if entry.ulps:
            raise NotNative(
                f"Pipeline step {name!r}: {type(step).__name__} is within"
                f" {entry.ulps} ulps, which no later step keeps bounded"
            )
        why = _float64_out(step) if k < len(steps) - 1 else None
        if why:
            raise NotNative(
                f"Pipeline step {name!r}: {type(step).__name__} is not the last"
                f" step and {why}, not float64"
            )
    if len(steps) > 1 and types and all(t == pa.bool_() for t in types):
        # The row is then a boolean array, which a selector hands on as is.
        raise NotNative("Pipeline over boolean features only")
    if not steps and pa.string() in types:
        # Every step passes: the step's float() of a string raises.
        raise NotNative("Pipeline of passthrough steps over a string feature")
    for _, name, step in steps:
        try:
            x = list(entries[type(step)].translate(step, x, types))
        except NotNative as e:
            raise NotNative(f"Pipeline step {name!r}: {e}") from None
        types = [pa.float64()] * len(x)
    return x
