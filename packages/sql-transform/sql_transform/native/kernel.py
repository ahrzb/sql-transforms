"""Kernel approximations (sklearn 1.9, `sklearn/kernel_approximation.py`):
`AdditiveChi2Sampler`, which maps each feature on its own.

For each feature x, with s the sample interval, the twin
(`_transform_dense`) answers 0 where x is 0, else `sqrt(x * s)`; then, for
each j from 1 to `sample_steps - 1`, two more fields, `f * cos(j * (s *
log(x)))` and `f * sin(...)`, with `f = sqrt(2 * x * s / cosh(pi * j * s))`.
The first field is a product and a square root, each correctly rounded on
both sides, so `sample_steps=1` is bit-exact. The other fields read numpy's
`log`, which on x86-64 with AVX-512 is SVML's kernel, up to 1 ulp from
glibc's (DuckDB's and confit's `ln`), and a rounding apart there moves `cos`
and `sin` by any number of ulps near their zeros. They wait on the error
scale the owner ruled (decisions/closed/additive-chi2-parity-bound.md).

Each output field reads its own feature, so a step builds about as fast as
a `StandardScaler` of its width: at 1,024 features, in 1.1 s as a list and
13.2 s as a struct, against 1.0 s and 12.6 s (warm `to_native`, release
build, master 5edd450, 2026-10-16).
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
from confit import sql as S
from sklearn.kernel_approximation import AdditiveChi2Sampler

from sql_transform.native._helpers import f64
from sql_transform.native._registry import NotNative, rejects, translates

# The sample interval sklearn takes where `sample_interval` is None, for
# `sample_steps` 1 to 3 (Vedaldi and Zisserman, 2011, figure 2 c).
_INTERVAL = {1: 0.8, 2: 0.5, 3: 0.4}


@translates(AdditiveChi2Sampler)
def _additive_chi2(est: Any, x: list[S.Expr], types: list[pa.DataType]) -> list[S.Expr]:
    # sklearn: `validate_data(..., ensure_non_negative=True)`, where NaN,
    # infinity and any x < 0 raise (-0.0 passes: `X.min() < 0` reads it as
    # 0), then 0 where `X == 0`, else `np.sqrt(X * s)`, in X's dtype.
    steps = est.sample_steps
    if steps != 1:
        raise NotNative(
            f"AdditiveChi2Sampler(sample_steps={steps}): its cos and sin fields"
            " read numpy's log, an ulp from glibc's on some hosts, which moves"
            " a field near a zero of cos or sin by any number of ulps; they"
            " wait on the error scale"
            " (decisions/closed/additive-chi2-parity-bound.md)"
        )
    given = est.sample_interval
    s = float(_INTERVAL[steps] if given is None else given)
    if not (math.isfinite(s) and s >= 0.0):
        # The fit rejects such an interval, which `transform` does not read
        # again. At inf, the entry's `0 * s` is NaN where the twin answers 0.
        raise NotNative(f"AdditiveChi2Sampler(sample_interval={given!r})")
    if all(t == pa.bool_() for t in types) and math.sqrt(s) not in (0.0, 1.0):
        # A row of boolean features none NULL reaches the twin as a boolean
        # array, which `np.zeros_like(X)` keeps boolean: True answers 1.0,
        # not sqrt(s). Any number or NULL in the row makes it float64.
        raise NotNative(
            "AdditiveChi2Sampler over boolean features only: the twin answers"
            f" True as 1.0, not sqrt({s!r})"
        )
    # NaN, above every number in DuckDB, is the least only where every
    # feature is NaN, which the probe's test traps.
    rejects((x[0] if len(x) == 1 else S.fn("least", *x)) < f64(0.0))
    out = []
    for xi in x:
        p = xi * f64(s)
        # `p > 0` is `x != 0` on what the guard passes, but where `p`
        # underflows to 0, whose root is 0 (x = -0.0 makes `p` -0.0, and the
        # twin's `zeros_like` answers 0.0). It keeps the root off the
        # negatives, which the guard traps: confit then knows that the root
        # cannot raise, as the test reads the root's own argument, so a read
        # of one field of a struct return leaves the others unevaluated
        # (confit #362).
        out.append(S.case(p > f64(0.0), S.fn("sqrt", p)).otherwise(f64(0.0)))
    return out
