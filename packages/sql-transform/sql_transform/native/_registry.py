"""The catalog and the translation of one `PythonTransform`.

A translator turns ONE fitted estimator into its output lanes, as
`confit.sql` expressions over the feature expressions it is handed:

    @translates(StandardScaler)
    def _(est, x: list[S.Expr]) -> list[S.Expr]: ...

`x` holds one expression per declared feature, already in the form the
estimator receives it from `PythonTransform`: a numeric feature is a DOUBLE
with NULL read as NaN, a string or boolean passes as is. A translator that
cannot serve this estimator's configuration raises `NotNative` naming why.

The framework does the rest, the same way for every entry: one
`SqlFunction` named like the step, taking the instance id then the
features, whose every lane selects the instance's expression by id
(`CASE WHEN id = 0 THEN ... WHEN id = 1 THEN ... END`; a NULL id matches no
arm and is NULL, as the step's own NULL id is).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from confit import Function, SqlFunction
from confit import sql as S

from sql_transform._udf import PythonTransform

Translator = Callable[[Any, list[S.Expr]], list[S.Expr]]


@dataclass(frozen=True)
class Entry:
    """A catalog entry: the translator and its declared parity bound, in
    doubles per lane (0 = bit-exact), fixed by the measurement its module
    cites."""

    translate: Translator
    ulps: int


_CATALOG: dict[type, Entry] = {}

# The instance-id parameter. Double underscores keep it apart from any
# feature name a step can declare (features come from user columns).
_ID = "__iid"


class NotNative(Exception):  # noqa: N818 — a reason, raised and caught
    """This step has no native translation; the message says why."""


def translates(*classes: type, ulps: int = 0) -> Callable[[Translator], Translator]:
    """Register a translator for exactly these estimator classes, bit-exact
    unless `ulps` says otherwise. A subclass is not covered: it may override
    what `transform` computes."""

    def deco(fn: Translator) -> Translator:
        for c in classes:
            if c in _CATALOG:
                raise ValueError(f"{c.__name__} is already in the catalog")
            _CATALOG[c] = Entry(fn, ulps)
        return fn

    return deco


def catalog() -> dict[type, Entry]:
    """The registered estimator classes and their entries."""
    return dict(_CATALOG)


def _feature(param: S.Expr, t: pa.DataType) -> S.Expr:
    """A declared feature as `PythonTransform` hands it to `transform`:
    numbers (and booleans, as 0/1) as DOUBLE with NULL read as NaN, a string
    as is."""
    if t == pa.string():
        return param
    return S.coalesce(param.cast(pa.float64()), S.lit(float("nan")))


def _lanes(step: PythonTransform) -> list[tuple[str | None, pa.DataType]]:
    r = step.returns
    if pa.types.is_struct(r):
        return [(r.field(i).name, r.field(i).type) for i in range(r.num_fields)]
    if pa.types.is_fixed_size_list(r):
        raise NotNative(
            "an unnamed width-k output (a list return) waits on confit serving"
            " list-valued SQL functions"
        )
    return [(None, r)]


def _translate(step: Any) -> SqlFunction:
    if not isinstance(step, PythonTransform):
        raise NotNative(f"{type(step).__name__} is not a PythonTransform")
    if not step.instances:
        raise NotNative("the step has no fitted instances")
    kinds = {type(e) for e in step.instances.values()}
    missing = sorted(k.__name__ for k in kinds if k not in _CATALOG)
    if missing:
        raise NotNative(f"no translation for {', '.join(missing)}")
    lanes = _lanes(step)
    types = list(step.takes.types)
    takes = pa.schema([pa.field(_ID, pa.int64()), *step.takes])

    def body(iid: S.Expr, *params: S.Expr) -> Any:
        feats = [_feature(p, t) for p, t in zip(params, types, strict=True)]
        per_id: list[tuple[int, list[S.Expr]]] = []
        for k, est in sorted(step.instances.items()):
            out = list(_CATALOG[type(est)].translate(est, feats))
            if len(out) != len(lanes):
                raise NotNative(
                    f"instance {k}: {type(est).__name__} translates to"
                    f" {len(out)} lanes, the step declares {len(lanes)}"
                )
            per_id.append((k, out))

        def select(j: int) -> S.Expr:
            (k0, out0), *rest = per_id
            e = S.case(iid == S.lit(k0), out0[j])
            for k, out in rest:
                e = e.when(iid == S.lit(k), out[j])
            return e

        if lanes[0][0] is None:
            return select(0)
        return {name: select(j) for j, (name, _) in enumerate(lanes)}

    return SqlFunction(step.name, takes, step.returns, body)


def bound(step: Any) -> int:
    """The parity bound of a step's translation: the loosest of its
    instances' entries."""
    return max(_CATALOG[type(e)].ulps for e in step.instances.values())


def to_native(step: Any, *, strict: bool = False) -> Function | Any:
    """The native twin of `step`, or `step` itself when there is none.

    With `strict`, a step without a translation raises `NotNative` instead.
    Idempotent: a step that is already native comes back unchanged."""
    if isinstance(step, Function):
        return step
    try:
        return _translate(step)
    except NotNative:
        if strict:
            raise
        return step


def explain_native(step: Any) -> str:
    """What `to_native(step)` returns, or why it returns the step."""
    if isinstance(step, Function):
        return f"{step!r} is already a confit function"
    try:
        fn = _translate(step)
    except NotNative as e:
        return f"{getattr(step, 'name', step)!r} stays Python: {e}"
    kinds = sorted({type(e).__name__ for e in step.instances.values()})
    return f"{fn.name!r}: {', '.join(kinds)} -> SqlFunction"
