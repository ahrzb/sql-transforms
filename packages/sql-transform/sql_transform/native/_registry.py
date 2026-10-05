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
(`CASE WHEN id = 0 THEN ... WHEN id IN (1, 2) THEN ... END`, instances
whose lane is the same SQL sharing an arm). As in the step, a NULL id
answers NULL (a NULL struct or list, for a struct or list return) and an
id the step does not know raises. Before it is returned, confit builds it into the query
reading every lane (`query`): a translation confit refuses leaves the step
Python.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from confit import DuckDBInferFn, Function, SqlFunction
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

# What an id the step does not know raises with (`error()` takes a constant
# message, so the id itself is not in it).
_UNKNOWN = (
    "{}: an instance id not in the fitted instances (params table and"
    " instances are from different fits)"
)


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
    # `SqlFunction` already cast the parameter to its declared type.
    x = param if t == pa.float64() else param.cast(pa.float64())
    return S.coalesce(x, S.lit(float("nan")))


def _lanes(step: PythonTransform) -> list[tuple[str | None, pa.DataType]]:
    """The output lanes, each `(field name, type)`; unnamed in a scalar
    or a fixed-size list return."""
    r = step.returns
    if pa.types.is_struct(r):
        return [(r.field(i).name, r.field(i).type) for i in range(r.num_fields)]
    if pa.types.is_fixed_size_list(r):
        return [(None, r.value_type)] * r.list_size
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
            # Instances whose lane is the same SQL (a stateless estimator,
            # or equal fits) share one arm.
            arms: dict[str, tuple[list[int], S.Expr]] = {}
            for k, out in per_id:
                arms.setdefault(out[j].sql(), ([], out[j]))[0].append(k)

            def hit(ks: list[int]) -> S.Expr:
                return iid == S.lit(ks[0]) if len(ks) == 1 else iid.isin(*ks)

            (ks0, v0), *rest = arms.values()
            e = S.case(hit(ks0), v0)
            for ks, v in rest:
                e = e.when(hit(ks), v)
            if j == 0:
                # An unknown id raises, from the first lane only: DuckDB
                # builds every field of a struct, so a read of any lane
                # fires it, and the other lanes carry no trap that every
                # field read would have to keep.
                e = e.when(iid.isnull(), S.lit(None, lanes[0][1])).otherwise(
                    S.fn("error", S.lit(_UNKNOWN.format(step.name)))
                )
            return e

        if pa.types.is_struct(step.returns):
            return {name: select(j) for j, (name, _) in enumerate(lanes)}
        if pa.types.is_fixed_size_list(step.returns):
            return [select(j) for j in range(len(lanes))]
        return select(0)

    whole = pa.types.is_struct(step.returns) or pa.types.is_fixed_size_list(
        step.returns
    )
    fn = SqlFunction(
        step.name,
        takes,
        step.returns,
        body,
        # A NULL id is a NULL struct or list, not one of NULL lanes.
        null_when=(lambda iid, *_: iid.isnull()) if whole else None,
    )
    _builds(step, fn)
    return fn


def query(step: Any, id_col: str = _ID) -> str:
    """The query reading every output lane of one call of `step`: what
    `to_native` builds and `check` serves with both entries."""
    args = ", ".join([S.col(id_col).sql(), *(S.col(n).sql() for n in step.takes.names)])
    call = f"{step.name}({args})"
    r = step.returns
    if pa.types.is_struct(r):
        items = [
            f"{call}.{S.col(r.field(i).name).sql()} AS {S.col(r.field(i).name).sql()}"
            for i in range(r.num_fields)
        ]
    else:
        items = [f"{call} AS o"]
    return f"SELECT {', '.join(items)} FROM __THIS__"  # noqa: S608


def _builds(step: PythonTransform, fn: SqlFunction) -> None:
    """Raise `NotNative` unless confit builds `fn` into `query(step)`, so a
    translation it refuses (one too large to expand, say) leaves the step
    Python rather than failing where it is served."""
    schema = pa.schema([pa.field(_ID, pa.int64()), *step.takes])
    try:
        DuckDBInferFn(
            query(step, _ID),
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[fn],
        )
    except (ValueError, RuntimeError) as e:
        # A refusal (ValueError), or a limit met while compiling, such as
        # Cranelift's function size (RuntimeError): the step stays Python.
        raise NotNative(f"confit does not build it: {e}") from None


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
