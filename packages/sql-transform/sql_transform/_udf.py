"""The UDF protocol: declared scalar functions the serving SQL may call.

A UDF is a name plus declared ``takes``/``returns`` types in the engine
type vocabulary (``"i1" | "i64" | "f64" | "str"``) and two call forms:

- ``__call__`` — THE semantic contract. Plain Python values in (``None``
  for NULL), a tuple matching ``returns`` out (or ``None`` for an all-NULL
  result). Every engine binding must agree with this form bit-for-bit.
- ``apply_batch`` — pyarrow arrays in, pyarrow arrays out. The default
  implementation loops ``__call__`` row by row *deliberately*: it amortizes
  the boundary (one crossing per chunk) without vectorizing the math, so
  batch results are bit-identical to row results. Vectorized overrides are
  an explicit opt-in with a documented ulp caveat.

The contract every UDF signs: **deterministic**. Fit-time DuckDB, batch
serving, and (later) Confit row serving all call the same object and are
compared bit-exact; randomness or hidden state breaks the round-trip
invariant in ways no gate can localize.

Scalar UDFs only — a user-defined aggregate over ``__THIS__`` is what
transformers already are (an object with fit/transform, window syntax).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

_TYPES = frozenset({"i1", "i64", "f64", "str"})
_DUCK = {"i1": "BOOLEAN", "i64": "BIGINT", "f64": "DOUBLE", "str": "VARCHAR"}


class UDFError(ValueError):
    """A UDF declaration or result violated its contract; names how."""


def _check_types(name: str, label: str, types: tuple[str, ...]) -> None:
    for t in types:
        if t not in _TYPES:
            raise UDFError(
                f"UDF {name}: {label} type {t!r} is not one of {sorted(_TYPES)}"
            )


class UDF:
    """Base protocol; subclasses set name/takes/returns and __call__."""

    name: str
    takes: tuple[str, ...]
    returns: tuple[str, ...]

    def __call__(self, *args: Any) -> tuple | None:
        raise NotImplementedError

    def _scalar(self, *args: Any) -> Any:
        """The single-SQL-value form: unwraps a width-1 tuple to its value,
        checks the result against ``returns`` (a violated declaration is a
        broken artifact — raise, never a silent NULL)."""
        out = self(*args)
        if out is None:
            return None
        if not isinstance(out, tuple) or len(out) != len(self.returns):
            raise UDFError(
                f"UDF {self.name} returned {out!r}, declared returns {self.returns}"
            )
        return out[0] if len(out) == 1 else list(out)

    def apply_batch(self, *cols: pa.Array) -> tuple[pa.Array, ...]:
        """Boundary-amortized form; the default loops ``__call__``."""
        rows = zip(*(c.to_pylist() for c in cols), strict=True)
        outs = [self(*r) for r in rows]
        width = len(self.returns)
        return tuple(
            pa.array([None if o is None else o[j] for o in outs]) for j in range(width)
        )

    def _duck_signature(self) -> tuple[list[str], str]:
        params = [_DUCK[t] for t in self.takes]
        ret = _DUCK[self.returns[0]] if len(self.returns) == 1 else "DOUBLE[]"
        return params, ret

    def register(self, con: Any) -> None:
        """Bind to a DuckDB connection: the same object DuckDB executes is
        the one every other engine must match."""
        params, ret = self._duck_signature()
        # DuckDB inspects the callable's arity, so *args won't do — generate
        # a wrapper with exactly len(params) positional parameters.
        names = ", ".join(f"a{i}" for i in range(len(params)))
        ns: dict[str, Any] = {"call": self._scalar}
        exec(f"def w({names}): return call({names})", ns)  # noqa: S102
        con.create_function(self.name, ns["w"], params, ret, null_handling="special")


@dataclass(frozen=True)
class PythonUDF(UDF):
    """Any plain deterministic function as a declared scalar UDF.

    ``fn`` receives plain Python values (``None`` for NULL) and returns the
    single value (``returns`` must declare exactly one type)."""

    name: str
    fn: Callable[..., Any]
    takes: tuple[str, ...]
    returns: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_types(self.name, "takes", self.takes)
        _check_types(self.name, "returns", self.returns)
        if len(self.returns) != 1:
            raise UDFError(
                f"UDF {self.name}: a scalar UDF declares exactly one return"
                f" type, got {self.returns}"
            )

    def __call__(self, *args: Any) -> tuple | None:
        return (self.fn(*args),)


@dataclass(frozen=True)
class PythonTransform(UDF):
    """A fitted transformer as a UDF: per-group weights arrive as data.

    The implicit leading argument (never written in ``takes``) is a
    nullable instance id — the ``__cf_est`` column of the params table the
    serving SQL joins. NULL id = the LEFT JOIN missed = unseen group =
    NULL output. An id *missing from* ``instances`` means the params table
    and the instances are from different fits: that raises."""

    name: str
    instances: dict[int, Any] = field(hash=False)
    takes: tuple[str, ...]
    returns: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_types(self.name, "takes", self.takes)
        _check_types(self.name, "returns", self.returns)

    def __call__(self, iid: int | None, *feats: Any) -> tuple | None:
        if iid is None:
            return None
        try:
            est = self.instances[iid]
        except KeyError:
            raise UDFError(
                f"UDF {self.name}: instance id {iid} not in the fitted"
                " instances — params table and instances are from"
                " different fits"
            ) from None
        row = est.transform([[float("nan") if f is None else float(f) for f in feats]])[
            0
        ]
        out = tuple(float(v) for v in _flatten_row(row))
        if len(out) != len(self.returns):
            raise UDFError(
                f"UDF {self.name} produced {len(out)} values, declared"
                f" {len(self.returns)}"
            )
        return out

    def _duck_signature(self) -> tuple[list[str], str]:
        params, ret = super()._duck_signature()
        return ["BIGINT", *params], ret


def _flatten_row(row: Any) -> list:
    """One transform() output row as a flat list (0-d scalars included)."""
    try:
        return list(row)
    except TypeError:
        return [row]
