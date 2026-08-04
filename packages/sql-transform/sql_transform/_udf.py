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
    """Base protocol; subclasses set name/takes/returns and __call__.

    ``take_names``/``return_names`` carry the *struct* half of the type
    (DRAFT-24): a fitted transform is a function ``S -> T`` between named
    structs, and the names are matched name-keyed, never positionally — a
    refit that renumbers lanes (OneHotEncoder gaining a category) must
    break loudly, not rewire silently. An addressed field is validated at
    fit and served as a field read over the ONE whole-value call (TASK-63);
    the names ride as string literals, so any spelling is safe."""

    name: str
    takes: tuple[str, ...]
    returns: tuple[str, ...]
    take_names: tuple[str, ...] = ()
    return_names: tuple[str, ...] = ()

    def lane_of(self, field_name: str) -> int:
        """Index of an output field, by name; raises naming what exists."""
        try:
            return self.return_names.index(field_name)
        except ValueError:
            raise UDFError(
                f"UDF {self.name} has no output field {field_name!r};"
                f" fitted output is {list(self.return_names)}"
            ) from None

    def __call__(self, *args: Any) -> tuple | None:
        raise NotImplementedError

    def _scalar(self, *args: Any) -> Any:
        """The single-SQL-value form: unwraps a width-1 tuple to its value,
        checks the result against ``returns`` (a violated declaration is a
        broken artifact — raise, never a silent NULL). Width-k is a dict
        (STRUCT — one call, field reads, TASK-63) when field names are
        declared, else a list."""
        out = self(*args)
        if out is None:
            return None
        if not isinstance(out, tuple) or len(out) != len(self.returns):
            raise UDFError(
                f"UDF {self.name} returned {out!r}, declared returns {self.returns}"
            )
        if len(out) == 1:
            return out[0]
        if self.return_names:
            return dict(zip(self.return_names, out, strict=True))
        return list(out)

    def apply_batch(self, *cols: pa.Array) -> tuple[pa.Array, ...]:
        """Boundary-amortized form; the default loops ``__call__``."""
        rows = zip(*(c.to_pylist() for c in cols), strict=True)
        outs = [self(*r) for r in rows]
        width = len(self.returns)
        return tuple(
            pa.array([None if o is None else o[j] for o in outs]) for j in range(width)
        )

    def _duck_signature(self) -> tuple[list[str], Any]:
        params = [_DUCK[t] for t in self.takes]
        if len(self.returns) == 1:
            return params, _DUCK[self.returns[0]]
        if self.return_names:
            # Width-k with declared names: a STRUCT return, so field access
            # reads lanes off ONE call (TASK-63) — DuckDB CSEs the
            # identical pure calls; confit shares the ecall site.
            import duckdb

            return params, duckdb.struct_type(
                {
                    n: _DUCK[t]
                    for n, t in zip(self.return_names, self.returns, strict=True)
                }
            )
        return params, "DOUBLE[]"

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
    take_names: tuple[str, ...] = ()
    return_names: tuple[str, ...] = ()

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
    take_names: tuple[str, ...] = ()
    return_names: tuple[str, ...] = ()

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
        vals = [_as_feature(f, t) for f, t in zip(feats, self.takes, strict=True)]
        row = est.transform([vals])[0]
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


class Named:
    """An estimator with author-declared output field names (DRAFT-24).

    >>> transformers={"pca": Named(PCA(n_components=2), returns=("size", "cost"))}
    >>> "SELECT pca(struct_pack(a := age, f := fare)) OVER ().size AS x ..."

    The declaration is *authoritative*, unlike sklearn's own
    ``get_feature_names_out`` (advisory: ignored when it disagrees with the
    learned width). So a fixed override on a learned-width transform — an
    encoder whose categories change with the data — refuses at fit rather
    than mislabelling lanes.

    Not required for correctness: SQL is already a rename operator
    (``struct_pack(x := p.pca0, ...)``, or a wrapper projection level)."""

    def __init__(self, estimator: Any, returns: tuple[str, ...]) -> None:
        # Keep the caller's object when it is already the normal form:
        # sklearn's clone asserts the constructor stores params unchanged
        # (identity, not equality).
        names = (
            returns
            if isinstance(returns, tuple) and all(isinstance(n, str) for n in returns)
            else tuple(str(n) for n in returns)
        )
        if not names:
            raise UDFError("Named(...): declare at least one output name")
        if len(set(names)) != len(names):
            raise UDFError(f"Named(...): duplicate output names in {list(names)}")
        self.estimator = estimator
        self.returns = names

    @property
    def declared_output_names(self) -> tuple[str, ...]:
        return self.returns

    # sklearn's clone protocol, so per-group cloning keeps its usual meaning
    # (a fresh unfitted copy) instead of falling back to deepcopy.
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"estimator": self.estimator, "returns": self.returns}

    def set_params(self, **params: Any) -> Named:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: Any, y: Any = None) -> Named:
        self.estimator.fit(X)
        return self

    def transform(self, X: Any) -> Any:
        return self.estimator.transform(X)

    def __repr__(self) -> str:
        return f"Named({self.estimator!r}, returns={list(self.returns)})"


def _as_feature(value: Any, ty: str) -> Any:
    """One feature value on its way into ``transform``, per its declared
    type: numerics go through float (NULL as NaN, the estimator's own
    missing-value convention); strings and booleans pass through."""
    if ty == "str":
        return value
    if value is None:
        return float("nan")
    return value if ty == "i1" else float(value)


def _flatten_row(row: Any) -> list:
    """One transform() output row as a flat list (0-d scalars included)."""
    try:
        return list(row)
    except TypeError:
        return [row]


def output_names(
    est: Any, take_names: tuple[str, ...], width: int, label: str = ""
) -> tuple[str, ...]:
    """The fitted output field names (T), by DRAFT-24's source order: an
    author declaration (``Named``) — authoritative, so a width disagreement
    refuses; else sklearn's ``get_feature_names_out`` — advisory, ignored
    when it disagrees; else canonical ``f0..``."""
    declared = getattr(est, "declared_output_names", None)
    if declared is not None:
        if len(declared) != width:
            raise UDFError(
                f"transformer {label or est!r} declares"
                f" {len(declared)} output names {list(declared)} but fits to"
                f" width {width}"
            )
        return tuple(declared)
    getter = getattr(est, "get_feature_names_out", None)
    if getter is not None:
        for args in ((take_names,) if take_names else (), ()):
            names: tuple[str, ...] = ()
            try:
                names = tuple(str(n) for n in getter(*args))
            except Exception:  # noqa: BLE001,S110 — metadata is advisory, never fatal
                names = ()
            if len(names) == width and len(set(names)) == width:
                return names
    return tuple(f"f{i}" for i in range(width))
