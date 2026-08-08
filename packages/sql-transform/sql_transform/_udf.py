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

_DUCK = {"i1": "BOOLEAN", "i64": "BIGINT", "f64": "DOUBLE", "str": "VARCHAR"}
_ARROW = {
    "i1": pa.bool_(),
    "i64": pa.int64(),
    "f64": pa.float64(),
    "str": pa.string(),
}
# The engine computes in exactly i64 / f64 / string / bool, so these four are
# the whole vocabulary. A narrower arrow type refuses rather than widening
# silently, which would make the declared schema a lie about what is served.
_CODE = {v: k for k, v in _ARROW.items()}


class UDFError(ValueError):
    """A UDF declaration or result violated its contract; names how."""


def _code(name: str, label: str, t: pa.DataType) -> str:
    try:
        return _CODE[t]
    except KeyError:
        raise UDFError(
            f"UDF {name}: {label} type {t} is not one of"
            f" {', '.join(str(a) for a in _ARROW.values())}"
        ) from None


def _lanes(name: str, returns: pa.DataType) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(field names, type codes)` for a declared return type — see
    [`UDF.returns`] for what each arrow shape means."""
    if pa.types.is_struct(returns):
        if returns.num_fields == 0:
            raise UDFError(f"UDF {name}: returns struct declares no fields")
        fields = [returns.field(i) for i in range(returns.num_fields)]
        return (
            tuple(f.name for f in fields),
            tuple(_code(name, "returns", f.type) for f in fields),
        )
    if pa.types.is_fixed_size_list(returns):
        # k == 1 is the one shape where the lane COUNT and the arrow SHAPE
        # disagree: width-1 unnamed is a plain scalar everywhere downstream
        # (the engine serves it as one, DuckDB is registered for one), so a
        # value declared as a list would cross as its element.
        if returns.list_size < 2:
            raise UDFError(
                f"UDF {name}: a width-1 list return is a scalar — declare"
                f" {returns.value_type} rather than pa.list_({returns.value_type},"
                f" {returns.list_size})"
            )
        return (), (_code(name, "returns", returns.value_type),) * returns.list_size
    if pa.types.is_list(returns) or pa.types.is_large_list(returns):
        raise UDFError(
            f"UDF {name}: returns list must declare its width —"
            f" pa.list_({returns.value_type}, k), not pa.list_({returns.value_type})"
        )
    return (), (_code(name, "returns", returns),)


class UDF:
    """Base protocol; subclasses set name/takes/returns and __call__.

    The schema is **arrow**, and it is the whole declaration — one field per
    argument, one type for the result:

    ``takes`` is a ``pa.Schema``: names and types together, in call order.
    Arguments bind by POSITION; the names are the *struct* half of the type
    (DRAFT-24), a fitted transform being a function ``S -> T`` between named
    structs.

    ``returns`` is the SQL return TYPE, which also says how wide the call is
    and whether its lanes are addressable:

    ===========================  =======================================
    ``pa.float64()``             an ordinary scalar expression
    ``pa.struct([...])``         width-k with addressable field names
                                 (TASK-63) — struct-valued at EVERY width,
                                 so ``f(x).a`` reads a lane off ONE call
    ``pa.list_(t, k)``           width-k unnamed (the DRAFT-22 list
                                 boundary); fixed size because the width
                                 is part of the declaration
    ===========================  =======================================

    Names are matched name-keyed, never positionally, so a refit that
    renumbers lanes (OneHotEncoder gaining a category) breaks loudly rather
    than rewiring silently. An addressed field is validated at fit and served
    as a field read over the one whole-value call; the names ride as string
    literals, so any spelling is safe.
    """

    name: str
    takes: pa.Schema
    returns: pa.DataType

    @property
    def take_names(self) -> tuple[str, ...]:
        return tuple(self.takes.names)

    @property
    def take_types(self) -> tuple[str, ...]:
        """Argument types as engine codes, in call order."""
        return tuple(_code(self.name, "takes", t) for t in self.takes.types)

    @property
    def return_names(self) -> tuple[str, ...]:
        """Output field names — empty unless ``returns`` is a struct."""
        return _lanes(self.name, self.returns)[0]

    @property
    def return_types(self) -> tuple[str, ...]:
        """Output lane types as engine codes; its length is the call width."""
        return _lanes(self.name, self.returns)[1]

    def _check_schema(self) -> None:
        """Force every type through the vocabulary at declaration, so a bad
        one names itself while the author is still holding the object rather
        than at prepare. The lane readings raise; their values are not wanted
        here."""
        _ = self.take_types, self.return_types
        names = self.return_names
        # Field binding is ASCII-case-insensitive (here and in DuckDB's struct
        # keys) — colliding names would bind silently wrong.
        for i, a in enumerate(names):
            if any(b.lower() == a.lower() for b in names[:i]):
                raise UDFError(
                    f"UDF {self.name}: returns field names collide"
                    f" case-insensitively ({a!r})"
                )

    def lane_of(self, field_name: str) -> int:
        """Index of an output field, by name — ASCII-case-insensitive like
        both engines' struct reads (case collisions refuse at declaration);
        raises naming what exists."""
        try:
            return [n.lower() for n in self.return_names].index(field_name.lower())
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
        rets = self.return_types
        if not isinstance(out, tuple) or len(out) != len(rets):
            raise UDFError(
                f"UDF {self.name} returned {out!r}, declared returns {self.returns}"
            )
        names = self.return_names
        if names:
            # Struct-valued at EVERY width (the subtraction loop): the
            # registration is a STRUCT, so the value is a dict.
            return dict(zip(names, out, strict=True))
        return out[0] if len(out) == 1 else list(out)

    def apply_batch(self, *cols: pa.Array) -> tuple[pa.Array, ...]:
        """Boundary-amortized form; the default loops ``__call__``."""
        rows = zip(*(c.to_pylist() for c in cols), strict=True)
        outs = [self(*r) for r in rows]
        width = len(self.return_types)
        return tuple(
            pa.array([None if o is None else o[j] for o in outs]) for j in range(width)
        )

    def _duck_signature(self) -> tuple[list[str], Any]:
        params = [_DUCK[t] for t in self.take_types]
        names, rets = self.return_names, self.return_types
        if names:
            # Declared names => a STRUCT return at EVERY width, so field
            # access reads lanes off ONE call (TASK-63) — DuckDB CSEs the
            # identical pure calls; confit shares the ecall site.
            import duckdb

            return params, duckdb.struct_type(
                {n: _DUCK[t] for n, t in zip(names, rets, strict=True)}
            )
        if pa.types.is_fixed_size_list(self.returns):
            # A LIST at every width including 1 — the declaration's SHAPE is
            # what crosses, not its lane count, and the lane TYPE is the one
            # declared. Registering this as `DOUBLE[]` regardless (as it was)
            # rounds an int64 lane through a double on the DuckDB side while
            # the engine serves the integer, and at k=1 leaves the two sides
            # disagreeing about the shape as well.
            return params, f"{_DUCK[rets[0]]}[]"
        return params, _DUCK[rets[0]]

    def _arrow_ret(self) -> pa.DataType:
        """The return type as arrow, mirroring ``_duck_signature``.

        This is ``returns`` itself except for the unnamed width-k case, where
        the declaration is a FIXED-size list and the value crossing the
        boundary is a variable one (DuckDB registers `DOUBLE[]`)."""
        if pa.types.is_fixed_size_list(self.returns):
            return pa.list_(self.returns.value_type)
        return self.returns

    def _arrow_scalar_batch(self, *cols: Any) -> pa.Array:
        """One chunk of ``_scalar`` calls as an arrow array. This is the
        DuckDB-facing form: the native return conversion maps NaN to NULL
        (losing a value both engines represent — 2026-08-05 review), while
        arrow arrays carry NaN as a real float, so the ``__call__`` contract
        survives the boundary bit-exact."""
        rows = zip(*(c.to_pylist() for c in cols), strict=True)
        return pa.array([self._scalar(*r) for r in rows], type=self._arrow_ret())

    def register(self, con: Any) -> None:
        """Bind to a DuckDB connection: the same object DuckDB executes is
        the one every other engine must match."""
        params, ret = self._duck_signature()
        # DuckDB inspects the callable's arity, so *args won't do — generate
        # a wrapper with exactly len(params) positional parameters.
        names = ", ".join(f"a{i}" for i in range(len(params)))
        ns: dict[str, Any] = {"call": self._arrow_scalar_batch}
        exec(f"def w({names}): return call({names})", ns)  # noqa: S102
        con.create_function(
            self.name, ns["w"], params, ret, type="arrow", null_handling="special"
        )


@dataclass(frozen=True)
class PythonUDF(UDF):
    """Any plain deterministic function as a declared scalar UDF.

    ``fn`` receives plain Python values (``None`` for NULL) and returns the
    single value (``returns`` must declare exactly one type)."""

    name: str
    fn: Callable[..., Any]
    takes: pa.Schema = field(hash=False)
    returns: pa.DataType = pa.float64()

    def __post_init__(self) -> None:
        self._check_schema()
        if len(self.return_types) != 1 or self.return_names:
            raise UDFError(
                f"UDF {self.name}: a scalar UDF declares one plain return"
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
    takes: pa.Schema = field(hash=False)
    returns: pa.DataType = pa.float64()

    def __post_init__(self) -> None:
        self._check_schema()

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
        vals = [_as_feature(f, t) for f, t in zip(feats, self.take_types, strict=True)]
        row = est.transform([vals])[0]
        out = tuple(float(v) for v in _flatten_row(row))
        width = len(self.return_types)
        if len(out) != width:
            raise UDFError(
                f"UDF {self.name} produced {len(out)} values, declared {width}"
            )
        return out

    def _duck_signature(self) -> tuple[list[str], str]:
        params, ret = super()._duck_signature()
        return ["BIGINT", *params], ret


class Named:
    """An estimator with author-declared output field names (DRAFT-24).

    >>> transformers={"pca": Named(PCA(n_components=2), returns=("size", "cost"))}
    >>> "SELECT pca(struct_pack(a := age, f := fare)).size AS x ..."

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
        if len({n.lower() for n in names}) != len(names):
            # Field matching is case-insensitive on both engines.
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

    def __getattr__(self, name: str) -> Any:
        # Forward inner declarations (order_sensitive, feature-name probes)
        # so wrapper nesting order never cancels a contract (review round
        # 2026-08-05: Named(OrderSensitive(est)) ran order-blind). Dunders
        # never forward: protocol probes (__sklearn_clone__, __deepcopy__,
        # pickle) must see the WRAPPER, or clone() strips it.
        est = self.__dict__.get("estimator")
        if est is None or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(est, name)

    def __repr__(self) -> str:
        return f"Named({self.estimator!r}, returns={list(self.returns)})"


class OrderSensitive:
    """Declares a fit ORDER-sensitive (2026-08-05 spec, fit lawfulness):
    the default contract makes a fit a multiset aggregate — order-blind,
    author-signed. Wrapping an estimator in ``OrderSensitive`` flips the
    contract: the query must name the order via in-call ``ORDER BY``
    (``sm_fit(bundle ORDER BY ts) OVER (...)``), and the fit scope is
    stably sorted by it before ``fit``. Mechanism promise only: we sort
    by what you name."""

    order_sensitive = True

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    # sklearn's clone protocol, like Named.
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"estimator": self.estimator}

    def set_params(self, **params: Any) -> OrderSensitive:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: Any, y: Any = None) -> OrderSensitive:
        self.estimator.fit(X)
        return self

    def transform(self, X: Any) -> Any:
        return self.estimator.transform(X)

    def __getattr__(self, name: str) -> Any:
        # Forward declared_output_names / get_feature_names_out / anything
        # else the naming machinery probes on the inner estimator. The .get
        # guard keeps pickle/copy (which probe on empty instance state) on
        # AttributeError, never KeyError; dunders never forward — protocol
        # probes (__sklearn_clone__, __deepcopy__) must see the WRAPPER, or
        # clone() strips it.
        est = self.__dict__.get("estimator")
        if est is None or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(est, name)

    def __repr__(self) -> str:
        return f"OrderSensitive({self.estimator!r})"


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
