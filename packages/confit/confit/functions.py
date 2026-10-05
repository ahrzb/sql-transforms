"""The functions a query may call, as classes: what `udfs=` accepts.

    Function                 name, takes, returns; deterministic
    ├── SqlFunction          defined by a SQL expression over its parameters
    └── ExternFunction       defined by a Python callable, which the engine calls
        └── Ensemble         plus packed tree tables, which the engine scores
                             natively, bit-equal to the callable

The rule that sorts them: a function's DEFINITION is what the DuckDB oracle
runs (`register`), and a subclass only adds what the engine may know about
it, never a different meaning. An `Ensemble` is an extern whose callable is
the reference walk of its own tables; that the engine scores those tables
natively instead of calling back into Python is a fast path, held to the
callable at `==`.

How a call is evaluated -- called, folded at bind, scored natively -- is the
engine's choice and not part of the contract. The contract is the value: as
if the definition ran once per call.

These classes are the documented spelling of the structural protocol the
engine reads (`name`, `takes`, `returns`, `__call__`, and optionally
`instances`, `side_effects` and `tree_tables()`), so any object with that
shape is still accepted. See docs/specs/serving-contract.md, "UDF and model
boundary".
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = ["Ensemble", "ExternFunction", "Function", "FunctionError", "SqlFunction"]

# The engine computes in these four, and nothing narrower crosses a call: a
# narrower arrow type is refused rather than widened, which would make the
# declaration a lie about what is served.
_DUCK = {
    pa.bool_(): "BOOLEAN",
    pa.int64(): "BIGINT",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
}


class FunctionError(ValueError):
    """A function's declaration violates the protocol; the message names how."""


def _check_type(name: str, label: str, t: pa.DataType) -> None:
    if t not in _DUCK:
        raise FunctionError(
            f"function {name}: {label} type {t} is not one of"
            f" {', '.join(str(a) for a in _DUCK)}"
        )


class Function:
    """The base: a deterministic function of its arguments.

    `takes` is a `pa.Schema`, one field per argument in call order; arguments
    bind by position. `returns` is the SQL result type, and also its width:

    * a scalar type -- an ordinary scalar value;
    * `pa.struct([...])` -- named lanes, struct-valued at every width, so
      `f(x).a` reads one lane of one call;
    * `pa.list_(t, k)` with `k >= 2` -- unnamed lanes.
    """

    name: str
    takes: pa.Schema
    returns: pa.DataType

    def _declare(self, name: str, takes: pa.Schema, returns: pa.DataType) -> None:
        if not isinstance(name, str) or not name:
            raise FunctionError("a function needs a non-empty string name")
        if not isinstance(takes, pa.Schema):
            raise FunctionError(f"function {name}: takes must be a pa.Schema")
        if not isinstance(returns, pa.DataType):
            raise FunctionError(f"function {name}: returns must be a pa.DataType")
        for t in takes.types:
            _check_type(name, "takes", t)
        for t in _lanes(name, returns)[1]:
            _check_type(name, "returns", t)
        self.name, self.takes, self.returns = name, takes, returns

    @property
    def return_names(self) -> tuple[str, ...]:
        """The output lane names: empty unless `returns` is a struct."""
        return _lanes(self.name, self.returns)[0]

    def register(self, con: Any) -> None:
        """Register the definition on a DuckDB connection (or an
        `confit.oracle.Oracle`): the reading every engine must match."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _lanes(
    name: str, returns: pa.DataType
) -> tuple[tuple[str, ...], list[pa.DataType]]:
    if pa.types.is_struct(returns):
        if returns.num_fields == 0:
            raise FunctionError(f"function {name}: a struct return declares no fields")
        fields = [returns.field(i) for i in range(returns.num_fields)]
        names = tuple(f.name for f in fields)
        # Lane reads are ASCII-case-insensitive in both engines.
        for i, a in enumerate(names):
            if any(b.lower() == a.lower() for b in names[:i]):
                raise FunctionError(
                    f"function {name}: return fields collide case-insensitively ({a!r})"
                )
        return names, [f.type for f in fields]
    if pa.types.is_fixed_size_list(returns):
        if returns.list_size < 2:
            raise FunctionError(
                f"function {name}: a width-1 list return is a scalar; declare"
                f" {returns.value_type}"
            )
        return (), [returns.value_type] * returns.list_size
    if pa.types.is_list(returns) or pa.types.is_large_list(returns):
        raise FunctionError(
            f"function {name}: a list return must declare its width,"
            f" pa.list_({returns.value_type}, k)"
        )
    return (), [returns]


class SqlFunction(Function):
    """A function defined by a SQL expression over its parameters.

    `body` receives one `confit.sql` expression per parameter, in `takes`
    order, each already cast to its declared type, and returns the result:
    an expression for a scalar `returns`, a dict of one expression per field
    for a struct `returns`, or a list of k expressions for a
    `pa.list_(t, k)` return. For a struct or a list, `null_when` (called like
    `body`) is the condition under which the whole value is NULL rather than
    a value of NULL parts: `CASE WHEN <null_when> THEN NULL ELSE ... END`.

        scale = SqlFunction(
            "scale", pa.schema([("x", pa.float64())]), pa.float64(),
            lambda x: (x - S.lit(3.5)) * S.lit(2.0),
        )
        DuckDBInferFn("SELECT scale(a) AS z FROM __THIS__", ..., udfs=[scale])

    The definition is `sql_body`, text with each parameter reference spelled
    `CAST("x" AS <type>)` and the result cast to `returns`. DuckDB registers
    it as a macro (`register`), which binds a call by substituting the
    argument for each parameter; confit performs the same substitution, so
    both engines bind one expression. That the engine inlines a call is how
    it is served today, not part of the contract.

    Unlike a declared extern, a parameter or field may have any type the
    engine serves, since the body is ordinary SQL. A body reads only its
    parameters; another column, which DuckDB would bind against the calling
    query, refuses here.
    """

    def __init__(
        self,
        name: str,
        takes: pa.Schema,
        returns: pa.DataType,
        body: Callable[..., Any],
        *,
        null_when: Callable[..., Any] | None = None,
    ) -> None:
        from confit import sql as S

        if not isinstance(name, str) or not S._IDENT.match(name):
            raise FunctionError(
                f"a sql function needs an identifier name, not {name!r}"
            )
        if not isinstance(takes, pa.Schema):
            raise FunctionError(f"function {name}: takes must be a pa.Schema")
        if not isinstance(returns, pa.DataType):
            raise FunctionError(f"function {name}: returns must be a pa.DataType")
        if len({n.lower() for n in takes.names}) != len(takes):
            raise FunctionError(f"function {name}: parameter names collide")
        self.name, self.takes, self.returns = name, takes, returns
        params = [S.col(f.name).cast(_type_name(name, f.type)) for f in takes]
        out = body(*params)
        if pa.types.is_struct(returns):
            fields = [returns.field(i) for i in range(returns.num_fields)]
            if not isinstance(out, dict) or [k.lower() for k in out] != [
                f.name.lower() for f in fields
            ]:
                raise FunctionError(
                    f"function {name}: a struct return's body is a dict of"
                    f" {[f.name for f in fields]}, in order"
                )
            exprs = [S._wrap(out[k]) for k in out]
            rendered = ", ".join(
                f"{S._quote_ident(f.name)} := {e.cast(_type_name(name, f.type)).sql()}"
                for f, e in zip(fields, exprs, strict=True)
            )
            text = f"struct_pack({rendered})"
        elif pa.types.is_fixed_size_list(returns):
            _lanes(name, returns)  # a width of at least 2
            k, t = returns.list_size, _type_name(name, returns.value_type)
            if not isinstance(out, list | tuple) or len(out) != k:
                raise FunctionError(
                    f"function {name}: a width-{k} list return's body is a list of"
                    f" {k} expressions"
                )
            exprs = [S._wrap(x) for x in out]
            text = "[" + ", ".join(e.cast(t).sql() for e in exprs) + "]"
        elif pa.types.is_list(returns) or pa.types.is_large_list(returns):
            raise FunctionError(
                f"function {name}: a list return declares its width,"
                f" pa.list_({returns.value_type}, k)"
            )
        else:
            if isinstance(out, dict):
                raise FunctionError(
                    f"function {name}: a scalar return's body is one expression"
                )
            exprs = [S._wrap(out)]
            text = exprs[0].cast(_type_name(name, returns)).sql()
        if null_when is not None:
            if not (
                pa.types.is_struct(returns) or pa.types.is_fixed_size_list(returns)
            ):
                raise FunctionError(
                    f"function {name}: null_when is for a struct or list return"
                )
            cond = S._wrap(null_when(*params))
            exprs.append(cond)
            text = f"CASE WHEN {cond.sql()} THEN NULL ELSE {text} END"
        allowed = set(takes.names)
        for e in exprs:
            for node in e.walk():
                if isinstance(node, S.Column) and (
                    len(node.path) != 1 or node.path[0] not in allowed
                ):
                    raise FunctionError(
                        f"function {name}: the body reads {node.sql()}, which is not a"
                        " parameter"
                    )
        self.sql_body = text

    def register(self, con: Any) -> None:
        from confit import sql as S

        params = ", ".join(S._quote_ident(n) for n in self.takes.names)
        con.execute(
            f"CREATE MACRO {S._quote_ident(self.name)}({params}) AS {self.sql_body}"
        )


def _type_name(name: str, t: pa.DataType) -> str:
    from confit import sql as S

    try:
        return S.type_name(t)
    except TypeError as e:
        raise FunctionError(f"function {name}: {e}") from None


class ExternFunction(Function):
    """A function defined by a Python callable.

    `fn` takes one Python value per argument (`None` for NULL) and returns
    the output lanes as a tuple, or `None` for an all-NULL result. The engine
    calls it per row (or folds a call over constants at bind, unless
    `side_effects`); DuckDB calls the same callable.

        double = ExternFunction(
            "double", pa.schema([("x", pa.float64())]), pa.float64(),
            lambda x: None if x is None else (2 * x,),
        )
        DuckDBInferFn("SELECT double(a) AS d FROM __THIS__", ..., udfs=[double])
    """

    def __init__(
        self,
        name: str,
        takes: pa.Schema,
        returns: pa.DataType,
        fn: Callable[..., tuple | None],
        *,
        side_effects: bool = False,
    ) -> None:
        self._declare(name, takes, returns)
        if not callable(fn):
            raise FunctionError(f"function {name}: fn must be callable")
        if not isinstance(side_effects, bool):
            raise FunctionError(f"function {name}: side_effects must be a bool")
        self.fn = fn
        self.side_effects = side_effects

    def __call__(self, *args: Any) -> tuple | None:
        return self.fn(*args)

    def _params(self) -> list[str]:
        return [_DUCK[t] for t in self.takes.types]

    def register(self, con: Any) -> None:
        import duckdb

        names, types = _lanes(self.name, self.returns)
        listy = pa.types.is_fixed_size_list(self.returns)
        if names:
            ret: Any = duckdb.struct_type(
                {n: _DUCK[t] for n, t in zip(names, types, strict=True)}
            )
        elif listy:
            ret = f"{_DUCK[types[0]]}[]"
        else:
            ret = _DUCK[types[0]]

        def unwrap(out: tuple | None) -> Any:
            if out is None:
                return None
            if not isinstance(out, tuple) or len(out) != len(types):
                raise FunctionError(
                    f"function {self.name} returned {out!r}, declared {self.returns}"
                )
            if names:
                return dict(zip(names, out, strict=True))
            return list(out) if listy else out[0]

        params = self._params()
        # DuckDB reads the arity off the Python signature, so *args won't do.
        argl = ", ".join(f"a{i}" for i in range(len(params)))
        ns: dict[str, Any] = {"call": self, "unwrap": unwrap}
        exec(f"def w({argl}): return unwrap(call({argl}))", ns)  # noqa: S102
        con.create_function(
            self.name,
            ns["w"],
            params,
            ret,
            null_handling="special",
            side_effects=self.side_effects,
        )


_NODE_TYPES = {
    "model_id": pa.int64(),
    "tree_id": pa.int64(),
    "node_id": pa.int64(),
    "feature": pa.int32(),
    "threshold": pa.float64(),
    "left": pa.int32(),
    "right": pa.int32(),
    "missing_left": pa.bool_(),
    "value": pa.float64(),
}
_MODEL_TYPES = {
    "model_id": pa.int64(),
    "base": pa.float64(),
    "agg": pa.string(),
    "link": pa.string(),
}


def _check_table(name: str, label: str, t: pa.Table, want: dict) -> None:
    if not isinstance(t, pa.Table):
        raise FunctionError(f"function {name}: {label} must be a pa.Table")
    for col, ty in want.items():
        if col not in t.column_names:
            raise FunctionError(f"function {name}: {label} has no column {col!r}")
        if t.schema.field(col).type != ty:
            raise FunctionError(
                f"function {name}: {label} column {col!r} is"
                f" {t.schema.field(col).type}, not {ty}"
            )
        if t.column(col).null_count:
            raise FunctionError(f"function {name}: {label} column {col!r} has a NULL")


def _int_to_f32(n: int) -> float:
    """`n` rounded to float32 in ONE rounding (nearest, ties to even), as the
    kernel's integer-to-float32 conversion does. `float(n)` first would round
    twice and land a float32 ulp away above 2**53."""
    a = abs(n)
    excess = a.bit_length() - 24
    if excess > 0:
        q, r = divmod(a, 1 << excess)
        half = 1 << (excess - 1)
        if r > half or (r == half and q & 1):
            q += 1
        a = q << excess
    # `a` now has at most 25 significant bits, so `float` is exact.
    return math.copysign(float(a), n) if n else 0.0


class Ensemble(ExternFunction):
    """A tree ensemble, scored natively from its packed tables.

    `nodes` holds every split and leaf, grouped by model and then by tree, in
    the order the trees accumulate: `model_id`, `tree_id`, `node_id` (dense
    from 0 per tree), `feature` (-1 on a leaf), `threshold`, `left`/`right`
    (tree-local node ids), `missing_left`, `value`. `models` holds one header
    per model, `model_id` dense from 0: `base`, `agg` (`"sum"` seeds the
    accumulator with `base`; `"mean"` adds `base` to the average), `link`
    (`"identity"` or `"sigmoid"`).

    `compare_grid` is the grid the thresholds live on. On `"float32"` an
    integer feature narrows to float32 in one rounding before the compare
    and a double compares as it is; on `"float64"` both compare as doubles.

    The call takes the model id first (NULL id = NULL result; an id with no
    model raises), then the features in `takes` order, and returns one
    double. The callable is the reference walk of the tables, which is what
    DuckDB runs; the native kernel is held to it at `==`.

    Confit knows no ML library: a packer (sql-transform's, for sklearn)
    builds the tables.
    """

    def __init__(
        self,
        name: str,
        takes: pa.Schema,
        nodes: pa.Table,
        models: pa.Table,
        compare_grid: str,
    ) -> None:
        self._declare(name, takes, pa.float64())
        if not len(takes):
            raise FunctionError(
                f"function {name}: an ensemble scores at least one feature"
            )
        for t in takes.types:
            if t not in (pa.int64(), pa.float64()):
                raise FunctionError(
                    f"function {name}: a tree feature is a number, not {t}"
                )
        if compare_grid not in ("float32", "float64"):
            raise FunctionError(
                f"function {name}: compare_grid {compare_grid!r} is not"
                " 'float32' or 'float64'"
            )
        _check_table(name, "nodes", nodes, _NODE_TYPES)
        _check_table(name, "models", models, _MODEL_TYPES)
        if models.column("model_id").to_pylist() != list(range(models.num_rows)):
            raise FunctionError(f"function {name}: model ids must be dense from 0")
        self.nodes, self.models, self.compare_grid = nodes, models, compare_grid
        self.side_effects = False
        self.fn = self._walk
        self._ints = [t == pa.int64() for t in takes.types]
        self._forest = self._unpack()

    @property
    def instances(self) -> range:
        """The model ids. Its presence is the protocol's marker for the
        implicit leading id argument."""
        return range(self.models.num_rows)

    def tree_tables(self) -> tuple[pa.Table, pa.Table, str]:
        """The engine hook: its presence routes calls to the native kernel."""
        return self.nodes, self.models, self.compare_grid

    def _params(self) -> list[str]:
        return ["BIGINT", *super()._params()]

    def _unpack(self) -> list[tuple[float, str, str, list[list[tuple]]]]:
        """Per model `(base, agg, link, trees)`, a tree being its node tuples
        `(feature, threshold, left, right, missing_left, value)` by node id.
        Structural validity is the engine's to refuse at build."""
        n = self.nodes.to_pydict()
        h = self.models.to_pydict()
        trees: list[list[list[tuple]]] = [[] for _ in h["model_id"]]
        last = None
        for i, (m, t) in enumerate(zip(n["model_id"], n["tree_id"], strict=True)):
            if not 0 <= m < len(trees):
                raise FunctionError(
                    f"function {self.name}: node row {i} names model {m}"
                )
            if (m, t) != last:
                trees[m].append([])
                last = (m, t)
            trees[m][-1].append(
                (
                    n["feature"][i],
                    n["threshold"][i],
                    n["left"][i],
                    n["right"][i],
                    n["missing_left"][i],
                    n["value"][i],
                )
            )
        return [
            (b, a, ln, ts)
            for b, a, ln, ts in zip(h["base"], h["agg"], h["link"], trees, strict=True)
        ]

    def _feature(self, i: int, v: Any) -> float:
        if v is None:
            return math.nan
        if self._ints[i] and self.compare_grid == "float32":
            return _int_to_f32(v)
        return float(v)

    def _walk(self, iid: int | None, *feats: Any) -> tuple | None:
        if iid is None:
            return None
        if not 0 <= iid < len(self._forest):
            raise ValueError(f"predict: no model with id {iid}")
        x = [self._feature(i, v) for i, v in enumerate(feats)]
        base, agg, link, trees = self._forest[iid]
        acc = base if agg == "sum" else 0.0
        for nodes in trees:
            f, thr, left, right, miss, value = nodes[0]
            while f >= 0:
                v = x[f]
                go_left = miss if math.isnan(v) else v <= thr
                f, thr, left, right, miss, value = nodes[left if go_left else right]
            acc += value
        if agg == "mean":
            acc = base + acc / len(trees)
        if link == "sigmoid":
            try:
                e = math.exp(-acc)
            except OverflowError:
                e = math.inf
            acc = 1.0 / (1.0 + e)
        return (acc,)
