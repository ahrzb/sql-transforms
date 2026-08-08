"""A transform is a function ``(F, T) -> R`` over relations.

``__FIT__`` and ``__THIS__`` are its two parameters. At the top level ``fit``
binds one and ``transform`` binds the other. Which half is learned and which
is live is read off the text — there is no annotation to remember and none to
forget.

This module is the surface — resolution, binding and the two classes. The
parts it stands on live next door: ``_ast`` (the oracle as parser and printer),
``_analysis`` (what a subtree reads), ``_plan`` (freezing), ``_foreign`` (the
supplied pair), ``_errors``.

Implements `docs/superpowers/specs/2026-08-07-datamodel-redesign-design.md`.
DuckDB is both the parser and the oracle — a construct means what DuckDB
computes.
"""

import itertools
import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import duckdb
import pyarrow as pa

from sql_transform.model._ast import (
    _ALL_FUNCTIONS,
    _TABLE_FUNCTIONS,
    FIT,
    THIS,
    Bindings,
    Captured,
    Connection,
    LazyRelation,
    Params,
    Relation,
    _base_table,
    _bind_parameters,
    _catalog,
    _deserialize,
    _functions,
    _is_recursive_cte,
    _list_of,
    _parse,
    _rename_free,
    _rename_functions,
    _statement,
    _subquery_ref,
    _table_function_ref,
)
from sql_transform.model._correlate import refuse_if_shadowed
from sql_transform.model._errors import (
    NestingTooDeep,
    NotFitted,
    TransformError,
    UnknownName,
)
from sql_transform.model._foreign import (
    Foreign,
    Transform,
    _execute,
    _Registry,
)
from sql_transform.model._nodes import (
    AstNode,
    BaseTable,
    ColumnRef,
    Document,
    Function,
    Node,
    SubqueryExpr,
    TableFunction,
    cte_entries,
    descendants,
    is_query,
    is_ref,
    rebuild,
    with_cte_entries,
)
from sql_transform.model._nodes import field as node_field
from sql_transform.model._plan import _plan

MAX_DEPTH = 8


def _splice(
    call: TableFunction, scope: dict[str, Any], captured: Captured
) -> tuple[Node, int]:
    """A member call, as the spliced relation it denotes, and its depth.

    Splice, never emit a DuckDB macro: measured, a table macro invoked under
    ``LATERAL`` does not see the correlation and silently returns the
    whole-table answer for every group.
    """
    function = call.function
    name = function.function_name
    member = scope.get(name)
    if member is None:
        raise UnknownName(
            f"{name} is not a table function and resolves to nothing in the "
            "caller's frame"
        )
    if not isinstance(member, SQLTransform):
        raise TransformError(
            f"{name} resolves to a {type(member).__name__}, not a transform"
        )
    args = function.children
    if len(args) != 2:
        raise TransformError(
            f"a transform takes two arguments ({FIT}, {THIS}); "
            f"{name} was called with {len(args)}"
        )

    depth = member.depth
    bound = {}
    for parameter, arg in zip((FIT, THIS), args, strict=True):
        relation, arg_depth = _argument(arg, scope, captured)
        bound[parameter] = relation
        depth = max(depth, arg_depth)
    if depth + 1 > MAX_DEPTH:
        raise NestingTooDeep(
            f"{name} nests deeper than {MAX_DEPTH} levels of member calls"
        )

    captured[name] = member  # spliced away, but a clone has to find it again

    body = member.node
    renames = {}
    for free, obj in member.bindings.items():
        renames[free] = f"{name}__{free}"
        captured[f"{name}__{free}"] = obj
    body = _rename_free(body, renames)

    function_renames = {}
    for stem, leaf in member.foreign.items():
        function_renames[stem] = f"{name}__{stem}"
        captured[f"{name}__{stem}"] = leaf
    body = _rename_functions(body, function_renames)

    body = _bind_parameters(body, bound)

    # Returned rather than smuggled back on the node: the old version parked
    # `_depth` on the ref dict for the caller to pop, which a typed node has
    # nowhere to put and nothing should have relied on anyway.
    return _subquery_ref(body, node_field(call, "alias", "") or ""), depth + 1


def _argument(arg: Node, scope: dict[str, Any], captured: Captured) -> tuple[Node, int]:
    """An argument expression, as the relation it denotes."""
    match arg:
        case ColumnRef(column_names=[name]):
            return _base_table(name), 0
        case SubqueryExpr(subquery=box):
            return _subquery_ref(box.node, ""), 0
        case Function():
            return _splice(_table_function_ref(arg, ""), scope, captured)
    raise TransformError(
        f"a transform argument is a relation — {FIT}, {THIS}, a parenthesised "
        "query, or another transform call"
    )


RESERVED = "__"


def _reserve(name: str, what: str) -> None:
    """Refuse a name under the model's own prefix.

    P8, finally implemented for this model: everything synthesized lives under
    ``__`` — ``__param_0``, ``__param_fit``, ``{name}__x{token}`` — so an
    authored name there can silently mean the model's relation instead of the
    author's. It did: a captured binding called ``__param_0`` lost to the
    frozen parameter with no error at all.

    The whole prefix rather than ``__param_`` alone, so nothing has to be kept
    in step as more names get synthesized. ``__FIT__`` and ``__THIS__`` are
    the exception — they are the two parameters, and are the only ``__`` names
    an author may write.
    """
    if name.startswith(RESERVED) and name.upper() not in (FIT, THIS):
        raise TransformError(
            f"{what} {name!r} starts with {RESERVED!r}, which is reserved: "
            f"every name the model synthesizes lives there. Only {FIT} and "
            f"{THIS} are yours to write."
        )


def _resolve(
    doc: Document,
    scope: dict[str, Any],
    captured: Captured,
    catalog: frozenset[str] = frozenset(),
    con: Connection | None = None,
) -> tuple[Document, int]:
    """Splice every member call and resolve every free name.

    Returns the rewritten document and the nesting depth. Children are resolved
    before their parent, so a splice at this level always grafts an
    already-resolved body.
    """
    depth = 0

    def foreign_call(call: Function) -> Function:
        """``x_fit``/``x_transform``: the stem resolves, the suffix says half."""
        name = call.function_name
        stem, _, half = name.rpartition("_")
        member = scope.get(stem) if half in ("fit", "transform") else None
        if not isinstance(member, Transform):
            raise UnknownName(
                f"{name} is not a DuckDB function, and "
                + (
                    f"{stem} resolves to nothing in the caller's frame"
                    if member is None
                    else f"{stem} resolves to a {type(member).__name__}, "
                    "not a Transform"
                )
            )
        captured[stem] = member
        if half != "fit":
            return call
        # The UDAF half is a scalar function over a collected list.
        return call.model_copy(
            update={"children": [_list_of(child) for child in call.children]}
        )

    def walk(node: Node, ctes: frozenset[str]) -> Node:
        nonlocal depth
        rewritten = []
        for entry in cte_entries(node):
            # DuckDB would let such a CTE win, and we would go on rewriting the
            # reference to the training set — two meanings for one name, and
            # the row count changed with no error. Refused where it is
            # defined, so `__FIT__` means the parameter everywhere or the text
            # does not compile.
            if entry.key.upper() in (FIT, THIS):
                raise TransformError(
                    f"a CTE may not be named {entry.key!r}: {FIT} and "
                    f"{THIS} are the transform's two parameters"
                )
            _reserve(entry.key, "a CTE named")
            body = entry.value.query.node
            # A RECURSIVE CTE is in scope inside its own body; a plain one is
            # not, where the same name means whatever the caller's frame binds.
            # The inner node type is the only thing that tells them apart.
            visible = ctes
            if _is_recursive_cte(body):
                visible = ctes | {entry.key.lower()}
            body = walk(body, visible)
            rewritten.append(
                entry.model_copy(
                    update={
                        "value": entry.value.model_copy(
                            update={
                                "query": entry.value.query.model_copy(
                                    update={"node": body}
                                )
                            }
                        )
                    }
                )
            )
            # Folded, because DuckDB's binder is case-insensitive: `WITH Sales`
            # then `FROM sales` resolves for the oracle, and comparing exact
            # strings refused valid SQL as an unknown free name.
            ctes = ctes | {entry.key.lower()}
        node = with_cte_entries(node, rewritten)

        node = rebuild(
            node, lambda v: walk(v, ctes) if is_query(v) else None, deep=False
        )

        for v in descendants(node, deep=False):
            if is_ref(v):
                if alias := node_field(v, "alias"):
                    _reserve(alias, "an alias named")
                named = node_field(v, "table_name")
                if named and named.lower() not in ctes:
                    _reserve(named, "a relation named")

        def resolve_ref(v: AstNode) -> AstNode | None:
            nonlocal depth
            match v:
                case TableFunction(function=Function(function_name=call)):
                    if call.lower() in _functions(_TABLE_FUNCTIONS, con):
                        return None
                    ref, at = _splice(v, scope, captured)
                    depth = max(depth, at)
                    return ref
                # A qualified name is the connection's own, and the catalog
                # listing is not the test for it: everything captured is
                # registered under a bare name, so `side.main.far` can never
                # mean a frame object however the listing is filtered. Without
                # a connection there is no catalog for it to be in — `fit`
                # makes a fresh one per call — so that refuses here rather than
                # at fit in DuckDB's words.
                case BaseTable(table_name=name) if v.schema_name or v.catalog_name:
                    if con is None:
                        path = ".".join(
                            p for p in (v.catalog_name, v.schema_name, name) if p
                        )
                        raise UnknownName(
                            f"{path} is qualified, so it names something in a "
                            "catalog, and this transform has none of its own; "
                            "pass connection= to say whose"
                        )
                # Folded against `ctes` and `catalog` because DuckDB binds
                # those; *not* against `captured` and `scope`, which are
                # Python's own namespace, where `codes` and `Codes` are two
                # different variables and folding would merge them.
                case BaseTable(table_name=name) if (
                    name not in (FIT, THIS)
                    and name.lower() not in ctes
                    and name not in captured
                    and name.lower() not in catalog
                ):
                    match scope.get(name):
                        case None:
                            raise UnknownName(
                                f"{name} resolves to nothing in the caller's frame"
                            )
                        case SQLTransform():
                            raise TransformError(
                                f"{name} is a transform; call it as "
                                f"{name}({FIT}, {THIS})"
                            )
                        case obj:
                            captured[name] = obj
            return None

        node = rebuild(node, resolve_ref, deep=False)

        # Member calls are gone by now, so every FUNCTION left at this level is
        # a scalar one — no need to tell the table call's own function apart.
        def scalar_call(v: AstNode) -> AstNode | None:
            if (
                isinstance(v, Function)
                and not v.is_operator
                and v.function_name.lower() not in _functions(_ALL_FUNCTIONS, con)
            ):
                return foreign_call(v)
            return None

        return rebuild(node, scalar_call, deep=False)

    box = doc.statements[0]
    resolved = walk(box.node, frozenset())
    return doc.model_copy(
        update={"statements": [box.model_copy(update={"node": resolved})]}
    ), depth


def _give_back(leases: list[Callable[[], None]]) -> None:
    """Release every lease in the list, once.

    Module-level and taking the *list* rather than the ``Fitted``, because a
    finalizer that closes over the object keeps it alive: it never becomes
    unreachable, so the finalizer never runs and the release silently never
    happens. The list is shared with the Fitted; holding it strongly is fine.
    """
    for release in leases:
        release()
    leases.clear()


@dataclass(slots=True, eq=False, repr=False, weakref_slot=True)
class Fitted:
    """``T -> R``, with the captured environment reified as data.

    A plain closure would be type-correct and unshippable — it could retain
    the whole training set and nothing outside could tell. ``params`` makes
    that a measurement instead of a rule.
    """

    node: Node
    params: Params
    bindings: Bindings
    foreign: Foreign
    instances: dict[int, Any]
    connection: Connection | None = None
    # Leases handed out by `relation()` and not yet given back. See `relation`.
    _leases: list[Callable[[], None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        weakref.finalize(self, _give_back, self._leases)

    def __repr__(self) -> str:
        # The generated one prints the whole residual AST: unreadable, and
        # it buries the two numbers that actually say what you are holding.
        shape = ", ".join(f"{k}[{len(v)}]" for k, v in self.params.items())
        return f"Fitted(params={shape or 'none'}, instances={len(self.instances)})"

    @property
    def sql(self) -> str:
        """The residual, under the names ``params`` uses. What you read."""
        return _deserialize(_statement(self.node))

    def _bind(
        self, data: Relation
    ) -> tuple[Connection, str, _Registry, Callable[[], None]]:
        """Register everything this residual needs, and say what to execute
        and how to clean up.

        On a connection we own, names stay the readable ones — a fresh
        connection per call makes collisions impossible, and it dies with the
        call, so there is nothing to give back. On a *shared* connection every
        execution is renamed and leased; see ``_lease``.
        """
        registry = _Registry(self.instances)
        tables = {THIS: data} | self.bindings | self.params
        own = self.connection is None
        con = duckdb.connect() if own else self.connection
        names, stems, release = _lease(
            con, tables, self.foreign, registry, rename=not own
        )
        return con, _rendered(self.node, names, stems), registry, release

    def transform(self, data: Relation) -> pa.Table:
        con, sql, registry, release = self._bind(data)
        try:
            return _execute(con, sql, registry)
        finally:
            release()

    __call__ = transform

    def relation(self, data: Relation) -> LazyRelation:
        """The residual as an unexecuted ``DuckDBPyRelation``.

        Nothing is materialised: bind, plan, hand it back. DuckDB still
        *binds* eagerly, so an unknown column refuses here; a foreign
        transform's refusal only surfaces when the relation is consumed, which
        is the price of not materialising.

        This is the one path that cannot release at the end of the call — the
        tables have to outlive it or there would be nothing left to execute.

        The lease therefore lives on *this artifact*, not on the relation.
        Tying it to the relation was wrong and crashed: a relation derived
        from this one still needs the tables but holds no reference to its
        parent, so ``t.transform(D).limit(2)`` lost them the moment the parent
        was collected.

        The cost is real and bounded rather than free — one registration per
        outstanding relation until this artifact is released, refit or
        dropped. ``release()`` is the deterministic way out; the eager
        ``transform`` path never accumulates at all, and is the right tool for
        serving in a loop.

        A relation belongs to the connection that built it and cannot be
        handed to another one, not even to a cursor of the same connection.
        Chaining lazily therefore means giving both transforms the same
        ``connection=``.
        """
        con, sql, _, release = self._bind(data)
        self._leases.append(release)
        return con.sql(sql)

    def release(self) -> None:
        """Give back every table this artifact still has registered.

        Only lazy output leaves anything to give back. Idempotent, so a caller
        can put it in a ``finally`` without checking.
        """
        _give_back(self._leases)


OUTPUTS = ("default", "arrow", "duckdb", "pandas", "numpy")


# One counter per process, so a shared connection never sees two executions
# under the same name. See _lease.
_EXECUTIONS = itertools.count()


def _lease(
    con: Connection,
    tables: Bindings,
    foreign: Foreign,
    registry: _Registry,
    *,
    rename: bool,
) -> tuple[dict[str, str], dict[str, str], Callable[[], None]]:
    """Register one execution's tables and functions, and say how to give
    them back.

    Two rules, both learned the hard way. **Renamed**, because two transforms
    sharing a connection both bind ``__THIS__`` and both call a parameter
    ``__param_0``; eagerly that is harmless, but a lazy relation is not
    executed yet, so one stage would read the other's tables — same shape,
    different numbers, no error. **Released**, because the rename alone turned
    that correctness bug into a resource one: every execution added names
    nobody ever took away, so a serving loop pinned every batch it had seen,
    and the leftovers were visible to ``_catalog``, which made the *next*
    transform bind to them instead of capturing from its caller's frame.

    Returned rather than a context manager because the lazy path cannot
    release at the end of the call — it releases when the relation it handed
    back is collected.

    ``names`` is live: ``fit`` adds each parameter as it lands, and the
    release closes over the same dict, so those come back too.
    """
    token = next(_EXECUTIONS)

    def under(name: str) -> str:
        return f"{name}__x{token}" if rename else name

    names = {name: under(name) for name in tables}
    stems = {stem: under(stem) for stem in foreign}
    for name, table in tables.items():
        con.register(names[name], table)
    for stem, leaf in foreign.items():
        leaf.register(con, stems[stem], registry)

    def release() -> None:
        if not rename:
            return  # a connection we own dies with the call; θ keeps its name
        for name in names.values():
            con.unregister(name)
        for stem in stems.values():
            con.remove_function(f"{stem}_fit")
            con.remove_function(f"{stem}_transform")

    return names, stems, release


def _rendered(node: Node, names: dict[str, str], stems: dict[str, str]) -> str:
    """``node`` under the names this execution actually registered."""
    doc = _rename_free(_statement(node), names)
    return _deserialize(_rename_functions(doc, stems))


def _as_output(
    table: pa.Table, output: str, source: Relation = None, aligned: bool = True
) -> Any:
    """The result in the caller's currency.

    ``pandas`` carries the caller's index when there is one to carry, which is
    what every sklearn transformer does. Resetting it was silent: in a
    ``FeatureUnion`` alongside an estimator that preserves the index, pandas
    aligns on index rather than position and NaN-pads the difference — four
    rows in, seven out, no error.

    A SQL transform may change cardinality, which an sklearn one cannot. When
    the row counts disagree there is no row correspondence to express, so no
    index is attached rather than one invented.
    """
    match output:
        case "default" | "arrow" | "duckdb":
            return table
        case "pandas":
            return _with_index(table.to_pandas(), source, aligned)
        case "numpy":
            return _with_index(table.to_pandas(), source, aligned).to_numpy()
    raise TransformError(f"output must be one of {OUTPUTS}; got {output!r}")


def _keeps_row_order(node: Node) -> bool:
    """Whether output row *i* still stands for input row *i*.

    An index is a claim about which input row each output row came from, and
    positional correspondence is the only evidence available. Any ORDER BY or
    LIMIT in the residual destroys it — measured, and silently: a three-row
    frame indexed a/b/c through ``ORDER BY v`` came back with a's label on b's
    value, no error.

    The query's own ORDER BY / LIMIT lives in the top-level ``modifiers``,
    which is what this reads. A deep scan is wrong: an ordinary projection
    carries an empty nested ORDER_MODIFIER, so scanning everything never
    carries an index at all.

    Even so this is a good-faith reading rather than a proof — SQL guarantees
    no row order without ORDER BY. Losing the index is loud (a FeatureUnion
    misaligns visibly) while attaching a wrong one is not, so where the two
    compete the doubt resolves toward dropping it.
    """
    return not node_field(node, "modifiers")


def _with_index(frame: Any, source: Relation, aligned: bool) -> Any:
    index = getattr(source, "index", None)
    if aligned and index is not None and len(index) == len(frame):
        frame.index = index
    return frame


class SQLTransform:
    """``F -> Fitted``, and an sklearn estimator.

    ``fit`` returns the ``Fitted`` artifact rather than ``self``. That is the
    currying the model is built on — ``.params`` is a thing you can ship —
    and it costs nothing with sklearn, which never reads what ``fit``
    returned: ``Pipeline`` keeps the object it called and asks *it* to
    ``transform`` later. So ``fit`` also remembers, and both spellings agree:

        t.fit(D).transform(X)     # curried: the artifact transforms
        t.fit(D); t.transform(X)  # stateful: the estimator transforms

    ``bindings`` and ``foreign`` are constructor parameters as well as frame
    lookups, because ``clone`` rebuilds an estimator inside sklearn's own
    frame, where a member or a lookup table is not in scope. They ride along
    in ``get_params`` so a clone resolves to the very same objects.

    Those two mappings are *adopted*, not copied, and completed in place with
    whatever the frame supplied. ``clone`` demands that ``get_params`` hand
    back the very object the constructor was given — a defensive copy fails
    its identity check — and carrying the completed set is the whole point.

    Construction parses, plans and refuses; nothing else does.
    """

    def __init__(
        self,
        sql: str,
        output: str = "default",
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> None:
        doc = _parse(sql)
        if len(doc.statements) != 1:
            raise TransformError(
                f"a transform is one statement, got {len(doc.statements)}"
            )
        # Resolution happens once, here, and captures by value: `scope` is a
        # local that dies with this call, so no caller frame is retained and
        # rebinding a member afterwards cannot change what was built.
        frame = sys._getframe(1)
        scope = frame.f_globals | frame.f_locals
        del frame

        if output not in OUTPUTS:
            raise TransformError(f"output must be one of {OUTPUTS}; got {output!r}")
        self.output = output
        # Given rather than conjured. A transform that makes its own hidden
        # connection cannot compose with anything: a DuckDBPyRelation belongs
        # to the connection that built it, so lazy output only chains when
        # both stages share one. Pass it and you own it.
        self.connection = connection
        # Adopted, not copied: see the class docstring. Explicit entries win,
        # and are how a clone keeps names the frame it was rebuilt in cannot
        # see.
        self.captured: Captured = {} if captured is None else captured
        scope = scope | self.captured

        doc, self.depth = _resolve(
            doc, scope, self.captured, _catalog(connection), connection
        )
        # Two runtime views. A member is spliced away, so it is neither.
        self.foreign: Foreign = {
            k: v for k, v in self.captured.items() if isinstance(v, Transform)
        }
        self.bindings: Bindings = {
            k: v
            for k, v in self.captured.items()
            if not isinstance(v, Transform | SQLTransform)
        }
        self.node = doc.statements[0].node
        self.source = sql  # the exact object, so clone's identity check passes
        self.sql = _deserialize(doc)
        # No copy: the models are frozen, so `_plan` cannot reach back into
        # `doc` and `self.node` stays the text the caller wrote.
        self._steps, self._residual, self._shadowable = _plan(doc)
        self._own = connection is None
        self.fitted_: Fitted | None = None
        self.feature_names_out_: list[str] | None = None

    # -- the model's own surface ----------------------------------------------

    def __repr__(self) -> str:
        state = "fitted" if self.fitted_ is not None else "unfitted"
        return f"SQLTransform({self.sql!r}, output={self.output!r}, {state})"

    def _connect(self) -> Connection:
        return duckdb.connect() if self._own else self.connection

    def fit(self, data: Relation, y: Any = None) -> Fitted:
        """Partial application — and the estimator remembers the result.

        ``y`` is accepted and ignored: a target belongs in the relation, as a
        column ``__FIT__`` can read, not in a second argument the SQL cannot
        name.

        Every step runs under leased names and they are all given back, so a
        shared connection is the caller's again when this returns — including
        ``__FIT__``, which used to stay bound to the whole training relation
        for the life of the connection.
        """
        registry = _Registry()
        con = self._connect()
        params: Params = {}
        names, stems, release = _lease(
            con,
            {FIT: data} | self.bindings,
            self.foreign,
            registry,
            rename=not self._own,
        )
        try:
            # Before any step: a lifted correlation read some qualifier as
            # *outer*, and if `__FIT__` turns out to have a nested column of
            # that name DuckDB would have bound it inward instead. The AST
            # cannot tell; the schema can, and this is the first place it
            # exists.
            refuse_if_shadowed(
                lambda: [
                    (name, kind)
                    for name, kind, *_ in con.execute(
                        f'DESCRIBE SELECT * FROM "{names[FIT]}"'  # noqa: S608
                    ).fetchall()
                ],
                self._shadowable,
            )
            for param, node in self._steps:
                try:
                    params[param] = _execute(
                        con, _rendered(node, names, stems), registry
                    )
                except duckdb.Error as exc:
                    # A leaf's own refusal comes back through here wearing
                    # DuckDB's coat: a Python exception raised inside a UDF is
                    # rewrapped as InvalidInputException. `_Registry` kept the
                    # original precisely so a refusal keeps its name, and
                    # dressing it up as a correlation problem was both wrong
                    # and unactionable.
                    if registry.error is not None:
                        raise registry.error from exc
                    # Whether an *unqualified* name resolves inward or outward
                    # cannot be known at construction — `__FIT__` has no schema
                    # until there is data — so this is the one refusal that
                    # cannot be hoisted to P7's construction time. It can at
                    # least carry our name rather than DuckDB's.
                    raise TransformError(
                        f"{param}: this {FIT} subquery does not stand on its "
                        f"own, so it cannot be evaluated once into a table "
                        f"({exc}). If the name comes from the outer query it "
                        f"is a correlated {FIT} subquery; qualifying it "
                        f"(f.x = t.x) makes that a refusal at construction."
                    ) from exc
                # Into the same dict the release closes over: later steps see
                # it, and it comes back with everything else.
                names[param] = param if self._own else f"{param}__x{next(_EXECUTIONS)}"
                con.register(names[param], params[param])
        finally:
            release()
        self.fitted_ = Fitted(
            self._residual,
            params,
            self.bindings,
            self.foreign,
            registry.instances,
            self.connection,
        )
        return self.fitted_

    __call__ = fit

    @property
    def params_(self) -> Params:
        return self._require_fit().params

    @property
    def instances_(self) -> dict[int, Any]:
        return self._require_fit().instances

    # -- the sklearn surface ---------------------------------------------------

    def _require_fit(self) -> Fitted:
        if self.fitted_ is None:
            raise NotFitted("this transform has not been fit; call fit first")
        return self.fitted_

    def transform(self, data: Relation) -> Any:
        fitted = self._require_fit()
        if self.output == "duckdb":
            lazy = fitted.relation(data)  # the whole point: never materialise
            self.feature_names_out_ = list(lazy.columns)
            return lazy
        out = fitted.transform(data)
        self.feature_names_out_ = out.column_names
        return _as_output(out, self.output, data, _keeps_row_order(fitted.node))

    def fit_transform(self, data: Relation, y: Any = None) -> Any:
        """On the training relation this is exactly ``run(t, D)`` — that is
        the *freezing is faithful* law, not a coincidence."""
        self.fit(data)
        return self.transform(data)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        if self.feature_names_out_ is None:
            raise NotFitted(
                "output column names are only known once something has been "
                "transformed; call transform or fit_transform first"
            )
        return list(self.feature_names_out_)

    def set_output(self, *, transform: str | None = None) -> Self:
        """sklearn's opt-in: ``pandas`` or ``numpy`` for a downstream
        estimator, ``default`` for the model's own arrow tables."""
        if transform is not None:
            if transform not in OUTPUTS:
                raise TransformError(
                    f"output must be one of {OUTPUTS}; got {transform!r}"
                )
            self.output = transform
        return self

    def __sklearn_clone__(self) -> Self:
        """sklearn's own hook, because the default clones by deep-copying
        every parameter and a live DuckDB connection cannot be deep-copied —
        ``clone``, and so ``GridSearchCV``/``cross_val_score``/``Pipeline``,
        died with a raw TypeError on any transform built with ``connection=``.

        A connection is a resource, not a value: the clone shares it. Rebuilt
        from ``source`` so the plan is derived rather than copied.
        """
        return type(self)(
            self.source,
            output=self.output,
            connection=self.connection,
            captured=self.captured,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "sql": self.source,
            "output": self.output,
            "connection": self.connection,
            "captured": self.captured,
        }

    def set_params(self, **params: Any) -> Self:
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise TransformError(f"unknown parameters {sorted(unknown)}")
        if {"sql", "captured", "connection"} & set(params):
            # The plan is derived from all three, so rebuild rather than let
            # them drift apart.
            rebuilt = type(self)(
                params.get("sql", self.source),
                output=params.get("output", self.output),
                connection=params.get("connection", self.connection),
                captured=params.get("captured", self.captured),
            )
            self.__dict__.update(rebuilt.__dict__)
        elif "output" in params:
            self.set_output(transform=params["output"])
        return self


def run(transform: SQLTransform, data: Relation) -> pa.Table:
    """Both parameters bound to the same relation, with no freezing at all.

    The reference side of "freezing is faithful". It is a *binding*, not a
    rewrite, which is what keeps that law from restating the implementation.
    """
    registry = _Registry()
    con = transform._connect()
    names, stems, release = _lease(
        con,
        {FIT: data, THIS: data} | transform.bindings,
        transform.foreign,
        registry,
        rename=not transform._own,
    )
    try:
        return _execute(con, _rendered(transform.node, names, stems), registry)
    finally:
        release()
