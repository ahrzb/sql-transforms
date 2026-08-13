"""The compiled two-parameter text: resolution, binding, ``Fitted``, ``Program``.

``__FIT__`` and ``__THIS__`` are the two parameters. ``Program.fit`` binds one
and ``Program.run`` binds both to the same relation. Which half is learned and
which is live is read off the text — there is no annotation to remember and
none to forget.

No estimator surface: this module is the part of ``SQLTransform`` that
``SQLProjection`` also needs, held as a value rather than inherited (TASK-87,
`docs/superpowers/specs/2026-08-11-row-wise-projections-design.md`). The parts
it stands on live next door: ``_ast`` (the oracle as parser and printer),
``_analysis`` (what a subtree reads), ``_plan`` (freezing), ``_foreign`` (the
supplied pair), ``_errors``.

``compile`` takes ``scope`` as a parameter rather than reading the stack:
each public class reads its own caller with ``sys._getframe(1)`` and passes
the mapping in. Moving the frame read here would silently break every
``FROM df`` replacement-scan idiom.
"""

import itertools
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
    _aliased,
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
    Opaque,
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


def _surface() -> type:
    """The estimator class, imported late: ``_transform`` imports this module,
    so importing it at the top would be circular. A member in the caller's
    frame is an ``SQLTransform``; the splice reads its compiled attributes."""
    from sql_transform.model._transform import SQLTransform  # noqa: PLC0415

    return SQLTransform


def _projection_type() -> type:
    """The projection class, late for the same circularity reason: a stem
    resolving to one is spliced as a leaf (``_leaf``), never registered."""
    from sql_transform.model._projection import SQLProjection  # noqa: PLC0415

    return SQLProjection


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
    if not isinstance(member, _surface()):
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

    def foreign_call(call: Function) -> Node:
        """``x_fit``/``x_transform``: the stem resolves, the suffix says half.
        A bare ``x`` resolving to a projection is the ONE sugar —
        ``x_transform(x_fit(...) OVER (), ...)``, the global fit scope."""
        name = call.function_name
        whole = scope.get(name)
        if isinstance(whole, _projection_type()):
            from sql_transform.model import _leaf  # noqa: PLC0415

            captured[name] = whole
            out = _leaf.bare_call(name, whole, call)
            return _aliased(out, call.alias) if call.alias else out
        stem, _, half = name.rpartition("_")
        member = scope.get(stem) if half in ("fit", "transform") else None
        if isinstance(member, _projection_type()):
            # A projection leaf is spliced, never registered (D2): both
            # halves become ordinary SQL, and θ carries the parameters. The
            # author's alias survives the rewrite — it names their column.
            from sql_transform.model import _leaf  # noqa: PLC0415

            captured[stem] = member
            if half == "fit":
                out = _leaf.fit_call(stem, member, list(call.children), None)
            else:
                out = _leaf.transform_call(stem, member, call)
            return _aliased(out, call.alias) if call.alias else out
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
        # Output columns too: an authored `AS __cf_row` would collide with the
        # ordinal a projection threads through the spine, silently — the same
        # P8 hole `_reserve` already closes for relations and aliases.
        for item in node_field(node, "select_list") or []:
            if alias := node_field(item, "alias"):
                _reserve(alias, "an output column named")

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
                        case obj if isinstance(obj, _surface()):
                            raise TransformError(
                                f"{name} is a transform; call it as "
                                f"{name}({FIT}, {THIS})"
                            )
                        case obj if isinstance(obj, _projection_type()):
                            raise TransformError(
                                f"{name} is a projection; use its halves, "
                                f"{name}_fit(...) and {name}_transform(...)"
                            )
                        case obj:
                            captured[name] = obj
            return None

        node = rebuild(node, resolve_ref, deep=False)

        # Member calls are gone by now, so every FUNCTION left at this level is
        # a scalar one — no need to tell the table call's own function apart.
        def scalar_call(v: AstNode) -> AstNode | None:
            # A projection's fit half under OVER parses as a window of an
            # unknown aggregate — an Opaque, so the Function branch below
            # never sees it. The author's window rides onto every θ field.
            if isinstance(v, Opaque) and v.fields.get("class") == "WINDOW":
                name = str(v.fields.get("function_name") or "")
                stem, _, half = name.rpartition("_")
                member = scope.get(stem) if half == "fit" else None
                if isinstance(member, _projection_type()):
                    from sql_transform.model import _leaf  # noqa: PLC0415

                    captured[stem] = member
                    out = _leaf.fit_call(
                        stem, member, list(v.fields.get("children") or []), v
                    )
                    alias = str(v.fields.get("alias") or "")
                    return _aliased(out, alias) if alias else out
                # `p(x) OVER w` — the deleted sugar (fit-transform-split
                # spec: no oracle reading). Refused here by name, not left
                # for DuckDB to reject as an unknown aggregate at fit.
                if isinstance(scope.get(name), _projection_type()):
                    raise TransformError(
                        f"{name} is a projection, and a fit scope is spelled "
                        f"on the fit half: "
                        f"{name}_transform({name}_fit(...) OVER (...), ...)"
                    )
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


@dataclass(frozen=True, slots=True)
class Program:
    """The compiled two-parameter text, as a value.

    Everything construction learns — the resolved text, the fit DAG, the
    residual, the captured environment — and the two bindings over it:
    ``fit`` binds ``__FIT__``, ``run`` binds both parameters to one relation.

    A value rather than a base class: ``SQLTransform`` adds the sklearn
    surface and ``SQLProjection`` adds the row-wise gates, and neither wants
    what the other adds. Both hold one of these.
    """

    node: Node  # resolved text, both parameters live
    depth: int  # member-call nesting, bounded by MAX_DEPTH
    steps: list[tuple[str, Node]]  # the fit DAG, in dependency order
    residual: Node
    shadowable: set[str]
    bindings: Bindings
    foreign: Foreign
    captured: Captured
    source: str  # the exact object given, so clone's identity check passes
    sql: str  # the resolved text, printed
    connection: Connection | None

    @classmethod
    def compile(
        cls,
        sql: str,
        scope: dict[str, Any],
        *,
        connection: Connection | None = None,
        captured: Captured | None = None,
    ) -> Self:
        """Parse, splice, resolve and plan; every refusal fires here.

        ``captured`` is *adopted*, not copied, and completed in place with
        whatever ``scope`` supplied: sklearn's ``clone`` demands that
        ``get_params`` hand back the very object the constructor was given,
        and carrying the completed set is the whole point.
        """
        doc = _parse(sql)
        if len(doc.statements) != 1:
            raise TransformError(
                f"a transform is one statement, got {len(doc.statements)}"
            )
        captured = {} if captured is None else captured
        scope = scope | captured

        doc, depth = _resolve(doc, scope, captured, _catalog(connection), connection)
        # Two runtime views. A member or a projection leaf is spliced away,
        # so it is neither.
        foreign: Foreign = {
            k: v for k, v in captured.items() if isinstance(v, Transform)
        }
        bindings: Bindings = {
            k: v
            for k, v in captured.items()
            if not isinstance(v, (Transform, _surface(), _projection_type()))
        }
        # No copy: the models are frozen, so `_plan` cannot reach back into
        # `doc` and `node` stays the text the caller wrote.
        steps, residual, shadowable = _plan(doc)
        return cls(
            node=doc.statements[0].node,
            depth=depth,
            steps=steps,
            residual=residual,
            shadowable=shadowable,
            bindings=bindings,
            foreign=foreign,
            captured=captured,
            source=sql,
            sql=_deserialize(doc),
            connection=connection,
        )

    def fit(self, data: Relation) -> Fitted:
        """Partial application: evaluate the fit DAG, return the artifact.

        Every step runs under leased names and they are all given back, so a
        shared connection is the caller's again when this returns — including
        ``__FIT__``, which used to stay bound to the whole training relation
        for the life of the connection.
        """
        registry = _Registry()
        own = self.connection is None
        con = duckdb.connect() if own else self.connection
        params: Params = {}
        names, stems, release = _lease(
            con,
            {FIT: data} | self.bindings,
            self.foreign,
            registry,
            rename=not own,
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
                self.shadowable,
            )
            for param, node in self.steps:
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
                names[param] = param if own else f"{param}__x{next(_EXECUTIONS)}"
                con.register(names[param], params[param])
        finally:
            release()
        return Fitted(
            self.residual,
            params,
            self.bindings,
            self.foreign,
            registry.instances,
            self.connection,
        )

    def run(self, data: Relation) -> pa.Table:
        """Both parameters bound to the same relation, with no freezing at all.

        The reference side of "freezing is faithful". It is a *binding*, not a
        rewrite, which is what keeps that law from restating the implementation.
        """
        registry = _Registry()
        own = self.connection is None
        con = duckdb.connect() if own else self.connection
        names, stems, release = _lease(
            con,
            {FIT: data, THIS: data} | self.bindings,
            self.foreign,
            registry,
            rename=not own,
        )
        try:
            return _execute(con, _rendered(self.node, names, stems), registry)
        finally:
            release()
