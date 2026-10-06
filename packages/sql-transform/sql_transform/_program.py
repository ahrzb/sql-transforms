"""The compiled two-parameter text: resolution, binding, ``Fitted``, ``Program``.

``__FIT__`` and ``__THIS__`` are the two parameters. ``Program.fit`` binds one
and ``Program.run`` binds both to the same relation. Which half is learned and
which is live is read off the text — there is no annotation to remember and
none to forget.

This is the sole executor shared by ``SQLTransform`` and ``SQLProjection``,
held as a value rather than inherited. The authoring rules live in
``packages/sql-transform/docs/contract.md``. Resolution, fit freezing, raw
estimator fitting, learned UDF publishing, and registration leases share
this execution path.

``compile`` takes ``scope`` as a parameter rather than reading the stack:
each public class reads its own caller with ``sys._getframe(1)`` and passes
the mapping in. Moving the frame read here would silently break every
``FROM df`` replacement-scan idiom.
"""

import itertools
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Self

import duckdb
import pyarrow as pa
import pyarrow.compute as pc

from sql_transform._ast import (
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
    _aggregates,
    _aliased,
    _base_table,
    _bind_parameters,
    _catalog,
    _deserialize,
    _functions,
    _is_recursive_cte,
    _list_of,
    _parse,
    _print_expr,
    _rename_calls,
    _rename_free,
    _rename_functions,
    _statement,
    _subquery_ref,
    _table_function_ref,
    _template,
    _unaliased,
)
from sql_transform._correlate import refuse_if_shadowed
from sql_transform._errors import (
    NestingTooDeep,
    TransformError,
    UnknownName,
)
from sql_transform._foreign import (
    Foreign,
    Transform,
    _Estimator,
    _execute,
    _Registry,
)
from sql_transform._nodes import (
    AstNode,
    BaseTable,
    ColumnRef,
    Document,
    Function,
    Join,
    Node,
    Opaque,
    Select,
    SubqueryExpr,
    SubqueryRef,
    TableFunction,
    cte_entries,
    descendants,
    is_query,
    is_ref,
    rebuild,
    with_cte_entries,
)
from sql_transform._nodes import field as node_field
from sql_transform._plan import _plan, _referenced
from sql_transform._udf import UDF

MAX_DEPTH = 8


def _surface() -> type:
    """The estimator class, imported late: ``_transform`` imports this module,
    so importing it at the top would be circular. A member in the caller's
    frame is an ``SQLTransform``; the splice reads its compiled attributes."""
    from sql_transform._transform import SQLTransform  # noqa: PLC0415

    return SQLTransform


def _projection_type() -> type:
    """The projection class, late for the same circularity reason: a stem
    resolving to one is spliced as a leaf (``_leaf``), never registered."""
    from sql_transform._projection import SQLProjection  # noqa: PLC0415

    return SQLProjection


def _estimator(value: Any) -> bool:
    """A raw row-local estimator, not one of the model's declared members."""
    return (
        not isinstance(value, (type, _projection_type(), _surface(), Transform, UDF))
        and callable(getattr(value, "fit", None))
        and callable(getattr(value, "transform", None))
    )


def _call_member(scope: dict[str, Any], name: str) -> Any:
    """Function names fold in SQL; ambiguous Python spellings cannot merge."""
    matches = [
        (key, value)
        for key, value in scope.items()
        if key.lower() == name.lower()
        and (
            isinstance(value, (_projection_type(), _surface(), Transform, UDF))
            or _estimator(value)
        )
    ]
    if len(matches) > 1:
        raise TransformError(
            f"{name}: captured function names collide case-insensitively"
        )
    return matches[0][1] if matches else scope.get(name)


def _raw_names(
    node: Node, scope: dict[str, Any], con: Connection | None = None
) -> None:
    """A raw member reserves its fit/apply function names before lowering."""
    vocabulary = _functions(_ALL_FUNCTIONS, con)
    for part in (node, *descendants(node, deep=True)):
        if not (isinstance(part, Function) or node_field(part, "class") == "WINDOW"):
            continue
        name = str(node_field(part, "function_name", "") or "")
        if name in vocabulary and _estimator(_call_member(scope, name)):
            raise TransformError(
                f"{name}: captured function collides with a DuckDB builtin"
            )
        stem, _, half = name.rpartition("_")
        if half not in ("fit", "transform") or not _estimator(
            _call_member(scope, stem)
        ):
            continue
        whole = _call_member(scope, name)
        if isinstance(
            whole, (_projection_type(), _surface(), Transform, UDF)
        ) or _estimator(whole):
            raise TransformError(f"{stem}: {name} is reserved for its fit/apply pair")


def _capture(captured: Captured, name: str, value: Any) -> None:
    """Keep an explicit author's spelling and mapping identity on replay."""
    if not any(
        key.lower() == name.lower() and obj is value for key, obj in captured.items()
    ):
        captured[name] = value


def _raw_bundle(name: str, children: list[Node]) -> list[tuple[str, Node]]:
    """The one raw named-bundle grammar, shared with window derivation."""
    if len(children) != 1:
        raise TransformError(f"{name}: expected one named scalar bundle")
    node = children[0]
    if isinstance(node, ColumnRef) and 1 <= len(node.column_names) <= 2:
        return [(node.column_names[-1], _unaliased(node))]
    if not isinstance(node, Function) or node.function_name.lower() != "struct_pack":
        raise TransformError(
            f"{name}: bundle must be named struct_pack(...) or a bare column"
        )
    names = tuple(str(node_field(v, "alias", "") or "") for v in node.children)
    if not names or any(not n for n in names):
        raise TransformError(f"{name}: bundle fields must be named")
    if len({n.lower() for n in names}) != len(names):
        raise TransformError(f"{name}: bundle field names collide case-insensitively")
    for child in node.children:
        for part in (child, *descendants(child, deep=True)):
            if (
                is_query(part)
                or isinstance(part, ColumnRef)
                and len(part.column_names) > 2
                or node_field(part, "class") == "WINDOW"
                or node_field(part, "type") == "STRUCT_EXTRACT"
                or isinstance(part, Function)
                and part.function_name.lower() in ("struct_pack", "struct_extract")
            ):
                raise TransformError(
                    f"{name}: nested bundles and field access are not scalar features"
                )
    return [
        (field, _unaliased(child))
        for field, child in zip(names, node.children, strict=True)
    ]


def _raw_resolve(
    doc: Document, scope: dict[str, Any], captured: Captured, *, admitted: bool
) -> tuple[Document, dict[str, _Estimator]]:
    """Bind each canonical raw params source to its own learned function."""
    estimators: dict[str, _Estimator] = {}

    def raw(node: Node, half: str) -> tuple[str, Any] | None:
        name = str(node_field(node, "function_name", "") or "")
        stem, _, suffix = name.rpartition("_")
        value = _call_member(scope, stem) if suffix == half else None
        if not _estimator(value):
            return None
        if not admitted:
            raise TransformError(
                f"{stem}: raw estimators are admitted only in SQLProjection"
            )
        schema = (
            node.schema_ if isinstance(node, Function) else node_field(node, "schema")
        )
        if schema or node_field(node, "catalog"):
            raise TransformError(
                f"{stem}: namespaced raw estimator calls are unsupported"
            )
        _capture(captured, stem, value)
        return stem, value

    def bundle(
        node: Node, stem: str, qualifiers: set[str]
    ) -> tuple[Function, tuple[str, ...]]:
        fields = _raw_bundle(stem, [node])
        names = tuple(name for name, _ in fields)
        packed = _template("SELECT struct_pack(v := 1)").select_list[0]
        node = packed.model_copy(
            update={"children": [_aliased(child, name) for name, child in fields]}
        )
        for child in node.children:
            for part in (child, *descendants(child, deep=True)):
                if (
                    isinstance(part, ColumnRef)
                    and len(part.column_names) == 2
                    and part.column_names[0].lower() not in qualifiers
                ):
                    raise TransformError(
                        f"{stem}: bundle column qualifiers must name a local source; "
                        "struct field access is unsupported"
                    )
                if isinstance(part, Function) and (
                    raw(part, "fit")
                    or raw(part, "transform")
                    or _estimator(_call_member(scope, part.function_name))
                    or isinstance(
                        _call_member(scope, part.function_name),
                        (Transform, _projection_type(), _surface()),
                    )
                    or any(
                        part.function_name == view.udf_name
                        for view in estimators.values()
                    )
                ):
                    raise TransformError(
                        f"{stem}: nested bundles, estimator calls and field access "
                        "are not scalar features"
                    )
        return node, names

    def flat(node: Node, *, grouped: bool) -> bool:
        return (
            isinstance(node, Select)
            and not cte_entries(node)
            and not node.modifiers
            and node.where_clause is None
            and node.having is None
            and node.qualify is None
            and node.sample is None
            and node.aggregate_handling == "STANDARD_HANDLING"
            and (grouped or not (node.group_expressions or node.group_sets))
        )

    def fit_source(ref: Node) -> bool:
        if isinstance(ref, BaseTable):
            return (
                ref.table_name.upper() == FIT
                and not (ref.schema_name or ref.catalog_name or ref.column_name_alias)
                and ref.sample is None
                and ref.at_clause is None
            )
        if (
            not isinstance(ref, SubqueryRef)
            or ref.sample is not None
            or ref.column_name_alias
        ):
            return False
        query = ref.subquery.node
        if not flat(query, grouped=False) or not isinstance(
            query.from_table, BaseTable
        ):
            return False
        if not fit_source(query.from_table):
            return False
        for item in query.select_list:
            for part in (item, *descendants(item, deep=False)):
                if (
                    isinstance(part, Function)
                    and part.function_name.lower() in _aggregates()
                ):
                    return False
            for part in (item, *descendants(item, deep=True)):
                if (
                    raw(part, "fit")
                    or raw(part, "transform")
                    or isinstance(part, BaseTable)
                    and part.table_name.upper() == THIS
                    or isinstance(part, Function)
                    and _estimator(_call_member(scope, part.function_name))
                ):
                    return False
        return True

    def binding(node: Node) -> tuple[_Estimator, str] | None:
        if isinstance(node, Select):
            for item in node.select_list:
                if isinstance(item, Function) and item.function_name in estimators:
                    return estimators[item.function_name], item.alias
        return None

    def source_refs(ref: Node) -> list[Node]:
        if isinstance(ref, Join):
            return source_refs(ref.left) + source_refs(ref.right)
        return [ref]

    def visit(node: Node, *, allow_fit: bool = False) -> Node:
        fits = [
            item
            for item in node_field(node, "select_list", [])
            if isinstance(item, Function) and raw(item, "fit")
        ]
        if fits:
            call = fits[0]
            stem, prototype = raw(call, "fit")
            if not (
                allow_fit
                and len(fits) == 1
                and flat(node, grouped=True)
                and fit_source(node.from_table)
                and len(call.children) == 1
                and not call.distinct
            ):
                raise TransformError(
                    f"{stem}: raw params must contain GROUP BY keys plus one fit item; "
                    f"read direct {FIT} or one inline projection, "
                    "directly consumed by requests"
                )
            groups = {_print_expr(v) for v in node.group_expressions}
            group_aliases = {
                v.column_names[0].lower()
                for v in node.group_expressions
                if isinstance(v, ColumnRef) and len(v.column_names) == 1
            }
            for position, item in enumerate(node.select_list, 1):
                if item is call:
                    continue
                if any(
                    node_field(part, "class") == "WINDOW"
                    or isinstance(part, Function)
                    and part.function_name.lower() in _aggregates()
                    for part in (item, *descendants(item, deep=False))
                ):
                    raise TransformError(
                        f"{stem}: raw params may select only scalar keys "
                        "and one fit item"
                    )
                if not (
                    _print_expr(item) in groups
                    or str(position) in groups
                    or str(node_field(item, "alias", "") or "").lower() in group_aliases
                ):
                    raise TransformError(
                        f"{stem}: raw params may select only keys and one fit item"
                    )
            if node.group_sets and node.group_sets != [
                list(range(len(node.group_expressions)))
            ]:
                raise TransformError(f"{stem}: raw fit does not admit grouping sets")
            source_name = (
                node_field(node.from_table, "alias")
                or node_field(node.from_table, "table_name")
                or ""
            )
            packed, names = bundle(call.children[0], stem, {source_name.lower()})
            if getattr(prototype, "order_sensitive", False) and not node_field(
                call.order_bys, "orders", []
            ):
                raise TransformError(
                    f"{stem}: OrderSensitive requires in-call ORDER BY"
                )
            for part in descendants(node, deep=True):
                if isinstance(part, BaseTable) and part.table_name.upper() == THIS:
                    raise TransformError(f"{stem}: raw fit cannot read {THIS}")
                if (
                    raw(part, "transform")
                    or node_field(part, "class") == "WINDOW"
                    and raw(part, "fit")
                ):
                    raise TransformError(
                        f"{stem}: raw applications cannot run in fit-only subtrees"
                    )
                if part is not call and raw(part, "fit"):
                    raise TransformError(f"{stem}: nested raw fits are not supported")
            index = len(estimators)
            descriptor = _Estimator(
                prototype, names, f"__cf_est_{index}_fit", f"__cf_est_{index}_transform"
            )
            estimators[descriptor.fit_name] = descriptor
            collected = _list_of(packed).model_copy(
                update={"filter_": call.filter_, "order_bys": call.order_bys}
            )
            lowered = call.model_copy(
                update={
                    "function_name": descriptor.fit_name,
                    "children": [collected],
                    "filter_": None,
                    "order_bys": _list_of(packed).order_bys,
                    "alias": call.alias or _print_expr(call),
                }
            )
            return node.model_copy(
                update={
                    "select_list": [
                        lowered if item is call else item for item in node.select_list
                    ]
                }
            )

        refs = source_refs(node_field(node, "from_table"))
        request = any(
            isinstance(ref, BaseTable) and ref.table_name.upper() == THIS
            for ref in refs
        )
        direct_ctes = {
            ref.table_name.lower() for ref in refs if isinstance(ref, BaseTable)
        }
        views: dict[str, tuple[_Estimator, str]] = {}
        entries = []
        for entry in cte_entries(node):
            body = visit(
                entry.value.query.node,
                allow_fit=request and entry.key.lower() in direct_ctes,
            )
            if bound := binding(body):
                for ref in refs:
                    if (
                        isinstance(ref, BaseTable)
                        and ref.table_name.lower() == entry.key.lower()
                    ):
                        views[(ref.alias or ref.table_name).lower()] = bound
            entries.append(
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
        node = with_cte_entries(node, entries)

        def nested(part: AstNode) -> AstNode | None:
            if is_query(part):
                return visit(part, allow_fit=request)
            return None

        node = rebuild(node, nested, deep=False)
        for ref in source_refs(node_field(node, "from_table")):
            if isinstance(ref, SubqueryRef) and (bound := binding(ref.subquery.node)):
                views[ref.alias.lower()] = bound

        def apply(part: AstNode) -> AstNode | None:
            extracts = (
                isinstance(part, Function)
                and part.function_name.lower()
                in ("struct_extract", "struct_extract_at")
            ) or node_field(part, "type") == "STRUCT_EXTRACT"
            if extracts:
                children = node_field(part, "children", [])
                base = children[0] if children else None
                learned = {view.udf_name for view in estimators.values()}
                direct = isinstance(base, Function) and base.function_name in learned
                chained = (
                    isinstance(base, Function)
                    and base.function_name.lower()
                    in ("struct_extract", "struct_extract_at")
                ) or node_field(base, "type") == "STRUCT_EXTRACT"
                nested = chained and any(
                    isinstance(value, Function) and value.function_name in learned
                    for value in descendants(base, deep=False)
                )
                if direct:
                    from sql_transform._projection import (
                        _constant_text,  # noqa: PLC0415
                    )

                    if len(children) != 2 or _constant_text(children[1]) is None:
                        raise TransformError(
                            "raw learned fields must be addressed by a literal name"
                        )
                elif nested:
                    raise TransformError(
                        "chained raw learned field access is unsupported; "
                        "learned fields are DOUBLE"
                    )
            if isinstance(part, Function) and (matched := raw(part, "transform")):
                stem, prototype = matched
                if (
                    not request
                    or len(part.children) != 2
                    or part.filter_ is not None
                    or part.distinct
                    or node_field(part.order_bys, "orders", [])
                ):
                    raise TransformError(
                        f"{stem}: raw transform requires a request-level ID "
                        "and scalar bundle"
                    )
                iid, features = part.children
                bound = None
                if isinstance(iid, ColumnRef) and len(iid.column_names) == 2:
                    candidate = views.get(iid.column_names[0].lower())
                    if (
                        candidate
                        and candidate[1].lower() == iid.column_names[1].lower()
                    ):
                        bound = candidate
                elif isinstance(iid, SubqueryExpr) and iid.subquery_type == "SCALAR":
                    query = iid.subquery.node
                    if len(
                        node_field(query, "select_list", [])
                    ) == 1 and not node_field(query, "group_expressions", []):
                        bound = binding(query)
                if bound is None or bound[0].prototype is not prototype:
                    raise TransformError(
                        f"{stem}: first argument must be its canonical direct "
                        "params ID or inline fit subquery"
                    )
                qualifiers = {
                    str(
                        node_field(ref, "alias") or node_field(ref, "table_name") or ""
                    ).lower()
                    for ref in source_refs(node_field(node, "from_table"))
                }
                packed, names = bundle(features, stem, qualifiers)
                if tuple(name.lower() for name in names) != tuple(
                    name.lower() for name in bound[0].feature_names
                ):
                    raise TransformError(
                        f"{stem}: apply bundle fields must match fit fields in order"
                    )
                return part.model_copy(
                    update={
                        "function_name": bound[0].udf_name,
                        "children": [
                            iid,
                            *[_unaliased(child) for child in packed.children],
                        ],
                    }
                )
            if raw(part, "fit"):
                raise TransformError(
                    "raw fit windows must be inline fit+transform "
                    "or explicit fit/request SQL"
                )
            if isinstance(part, Function) and _estimator(
                _call_member(scope, part.function_name)
            ):
                raise TransformError(
                    f"{part.function_name}: use SQLProjection.marginalize "
                    "for bare estimator syntax"
                )
            return None

        return rebuild(node, apply, deep=False)

    box = doc.statements[0]
    result = visit(box.node)
    return doc.model_copy(
        update={"statements": [box.model_copy(update={"node": result})]}
    ), estimators


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
    if member._program.estimators or member._program.udfs:
        raise TransformError(
            f"{name}: row-only Python members cannot be spliced as general transforms"
        )

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
    *,
    row_udfs: bool = False,
    estimators: dict[str, _Estimator] | None = None,
    udfs: dict[str, UDF] | None = None,
) -> tuple[Document, int]:
    """Splice every member call and resolve every free name.

    Returns the rewritten document and the nesting depth. Children are resolved
    before their parent, so a splice at this level always grafts an
    already-resolved body.
    """
    depth = 0
    generated = set(estimators or {}) | {
        value.udf_name for value in (estimators or {}).values()
    }
    vocabulary = _functions(_ALL_FUNCTIONS, con)

    def foreign_call(call: Function) -> Node:
        """``x_fit``/``x_transform``: the stem resolves, the suffix says half.
        A bare ``x`` resolving to a projection is the ONE sugar —
        ``x_transform(x_fit(...) OVER (), ...)``, the global fit scope."""
        name = call.function_name
        whole = _call_member(scope, name)
        if name in generated:
            return call
        if isinstance(whole, UDF):
            if not row_udfs:
                raise TransformError(
                    f"{name}: scalar UDFs are admitted only in SQLProjection"
                )
            if not name.isidentifier() or name.startswith(RESERVED):
                raise TransformError(f"{name}: scalar UDF name is not usable")
            if whole.name.lower() != name.lower():
                raise TransformError(
                    f"UDF {name} resolves to an object named {whole.name!r}"
                )
            if call.schema_ or call.catalog:
                raise TransformError(
                    f"{name}: namespaced captured scalar UDF calls are unsupported"
                )
            try:
                whole._check_schema()
                arity = len(whole._duck_signature()[0])
            except ValueError as exc:
                raise TransformError(f"{name}: {exc}") from exc
            if len(call.children) != arity:
                raise TransformError(
                    f"{name}: expected {arity} scalar arguments, "
                    f"got {len(call.children)}"
                )
            if (
                call.distinct
                or call.filter_ is not None
                or node_field(call.order_bys, "orders", [])
            ):
                raise TransformError(
                    f"{name}: a scalar UDF does not take aggregate modifiers"
                )
            _capture(captured, name, whole)
            udfs[name] = whole if whole.name == name else _AliasedUDF(whole, name)
            return call
        if isinstance(whole, _projection_type()):
            from sql_transform import _leaf  # noqa: PLC0415

            _capture(captured, name, whole)
            out = _leaf.bare_call(name, whole, call)
            return _aliased(out, call.alias) if call.alias else out
        stem, _, half = name.rpartition("_")
        member = _call_member(scope, stem) if half in ("fit", "transform") else None
        if isinstance(member, _projection_type()):
            # A projection leaf is spliced, never registered (D2): both
            # halves become ordinary SQL, and θ carries the parameters. The
            # author's alias survives the rewrite — it names their column.
            from sql_transform import _leaf  # noqa: PLC0415

            _capture(captured, stem, member)
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
        _capture(captured, stem, member)
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
                    and (name in captured or name.lower() not in catalog)
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
                        case obj if isinstance(obj, (Transform, UDF)) or _estimator(
                            obj
                        ):
                            raise TransformError(
                                f"{name} is a Python function member, not a relation"
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
                member = _call_member(scope, stem) if half == "fit" else None
                if isinstance(member, _projection_type()):
                    from sql_transform import _leaf  # noqa: PLC0415

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
            if isinstance(v, Function) and not v.is_operator:
                name = v.function_name.lower()
                whole = _call_member(scope, name)
                stem, _, half = name.rpartition("_")
                member = (
                    _call_member(scope, stem) if half in ("fit", "transform") else None
                )
                if name in vocabulary and (
                    isinstance(whole, (UDF, Transform, _projection_type(), _surface()))
                    or _estimator(whole)
                    or isinstance(member, (Transform, _projection_type()))
                    or _estimator(member)
                ):
                    raise TransformError(
                        f"{name}: captured function collides with a DuckDB builtin"
                    )
                if name not in vocabulary:
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

    node: Node  # the RESIDUAL, not the resolved text `Program.node` holds
    params: Params  # each fit step's evaluated table, under its parameter name
    bindings: Bindings
    foreign: Foreign
    # A leaf's fitted state, keyed by the θ id its params row carries. Two
    # fits mint two id spaces, so mixing an artifact's params with another's
    # instances is caught rather than silently scored.
    instances: dict[int, Any]
    connection: Connection | None = None
    # Scalar UDFs the residual calls, under the names it calls them by. Runtime
    # views only: a shared connection registers each under a leased alias.
    udfs: dict[str, UDF] = field(default_factory=dict)
    # A projection's runtime: every eager execution runs at threads=1. See
    # `_session`.
    _row_udfs: bool = False
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

    def _lease_on(
        self, con: Connection, own: bool, extra: Bindings
    ) -> tuple[Callable[[Node], str], _Registry, Callable[[], None]]:
        """Register everything this residual needs, and say how to render a
        node under those names and how to clean up.

        On a connection we own, names stay the readable ones — a fresh
        connection per call makes collisions impossible, and it dies with the
        call, so there is nothing to give back. On a *shared* connection every
        execution is renamed and leased; see ``_lease``.
        """
        registry = _Registry(self.instances)
        names, renames, release = _lease(
            con,
            extra | self.bindings | self.params,
            self.foreign,
            registry,
            rename=not own,
            udfs=self.udfs,
        )
        return partial(_rendered, names=names, renames=renames), registry, release

    @contextmanager
    def _leased(
        self, extra: Bindings | None = None
    ) -> Iterator[tuple[Connection, Callable[[Node], str], _Registry]]:
        """One eager execution over this artifact's tables and functions, plus
        ``extra``: the connection, a renderer for nodes under the leased
        names, and the registry. Everything is given back on exit, then the
        projection thread setting; see ``_session``. Fit-time probes run here
        too, so they see the caller's catalog exactly as batch does."""
        con, own = _connection(self.connection)
        with _session(con, single_threaded=self._row_udfs) as undo:
            render, registry, release = self._lease_on(con, own, extra or {})
            undo.append(release)
            yield con, render, registry

    def transform(self, data: Relation) -> pa.Table:
        with self._leased({THIS: data}) as (con, render, registry):
            return _execute(con, render(self.node), registry)

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
        con, own = _connection(self.connection)
        render, _, release = self._lease_on(con, own, {THIS: data})
        self._leases.append(release)
        return con.sql(render(self.node))

    def release(self) -> None:
        """Give back every table this artifact still has registered.

        Only lazy output leaves anything to give back. Idempotent, so a caller
        can put it in a ``finally`` without checking.
        """
        _give_back(self._leases)


# One counter per process, so a shared connection never sees two executions
# under the same name. See _lease.
_EXECUTIONS = itertools.count()


class _AliasedUDF(UDF):
    """``udf`` under another name — the one an execution leased on a shared
    connection. Everything but the name is the original's, so DuckDB calls
    exactly what serving calls: a subclass's own signature (the implicit id
    of ``PythonTransform``), its batch form, its instance or tree views.
    Registration is ``UDF.register``, inherited unchanged; this is a view,
    never a copy, so a UDF that is not a dataclass works the same."""

    def __init__(self, udf: UDF, name: str) -> None:
        self.udf = udf
        self.name = name

    @property
    def takes(self) -> pa.Schema:
        return self.udf.takes

    @property
    def returns(self) -> pa.DataType:
        return self.udf.returns

    def __call__(self, *args: Any) -> tuple | None:
        return self.udf(*args)

    def apply_batch(self, *cols: pa.Array) -> tuple[pa.Array, ...]:
        return self.udf.apply_batch(*cols)

    def _duck_signature(self) -> tuple[list[str], Any]:
        return self.udf._duck_signature()

    def _arrow_scalar_batch(self, *cols: Any) -> pa.Array:
        return self.udf._arrow_scalar_batch(*cols)

    def __getattr__(self, name: str) -> Any:
        if name == "udf":  # not set yet (copy/pickle probe): no recursion
            raise AttributeError(name)
        return getattr(self.udf, name)


class _Recording:
    """A connection that remembers each function it actually registered.

    A ``Transform`` registers two functions and its second can fail after the
    first landed; releasing by what *should* exist would either leak the first
    or try to remove a function that never was. Only ``create_function`` is
    needed: it is all ``Transform.register`` and ``UDF.register`` call.
    """

    def __init__(self, con: Connection, made: list[str]) -> None:
        self._con = con
        self._made = made

    def create_function(self, name: str, *args: Any, **kwargs: Any) -> None:
        self._con.create_function(name, *args, **kwargs)
        self._made.append(name)


def _settle(
    steps: Iterable[Callable[[], Any]], error: BaseException | None = None
) -> None:
    """Run every cleanup step, even after one fails.

    A cleanup failure never replaces ``error``, the execution's own failure
    already in flight — it is noted on it. With none in flight the first
    cleanup failure raises, carrying the rest as notes.
    """
    first: Exception | None = None
    for step in steps:
        try:
            step()
        except Exception as exc:
            owner = error if error is not None else first
            if owner is None:
                first = exc
            else:
                owner.add_note(f"cleanup also failed: {exc!r}")
    if first is not None:
        raise first


def _connection(con: Connection | None) -> tuple[Connection, bool]:
    """The caller's connection, or a fresh one we own, and which it is."""
    return (duckdb.connect(), True) if con is None else (con, False)


@contextmanager
def _session(
    con: Connection, *, single_threaded: bool
) -> Iterator[list[Callable[[], Any]]]:
    """One eager execution's cleanup, however it ends.

    Yields a list to append cleanups to; on exit they run last-in first-out,
    all of them (``_settle``), and only then does the thread setting go back.

    ``single_threaded`` is the projection runtime: a parallel float reduction
    is not bit-reproducible, so fit, probes and batch run at ``threads = 1``.
    The setting is database-wide, so the *observed* value goes back with
    ``SET`` — never ``RESET``, which would discard the caller's own setting.
    ponytail: a caller changing ``threads`` concurrently on the same database
    is not supported; it would need a lock the caller shares.
    """
    undo: list[Callable[[], Any]] = []
    if single_threaded:
        (observed,) = con.execute("SELECT current_setting('threads')").fetchone()
        con.execute("SET threads = 1")
        undo.append(partial(con.execute, f"SET threads = {int(observed)}"))
    try:
        yield undo
    except BaseException as exc:
        _settle(reversed(undo), exc)
        raise
    _settle(reversed(undo))


def _arrow(data: Relation) -> pa.Table:
    """``data`` read once into one Arrow table.

    A projection's artifact holds its captured relations as this snapshot, so
    fit, probes, batch and serving all read one mapping. Anything that is not
    already Arrow, a DuckDB relation or a plain dict is read the way DuckDB
    reads it when registered — pandas' index ignored, exactly what batch
    bound before.
    """
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, duckdb.DuckDBPyRelation):
        # to_arrow_table, never .arrow(): see `_execute`.
        return data.to_arrow_table()
    if isinstance(data, dict):
        return pa.table(data)
    con = duckdb.connect()
    try:
        con.register("data", data)
        return con.execute('SELECT * FROM "data"').to_arrow_table()
    finally:
        con.close()


def _lease(
    con: Connection,
    tables: Bindings,
    foreign: Foreign,
    registry: _Registry,
    *,
    rename: bool,
    udfs: dict[str, UDF] | None = None,
    estimators: dict[str, _Estimator] | None = None,
    _made: list[str] | None = None,
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

    ``names`` maps each table to its registered name; ``renames`` maps each
    call name — a UDF's whole name, a ``Transform`` stem's ``_fit`` and
    ``_transform`` halves — to its registered one. Only what actually
    registered is released: a failure part-way gives back what landed and
    re-raises.

    ``names`` is live: ``fit`` adds each parameter as it lands, and the
    release closes over the same dict, so those come back too.
    """
    token = next(_EXECUTIONS)

    def under(name: str) -> str:
        return f"{name}__x{token}" if rename else name

    names: dict[str, str] = {}
    renames: dict[str, str] = {}
    made: list[str] = [] if _made is None else _made

    def release() -> None:
        if not rename:
            return  # a connection we own dies with the call; θ keeps its name
        _settle(
            [
                *(partial(con.unregister, name) for name in names.values()),
                *(partial(con.remove_function, name) for name in made),
            ]
        )

    recording = _Recording(con, made)
    try:
        for name, table in tables.items():
            con.register(under(name), table)
            names[name] = under(name)
        for stem, leaf in foreign.items():
            leaf.register(recording, under(stem), registry)  # type: ignore[arg-type]
            for half in ("fit", "transform"):
                renames[f"{stem}_{half}"] = f"{under(stem)}_{half}"
        for name, descriptor in (estimators or {}).items():
            descriptor.register(recording, under(name), registry)
            renames[name] = under(name)
        for name, udf in (udfs or {}).items():
            view = udf if udf.name == under(name) else _AliasedUDF(udf, under(name))
            view.register(recording)
            renames[name] = under(name)
    except BaseException as exc:
        _settle([release], exc)
        raise
    return names, renames, release


def _rendered(node: Node, names: dict[str, str], renames: dict[str, str]) -> str:
    """``node`` under the names this execution actually registered."""
    doc = _rename_free(_statement(node), names)
    return _deserialize(_rename_calls(doc, renames))


def _raw_step(node: Select, numbered: str, fit_names: set[str]) -> Select:
    """Number only the canonical source; nested scalar FIT queries stay intact."""
    source = node.from_table
    if isinstance(source, SubqueryRef):
        projected = source.subquery.node
        base = projected.from_table
        ordinal = _template("SELECT __cf_fit_row").select_list[0]
        projected = projected.model_copy(
            update={
                "from_table": base.model_copy(
                    update={
                        "table_name": numbered,
                        "alias": base.alias or FIT,
                    }
                ),
                "select_list": [*projected.select_list, ordinal],
            }
        )
        source = source.model_copy(
            update={"subquery": source.subquery.model_copy(update={"node": projected})}
        )
        qualifier = source.alias
    else:
        qualifier = source.alias or FIT
        source = source.model_copy(update={"table_name": numbered, "alias": qualifier})
    tie = _template("SELECT list(1 ORDER BY __cf_fit_row)").select_list[0].order_bys
    tie_order = node_field(tie, "orders")[0]
    tie_order = rebuild(
        tie_order,
        lambda part: (
            part.model_copy(update={"column_names": [qualifier, "__cf_fit_row"]})
            if isinstance(part, ColumnRef)
            else None
        ),
        deep=True,
    )
    items = []
    for item in node.select_list:
        if isinstance(item, Function) and item.function_name in fit_names:
            collected = item.children[0]
            ordering = collected.order_bys
            ordering = ordering.model_copy(
                update={
                    "fields": ordering.fields
                    | {"orders": [*node_field(ordering, "orders", []), tie_order]}
                }
            )
            collected = collected.model_copy(update={"order_bys": ordering})
            item = item.model_copy(update={"children": [collected]})
        items.append(item)
    return node.model_copy(update={"from_table": source, "select_list": items})


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
    # Runtime-only views; authored captures never acquire generated bindings.
    udfs: dict[str, UDF] = field(default_factory=dict)
    estimators: dict[str, _Estimator] = field(default_factory=dict)
    # Compiled for a projection: its fit and every execution of its artifact
    # run single-threaded, and its captured relations become artifact data.
    row_udfs: bool = False

    @classmethod
    def compile(
        cls,
        sql: str,
        scope: dict[str, Any],
        *,
        connection: Connection | None = None,
        captured: Captured | None = None,
        row_udfs: bool = False,
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
        if row_udfs:
            _raw_names(doc.statements[0].node, scope, connection)
        for part in descendants(doc, deep=True):
            if isinstance(part, Function) or node_field(part, "class") == "WINDOW":
                _reserve(
                    str(node_field(part, "function_name", "") or ""), "a function named"
                )

        doc, estimators = _raw_resolve(doc, scope, captured, admitted=row_udfs)
        udfs: dict[str, UDF] = {}
        doc, depth = _resolve(
            doc,
            scope,
            captured,
            _catalog(connection),
            connection,
            row_udfs=row_udfs,
            estimators=estimators,
            udfs=udfs,
        )
        # Two runtime views. A member or a projection leaf is spliced away,
        # so it is neither.
        foreign: Foreign = {
            k: v for k, v in captured.items() if isinstance(v, Transform)
        }
        bindings: Bindings = {
            k: v
            for k, v in captured.items()
            if not isinstance(v, (Transform, UDF, _surface(), _projection_type()))
            and not _estimator(v)
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
            row_udfs=row_udfs,
            udfs=udfs,
            estimators=estimators,
        )

    def fit(self, data: Relation) -> Fitted:
        """Partial application: evaluate the fit DAG, return the artifact.

        Every step runs under leased names and they are all given back, so a
        shared connection is the caller's again when this returns — including
        ``__FIT__``, which used to stay bound to the whole training relation
        for the life of the connection.

        A projection's captured relations are read once, before any step, into
        a local ``Program`` value with the same author captures; the artifact
        holds that snapshot in the same ``params`` mapping as what fit learned
        — one mapping that batch, probes and serving all read, and nothing
        live left in ``bindings``. A general ``Fitted`` keeps learned-only
        ``params`` and its live bindings.

        Only the parameters the residual reads are shipped. Any other step
        result is an intermediate — a carrier a later step picks from — and
        is given back right after the last step that reads it.
        """
        program = (
            replace(self, bindings={k: _arrow(v) for k, v in self.bindings.items()})
            if self.row_udfs
            else self
        )
        runtimes = {
            name: descriptor.fresh() for name, descriptor in program.estimators.items()
        }
        runtime_udfs = dict(program.udfs)
        numbered: dict[str, pa.Table] = {}
        if runtimes:
            data = _arrow(data)
            if {"__cf_fit_row", "__cf_row"} & {
                name.lower() for name in data.column_names
            }:
                raise TransformError(
                    "raw FIT schema carries a reserved ordinal column "
                    "(__cf_fit_row or __cf_row)"
                )
            numbered["__cf_numbered_fit"] = data.append_column(
                "__cf_fit_row", pa.array(range(len(data)), type=pa.int64())
            )
        made: list[str] = []
        order = [param for param, _ in program.steps]
        shipped = _referenced(program.residual, order)
        last_use = {
            read: i
            for i, (_, node) in enumerate(program.steps)
            for read in _referenced(node, order)
        }
        registry = _Registry()
        con, own = _connection(program.connection)
        params: Params = {}
        with _session(con, single_threaded=program.row_udfs) as undo:
            names, renames, release = _lease(
                con,
                {FIT: data} | program.bindings | numbered,
                program.foreign,
                registry,
                rename=not own,
                udfs=program.udfs,
                estimators=runtimes,
                _made=made,
            )
            undo.append(release)
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
                program.shadowable,
            )
            for i, (param, node) in enumerate(program.steps):
                fit_items = [
                    item
                    for item in node_field(node, "select_list", [])
                    if isinstance(item, Function) and item.function_name in runtimes
                ]
                executable = (
                    _raw_step(node, "__cf_numbered_fit", set(runtimes))
                    if fit_items
                    else node
                )
                try:
                    params[param] = _execute(
                        con, _rendered(executable, names, renames), registry
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
                for item in fit_items:
                    descriptor = runtimes[item.function_name]
                    table = params[param]
                    params[param] = table.filter(pc.is_valid(table[item.alias]))
                    udf = descriptor.publish()
                    runtime_udfs[udf.name] = udf
                    leased_udf = (
                        udf.name if own else f"{udf.name}__x{next(_EXECUTIONS)}"
                    )
                    _AliasedUDF(udf, leased_udf).register(_Recording(con, made))
                    renames[udf.name] = leased_udf
                # Into the same dict the release closes over, once it has
                # registered: later steps see it, and it comes back with
                # everything else.
                leased = param if own else f"{param}__x{next(_EXECUTIONS)}"
                con.register(leased, params[param])
                names[param] = leased
                # An intermediate nothing later reads: given back now, not
                # at the end, so a carrier does not outlive its last pick.
                for done in [p for p, j in last_use.items() if j == i]:
                    if done not in shipped:
                        con.unregister(names.pop(done))
                        del params[done]
        if program.row_udfs:
            return Fitted(
                program.residual,
                program.bindings | params,
                {},
                program.foreign,
                registry.instances,
                program.connection,
                runtime_udfs,
                _row_udfs=True,
            )
        return Fitted(
            program.residual,
            params,
            program.bindings,
            program.foreign,
            registry.instances,
            program.connection,
            runtime_udfs,
        )

    def run(self, data: Relation) -> pa.Table:
        """Both parameters bound to the same relation, with no freezing at all.

        The reference side of "freezing is faithful". It is a *binding*, not a
        rewrite, which is what keeps that law from restating the implementation.
        """
        if self.estimators:
            raise TransformError(
                "raw SQLProjection Program.run is unsupported; "
                "use independent estimator execution"
            )
        registry = _Registry()
        con, own = _connection(self.connection)
        with _session(con, single_threaded=self.row_udfs) as undo:
            names, renames, release = _lease(
                con,
                {FIT: data, THIS: data} | self.bindings,
                self.foreign,
                registry,
                rename=not own,
                udfs=self.udfs,
            )
            undo.append(release)
            return _execute(con, _rendered(self.node, names, renames), registry)
