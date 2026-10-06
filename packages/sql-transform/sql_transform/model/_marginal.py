"""``SQLProjection.marginalize`` — the ``__FIT__`` half derived from a
``__THIS__``-only text.

A bounded convenience in front of the ordinary constructor, not the core: the
core is authored two-parameter SQL, and this rewrite only writes that SQL for
one SELECT over ``__THIS__`` whose window scopes become lookups. The output is
*text*, and it is an ordinary author text: the derived names live in author
space under a fresh prefix (gensym'd against the author's own identifiers)
rather than under ``__cf_``, precisely so the ordinary constructor — which
reserves ``__`` — accepts what the rewrite emits. That is the attribution
gate: every refusal here fires against the author's own spelling, before the
rewrite, and a refusal escaping from the derived text is our bug.

One numerical lowering serves every admitted window. The *carrier*, a CTE
over ``__FIT__``, evaluates the author's own executable select items — top-level
stars omitted, nothing else — then every window value in first-occurrence
order, then every lookup key. Evaluating the windows beside the original items
keeps the original query's window operator chain, and with it DuckDB's
floating reduction order: measured, seed 20260729 case 206's ``avg(x) OVER
()`` differs in its last bits when evaluated alone instead of beside the
query's ordered ``sum``. Each scope then picks its keys and values from the
carrier with a flat ``SELECT DISTINCT``, which the planner freezes into a
params table of one row per fitted key; the carrier itself never ships.
Lookups are NULL-safe, because a window puts NULL keys in one partition.

Keyed projection leaves keep their own grouped composition (``keyed_call``),
and uncorrelated scalar and EXISTS subqueries over ``__THIS__`` freeze
verbatim over ``__FIT__``.
"""

import itertools
import re
from typing import Any, NoReturn

from sql_transform.model._ast import (
    FIT,
    THIS,
    _aggregates,
    _aliased,
    _parse,
    _print_expr,
    _template,
)
from sql_transform.model._errors import TransformError
from sql_transform.model._nodes import (
    BaseTable,
    ColumnRef,
    Function,
    Node,
    Opaque,
    Select,
    SubqueryExpr,
    SubqueryRef,
    cte_entries,
    descendants,
    field,
    is_query,
    rebuild,
)

_MODIFIERS = {
    "DISTINCT_MODIFIER": "DISTINCT",
    "ORDER_MODIFIER": "ORDER BY",
    "LIMIT_MODIFIER": "LIMIT",
    "LIMIT_PERCENT_MODIFIER": "LIMIT",
}

# The marginalizability rule: a window's value must be a function of
# row-visible values — the partition keys plus, when the frame moves with
# them, the order values (RANGE/GROUPS peers share values). Physical position
# is the one thing a lookup key cannot carry, so positional windows refuse.
_RANK = frozenset(
    {"WINDOW_RANK", "WINDOW_RANK_DENSE", "WINDOW_PERCENT_RANK", "WINDOW_CUME_DIST"}
)
_POSITIONAL = frozenset(
    {"WINDOW_ROW_NUMBER", "WINDOW_NTILE", "WINDOW_LAG", "WINDOW_LEAD"}
)
_VALUE = frozenset({"WINDOW_FIRST_VALUE", "WINDOW_LAST_VALUE", "WINDOW_NTH_VALUE"})
_WHOLE = ("UNBOUNDED_PRECEDING", "UNBOUNDED_FOLLOWING")
# Every WINDOW field the admission below reads or carries verbatim; any other
# truthy field is a window feature nobody checked, refused by its own name.
_KNOWN = frozenset(
    {"class", "type", "alias", "query_location", "function_name"}
    | {"schema", "catalog", "children", "partitions", "orders", "arg_orders"}
    | {"start", "end", "start_expr", "end_expr", "exclude_clause"}
    | {"filter_expr", "distinct", "ignore_nulls", "offset_expr", "default_expr"}
)
_STAR_MODIFIERS = (
    "exclude_list",
    "qualified_exclude_list",
    "replace_list",
    "rename_list",
)


def _refuse(detail: str) -> NoReturn:
    raise TransformError(detail)


def _is_window(v: Any) -> bool:
    return isinstance(v, Opaque) and v.fields.get("class") == "WINDOW"


def _is_star(v: Any) -> bool:
    return isinstance(v, Opaque) and v.fields.get("class") == "STAR"


def _projection(name: str, scope: dict[str, Any]) -> Any | None:
    """The projection ``name`` resolves to, or None. Late import:
    ``_projection`` imports this module."""
    from sql_transform.model._projection import SQLProjection  # noqa: PLC0415

    obj = scope.get(name)
    return obj if isinstance(obj, SQLProjection) else None


def _raw(name: str, scope: dict[str, Any]) -> Any | None:
    from sql_transform.model._program import _estimator  # noqa: PLC0415

    obj = scope.get(name)
    return obj if _estimator(obj) else None


def _pure(
    what: str,
    parts: list[Node],
    scope: dict[str, Any],
    spine: frozenset[str],
    laterals: frozenset[str],
) -> None:
    """A fit scope's arguments must move intact into the carrier over
    ``__FIT__``: no nested scope (it has no carrier column of its own), no
    name that only resolves in the spine's own SELECT."""
    for c in parts:
        for d in (c, *descendants(c, deep=True)):
            if _is_window(d):
                _refuse(f"{what} nests {_print_expr(d)} inside a fit scope")
            if is_query(d) or isinstance(d, SubqueryExpr):
                _refuse(f"{what} nests a subquery inside a fit scope")
            if isinstance(d, Opaque) and d.fields.get("class") == "LAMBDA":
                _refuse(
                    f"{what} carries a lambda, whose parameter the rewrite "
                    "cannot tell from a batch column — no frozen spelling yet"
                )
            if isinstance(d, ColumnRef) and len(d.column_names) == 1:
                name = d.column_names[0]
                if name.lower() in spine:
                    _refuse(
                        f"{what} reads the whole {name} row, "
                        "which has no frozen spelling yet"
                    )
                if name.lower() in laterals:
                    _refuse(
                        f"{what}: {name} is a sibling select item's alias, "
                        f"which {FIT} does not have — inline the expression"
                    )
            if isinstance(d, Function) and not d.is_operator:
                if d.function_name.lower() in _aggregates():
                    _refuse(
                        f"{what} nests the aggregate {d.function_name} "
                        "inside a fit scope"
                    )
                if _projection(d.function_name, scope):
                    _refuse(
                        f"{what} nests the projection call {d.function_name} "
                        "inside a fit scope"
                    )
                stem, _, half = d.function_name.rpartition("_")
                if _raw(d.function_name, scope) or (
                    half in ("fit", "transform") and _raw(stem, scope)
                ):
                    _refuse(f"{what} nests an estimator call inside a fit bundle")


def _keyed(projection: Any) -> bool:
    """Whether the projection's own text joins ``__FIT__`` through keys."""
    return any(probe.keys for probe in projection._probes)


def _constant(v: Any) -> bool:
    return isinstance(v, Opaque) and v.fields.get("class") == "CONSTANT"


def _moves(w: Opaque, what: str) -> bool:
    """Whether the frame moves with the order values (so they join the key
    set), or the refusal. False when it covers the whole partition."""
    f = w.fields
    if f.get("exclude_clause") != "NO_OTHER":
        _refuse(
            f"{what}: EXCLUDE splits value peers, so the value is no "
            "longer a function of the keys"
        )
    for key in ("start_expr", "end_expr"):
        if f.get(key) is not None and not _constant(f[key]):
            _refuse(f"{what}: a non-constant frame bound has no frozen spelling")
    start, end = str(f.get("start") or ""), str(f.get("end") or "")
    if (start, end) == _WHOLE:
        return False
    if start.endswith("_ROWS") or end.endswith("_ROWS"):
        _refuse(
            f"{what}: a bounded ROWS frame is positional — only value peers "
            "(RANGE/GROUPS) or the whole partition freeze"
        )
    return bool(f.get("orders"))


def _nth(w: Opaque) -> bool:
    """``nth_value``'s n is a positive integer constant."""
    children = w.fields.get("children") or []
    if len(children) < 2 or not _constant(children[1]):
        return False
    value = field(children[1], "value")
    n = field(value, "value")
    return not field(value, "is_null") and isinstance(n, int) and n >= 1


def _admit(w: Opaque, scope: dict[str, Any], keyed_ok: bool = False) -> list[Node]:
    """The lookup keys the scope's value is a function of — its partitions,
    plus its order values when the frame moves with them — or the refusal in
    the author's spelling."""
    f = w.fields
    name = str(f.get("function_name") or "")
    kind = str(f.get("type") or "")
    what = _print_expr(w)
    stem, _, half = name.rpartition("_")
    if _projection(name, scope):
        _refuse(
            f"{name} is a projection, and a fit scope is spelled on the fit "
            f"half: {name}_transform({name}_fit(...) OVER (...), ...)"
        )
    leaf = _projection(stem, scope) if half == "fit" else None
    if half == "transform" and _projection(stem, scope):
        _refuse(f"{name} is the scalar half — the OVER belongs on {stem}_fit")
    if leaf is not None and _keyed(leaf) and not keyed_ok:
        _refuse(
            f"{stem} is keyed, so its θ is a table — only "
            f"{stem}_transform({stem}_fit(...) OVER (...), ...) can read it "
            "as one scope"
        )
    if kind in _POSITIONAL:
        _refuse(
            f"{what} is positional: its value is a row position, "
            "which a join key cannot carry"
        )
    for key, value in f.items():
        if key not in _KNOWN and value not in (None, [], "", False):
            _refuse(f"{what}: its {key.upper()} has no frozen spelling")
    if f.get("offset_expr") is not None or f.get("default_expr") is not None:
        _refuse(f"{what}: an offset or default has no frozen spelling")
    raw = _raw(stem, scope) if half == "fit" else None
    if raw is not None:
        if f.get("schema") or f.get("catalog"):
            _refuse(f"{what}: namespaced raw estimator calls are unsupported")
        if f.get("orders"):
            _refuse(f"{what}: an ordered fit scope is a running fit")
        if (
            kind != "WINDOW_AGGREGATE"
            or f.get("distinct")
            or f.get("ignore_nulls")
            or f.get("start") != "UNBOUNDED_PRECEDING"
            or f.get("end") != "CURRENT_ROW_RANGE"
            or f.get("start_expr") is not None
            or f.get("end_expr") is not None
            or f.get("exclude_clause") != "NO_OTHER"
        ):
            _refuse(f"{what}: a raw estimator requires a whole partition fit")
        return list(f.get("partitions") or [])
    if leaf is not None:
        if f.get("orders"):
            _refuse(
                f"{what}: an ordered fit scope is a running fit — "
                "per-row θ, still a future feature"
            )
        if (
            f.get("filter_expr")
            or f.get("distinct")
            or f.get("arg_orders")
            or f.get("ignore_nulls")
        ):
            _refuse(
                f"{what}: FILTER, DISTINCT, IGNORE NULLS and an argument "
                "ORDER BY on a projection fit scope have no frozen spelling yet"
            )
    if f.get("ignore_nulls") and kind not in _VALUE:
        _refuse(f"{what}: IGNORE NULLS has no frozen spelling outside a value window")
    if kind in _RANK:
        moves = True  # a function of the order values; frames don't apply
    elif kind == "WINDOW_AGGREGATE":
        if leaf is None and name.lower() not in _aggregates():
            _refuse(f"{what}: {name} is not an aggregate the oracle knows")
        moves = _moves(w, what)
    elif kind in _VALUE:
        if kind == "WINDOW_NTH_VALUE" and not _nth(w):
            _refuse(f"{what}: nth_value's n must be a positive integer constant")
        moves = _moves(w, what)
        if (
            kind == "WINDOW_FIRST_VALUE"
            and not f.get("ignore_nulls")
            and f.get("start") == "UNBOUNDED_PRECEDING"
            and f.get("end") in ("CURRENT_ROW_RANGE", "UNBOUNDED_FOLLOWING")
        ):
            # The frame always starts at the partition start and is never
            # empty: the partition's first row, whatever the order value.
            moves = False
    else:
        _refuse(f"{what}: a {kind} window has no frozen spelling")
    keys = list(f.get("partitions") or [])
    if moves:
        keys += [o.fields["expression"] for o in (f.get("orders") or [])]
    return keys


def _admit_subquery(v: SubqueryExpr, spine: frozenset[str]) -> SubqueryExpr:
    """The frozen subquery — ``__THIS__`` re-bound to ``__FIT__``, every
    clause verbatim — or the refusal in the author's spelling. Admitted: one
    uncorrelated scalar or EXISTS SELECT over ``__THIS__``."""
    node = v.subquery.node
    what = _print_expr(v)
    kind = str(v.subquery_type).upper()
    if kind not in ("SCALAR", "EXISTS"):
        _refuse(
            f"{what}: only a scalar or EXISTS subquery is one frozen value — "
            f"an {kind} subquery (IN, ANY, ALL) has no frozen spelling; write "
            f"its fit query over {FIT} explicitly"
        )
    if not isinstance(node, Select) or cte_entries(node):
        _refuse(
            f"{what}: a subquery freezes only as one SELECT over {THIS} — "
            "anything wider has no frozen spelling yet"
        )
    inner = node.from_table
    if not (isinstance(inner, BaseTable) and inner.table_name.upper() == THIS):
        _refuse(
            f"{what}: a subquery freezes only as one SELECT over {THIS} — "
            "anything wider has no frozen spelling yet"
        )
    if inner.alias and inner.alias.lower() in spine:
        _refuse(f"{what} shadows the spine alias {inner.alias} — rename one")
    for d in descendants(node, deep=False):
        if is_query(d) or isinstance(d, SubqueryExpr):
            _refuse(f"{what} nests another subquery — no frozen spelling yet")
    bound = {inner.alias.lower()} if inner.alias else {THIS.lower()}
    for d in descendants(node, deep=True):
        if isinstance(d, ColumnRef) and len(d.column_names) > 1:
            q = d.column_names[0].lower()
            if q in spine and q not in bound:
                _refuse(
                    f"{what} is correlated: {'.'.join(d.column_names)} reads "
                    "the outer row, and a frozen subquery is one value for "
                    "every row"
                )
    renamed = node.model_copy(
        update={"from_table": inner.model_copy(update={"table_name": FIT})}
    )
    return v.model_copy(
        update={"subquery": v.subquery.model_copy(update={"node": renamed})}
    )


def _stripped(expr: Node, spine: frozenset[str]) -> Node:
    """``expr`` with the spine qualifier removed, so it reads the same columns
    when moved into a derived subquery over ``__FIT__``."""

    def strip(v: Node) -> Node | None:
        if (
            isinstance(v, ColumnRef)
            and len(v.column_names) > 1
            and v.column_names[0].lower() in spine
        ):
            # Drop exactly the qualifier: `t.p.v` is the struct path `p.v`,
            # and keeping only the last part would read a different column.
            return v.model_copy(update={"column_names": v.column_names[1:]})
        return None

    return strip(expr) or rebuild(expr, strip, deep=True)


def _uncollated(expr: Node) -> Node:
    """A lookup key without its COLLATE: the window already evaluated under
    the collation, and the params table carries the key's own value."""

    def strip(v: Node) -> Node | None:
        if isinstance(v, Opaque) and v.fields.get("class") == "COLLATE":
            return v.fields["child"]
        return None

    out = rebuild(expr, strip, deep=False)
    while (bare := strip(out)) is not None:
        out = bare
    return out


def _fit_window(stem: str, children: list[Node]) -> Opaque:
    """``stem_fit(children) OVER ()`` — the global scope of the bare sugar."""
    over = _template("SELECT avg(1) OVER ()").select_list[0]
    return over.model_copy(
        update={
            "fields": over.fields
            | {"function_name": f"{stem}_fit", "children": list(children), "alias": ""}
        }
    )


def _fresh_prefix(sql: str) -> str:
    """A prefix no identifier in the author's text starts with. The scan is a
    superset (strings and keywords too) — over-matching only moves the pick."""
    idents = {m.lower() for m in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", sql)}
    for n in itertools.count():
        prefix = f"cf{n or ''}_"
        if not any(i.startswith(prefix) for i in idents):
            return prefix
    raise AssertionError("unreachable")


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_CHAIN = (
    f"marginalize takes one SELECT over {THIS}; write a chain's fit and "
    f"request stages explicitly over {FIT} and {THIS} and pass that text to "
    "SQLProjection(...)"
)


def derive(sql: str, scope: dict[str, Any]) -> str:  # noqa: C901
    """The explicit-``__FIT__`` text a ``__THIS__``-only text means, or the
    refusal — always in the author's own spelling — that says why not."""
    doc = _parse(sql)
    from sql_transform.model._program import _raw_names  # noqa: PLC0415

    if len(doc.statements) != 1:
        _refuse("marginalize takes one statement at a time")
    _raw_names(doc.statements[0].node, scope)
    node = doc.statements[0].node
    for v in (node, *descendants(node, deep=True)):
        if isinstance(v, BaseTable) and v.table_name.upper() == FIT:
            _refuse(
                f"the text already reads {FIT}, so there is nothing to "
                "derive — call SQLProjection(...) directly"
            )
    if not isinstance(node, Select):
        _refuse(
            "a set operation reads the batch more than once — "
            "it has no single spine to join the derived params to"
        )
    if cte_entries(node):
        _refuse(f"a CTE chain has no marginal spelling — {_CHAIN}")
    spine_ref = node.from_table
    if isinstance(spine_ref, SubqueryRef):
        _refuse(f"a derived-table chain has no marginal spelling — {_CHAIN}")
    if not (isinstance(spine_ref, BaseTable) and spine_ref.table_name.upper() == THIS):
        _refuse(
            f"its FROM reads more than {THIS} — a marginalize text is "
            f"{THIS}-only, and the derived joins are marginalize's to write"
        )
    # The spine is re-emitted from table_name + alias alone, so every other
    # clause the ref could carry must refuse rather than silently vanish.
    if spine_ref.column_name_alias:
        _refuse(
            f"a column-alias list on {THIS} has no frozen spelling yet — "
            "alias in the select list instead"
        )
    if spine_ref.sample is not None:
        _refuse(f"a SAMPLE over {THIS} drops rows — a projection cannot")
    if spine_ref.at_clause is not None:
        _refuse(f"an AT clause on {THIS} has no frozen spelling")
    if node.where_clause is not None:
        _refuse(
            f"a WHERE over {THIS} filters the batch — a projection cannot drop rows"
        )
    if node.qualify is not None:
        _refuse(
            f"a QUALIFY over {THIS} filters the batch — a projection cannot drop rows"
        )
    if (
        node.group_expressions
        or node.group_sets
        or node.having is not None
        or node.aggregate_handling == "FORCE_AGGREGATES"
    ):
        _refuse(
            f"a GROUP BY (or HAVING) over {THIS} collapses the batch — "
            "a projection is one row out per row in"
        )
    if node.sample is not None:
        _refuse(f"a SAMPLE over {THIS} drops rows — a projection cannot")
    for m in node.modifiers:
        _refuse(
            f"{_MODIFIERS.get(str(field(m, 'type')), 'a modifier')} changes "
            "the batch's rows, which is the transform's business, not a "
            "projection's"
        )

    prefix = _fresh_prefix(sql)
    alias = spine_ref.alias
    spine = frozenset({THIS.lower()} | ({alias.lower()} if alias else set()))
    # The carrier reads `__FIT__` under the spine's own alias, so only an
    # explicit `__THIS__.` qualifier needs removing there.
    unthis = frozenset({THIS.lower()})
    laterals = frozenset(
        a.lower() for item in node.select_list if (a := str(field(item, "alias") or ""))
    )

    originals: list[str] = []  # the author's executable items, over __FIT__
    omitted: set[str] = set()  # aliases of items the carrier cannot evaluate
    carried: dict[str, str] = {}  # window text -> carrier column
    thetas: dict[str, Node] = {}  # SQL-leaf theta remains a compile-time struct
    window_items: list[str] = []
    key_columns: dict[str, str] = {}  # lookup key text -> carrier column
    key_items: list[str] = []
    # key texts -> (pick name, carrier value columns), first occurrence order
    picks: dict[tuple[str, ...], tuple[str, list[str]]] = {}
    picked: dict[str, str] = {}  # window text -> its pick
    subs: list[tuple[str, str]] = []  # (column, frozen subquery text), one join
    sub_alias: list[str] = []  # allocated on first frozen subquery
    keyed_joins: list[str] = []  # one self-contained LEFT JOIN per keyed scope
    m_names = itertools.count()
    w_numbers = itertools.count()
    items_text: list[str] = []
    raw_scopes: dict[str, Node] = {}
    raw_ctes: list[str] = []
    raw_joins: list[str] = []

    def raw_in(v: Node) -> bool:
        if _is_window(v):
            name = str(v.fields.get("function_name") or "")
        elif isinstance(v, Function) and not v.is_operator:
            name = v.function_name
        else:
            return False
        stem, _, half = name.rpartition("_")
        return _raw(name, scope) is not None or (
            half in ("fit", "transform") and _raw(stem, scope) is not None
        )

    def raw_ref(w: Opaque) -> Node:
        from sql_transform.model._program import _raw_bundle  # noqa: PLC0415

        text = _print_expr(w)
        if text in raw_scopes:
            return raw_scopes[text]
        stem = str(w.fields["function_name"]).removesuffix("_fit")
        keys = [_uncollated(k) for k in _admit(w, scope)]
        features = _raw_bundle(stem, list(w.fields.get("children") or []))
        number = len(raw_scopes)
        params, source = f"{prefix}r{number}", f"{prefix}f{number}"
        iid = f"{prefix}iid"
        projected: list[str] = []

        def project(expr: Node, suffix: str) -> ColumnRef:
            column = f"{prefix}r{number}_{suffix}"
            projected.append(f"({_print_expr(_stripped(expr, unthis))}) AS {column}")
            return _template(f"SELECT {source}.{column}").select_list[0]  # noqa: S608

        key_refs = [project(k, f"k{i}") for i, k in enumerate(keys)]
        feature_refs = [
            _aliased(project(expr, f"f{i}"), name)
            for i, (name, expr) in enumerate(features)
        ]
        packed = _template("SELECT struct_pack(v := 1)").select_list[0]
        packed = packed.model_copy(update={"children": feature_refs})
        predicate = w.fields.get("filter_expr")
        filter_ref = project(predicate, "filter") if predicate is not None else None
        orders = []
        for i, order in enumerate(w.fields.get("arg_orders") or []):
            expression = order.fields["expression"]
            ref = expression if _constant(expression) else project(expression, f"o{i}")
            orders.append(
                order.model_copy(update={"fields": order.fields | {"expression": ref}})
            )
        call = _template("SELECT count(1)").select_list[0]
        order_bys = call.order_bys.model_copy(
            update={"fields": call.order_bys.fields | {"orders": orders}}
        )
        call = call.model_copy(
            update={
                "function_name": f"{stem}_fit",
                "children": [packed],
                "filter_": filter_ref,
                "order_bys": order_bys,
                "alias": "",
            }
        )
        fit_from = FIT + (f" AS {_quoted(alias)}" if alias else "")
        columns = [_print_expr(k) for k in key_refs]
        columns.append(f"{_print_expr(call)} AS {iid}")
        inner = (
            f"SELECT {', '.join(columns)} FROM "  # noqa: S608
            f"(SELECT {', '.join(projected)} FROM {fit_from}) {source}"
        )
        if key_refs:
            inner += " GROUP BY " + ", ".join(_print_expr(k) for k in key_refs)
        raw_ctes.append(f"{params} AS ({inner})")
        on = " AND ".join(
            f"({_print_expr(k)}) IS NOT DISTINCT FROM {params}.{ref.column_names[-1]}"
            for k, ref in zip(keys, key_refs, strict=True)
        )
        raw_joins.append(f"LEFT JOIN {params} ON {on or '1 = 1'}")
        ref = _template(f"SELECT {params}.{iid}").select_list[0]  # noqa: S608
        raw_scopes[text] = ref
        return ref

    def carry(w: Opaque, keys: list[Node]) -> None:
        text = _print_expr(w)
        if text in carried or text in thetas:
            return
        stem, _, half = str(w.fields.get("function_name") or "").rpartition("_")
        if half == "fit" and (leaf := _projection(stem, scope)) is not None:
            from sql_transform.model import _leaf  # noqa: PLC0415

            theta = _leaf.fit_call(stem, leaf, list(w.fields.get("children") or []), w)
            thetas[text] = theta
            for value in (theta, *descendants(theta, deep=True)):
                if _is_window(value):
                    carry(value, _admit(value, scope))
            return
        column = f"{prefix}w{len(carried)}"
        carried[text] = column
        window_items.append(f"{_print_expr(_stripped(w, unthis))} AS {column}")
        texts: dict[str, None] = {}
        for k in keys:
            key = _uncollated(k)
            key_text = _print_expr(key)
            if key_text not in key_columns:
                key_columns[key_text] = f"{prefix}k{len(key_columns)}"
                key_items.append(
                    f"({_print_expr(_stripped(key, unthis))})"
                    f" AS {key_columns[key_text]}"
                )
            texts[key_text] = None
        scope_key = tuple(texts)
        if scope_key not in picks:
            picks[scope_key] = (f"{prefix}m{next(m_names)}", [])
        picks[scope_key][1].append(column)
        picked[text] = picks[scope_key][0]

    def keyed_in(v: Node) -> bool:
        """A keyed projection call: the carrier cannot evaluate its table θ."""
        if _is_window(v):
            name = str(v.fields.get("function_name") or "")
        elif isinstance(v, Function) and not v.is_operator:
            name = v.function_name
        else:
            return False
        stem, _, half = name.rpartition("_")
        p = _projection(name, scope) or (
            _projection(stem, scope) if half in ("fit", "transform") else None
        )
        return p is not None and _keyed(p)

    for item in node.select_list:
        # Every check runs on the author's tree, before any swap: a bottom-up
        # rebuild would replace an inner scope first, and the nesting the
        # refusal names would no longer be there to see. The spine-side walk
        # is deep=False — an admitted subquery freezes wholesale and keeps
        # its own inner scoping (its stars and aggregates are its own).
        walk = [item, *descendants(item, deep=False)]
        if _is_star(item):
            f = item.fields
            if (
                f.get("columns")
                or f.get("expr") is not None
                or any(f.get(k) for k in _STAR_MODIFIERS)
            ):
                _refuse(
                    "a * with COLUMNS, EXCLUDE, REPLACE or RENAME has no "
                    "marginal spelling — name the columns"
                )
            # Qualified by the spine, so no derived params column enters it.
            qualifier = f.get("relation_name") or alias or THIS
            star = item.model_copy(update={"fields": f | {"relation_name": qualifier}})
            items_text.append(_print_expr(star))
            continue
        for v in walk:
            if _is_star(v):
                _refuse(
                    "a * inside an expression has no marginal spelling — "
                    "name the columns"
                )
            if (
                isinstance(v, Opaque)
                and v.fields.get("class") == "POSITIONAL_REFERENCE"
            ):
                _refuse(
                    "a positional reference (#N) resolves by position, which "
                    "the derived joins shift — name the column"
                )
            if _is_window(v):
                fw = v.fields
                _pure(
                    _print_expr(v),
                    [
                        *(fw.get("children") or []),
                        *(fw.get("partitions") or []),
                        *[o.fields["expression"] for o in (fw.get("orders") or [])],
                        *[o.fields["expression"] for o in (fw.get("arg_orders") or [])],
                        *([fw["filter_expr"]] if fw.get("filter_expr") else []),
                        *([fw["start_expr"]] if fw.get("start_expr") else []),
                        *([fw["end_expr"]] if fw.get("end_expr") else []),
                    ],
                    scope,
                    spine,
                    laterals,
                )
            if (
                isinstance(v, Function)
                and not v.is_operator
                and (
                    _projection(v.function_name, scope)
                    or _raw(v.function_name, scope) is not None
                )
            ):
                _pure(_print_expr(v), list(v.children), scope, spine, laterals)

        # Subqueries validate against the author's tree too, before any
        # rebuild could replace an inner one and blind the nesting check.
        for v in (item, *descendants(item, deep=True)):
            if isinstance(v, SubqueryExpr):
                _admit_subquery(v, spine)

        # A projection fit scope is read only by its own transform, inline:
        # a parked θ (`... OVER (...) AS th`) has no one scope to stand for.
        inline: set[int] = set()
        for v in walk:
            if isinstance(v, Function) and not v.is_operator and v.children:
                stem, _, half = v.function_name.rpartition("_")
                c0 = v.children[0]
                if (
                    half == "transform"
                    and (_projection(stem, scope) or _raw(stem, scope) is not None)
                    and _is_window(c0)
                    and str(c0.fields.get("function_name") or "") == f"{stem}_fit"
                ):
                    inline.add(id(c0))
        for v in walk:
            if not _is_window(v):
                continue
            stem, _, half = str(v.fields.get("function_name") or "").rpartition("_")
            if (
                half == "fit"
                and (_projection(stem, scope) or _raw(stem, scope) is not None)
                and id(v) not in inline
            ):
                _refuse(
                    f"{_print_expr(v)} is a fit scope outside its transform — "
                    f"apply it inline, {stem}_transform({stem}_fit(...) OVER "
                    f"(...), ...), or write the fit as an explicit CTE over "
                    f"{FIT} joined to {THIS}"
                )

        # Window scopes, first occurrence first; keyed projection scopes
        # keep their own grouped composition below.
        for v in walk:
            if _is_window(v):
                if not keyed_in(v):
                    keys = _admit(v, scope)
                    if not raw_in(v):
                        carry(v, keys)
            elif (
                isinstance(v, Function)
                and not v.is_operator
                and (p := _projection(v.function_name, scope)) is not None
                and not _keyed(p)
            ):
                carry(_fit_window(v.function_name, list(v.children)), [])

        # The carrier evaluates what the original evaluated: the item itself
        # over __FIT__, its subqueries re-bound there too. An item the
        # carrier cannot evaluate — a keyed θ, or a lateral read of one — is
        # left out; its windows above are carried all the same.
        item_alias = str(field(item, "alias") or "")
        if any(keyed_in(v) or raw_in(v) for v in walk) or any(
            isinstance(v, ColumnRef)
            and len(v.column_names) == 1
            and v.column_names[0].lower() in omitted
            for v in walk
        ):
            if item_alias:
                omitted.add(item_alias.lower())
        else:

            def rebind(v: Node) -> Node | None:
                if isinstance(v, SubqueryExpr):
                    return _admit_subquery(v, spine)
                return None

            fit_item = rebind(item) or rebuild(item, rebind, deep=True)
            printed = _print_expr(_stripped(fit_item, unthis))
            originals.append(
                printed + (f" AS {_quoted(item_alias)}" if item_alias else "")
            )

        swapped_any = False

        def pick_ref(w: Opaque) -> Node:
            nonlocal swapped_any
            swapped_any = True
            text = _print_expr(w)
            if text in thetas:
                return rebuild(
                    thetas[text],
                    lambda value: pick_ref(value) if _is_window(value) else None,
                    deep=True,
                )
            ref = _template("SELECT a.b").select_list[0]
            return ref.model_copy(
                update={
                    "column_names": [picked[text], carried[text]],
                    "alias": str(field(w, "alias") or ""),
                }
            )

        def sub_swap(v: Node) -> Node | None:
            nonlocal swapped_any
            if not isinstance(v, SubqueryExpr):
                return None
            swapped_any = True
            if not sub_alias:
                sub_alias.append(f"{prefix}m{next(m_names)}")
            column = f"{prefix}s{next(w_numbers)}"
            subs.append((column, _print_expr(_admit_subquery(v, spine))))
            ref = _template("SELECT a.b").select_list[0]
            return ref.model_copy(
                update={"column_names": [sub_alias[0], column], "alias": ""}
            )

        def keyed_call(
            stem: str,
            projection: Any,
            fit_children: list[Node],
            this_children: list[Node],
            window: Opaque | None,
            alias: str,
        ) -> Node:
            """The flat keyed lowering (spec M5): effective key = scope keys
            ⊕ internal keys, the scope half NULL-safe, the internal half in
            the author's own operator — θ never carries a table."""
            nonlocal swapped_any
            from sql_transform.model import _leaf  # noqa: PLC0415

            kp = _leaf.keyed_plan(stem, projection)
            fit_fields = _leaf._bundle_fields(stem, fit_children, kp.fit_columns, "fit")
            this_fields = _leaf._bundle_fields(
                stem, this_children, kp.this_columns, "transform"
            )
            stripped_fit = {k: _stripped(v, spine) for k, v in fit_fields.items()}
            keys = _admit(window, scope, keyed_ok=True) if window is not None else []
            key_texts = [_print_expr(_stripped(k, spine)) for k in keys]

            m = f"{prefix}m{next(m_names)}"
            # Exported columns wear derived names: the leaf's own names
            # (store, m, ...) would be ambiguous against the spine's columns
            # in the ON clause and the outputs.
            rename = {
                name.lower(): f"{prefix}c{j}"
                for j, (name, _) in enumerate((*kp.key_items, *kp.agg_items))
            }
            cols = [f"({k}) AS {prefix}k{i}" for i, k in enumerate(key_texts)]
            cols += [
                f"({_print_expr(_leaf._substituted(expr, stripped_fit))})"
                f" AS {rename[name.lower()]}"
                for name, expr in kp.key_items
            ]
            cols += [
                _print_expr(
                    agg.model_copy(
                        update={
                            "children": [
                                _leaf._substituted(c, stripped_fit)
                                for c in agg.children
                            ],
                            "alias": "",
                        }
                    )
                )
                + f" AS {rename[name.lower()]}"
                for name, agg in kp.agg_items
            ]
            by = [f"{prefix}k{i}" for i in range(len(key_texts))]
            by += [rename[name.lower()] for name, _ in kp.key_items]
            inner = f"SELECT {', '.join(cols)} FROM {FIT} GROUP BY {', '.join(by)}"  # noqa: S608
            on = [
                f"({k}) IS NOT DISTINCT FROM {m}.{prefix}k{i}"
                for i, k in enumerate(key_texts)
            ]
            on += [
                f"({_print_expr(this_fields[tc])}) {op} {m}.{rename[pc]}"
                for tc, op, pc in kp.on_pairs
            ]
            keyed_joins.append(f"LEFT JOIN ({inner}) AS {m} ON {' AND '.join(on)}")

            def value(ref: ColumnRef) -> Node:
                if ref.column_names[0].lower() == kp.this_alias:
                    return this_fields[ref.column_names[-1].lower()]
                return ref.model_copy(
                    update={
                        "column_names": [m, rename[ref.column_names[-1].lower()]],
                        "alias": "",
                    }
                )

            def remap(v: Node) -> Node | None:
                return value(v) if isinstance(v, ColumnRef) else None

            packed = []
            for name, expr in kp.outputs:
                replaced = (
                    value(expr)
                    if isinstance(expr, ColumnRef)
                    else rebuild(expr, remap, deep=True)
                )
                packed.append(_aliased(replaced, name))
            swapped_any = True
            out = _leaf._struct_pack(packed)
            return _aliased(out, alias) if alias else out

        def keyed_swap(v: Node) -> Node | None:
            if not (isinstance(v, Function) and not v.is_operator):
                return None
            name = v.function_name
            alias = str(field(v, "alias") or "")
            if (bare := _projection(name, scope)) is not None and _keyed(bare):
                return keyed_call(
                    name, bare, list(v.children), list(v.children), None, alias
                )
            stem, _, half = name.rpartition("_")
            if half != "transform":
                return None
            proj = _projection(stem, scope)
            if proj is None or not _keyed(proj):
                return None
            c0 = v.children[0] if len(v.children) == 2 else None
            if (
                c0 is not None
                and _is_window(c0)
                and str(c0.fields.get("function_name") or "") == f"{stem}_fit"
            ):
                return keyed_call(
                    stem,
                    proj,
                    list(c0.fields.get("children") or []),
                    [v.children[1]],
                    c0,
                    alias,
                )
            _refuse(
                f"{stem} is keyed, so its θ is a table — spell the scope as "
                f"{stem}_transform({stem}_fit(...) OVER (...), ...) in one piece"
            )

        def swap(v: Node) -> Node | None:
            nonlocal swapped_any
            if isinstance(v, Function) and not v.is_operator:
                if _raw(v.function_name, scope) is not None:
                    swapped_any = True
                    theta = raw_ref(_fit_window(v.function_name, list(v.children)))
                    return v.model_copy(
                        update={
                            "function_name": f"{v.function_name}_transform",
                            "children": [theta, *v.children],
                        }
                    )
            if _is_window(v) and raw_in(v):
                swapped_any = True
                return raw_ref(v)
            if (
                isinstance(v, Function)
                and not v.is_operator
                and _projection(v.function_name, scope)
            ):
                # The ONE sugar inside a marginalize text: a bare projection
                # call is the global fit scope — θ crosses the derived join
                # as a value, the transform half stays.
                theta = pick_ref(_fit_window(v.function_name, list(v.children)))
                return v.model_copy(
                    update={
                        "function_name": f"{v.function_name}_transform",
                        "children": [theta, *v.children],
                    }
                )
            return pick_ref(v) if _is_window(v) else None

        # Subqueries first (frozen wholesale, so nothing walks into them),
        # then keyed scopes (their lowering consumes the whole
        # transform(fit OVER w, bundle) call), then the carried scopes.
        item_s = sub_swap(item) or rebuild(item, sub_swap, deep=True)
        item_k = keyed_swap(item_s) or rebuild(item_s, keyed_swap, deep=True)
        swapped = swap(item_k) or rebuild(item_k, swap, deep=True)
        for v in (swapped, *descendants(swapped, deep=True)):
            if not isinstance(v, Function) or v.is_operator:
                continue
            if v.function_name.lower() in _aggregates():
                _refuse(
                    f"{_print_expr(v)} has no OVER: without a scope it is one "
                    "value per batch, not one per row — spell the fit scope: "
                    f"{v.function_name}(...) OVER ()"
                )
            fstem, _, fhalf = v.function_name.rpartition("_")
            if fhalf == "fit" and _projection(fstem, scope):
                _refuse(
                    f"{_print_expr(v)} has no OVER: a fit scope needs one — "
                    "even the global scope is spelled OVER ()"
                )
        # An unaliased item keeps the name DuckDB gives the original: its
        # printed text, which the rewrite would otherwise change.
        name = item_alias or (_print_expr(item) if swapped_any else "")
        printed = _print_expr(swapped)
        items_text.append(printed + (f" AS {_quoted(name)}" if name else ""))

    ctes: list[str] = []
    joins: list[str] = []
    if carried:
        carrier, local = f"{prefix}c", f"{prefix}a"
        fit_from = FIT + (f" AS {_quoted(alias)}" if alias else "")
        # Not injectable: every fragment is either a constant, a gensym'd
        # name, or an expression the oracle itself printed (P9).
        carried_items = ", ".join([*originals, *window_items, *key_items])
        ctes.append(
            f"{carrier} AS (SELECT {carried_items} FROM {fit_from})"  # noqa: S608
        )
        for key_texts, (m, columns) in picks.items():
            cols = [f"{local}.{key_columns[k]}" for k in key_texts]
            cols += [f"{local}.{c}" for c in columns]
            # A flat DISTINCT pick over the carrier: one row per fitted key,
            # since every carried value is a function of its scope's keys.
            ctes.append(
                f"{m} AS (SELECT DISTINCT {', '.join(cols)}"  # noqa: S608
                f" FROM {carrier} AS {local})"
            )
            # Never CROSS JOIN: the printer re-emits it as a comma, which
            # binds looser than a following LEFT JOIN and regroups the tree.
            on = " AND ".join(
                f"({k}) IS NOT DISTINCT FROM {m}.{key_columns[k]}" for k in key_texts
            )
            joins.append(f"LEFT JOIN {m} ON {on or '1 = 1'}")
    ctes.extend(raw_ctes)
    joins.extend(raw_joins)

    if subs:
        cols = ", ".join(f"{text} AS {column}" for column, text in subs)
        joins.append(f"LEFT JOIN (SELECT {cols}) AS {sub_alias[0]} ON 1 = 1")  # noqa: S608

    spine_text = THIS + (f" AS {_quoted(alias)}" if alias else "")
    return (f"WITH {', '.join(ctes)} " if ctes else "") + " ".join(
        [f"SELECT {', '.join(items_text)}", f"FROM {spine_text}", *joins, *keyed_joins]
    )
