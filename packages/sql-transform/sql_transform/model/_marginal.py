"""``SQLProjection.marginalize`` — the ``__FIT__`` half derived from a
``__THIS__``-only text.

A rewrite in front of the ordinary constructor (spec M3). Every fit scope —
a window aggregate over the spine — is evaluated over ``__FIT__`` once, per
partition, and joined back NULL-safe: a window puts NULL keys in one
partition, so the join must too. The output is *text*, and it is an ordinary
author text: the derived names live in author space under a fresh prefix
(gensym'd against the author's own identifiers) rather than under ``__cf_``,
precisely so the ordinary constructor — which reserves ``__`` — accepts what
the rewrite emits. That is the attribution gate: every refusal here fires
against the author's own spelling, before the rewrite, and a refusal escaping
from the derived text is our bug.

Implements slices 2–5 of
`docs/specs/2026-08-13-marginalize-design.md`: plain aggregates
and ``PARTITION BY`` scopes; projection scopes (windowed ``tfm_fit``, the
bare ``tfm(x)`` sugar, keyed composition per RFC M5); order-discriminating
``RANGE``/``GROUPS`` scopes (keys = partitions ⊕ order values, DISTINCT
lowering); and uncorrelated scalar subqueries, frozen verbatim over
``__FIT__``.
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

# The whole-partition default frame — with no ORDER BY it covers the entire
# partition, which is exactly what makes the value a function of the keys.
_PLAIN_FRAME = {
    "start": "UNBOUNDED_PRECEDING",
    "end": "CURRENT_ROW_RANGE",
    "exclude_clause": "NO_OTHER",
}
# Everything a plain PARTITION BY scope may carry; any other truthy field is a
# window feature with no frozen spelling, refused by its own name.
_CARRIED = frozenset(
    {"class", "type", "alias", "query_location", "function_name"}
    | {"children", "partitions", "filter_expr", "distinct"}
    | {"schema", "catalog"}  # a qualified spelling of the same aggregate
    | {"orders", "start_expr", "end_expr"}  # value frames, checked in _admit
    | set(_PLAIN_FRAME)
)


def _refuse(detail: str) -> NoReturn:
    raise TransformError(detail)


def _is_window(v: Any) -> bool:
    return isinstance(v, Opaque) and v.fields.get("class") == "WINDOW"


def _projection(name: str, scope: dict[str, Any]) -> Any | None:
    """The projection ``name`` resolves to, or None. Late import:
    ``_projection`` imports this module."""
    from sql_transform.model._projection import SQLProjection  # noqa: PLC0415

    obj = scope.get(name)
    return obj if isinstance(obj, SQLProjection) else None


def _pure(
    what: str,
    parts: list[Node],
    scope: dict[str, Any],
    spine: frozenset[str],
    laterals: frozenset[str],
) -> None:
    """A fit scope's arguments must move intact into ``SELECT ... FROM
    __FIT__``: no nested scope (it would freeze into a column the subquery
    cannot see), no name that only resolves in the spine's own SELECT."""
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


def _keyed(projection: Any) -> bool:
    """Whether the projection's own text joins ``__FIT__`` through keys."""
    return any(probe.keys for probe in projection._probes)


def _admit(w: Opaque, scope: dict[str, Any], keyed_ok: bool = False) -> list[Node]:
    """The scope's partition keys, or the refusal in the author's spelling."""
    f = w.fields
    name = str(f.get("function_name") or "")
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
    if f.get("type") != "WINDOW_AGGREGATE":
        _refuse(
            f"{what} is positional: its value is a row position, "
            "which a join key cannot carry"
        )
    if f.get("orders"):
        # Order-discriminating scopes: RANGE/GROUPS peers share order values,
        # so the value is a function of (partition keys ⊕ order values) —
        # exactly what the frozen join can carry. ROWS is physical position.
        if leaf is not None:
            _refuse(
                f"{what}: an ordered fit scope is a running fit — "
                "per-row θ, still a future feature"
            )
        for key in ("start", "end"):
            bound = str(f.get(key) or "")
            if not (
                bound.startswith("UNBOUNDED") or bound.endswith(("_RANGE", "_GROUPS"))
            ):
                _refuse(
                    f"{what}: a ROWS frame is positional — only value peers "
                    "(RANGE/GROUPS) freeze"
                )
        if f.get("exclude_clause") != "NO_OTHER":
            _refuse(
                f"{what}: EXCLUDE splits value peers, so the value is no "
                "longer a function of the keys"
            )
    else:
        for key, expected in _PLAIN_FRAME.items():
            if f.get(key) != expected:
                _refuse(
                    f"{what} names a frame, and a frame is positional — "
                    "only a whole-partition scope freezes"
                )
    for key, value in f.items():
        if key not in _CARRIED and value not in (None, [], "", False):
            _refuse(f"{what}: its {key.upper()} has no frozen spelling")
    if leaf is not None and (f.get("filter_expr") or f.get("distinct")):
        _refuse(
            f"{what}: FILTER and DISTINCT on a projection fit scope have "
            "no frozen spelling yet"
        )
    if leaf is None and name.lower() not in _aggregates():
        _refuse(f"{what}: {name} is not an aggregate the oracle knows")
    return list(f.get("partitions") or [])


def _admit_subquery(v: SubqueryExpr, spine: frozenset[str]) -> str:
    """The frozen text of a scalar subquery — ``__THIS__`` re-bound to
    ``__FIT__``, everything else verbatim — or the refusal in the author's
    spelling. Admitted, v1: one SELECT over ``__THIS__``, uncorrelated."""
    node = v.subquery.node
    what = _print_expr(v)
    if str(v.subquery_type).upper() != "SCALAR":
        _refuse(
            f"{what}: only a scalar subquery is one frozen value — "
            f"{v.subquery_type} has no frozen spelling yet"
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
    return _print_expr(
        v.model_copy(
            update={"subquery": v.subquery.model_copy(update={"node": renamed})}
        )
    )


def _stripped(expr: Node, spine: frozenset[str]) -> Node:
    """``expr`` with the spine qualifier removed, so it reads the same columns
    when moved into the derived subquery over ``__FIT__``."""

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


def _printed_call(
    name: str,
    children: list[Node],
    spine: frozenset[str],
    distinct: bool = False,
    filter_expr: Node | None = None,
    schema: str = "",
    catalog: str = "",
) -> str:
    """A call with spine qualifiers stripped, printed by the oracle."""
    tpl = _template("SELECT count(1)").select_list[0]
    call = tpl.model_copy(
        update={
            "function_name": name,
            "children": [_stripped(c, spine) for c in children],
            "distinct": distinct,
            "filter_": _stripped(filter_expr, spine) if filter_expr else None,
            "schema_": schema,
            "catalog": catalog,
            "alias": "",
        }
    )
    return _print_expr(call)


def _call_text(w: Opaque, spine: frozenset[str]) -> str:
    """The scope's aggregate as a grouped call — the OVER removed, FILTER,
    DISTINCT and any schema/catalog qualification carried, spine qualifiers
    stripped — printed by the oracle."""
    f = w.fields
    return _printed_call(
        str(f["function_name"]),
        list(f.get("children") or []),
        spine,
        distinct=bool(f.get("distinct")),
        filter_expr=f.get("filter_expr"),
        schema=str(f.get("schema") or ""),
        catalog=str(f.get("catalog") or ""),
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


def derive(sql: str, scope: dict[str, Any]) -> str:  # noqa: C901
    """The explicit-``__FIT__`` text a ``__THIS__``-only text means, or the
    refusal — always in the author's own spelling — that says why not."""
    doc = _parse(sql)
    if len(doc.statements) != 1:
        _refuse("marginalize takes one statement at a time")
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
        _refuse("a CTE has no frozen spelling yet — inline it")
    spine_ref = node.from_table
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
    spine = frozenset(
        {THIS.lower()} | ({spine_ref.alias.lower()} if spine_ref.alias else set())
    )
    laterals = frozenset(
        alias.lower()
        for item in node.select_list
        if (alias := str(field(item, "alias") or ""))
    )
    # key texts (stripped, printed) -> (join alias, [(column, aggregate text)])
    scopes: dict[tuple[str, ...], tuple[str, list[tuple[str, str]]]] = {}
    # (partition ⊕ order) texts -> (join alias, [(column, window text)])
    ordered: dict[tuple[str, ...], tuple[str, list[tuple[str, str]]]] = {}
    subs: list[tuple[str, str]] = []  # (column, frozen subquery text), one join
    sub_alias: list[str] = []  # allocated on first frozen subquery
    keyed_joins: list[str] = []  # one self-contained LEFT JOIN per keyed scope
    m_names = itertools.count()
    w_numbers = itertools.count()
    items_text: list[str] = []

    for item in node.select_list:
        # Purity runs on the author's tree, before any swap: a bottom-up
        # rebuild would replace an inner scope first, and the nesting the
        # refusal names would no longer be there to see. The spine-side walk
        # is deep=False — an admitted subquery freezes wholesale and keeps
        # its own inner scoping (its stars and aggregates are its own).
        for v in (item, *descendants(item, deep=False)):
            if isinstance(v, Opaque) and v.fields.get("class") == "STAR":
                _refuse(
                    "a * would read the derived params tables too — name the columns"
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
                and _projection(v.function_name, scope)
            ):
                _pure(_print_expr(v), list(v.children), scope, spine, laterals)

        # Subqueries validate against the author's tree too, before any
        # rebuild could replace an inner one and blind the nesting check.
        for v in (item, *descendants(item, deep=True)):
            if isinstance(v, SubqueryExpr):
                _admit_subquery(v, spine)

        swapped_any = False

        def sub_swap(v: Node) -> Node | None:
            nonlocal swapped_any
            if not isinstance(v, SubqueryExpr):
                return None
            swapped_any = True
            if not sub_alias:
                sub_alias.append(f"{prefix}m{next(m_names)}")
            column = f"{prefix}w{next(w_numbers)}"
            subs.append((column, _admit_subquery(v, spine)))
            ref = _template("SELECT a.b").select_list[0]
            return ref.model_copy(
                update={"column_names": [sub_alias[0], column], "alias": v.alias}
            )

        def frozen_ref(key_texts: tuple[str, ...], agg_text: str, alias: str) -> Node:
            nonlocal swapped_any
            swapped_any = True
            if key_texts not in scopes:
                scopes[key_texts] = (f"{prefix}m{next(m_names)}", [])
            m, columns = scopes[key_texts]
            column = f"{prefix}w{next(w_numbers)}"
            columns.append((column, agg_text))
            ref = _template("SELECT a.b").select_list[0]
            return ref.model_copy(update={"column_names": [m, column], "alias": alias})

        def ordered_ref(key_texts: tuple[str, ...], w_text: str, alias: str) -> Node:
            nonlocal swapped_any
            swapped_any = True
            if key_texts not in ordered:
                ordered[key_texts] = (f"{prefix}m{next(m_names)}", [])
            m, columns = ordered[key_texts]
            column = f"{prefix}w{next(w_numbers)}"
            columns.append((column, w_text))
            ref = _template("SELECT a.b").select_list[0]
            return ref.model_copy(update={"column_names": [m, column], "alias": alias})

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
            if (
                isinstance(v, Function)
                and not v.is_operator
                and _projection(v.function_name, scope)
            ):
                # The ONE sugar inside a marginalize text: a bare projection
                # call is the global fit scope, frozen keyless — θ crosses
                # the derived join as a value, the transform half stays.
                theta = frozen_ref(
                    (),
                    _printed_call(f"{v.function_name}_fit", list(v.children), spine),
                    "",
                )
                return v.model_copy(
                    update={
                        "function_name": f"{v.function_name}_transform",
                        "children": [theta, *v.children],
                    }
                )
            if not _is_window(v):
                return None
            keys = _admit(v, scope)
            order_keys = [
                o.fields["expression"] for o in (v.fields.get("orders") or [])
            ]
            if order_keys:
                # An order-discriminating scope: keys = partitions ⊕ order
                # values, and the window itself reruns over __FIT__ — the
                # DISTINCT rows are unique per key by the value-peer rule.
                key_texts = tuple(
                    _print_expr(_stripped(k, spine)) for k in (*keys, *order_keys)
                )
                return ordered_ref(
                    key_texts,
                    _print_expr(_stripped(v, spine)),
                    str(v.fields.get("alias") or ""),
                )
            key_texts = tuple(_print_expr(_stripped(k, spine)) for k in keys)
            return frozen_ref(
                key_texts, _call_text(v, spine), str(v.fields.get("alias") or "")
            )

        # Subqueries first (frozen wholesale, so nothing walks into them),
        # then keyed scopes (their lowering consumes the whole
        # transform(fit OVER w, bundle) call), then the generic swap.
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
        alias = str(field(item, "alias") or "")
        if swapped_any and not alias:
            _refuse(f"{_print_expr(item)} holds a fit scope and no name — alias it")
        items_text.append(
            _print_expr(swapped) + (f" AS {_quoted(alias)}" if alias else "")
        )

    joins = []
    for key_texts, (alias, columns) in scopes.items():
        cols = [f"({k}) AS {prefix}k{i}" for i, k in enumerate(key_texts)]
        cols += [f"{agg} AS {column}" for column, agg in columns]
        # Not injectable: every fragment is either a constant, a gensym'd
        # name, or an expression the oracle itself printed (P9).
        inner = f"SELECT {', '.join(cols)} FROM {FIT}"  # noqa: S608
        if key_texts:
            # GROUP BY the derived alias, never the raw expression: the text
            # is re-printed with redundant parens dropped, and a bare integer
            # literal would decay into a positional ordinal.
            inner += " GROUP BY " + ", ".join(
                f"{prefix}k{i}" for i in range(len(key_texts))
            )
            on = " AND ".join(
                f"({k}) IS NOT DISTINCT FROM {alias}.{prefix}k{i}"
                for i, k in enumerate(key_texts)
            )
        else:
            # Never CROSS JOIN: the printer re-emits it as a comma, which
            # binds looser than a following LEFT JOIN and regroups the tree.
            # Against a one-row side the two spellings are the same relation —
            # the same respelling the row path does (`_flattened`).
            on = "1 = 1"
        joins.append(f"LEFT JOIN ({inner}) AS {alias} ON {on}")

    for key_texts, (alias, columns) in ordered.items():
        cols = [f"({k}) AS {prefix}k{i}" for i, k in enumerate(key_texts)]
        cols += [f"{w} AS {column}" for column, w in columns]
        # DISTINCT, not GROUP BY: the windows rerun verbatim over __FIT__,
        # and every column is a function of the key tuple, so DISTINCT
        # collapses to exactly one row per key — uniqueness by construction.
        inner = f"SELECT DISTINCT {', '.join(cols)} FROM {FIT}"  # noqa: S608
        on = " AND ".join(
            f"({k}) IS NOT DISTINCT FROM {alias}.{prefix}k{i}"
            for i, k in enumerate(key_texts)
        )
        joins.append(f"LEFT JOIN ({inner}) AS {alias} ON {on}")

    if subs:
        cols = ", ".join(f"{text} AS {column}" for column, text in subs)
        joins.append(f"LEFT JOIN (SELECT {cols}) AS {sub_alias[0]} ON 1 = 1")  # noqa: S608

    spine_text = THIS + (f" AS {_quoted(spine_ref.alias)}" if spine_ref.alias else "")
    return " ".join(
        [f"SELECT {', '.join(items_text)}", f"FROM {spine_text}", *joins, *keyed_joins]
    )
