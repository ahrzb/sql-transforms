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

Implements slice 2 of `docs/superpowers/specs/2026-08-13-marginalize-design.md`
(plain aggregates, ``PARTITION BY`` scopes). Projection scopes are slice 3,
the widened window vocabulary (``RANGE``/``GROUPS``, subqueries) slice 5.
"""

import itertools
import re
from typing import Any, NoReturn

from sql_transform.model._ast import (
    FIT,
    THIS,
    _aggregates,
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
    | set(_PLAIN_FRAME)
)


def _refuse(detail: str) -> NoReturn:
    raise TransformError(detail)


def _is_window(v: Any) -> bool:
    return isinstance(v, Opaque) and v.fields.get("class") == "WINDOW"


def _projection_stem(name: str, scope: dict[str, Any]) -> str | None:
    """The projection a call name means — the name itself or its split-half
    stem — or None. Late import: ``_projection`` imports this module."""
    from sql_transform.model._projection import SQLProjection  # noqa: PLC0415

    stem, _, half = name.rpartition("_")
    for candidate in (name, stem if half in ("fit", "transform") else None):
        if candidate and isinstance(scope.get(candidate), SQLProjection):
            return candidate
    return None


def _admit(w: Opaque, scope: dict[str, Any]) -> list[Node]:
    """The scope's partition keys, or the refusal in the author's spelling."""
    f = w.fields
    name = str(f.get("function_name") or "")
    what = _print_expr(w)
    if stem := _projection_stem(name, scope):
        _refuse(
            f"{stem} is a projection, and marginalize does not freeze "
            f"projection scopes yet — write the {FIT} half yourself "
            "(SQLProjection(...))"
        )
    if f.get("type") != "WINDOW_AGGREGATE":
        _refuse(
            f"{what} is positional: its value is a row position, "
            "which a join key cannot carry"
        )
    if f.get("orders"):
        _refuse(
            f"{what} has an ORDER BY: a running scope is per-row, "
            "not per-partition — no join key carries it"
        )
    for key, expected in _PLAIN_FRAME.items():
        if f.get(key) != expected:
            _refuse(
                f"{what} names a frame, and a frame is positional — "
                "only a whole-partition scope freezes"
            )
    for key, value in f.items():
        if key not in _CARRIED and value not in (None, [], "", False):
            _refuse(f"{what}: its {key.upper()} has no frozen spelling")
    if name.lower() not in _aggregates():
        _refuse(f"{what}: {name} is not an aggregate the oracle knows")
    inner = [
        *(f.get("children") or []),
        *(f.get("partitions") or []),
        *([f["filter_expr"]] if f.get("filter_expr") else []),
    ]
    for c in inner:
        for d in (c, *descendants(c, deep=True)):
            if _is_window(d):
                _refuse(f"{what} nests {_print_expr(d)} inside a fit scope")
            if (
                isinstance(d, Function)
                and not d.is_operator
                and d.function_name.lower() in _aggregates()
            ):
                _refuse(
                    f"{what} nests the aggregate {d.function_name} inside a fit scope"
                )
    return list(f.get("partitions") or [])


def _stripped(expr: Node, spine: frozenset[str]) -> Node:
    """``expr`` with the spine qualifier removed, so it reads the same columns
    when moved into the derived subquery over ``__FIT__``."""

    def strip(v: Node) -> Node | None:
        if (
            isinstance(v, ColumnRef)
            and len(v.column_names) > 1
            and v.column_names[0].lower() in spine
        ):
            return v.model_copy(update={"column_names": [v.column_names[-1]]})
        return None

    return strip(expr) or rebuild(expr, strip, deep=True)


def _call_text(w: Opaque, spine: frozenset[str]) -> str:
    """The scope's aggregate as a grouped call — the OVER removed, FILTER and
    DISTINCT carried, spine qualifiers stripped — printed by the oracle."""
    f = w.fields
    tpl = _template("SELECT count(1)").select_list[0]
    call = tpl.model_copy(
        update={
            "function_name": str(f["function_name"]),
            "children": [_stripped(c, spine) for c in f.get("children") or []],
            "distinct": bool(f.get("distinct")),
            "filter_": _stripped(f["filter_expr"], spine)
            if f.get("filter_expr")
            else None,
            "alias": "",
        }
    )
    return _print_expr(call)


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
    # key texts (stripped, printed) -> [(column name, aggregate text)]
    scopes: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    w_numbers = itertools.count()
    items_text: list[str] = []

    for item in node.select_list:
        for v in (item, *descendants(item, deep=True)):
            if isinstance(v, Opaque) and v.fields.get("class") == "STAR":
                _refuse(
                    "a * would read the derived params tables too — name the columns"
                )
            if is_query(v):
                _refuse("a subquery expression has no frozen spelling yet")
            if isinstance(v, Function) and (
                stem := _projection_stem(v.function_name, scope)
            ):
                _refuse(
                    f"{stem} is a projection, and marginalize does not freeze "
                    f"projection scopes yet — write the {FIT} half yourself "
                    "(SQLProjection(...))"
                )

        swapped_any = False

        def swap(v: Node) -> Node | None:
            nonlocal swapped_any
            if not _is_window(v):
                return None
            keys = _admit(v, scope)
            key_texts = tuple(_print_expr(_stripped(k, spine)) for k in keys)
            columns = scopes.setdefault(key_texts, [])
            column = f"{prefix}w{next(w_numbers)}"
            columns.append((column, _call_text(v, spine)))
            swapped_any = True
            ref = _template("SELECT a.b").select_list[0]
            m = f"{prefix}m{list(scopes).index(key_texts)}"
            return ref.model_copy(update={"column_names": [m, column], "alias": ""})

        swapped = swap(item) or rebuild(item, swap, deep=True)
        for v in (swapped, *descendants(swapped, deep=True)):
            if (
                isinstance(v, Function)
                and not v.is_operator
                and v.function_name.lower() in _aggregates()
            ):
                _refuse(
                    f"{_print_expr(v)} has no OVER: without a scope it is one "
                    "value per batch, not one per row — spell the fit scope: "
                    f"{v.function_name}(...) OVER ()"
                )
        alias = str(field(item, "alias") or "")
        if swapped_any and not alias:
            _refuse(f"{_print_expr(item)} holds a fit scope and no name — alias it")
        items_text.append(
            _print_expr(swapped) + (f" AS {_quoted(alias)}" if alias else "")
        )

    joins = []
    for m, (key_texts, columns) in enumerate(scopes.items()):
        cols = [f"({k}) AS {prefix}k{i}" for i, k in enumerate(key_texts)]
        cols += [f"{agg} AS {column}" for column, agg in columns]
        # Not injectable: every fragment is either a constant, a gensym'd
        # name, or an expression the oracle itself printed (P9).
        inner = f"SELECT {', '.join(cols)} FROM {FIT}"  # noqa: S608
        alias = f"{prefix}m{m}"
        if key_texts:
            inner += " GROUP BY " + ", ".join(f"({k})" for k in key_texts)
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

    spine_text = THIS + (f" AS {_quoted(spine_ref.alias)}" if spine_ref.alias else "")
    return " ".join([f"SELECT {', '.join(items_text)}", f"FROM {spine_text}", *joins])
