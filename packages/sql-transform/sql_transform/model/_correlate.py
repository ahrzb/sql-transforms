"""A correlated ``__FIT__`` subquery, lifted into a keyed table.

Kim's ``NEST-JA`` (TODS 7(3), 1982) is the only rule in the decorrelation
lineage whose temporary relation names the *inner* relation alone::

    Rt(C1..Cn, Cn+1) = SELECT C1..Cn, AGG(Cn+1) FROM R2 GROUP BY C1..Cn

That is exactly a params table. Everything published since that repairs Kim —
Ganski & Wong's outer join for the count bug, Dayal's outerjoin-then-group,
Neumann & Kemper's dependent join, SQL Server's Apply — repairs him by pulling
the *outer* relation into the temporary relation. In an optimizer that is free.
Here the outer relation is ``__THIS__`` and does not exist at fit, so
materialisability is strictly narrower than soundness, and the narrowing is
precisely at *the correlation is a conjunction of equalities*.

Which conjunct goes where is the whole algorithm, and it is a partition — every
conjunct of the author's ``WHERE`` lands in exactly one of three places:

    reaches neither, or ``__FIT__`` only   the params query's own WHERE
    reaches ``__FIT__`` and outward        a grouping key, if it is an equality
    reaches outward only                   the lookup's WHERE

Dropping the third kind is the sharpest edge in the space and has no
counterpart in the literature — a plan rewrite holds both relations, so a
predicate over the outer one alone never needs moving. Measured, dropping it
returns a plausible number where the answer is NULL. It goes in the *lookup's*
WHERE rather than in a guard around it, so that a false one produces no rows,
which is what the author's own filter would have produced: an empty group takes
the empty-input value, and for ``count`` that is 0, not NULL.

Neumann & Kemper decide, Kim emits: the admission test is *does every free
attribute land in an equivalence class with an F-side expression*, which is an
AST check; the emission is Kim's, unmodified.
"""

from collections.abc import Callable, Iterator
from typing import NoReturn

from sql_transform.model._analysis import _names_in, _reads
from sql_transform.model._ast import (
    FIT,
    THIS,
    _aggregates,
    _aliased,
    _parse,
    _print_expr,
    _template,
)
from sql_transform.model._errors import CorrelatedFit
from sql_transform.model._nodes import (
    AstNode,
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
)

# The keyed table's columns. `__` is reserved (P8), so none of these can
# collide with a column the author named.
KEY = "__key_{}"
VALUE = "__value_{}"


# The two operators whose equivalence classes a `GROUP BY` can reproduce, each
# carrying its own null-rejection. `=` never matches a NULL key, so that group
# is unreachable at serving and is not shipped; `IS NOT DISTINCT FROM` does
# match it, so it is kept. P4 says *always use INDF* and P4 is a window rule —
# applied here it invents an answer for a key `=` cannot see.
_EQUALITIES = {
    "COMPARE_EQUAL": "=",
    "COMPARE_NOT_DISTINCT_FROM": "IS NOT DISTINCT FROM",
}


# Raised at serving, not at fit: two fitted groups that are one serving key.
# Predicting it needs `__THIS__`'s types, which construction does not have, so
# the choice is a named error or a quiet wrong number.
#
# It names the two relations rather than spelling them: the message is a string
# literal in the residual, and `__FIT__` inside one reads as an unfrozen
# reference to everything that looks for those.
_FAN_OUT = (
    "fit and serving compare the correlation key at different types, "
    "so one serving key matched more than one fitted group"
)


# Every refusal this module can raise. The set *is* the refusal list: short on
# purpose, and each entry has a row in docs/decorrelation-unsupported.md saying
# what lifting it would take. `test_every_refusal_reason_is_documented` holds
# the two together.
REASONS: dict[str, str] = {
    "not-a-scalar-subquery": (
        "it is not a scalar subquery — EXISTS, IN and the quantified "
        "comparisons each need their own rule"
    ),
    "not-a-select": "it is a set operation rather than a plain SELECT",
    "modifier": "it carries its own ORDER BY, LIMIT or DISTINCT",
    "grouping": "it does its own grouping",
    "sample": "it samples __FIT__, so fit would freeze one draw and ship it",
    "window": "it uses a window function",
    "not-aggregated": "its value is not an aggregate, so there is nothing to key",
    "outside-where": "it reads {ref} outside its WHERE clause",
    "not-an-equality": (
        "the correlation is not a conjunction of equalities — {ref} joins the "
        "two relations some other way"
    ),
}


def decorrelate(
    node: AstNode,
    outer: dict[str, bool],
    reading: dict[str, set[str]],
    freeze: Callable[[Node], str],
) -> AstNode | None:
    """``node`` rewritten to read a keyed table, or ``None`` if it is not ours.

    Claims both halves of DuckDB's ``SUBQUERY``: the scalar one in an
    expression, and the derived table in a ``FROM``. Splicing a member call
    produces the second — ``z((FROM __FIT__ WHERE store = g.store), ...)``
    lands as a correlated derived table — and that shape used to ship the whole
    training set.

    ``freeze`` takes a fit-time query node and returns the parameter name it
    was registered under: the same allocator every other step uses, so a
    generated name can never mean a user's relation.

    Only the subquery's own body is replaced, so the alias, the enclosing
    select list and the row count are untouched by construction.
    """
    if not isinstance(node, SubqueryExpr | SubqueryRef):
        return None
    sub = node.subquery.node
    if THIS in (reads := _reads(sub, reading)) or FIT not in reads:
        return None
    if isinstance(sub, Select):
        sub = _flatten(sub)
    inside = _names_in(sub)
    if not _outward(sub, inside, outer):
        return None  # uncorrelated: the maximal freeze already has this one

    def refuse(reason: str, **fmt: str) -> NoReturn:
        raise CorrelatedFit(
            f"{FIT} subquery correlates out of itself and "
            f"{REASONS[reason].format(**fmt)}, so it cannot be evaluated once "
            "into a keyed table",
            reason,
        )

    if isinstance(node, SubqueryExpr) and node.subquery_type != "SCALAR":
        refuse("not-a-scalar-subquery")
    if not isinstance(sub, Select):
        refuse("not-a-select")
    if sub.modifiers:
        refuse("modifier")
    if sub.group_expressions or sub.having or sub.qualify:
        refuse("grouping")
    if sub.aggregate_handling != "STANDARD_HANDLING":
        refuse("grouping")
    if sub.sample is not None:
        refuse("sample")
    if any(_is_window(v) for v in descendants(sub, deep=True)):
        refuse("window")
    # Every column has to collapse, or the lookup is not one row per key.
    if not all(_aggregated(item) for item in sub.select_list):
        refuse("not-aggregated")

    # Everything but the WHERE clause must be blind to the outer query. The
    # deliberately-broad one: `avg(f.price) - t.price` distributes and
    # `avg(abs(f.price - t.price))` does not, no cheap AST rule sits between
    # them, and the workaround is one edit.
    if stray := _outward(sub.model_copy(update={"where_clause": None}), inside, outer):
        refuse("outside-where", ref=stray[0])

    kept: list[AstNode] = []
    guards: list[AstNode] = []
    keys: list[tuple[AstNode, str, AstNode]] = []
    for conjunct in _conjuncts(sub.where_clause):
        if not (out := _outward(conjunct, inside, outer)):
            kept.append(conjunct)
        elif not _inward(conjunct, inside, outer):
            guards.append(conjunct)
        elif key := _equality(conjunct, inside, outer):
            keys.append(key)
        else:
            refuse("not-an-equality", ref=out[0])

    body = _emit(sub, kept, guards, keys, freeze)
    return node.model_copy(
        update={"subquery": node.subquery.model_copy(update={"node": body})}
    )


def _emit(
    sub: Select,
    kept: list[AstNode],
    guards: list[AstNode],
    keys: list[tuple[AstNode, str, AstNode]],
    freeze: Callable[[Node], str],
) -> Select:
    """Kim's temporary relation, its empty-input probe, and the lookup."""
    grouped = [inner for inner, _, _ in keys]
    values = [
        f"({_print_expr(v)}) AS {VALUE.format(i)}"
        for i, v in enumerate(sub.select_list)
    ]
    table = freeze(
        sub.model_copy(
            update={
                "select_list": _expressions(
                    [
                        f"({_print_expr(k)}) AS {KEY.format(i)}"
                        for i, k in enumerate(grouped)
                    ]
                    + values
                ),
                "where_clause": _where(
                    [_print_expr(c) for c in kept]
                    + [
                        f"({_print_expr(k)}) IS NOT NULL"
                        for k, op, _ in keys
                        if op == "="
                    ]
                ),
                "group_expressions": grouped,
                "group_sets": [list(range(len(grouped)))] if grouped else [],
            }
        )
    )

    # The empty-input value, read off `__FIT__`'s schema and none of its rows.
    # Four strands of the survey each produced a different list of "aggregates
    # that are non-NULL on empty input"; the category does not exist —
    # `count_if(x)` is NULL and `count(x) FILTER (WHERE x)` is 0, the same count
    # spelled twice. A probe needs no list and cannot rot.
    probe = freeze(
        sub.model_copy(
            update={
                "select_list": _expressions(values),
                "where_clause": _template("SELECT 1 WHERE false").where_clause,
                "group_expressions": [],
                "group_sets": [],
            }
        )
    )

    # Hit-ness is counted, never inferred from the value: a group that is
    # present and legitimately NULL is not a miss, and `COALESCE` cannot tell
    # the two apart. Kim's own shape — a correlated aggregate with no GROUP BY,
    # which DuckDB unnests natively. A LEFT JOIN against a one-row anchor would
    # read better and does not survive: "Non-inner join on correlated columns
    # not supported".
    def lookup(i: int) -> str:
        return (
            f"CASE WHEN count(*) > 1 THEN error('{_FAN_OUT}')"  # noqa: S608
            f" WHEN count(*) = 1 THEN any_value({table}.{VALUE.format(i)})"
            f" ELSE (SELECT {VALUE.format(i)} FROM {probe}) END"
        )

    columns = [
        _aliased(made, field(item, "alias") or _print_expr(item))
        for item, made in zip(
            sub.select_list,
            _expressions([lookup(i) for i in range(len(sub.select_list))]),
            strict=True,
        )
    ]
    on = [
        f"{table}.{KEY.format(i)} {op} ({_print_expr(o)})"
        for i, (_, op, o) in enumerate(keys)
    ] + [f"({_print_expr(g)})" for g in guards]
    return sub.model_copy(
        update={
            "select_list": columns,
            "from_table": _template(f"SELECT 1 FROM {table}").from_table,  # noqa: S608
            "where_clause": _where(on),
            "group_expressions": [],
            "group_sets": [],
        }
    )


# ------------------------------------------------------------- the AST parts


def _flatten(sub: Select) -> Select:
    """A ``SELECT * FROM x WHERE p`` in the ``FROM``, merged into its parent.

    Splicing a member call writes one: ``z((FROM __FIT__ WHERE store =
    g.store), ...)`` puts the correlating predicate a level below the
    aggregate. Without this the partition finds nothing to partition and the
    shape refuses for a reason that is an artefact of how it was written —
    which is how the per-group ``LATERAL``, a documented pattern, came to ship
    the whole training set.
    """
    while True:
        ref = sub.from_table
        if not isinstance(ref, SubqueryRef) or ref.column_name_alias or ref.sample:
            return sub
        inner = ref.subquery.node
        if not isinstance(inner, Select) or not _transparent(inner):
            return sub
        base = inner.from_table
        if not isinstance(base, BaseTable):
            return sub
        if ref.alias and base.alias and base.alias != ref.alias:
            return sub  # two names for one relation: leave it alone
        sub = sub.model_copy(
            update={
                "from_table": base.model_copy(
                    update={"alias": ref.alias or base.alias}
                ),
                "where_clause": _where(
                    [
                        _print_expr(c)
                        for c in (inner.where_clause, sub.where_clause)
                        if c
                    ]
                ),
            }
        )


def _transparent(inner: Select) -> bool:
    """Whether ``inner`` projects and reorders nothing, so merging it is safe."""
    return (
        not inner.modifiers
        and not inner.group_expressions
        and not inner.having
        and not inner.qualify
        and inner.sample is None
        and inner.aggregate_handling == "STANDARD_HANDLING"
        and not cte_entries(inner)
        and len(inner.select_list) == 1
        and _plain_star(inner.select_list[0])
    )


def _plain_star(item: AstNode) -> bool:
    """A bare ``*`` — no ``EXCLUDE``, ``REPLACE``, ``RENAME`` or qualifier."""
    if not isinstance(item, Opaque) or item.tag_name != "STAR":
        return False
    return not any(
        item.fields.get(k)
        for k in (
            "relation_name",
            "exclude_list",
            "replace_list",
            "qualified_exclude_list",
            "rename_list",
            "columns",
            "expr",
            "alias",
        )
    )


def _conjuncts(node: AstNode | None) -> Iterator[AstNode]:
    """The top-level ``AND`` operands, flattened. Only these can be moved:
    a correlating predicate under ``OR`` or ``NOT`` does not partition."""
    if node is None:
        return
    if isinstance(node, Opaque) and node.tag_name == "CONJUNCTION_AND":
        for child in node.fields.get("children", ()):
            yield from _conjuncts(child)
    else:
        yield node


def _outward(node: AstNode, inside: set[str], outer: dict[str, bool]) -> list[str]:
    """Qualified references in ``node`` that bind outside it, as written.

    Unqualified names are not outward: DuckDB resolves inward whenever it can,
    and where it cannot, the params query names a column ``__FIT__`` does not
    have and fails loudly at fit rather than answering wrongly.
    """
    return [
        ".".join(v.column_names)
        for v in (node, *descendants(node, deep=True))
        if isinstance(v, ColumnRef)
        and len(v.column_names) >= 2
        and v.column_names[0].lower() not in inside
        and v.column_names[0].lower() in outer
    ]


def _inward(node: AstNode, inside: set[str], outer: dict[str, bool]) -> bool:
    """Whether ``node`` reads anything the subquery itself binds."""
    for v in (node, *descendants(node, deep=True)):
        if not isinstance(v, ColumnRef):
            continue
        parts = v.column_names
        if (
            len(parts) < 2
            or parts[0].lower() in inside
            or parts[0].lower() not in outer
        ):
            return True
    return False


def _equality(
    conjunct: AstNode, inside: set[str], outer: dict[str, bool]
) -> tuple[AstNode, str, AstNode] | None:
    """``(inner expression, operator, outer expression)``, either way round.

    Muralikrishna (VLDB 1992) states the admissible form as ``f1(R) = f2(S)``
    where each function references one relation only. One side purely inner and
    the other purely outer is that condition, as an AST check.
    """
    if not isinstance(conjunct, Opaque) or conjunct.tag_name not in _EQUALITIES:
        return None
    left, right = conjunct.fields.get("left"), conjunct.fields.get("right")
    if not isinstance(left, AstNode) or not isinstance(right, AstNode):
        return None
    for inner, out in ((left, right), (right, left)):
        if not _outward(inner, inside, outer) and not _inward(out, inside, outer):
            return inner, _EQUALITIES[conjunct.tag_name], out
    return None


def _is_window(node: AstNode) -> bool:
    return isinstance(node, Opaque) and node.fields.get("class") == "WINDOW"


def _aggregated(node: AstNode) -> bool:
    """Whether the value is an aggregate over this level's own relation.

    ``deep=False``: an aggregate inside a nested subquery is that subquery's,
    and says nothing about whether *this* select item collapses a group.
    """
    known = _aggregates()
    return any(
        isinstance(v, Function) and v.function_name.lower() in known
        for v in (node, *descendants(node, deep=False))
    )


def _expressions(printed: list[str]) -> list[Node]:
    """Expressions cut from the oracle's own parse of them (P9)."""
    return _parse(f"SELECT {', '.join(printed)}").statements[0].node.select_list


def _where(printed: list[str]) -> Node | None:
    if not printed:
        return None
    return (
        _parse(f"SELECT 1 WHERE {' AND '.join(printed)}")
        .statements[0]
        .node.where_clause
    )
