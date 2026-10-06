"""Freezing: the rewrite from a two-parameter text into params + residual.

> Every maximal subquery whose leaves are all ``__FIT__`` and constants is
> evaluated once at fit and replaced by a table.

Planning is structural and happens before data exists. Schema-dependent
correlation binding checks are deferred to fit.

Nothing is mutated: each step returns a new subtree, so a node handed to
``_reads`` is the node that was analysed rather than whatever a later pass
turned it into.
"""

from sql_transform._analysis import _bindings_at, _correlation, _reads
from sql_transform._ast import (
    FIT,
    THIS,
    _deserialize,
    _is_recursive_cte,
    _one_item,
    _select_star,
    _statement,
)
from sql_transform._correlate import decorrelate
from sql_transform._errors import CorrelatedFit, WholeTrainingSet
from sql_transform._nodes import (
    AstNode,
    BaseTable,
    ColumnRef,
    CteEntry,
    Document,
    Node,
    Select,
    cte_entries,
    descendants,
    field,
    is_query,
    rebuild,
    with_cte_entries,
)


def _pin_derived_names[N](node: N) -> N:
    """Freeze the output column names before the expressions move.

    DuckDB derives unaliased expression names from printed SQL. Pinning names
    before freezing preserves the authored schema instead of exposing internal
    parameter names. Only expressions containing a query node need pinning;
    plain columns retain their last path part as the name.
    """
    items = field(node, "select_list")
    if not items:
        return node
    pinned = []
    for item in items:
        if not isinstance(item, AstNode) or field(item, "alias"):
            pinned.append(item)
            continue
        if not any(is_query(v) for v in descendants(item, deep=True)):
            pinned.append(item)
            continue
        printed = _deserialize(_statement(_one_item(item)))
        pinned.append(item.model_copy(update={"alias": printed[len("SELECT ") :]}))
    return node.model_copy(update={"select_list": pinned})


_RETAIN_HINT = (
    "Wrap the __FIT__ reference in a subquery selecting the rows and columns "
    "you need — `(SELECT ... FROM __FIT__) f` — so the artifact's size is "
    "visible in the text"
)


def _refuse_whole_fit(node: Node, why: str, *, deep: bool) -> None:
    """Refuse a bare FIT reference left outside a frozen subquery.

    Retaining training rows requires an explicit subquery selecting the rows
    and columns to keep, so artifact size remains visible in the authored SQL.
    """
    for v in descendants(node, deep=deep):
        if isinstance(v, BaseTable) and v.table_name == FIT:
            raise WholeTrainingSet(
                f"{why} would put the whole training set in the artifact. "
                + _RETAIN_HINT
            )


def _frozen_pick(sub: Node, ctes: list[CteEntry], params: set[str]) -> bool:
    """A flat ``SELECT DISTINCT`` of one frozen CTE's columns, and nothing else.

    This pick reads a previously frozen parameter rather than FIT directly.
    Window marginalization emits it: a carrier CTE over ``__FIT__`` is frozen whole,
    and each scope's lookup table is a distinct pick of its key and value
    columns. Left live, serving would ship the whole carrier — every fit row —
    to recompute a pick whose answer was fixed at fit.

    Structural and deliberately narrow: every item a column qualified by the
    one source's alias, no clause, expression or nested query, and the source
    an enclosing CTE already rewritten to ``SELECT * FROM`` an earlier step's
    parameter. A live or ``__THIS__``-reading CTE never qualifies.
    """
    if not (
        isinstance(sub, Select)
        and not cte_entries(sub)
        and len(sub.modifiers) == 1
        and field(sub.modifiers[0], "type") == "DISTINCT_MODIFIER"
        and not field(sub.modifiers[0], "distinct_on_targets")
        and sub.where_clause is None
        and not sub.group_expressions
        and not sub.group_sets
        and sub.aggregate_handling == "STANDARD_HANDLING"
        and sub.having is None
        and sub.sample is None
        and sub.qualify is None
        and sub.select_list
        and isinstance(src := sub.from_table, BaseTable)
        and not (src.schema_name or src.catalog_name or src.column_name_alias)
        and src.sample is None
        and src.at_clause is None
    ):
        return False
    alias = (src.alias or src.table_name).lower()
    names: set[str] = set()
    for item in sub.select_list:
        if not (
            isinstance(item, ColumnRef)
            and len(item.column_names) == 2
            and item.column_names[0].lower() == alias
        ):
            return False
        names.add((item.alias or item.column_names[1]).lower())
    if len(names) != len(sub.select_list):
        return False  # a duplicate output name is not one the pick can keep
    # The innermost definition in scope: `ctes` is in definition order.
    source = next(
        (e for e in reversed(ctes) if e.key.lower() == src.table_name.lower()), None
    )
    if source is None:
        return False
    body = source.value.query.node
    return (
        isinstance(body, Select)
        and isinstance(body.from_table, BaseTable)
        and body.from_table.table_name in params
        and body == _select_star(body.from_table.table_name)
    )


def _tables(node: Node) -> set[str]:
    """Every table name read anywhere inside ``node``, CTE references
    included, folded. Over-approximates — an inner CTE shadowing an outer one
    counts as a read of both — so pruning by it never drops a live name."""
    return {
        name.lower()
        for v in (node, *descendants(node, deep=True))
        if isinstance(name := field(v, "table_name"), str) and name
    }


def _referenced(node: Node, params: list[str]) -> set[str]:
    """The ``params`` that ``node`` reads."""
    read = _tables(node)
    return {p for p in params if p.lower() in read}


def _drop_dead_ctes[N](node: N) -> N:
    """``node`` without the CTE definitions nothing reads, at every level.

    After freezing, a CTE that only fed a frozen step is dead in what remains:
    left in, it would keep its parameter in the artifact for no reader.
    Definitions reached from the body, directly or through other kept
    definitions, all stay.
    """

    def prune(v: AstNode) -> AstNode | None:
        entries = cte_entries(v)
        if not entries:
            return None
        live = _tables(with_cte_entries(v, []))
        while True:
            kept = [e for e in entries if e.key.lower() in live]
            more = set().union(*(_tables(e.value.query.node) for e in kept))
            if more <= live:
                break
            live |= more
        return None if len(kept) == len(entries) else with_cte_entries(v, kept)

    pruned = rebuild(node, prune, deep=True)
    if not isinstance(pruned, AstNode):
        return pruned
    return prune(pruned) or pruned  # type: ignore[return-value]


def _plan(doc: Document) -> tuple[list[tuple[str, Node]], Node, set[str]]:
    """Rewrite ``doc`` into the residual, returning the fit steps.

    A step is ``(param_name, node)``, in dependency order: running them against
    a connection with ``__FIT__`` bound, registering each result as it lands,
    produces every table the residual needs. Structural refusals happen at
    construction; schema-dependent binding is checked during fit.

    The third return is the set of qualifiers a lifted correlation read as
    *outer*. Whether one of those is instead a nested column of ``__FIT__``,
    which DuckDB binds in preference, needs a schema; `refuse_if_shadowed`
    settles it at fit.

    The step is kept as a node rather than printed SQL because ``fit`` has to
    rename its free references per execution, the same way serving does, and
    renaming a node is a walk while renaming text is a guess.

    After the visit, dead CTE definitions are dropped from the residual and
    from every step, and only the steps the residual reaches — directly or
    through the steps that feed it — are returned.
    """
    steps: list[tuple[str, Node]] = []
    taken: set[str] = set()
    shadowable: set[str] = set()

    def name(hint: str | None) -> str:
        base = f"__param_{hint}" if hint else f"__param_{len(steps)}"
        candidate, n = base, 1
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        taken.add(candidate)
        return candidate

    def freeze_into(sub: Node, ctes: list[CteEntry], hint: str | None = None) -> str:
        # Carry the enclosing CTEs in: a frozen subtree may reference one, and
        # by now their own definitions have been rewritten to frozen tables.
        frozen = with_cte_entries(sub, list(ctes) + list(cte_entries(sub)))
        param = name(hint)
        steps.append((param, frozen))
        return param

    def freeze(sub: Node, ctes: list[CteEntry], hint: str | None) -> Node:
        return _select_star(freeze_into(sub, ctes, hint))

    def visit(
        sub: Node,
        ctes: list[CteEntry],
        outer: dict[str, bool],
        hint: str | None,
        reading: dict[str, set[str]],
    ) -> Node:
        reads = _reads(sub, reading)
        # A recursive CTE binds its own name outside its body. Hoisting any
        # part of that body would leave self-references unbound, so it stays
        # live; bare FIT retention still requires an explicit subquery.
        if _is_recursive_cte(sub):
            _refuse_whole_fit(sub, "a recursive CTE reading __FIT__", deep=True)
            return sub
        if _frozen_pick(sub, ctes, {param for param, _ in steps}):
            return freeze(sub, ctes, hint)
        if FIT in reads and THIS not in reads:
            match _correlation(sub, outer):
                case None:
                    return freeze(sub, ctes, hint)  # maximal: never refreezes
                case (reference, _):
                    # `decorrelate` has already had its turn on the shapes it
                    # claims — a scalar subquery or a derived table — so what
                    # reaches here is a correlated subtree in neither position:
                    # a CTE definition, or an arm of a set operation. There is
                    # nowhere to put a lookup, because there is no subquery
                    # body to replace.
                    raise CorrelatedFit(
                        f"{FIT} subtree references {reference} from the outer "
                        "query and is not a subquery — a CTE definition or a "
                        "set-operation arm has no body to replace with a "
                        "lookup, so it cannot be evaluated once into a table",
                        "not-in-a-subquery",
                    )
        return descend(sub, ctes, outer, reading)

    def descend(
        node: Node,
        ctes: list[CteEntry],
        outer: dict[str, bool],
        reading: dict[str, set[str]],
    ) -> Node:
        node = _pin_derived_names(node)
        ctes = list(ctes)
        reading = dict(reading)
        rewritten: list[CteEntry] = []
        for entry in cte_entries(node):
            body = visit(entry.value.query.node, ctes, outer, entry.key, reading)
            entry = entry.model_copy(
                update={
                    "value": entry.value.model_copy(
                        update={
                            "query": entry.value.query.model_copy(update={"node": body})
                        }
                    )
                }
            )
            # After visiting, not before: a frozen body reads nothing, and one
            # left live has had its `__FIT__` rewritten already. Either way this
            # is what a later reference to the name actually depends on.
            reading[entry.key.lower()] = _reads(body, reading)
            ctes.append(entry)
            rewritten.append(entry)
        node = with_cte_entries(node, rewritten)
        outer = outer | _bindings_at(node, reading)

        # Before `visit` descends, not after: what this claims is exactly what
        # `visit` would otherwise refuse. `_pin_derived_names` has run, so a
        # rewritten select item still carries the output column name it had.
        def lifted(v: AstNode) -> AstNode | None:
            return decorrelate(
                v, outer, reading, lambda sub: freeze_into(sub, ctes), shadowable
            )

        node = rebuild(node, lifted, deep=False)

        def nested(v: AstNode) -> AstNode | None:
            return visit(v, ctes, outer, None, reading) if is_query(v) else None

        node = rebuild(node, nested, deep=False)
        _refuse_whole_fit(node, f"a bare `FROM {FIT}` beside {THIS}", deep=False)
        return node

    residual = _drop_dead_ctes(visit(doc.statements[0].node, [], {}, None, {}))
    pruned = [(param, _drop_dead_ctes(node)) for param, node in steps]
    names = [param for param, _ in pruned]
    needed = _referenced(residual, names)
    for param, node in reversed(pruned):
        if param in needed:
            needed |= _referenced(node, names)
    return [s for s in pruned if s[0] in needed], residual, shadowable
