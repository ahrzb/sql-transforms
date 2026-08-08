"""Freezing: the rewrite from a two-parameter text into params + residual.

> Every maximal subquery whose leaves are all ``__FIT__`` and constants is
> evaluated once at fit and replaced by a table.

All the analysis — and every refusal freezing can raise — happens here, at
construction, before any data exists.

Nothing is mutated: each step returns a new subtree, so a node handed to
``_reads`` is the node that was analysed rather than whatever a later pass
turned it into.
"""

from sql_transform.model._analysis import _bindings_at, _correlation, _reads
from sql_transform.model._ast import (
    FIT,
    THIS,
    _deserialize,
    _is_recursive_cte,
    _one_item,
    _select_star,
    _statement,
)
from sql_transform.model._correlate import decorrelate
from sql_transform.model._errors import CorrelatedFit, WholeTrainingSet
from sql_transform.model._nodes import (
    AstNode,
    BaseTable,
    CteEntry,
    Document,
    Node,
    cte_entries,
    descendants,
    field,
    is_query,
    rebuild,
    with_cte_entries,
)


def _pin_derived_names[N](node: N) -> N:
    """Freeze the output column names before the expressions move.

    DuckDB names an unaliased select item after its own printed text, so
    replacing a frozen subquery with ``SELECT * FROM __param_0`` renamed the
    column to that — ``get_feature_names_out()`` handed back an internal
    parameter name, and the schema stopped matching ``run``'s.

    Written as an explicit alias first, so the name survives the rewrite. Only
    items that contain a query node are touched; a plain column reference is
    named after its last path part and is unaffected by any of this.
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


# The one edit that turns a refused retention into an allowed one, and the
# reason it is worth asking for: a subquery names the rows *and* drops the
# columns, so the artifact is smaller as well as legible.
_RETAIN_HINT = (
    "Wrap the __FIT__ reference in a subquery selecting the rows and columns "
    "you need — `(SELECT ... FROM __FIT__) f` — so the artifact's size is "
    "visible in the text"
)


def _refuse_whole_fit(node: Node, why: str, *, deep: bool) -> None:
    """A bare ``FROM __FIT__`` that no rewrite reached.

    It used to become a parameter holding every row and every column of the
    training set — correct, and reported by ``len(params)``, but the artifact's
    size was then a fact about freezing rather than about the text. Refused
    instead. Retention is still available and takes one edit: name the rows you
    want in a subquery, and the query's value *is* those rows.
    """
    for v in descendants(node, deep=deep):
        if isinstance(v, BaseTable) and v.table_name == FIT:
            raise WholeTrainingSet(
                f"{why} would put the whole training set in the artifact. "
                + _RETAIN_HINT
            )


def _plan(doc: Document) -> tuple[list[tuple[str, Node]], Node]:
    """Rewrite ``doc`` into the residual, returning the fit steps.

    A step is ``(param_name, node)``, in dependency order: running them against
    a connection with ``__FIT__`` bound, registering each result as it lands,
    produces every table the residual needs. All the analysis — and every
    refusal — happens here, at construction, before any data exists.

    The step is kept as a node rather than printed SQL because ``fit`` has to
    rename its free references per execution, the same way serving does, and
    renaming a node is a walk while renaming text is a guess.
    """
    steps: list[tuple[str, Node]] = []
    taken: set[str] = set()

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
        # A recursive CTE's self-reference is bound by the enclosing entry key,
        # not by anything inside the body, so hoisting the body into a
        # standalone statement leaves that name unbound. Left live instead: the
        # training set becomes the parameter, which costs params size and is
        # correct. Freezing it properly means reconstructing the WITH RECURSIVE
        # wrapper, which is worth doing only if it ever matters.
        #
        # Nothing inside may be frozen, not just the body as a whole: the
        # self-reference is visible to every arm but bound by none of them, so
        # hoisting any part of it out leaves that name dangling.
        if _is_recursive_cte(sub):
            _refuse_whole_fit(sub, "a recursive CTE reading __FIT__", deep=True)
            return sub
        if FIT in reads and THIS not in reads:
            match _correlation(sub, outer):
                case None:
                    return freeze(sub, ctes, hint)  # maximal: never refreezes
                case (reference, _):
                    # `decorrelate` has already had its turn on the shapes it
                    # claims — this is what it left. Which side the correlation
                    # reaches no longer separates two outcomes: reaching a
                    # `__FIT__`-only relation used to fall through and ship the
                    # training set, and that is not an outcome any more.
                    raise CorrelatedFit(
                        f"{FIT} subquery references {reference} from the "
                        "outer query, so it cannot be evaluated once into "
                        "a table"
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
            return decorrelate(v, outer, reading, lambda sub: freeze_into(sub, ctes))

        node = rebuild(node, lifted, deep=False)

        def nested(v: AstNode) -> AstNode | None:
            return visit(v, ctes, outer, None, reading) if is_query(v) else None

        node = rebuild(node, nested, deep=False)
        _refuse_whole_fit(node, f"a bare `FROM {FIT}` beside {THIS}", deep=False)
        return node

    residual = visit(doc.statements[0].node, [], {}, None, {})
    return steps, residual
