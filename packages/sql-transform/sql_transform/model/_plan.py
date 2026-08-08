"""Freezing: the rewrite from a two-parameter text into params + residual.

> Every maximal subquery whose leaves are all ``__FIT__`` and constants is
> evaluated once at fit and replaced by a table.

All the analysis — and every refusal freezing can raise — happens here, at
construction, before any data exists.

Nothing is mutated: each step returns a new subtree, so a node handed to
``_reads`` is the node that was analysed rather than whatever a later pass
turned it into.
"""

from collections.abc import Callable

from sql_transform.model._analysis import _bindings_at, _correlation, _reads
from sql_transform.model._ast import (
    FIT,
    THIS,
    _deserialize,
    _is_recursive_cte,
    _select_star,
    _statement,
    _template,
)
from sql_transform.model._errors import CorrelatedFit
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


def _one_item(item: Node) -> Node:
    """``SELECT <item>`` as a query node, for asking the oracle what it would
    call that column."""
    return _template("SELECT 1").model_copy(update={"select_list": [item]})


def _rewrite_fit_refs(node: Node, mint: "Callable[[], str]", *, deep: bool) -> Node:
    """Every bare ``FROM __FIT__`` under ``node``, pointed at ``mint()``.

    ``mint`` is called only when a reference is actually found, and it is
    idempotent — so a level with nothing to rewrite emits no step, which is
    what the mutating version got from only calling it inside the loop.
    """

    def point(v: AstNode) -> AstNode | None:
        if isinstance(v, BaseTable) and v.table_name == FIT:
            return v.model_copy(update={"table_name": mint()})
        return None

    return rebuild(node, point, deep=deep)


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
    whole: str | None = None

    def name(hint: str | None) -> str:
        base = f"__param_{hint}" if hint else f"__param_{len(steps)}"
        candidate, n = base, 1
        while candidate in taken:
            candidate, n = f"{base}_{n}", n + 1
        taken.add(candidate)
        return candidate

    def whole_fit() -> str:
        # A bare `FROM __FIT__` inside a relation that also reads `__THIS__`.
        # The training set itself is the parameter; `len(params)` says so.
        #
        # `whole` rather than a membership test on `taken`: hints come from CTE
        # keys, so a CTE named `fit` mints `__param_fit` too. Asking whether
        # the *name* was taken conflated "someone else has it" with "my step is
        # already emitted", and the bare `FROM __FIT__` was then aliased onto
        # that CTE's table — silently, and only when a CTE happened to be
        # called `fit`. Emitted-ness is its own fact; the name comes from the
        # same collision-avoiding allocator as every other one.
        nonlocal whole
        if whole is None:
            whole = name("fit")
            steps.append((whole, _select_star(FIT)))
        return whole

    def freeze(sub: Node, ctes: list[CteEntry], hint: str | None) -> Node:
        # Carry the enclosing CTEs in: a frozen subtree may reference one, and
        # by now their own definitions have been rewritten to frozen tables.
        frozen = with_cte_entries(sub, list(ctes) + list(cte_entries(sub)))
        param = name(hint)
        steps.append((param, frozen))
        return _select_star(param)

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
            return _rewrite_fit_refs(sub, whole_fit, deep=True)
        if FIT in reads and THIS not in reads:
            match _correlation(sub, outer):
                case None:
                    return freeze(sub, ctes, hint)  # maximal: never refreezes
                case (reference, True):
                    raise CorrelatedFit(
                        f"{FIT} subquery references {reference} from the "
                        "outer query, so it cannot be evaluated once into "
                        "a table"
                    )
                case _:
                    # Correlated into a `__FIT__`-only relation instead:
                    # still per-outer-row, so still unfreezable, but
                    # nothing is wrong. Fall through and the training set
                    # itself becomes the parameter; `len(params)` reports
                    # the cost. Marginalization makes it one row per group.
                    pass
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

        def nested(v: AstNode) -> AstNode | None:
            return visit(v, ctes, outer, None, reading) if is_query(v) else None

        node = rebuild(node, nested, deep=False)
        return _rewrite_fit_refs(node, whole_fit, deep=False)

    residual = visit(doc.statements[0].node, [], {}, None, {})
    return steps, residual
