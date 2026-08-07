"""Freezing: the rewrite from a two-parameter text into params + residual.

> Every maximal subquery whose leaves are all ``__FIT__`` and constants is
> evaluated once at fit and replaced by a table.

All the analysis — and every refusal freezing can raise — happens here, at
construction, before any data exists.
"""

import copy

from sql_transform.model._analysis import _bindings_at, _correlation, _reads
from sql_transform.model._ast import (
    _QUERY,
    _RECURSIVE_CTE,
    FIT,
    THIS,
    Node,
    _deserialize,
    _is_query,
    _select_star,
    _statement,
    _template,
    _under,
)
from sql_transform.model._errors import CorrelatedFit


def _pin_derived_names(node: Node) -> None:
    """Freeze the output column names before the expressions move.

    DuckDB names an unaliased select item after its own printed text, so
    replacing a frozen subquery with ``SELECT * FROM __param_0`` renamed the
    column to that — ``get_feature_names_out()`` handed back an internal
    parameter name, and the schema stopped matching ``run``'s.

    Written as an explicit alias first, so the name survives the rewrite. Only
    items that contain a query node are touched; a plain column reference is
    named after its last path part and is unaffected by any of this.
    """
    for item in node.get("select_list", []):
        if not isinstance(item, dict) or item.get("alias"):
            continue
        if not any(_is_query(v) for _, _, v in _under(item, deep=True)):
            continue
        printed = _deserialize(_statement(_one_item(item)))
        item["alias"] = printed[len("SELECT ") :]


def _one_item(item: Node) -> Node:
    """``SELECT <item>`` as a query node, for asking the oracle what it would
    call that column."""
    node = _template("SELECT 1")
    node["select_list"] = [copy.deepcopy(item)]
    return node


def _plan(doc: Node) -> tuple[list[tuple[str, Node]], Node]:
    """Rewrite ``doc`` in place into the residual, returning the fit steps.

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

    def freeze(sub: Node, ctes: list[Node], hint: str | None) -> Node:
        frozen = copy.deepcopy(sub)
        # Carry the enclosing CTEs in: a frozen subtree may reference one, and
        # by now their own definitions have been rewritten to frozen tables.
        frozen[_QUERY]["map"] = copy.deepcopy(ctes) + frozen[_QUERY]["map"]
        param = name(hint)
        steps.append((param, frozen))
        return _select_star(param)

    def visit(
        parent: Node,
        key: str,
        ctes: list[Node],
        outer: dict[str, bool],
        hint: str | None,
        reading: dict[str, set[str]],
    ) -> None:
        sub = parent[key]
        reads = _reads(sub, reading)
        # A recursive CTE's self-reference is bound by the enclosing entry key,
        # not by anything inside the body, so hoisting the body into a
        # standalone statement leaves that name unbound. Left live instead: the
        # training set becomes the parameter, which costs params size and is
        # correct. Freezing it properly means reconstructing the WITH RECURSIVE
        # wrapper, which is worth doing only if it ever matters.
        if sub.get("type") == _RECURSIVE_CTE:
            # Nothing inside may be frozen, not just the body as a whole: the
            # self-reference is visible to every arm but bound by none of them,
            # so hoisting any part of it out leaves that name dangling. The
            # training set becomes the parameter and the recursion stays live.
            for _, _, v in _under(sub, deep=True):
                if v.get("type") == "BASE_TABLE" and v.get("table_name") == FIT:
                    v["table_name"] = whole_fit()
            return
        if FIT in reads and THIS not in reads:
            match _correlation(sub, outer):
                case None:
                    parent[key] = freeze(sub, ctes, hint)
                    return  # maximal: a frozen subtree never refreezes
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
        descend(sub, ctes, outer, reading)

    def descend(
        node: Node,
        ctes: list[Node],
        outer: dict[str, bool],
        reading: dict[str, set[str]],
    ) -> None:
        _pin_derived_names(node)
        ctes = list(ctes)
        reading = dict(reading)
        for entry in node[_QUERY]["map"]:
            visit(entry["value"]["query"], "node", ctes, outer, entry["key"], reading)
            # After visiting, not before: a frozen body reads nothing, and one
            # left live has had its `__FIT__` rewritten already. Either way this
            # is what a later reference to the name actually depends on.
            reading[entry["key"].lower()] = _reads(
                entry["value"]["query"]["node"], reading
            )
            ctes.append(entry)
        outer = outer | _bindings_at(node, reading)
        sites = [(p, k) for p, k, v in _under(node, deep=False) if _is_query(v)]
        for parent, key in sites:
            visit(parent, key, ctes, outer, None, reading)
        for _, _, v in _under(node, deep=False):
            if v.get("type") == "BASE_TABLE" and v.get("table_name") == FIT:
                v["table_name"] = whole_fit()

    visit(doc["statements"][0], "node", [], {}, None, {})
    return steps, doc["statements"][0]["node"]
