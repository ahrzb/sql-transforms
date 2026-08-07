"""What a subtree reads, and what it reaches out to.

The two questions freezing turns on: *does this depend on ``__FIT__`` alone*,
and *does it correlate out of itself*. Both are answered from the AST, before
any data exists.
"""

from sql_transform.model._ast import (
    _QUERY,
    FIT,
    THIS,
    Node,
    _is_query,
    _is_ref,
    _under,
)


def _reads(node: Node, ctes: dict[str, set[str]] | None = None) -> set[str]:
    """Which of the two parameters the whole subtree reads. ``node`` itself
    counts — a bare ``FROM __THIS__`` ref is the whole subtree.

    A CTE reference reads whatever that CTE reads, which is why ``ctes`` is
    needed: matching literal ``__FIT__``/``__THIS__`` nodes alone let an
    enclosing CTE hide a ``__THIS__`` dependency, and the subtree was then
    frozen and evaluated at fit, where ``__THIS__`` does not exist.

    Deliberately over-approximating: an inner CTE shadowing an outer one of
    the same name contributes both. Reporting a read that is not there costs
    a freeze; missing one is wrong.
    """
    ctes = ctes or {}
    seen: set[str] = set()
    for v in (node, *(v for _, _, v in _under(node, deep=True))):
        if v.get("type") != "BASE_TABLE":
            continue
        name = v.get("table_name")
        if name in (FIT, THIS):
            seen.add(name)
        else:
            seen |= ctes.get(str(name).lower(), set())
    return seen


def _names_in(node: Node) -> set[str]:
    """Every name a column reference could be qualified by, anywhere inside."""
    names = {e["key"] for e in node[_QUERY]["map"]}
    for _, _, v in _under(node, deep=True):
        if _is_ref(v):
            names.add(v.get("alias") or v.get("table_name") or "")
        if _is_query(v):
            names.update(e["key"] for e in v[_QUERY]["map"])
    names.discard("")
    return names


def _bindings_at(
    node: Node, ctes: dict[str, set[str]] | None = None
) -> dict[str, bool]:
    """The names this level binds, each mapped to *does it read ``__THIS__``*.

    Which side a correlation lands on is what separates the one refusal from
    the case that merely costs params.
    """
    out = {
        e["key"]: THIS in _reads(e["value"]["query"]["node"], ctes)
        for e in node[_QUERY]["map"]
    }
    for _, _, v in _under(node, deep=False):
        if _is_ref(v):
            alias = v.get("alias") or v.get("table_name") or ""
            if alias:
                out[alias] = THIS in _reads(v, ctes)
    return out


def _correlation(node: Node, outer: dict[str, bool]) -> tuple[str, bool] | None:
    """A column reference reaching out of ``node``, and whether its target
    reads ``__THIS__``.

    Only qualified references are checked. An unqualified one is ambiguous
    without a binder, and DuckDB resolves it inward whenever it can.
    """
    inside = _names_in(node)
    found: tuple[str, bool] | None = None
    for _, _, v in _under(node, deep=True):
        if v.get("class") != "COLUMN_REF":
            continue
        parts = v["column_names"]
        if len(parts) >= 2 and parts[0] not in inside and parts[0] in outer:
            if outer[parts[0]]:
                return ".".join(parts), True  # into __THIS__: the one refusal
            found = found or (".".join(parts), False)
    return found
