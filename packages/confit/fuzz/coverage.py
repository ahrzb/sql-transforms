"""Coverage as (operator, argument-type, edge-class) triples.

A raw query count says how many cases ran, not what they exercised. Each
operator application in a generated query contributes one triple per
argument: the operator, the argument's type, and the argument's edge class.
The type is the literal's or column's declared type, or `expr` for a
compound argument whose type the AST does not carry. The edge class is read
off a literal's value (`null`, `zero`, `negative`, `extreme`, `nonfinite`,
`empty`, `non-ascii`, `ordinary`); a column is `column` (data-dependent) and
a compound argument `expr`.

Reporting only: no gate reads these, and no coverage-metadata scheme beyond
the report is implied.
"""

from __future__ import annotations

import dataclasses
import math

from . import gen as G

_SEP = "\t"  # operators include `||`, so the key cannot use a pipe


def key(t: tuple[str, str, str]) -> str:
    return _SEP.join(t)


def parse(k: str) -> tuple[str, str, str]:
    op, ty, edge = k.split(_SEP)
    return op, ty, edge


def _edge(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "ordinary"
    if isinstance(v, float) and not math.isfinite(v):
        return "nonfinite"
    if isinstance(v, (int, float)):
        if v == 0:
            return "zero"
        if isinstance(v, int) and abs(v) >= 2**31 - 1:
            return "extreme"
        if isinstance(v, float) and (abs(v) >= 1e300 or abs(v) <= 1e-300):
            return "extreme"
        return "negative" if v < 0 else "ordinary"
    if isinstance(v, str):
        if v == "":
            return "empty"
        return "ordinary" if v.isascii() else "non-ascii"
    return "ordinary"


def _arg(n) -> tuple[str, str]:
    if isinstance(n, G.Lit):
        return n.ty, _edge(n.val)
    if isinstance(n, G.Col):
        return n.ty, "column"
    return "expr", "expr"


def _applications(n):
    """`(operator, [arguments])` for one node, or None if it applies none."""
    if isinstance(n, G.Bin):
        return n.op, [n.lhs, n.rhs]
    if isinstance(n, G.Un):
        return ("unary -" if n.op == "-" else n.op), [n.e]
    if isinstance(n, G.Call):
        return n.name, list(n.args)
    if isinstance(n, G.Cast):
        return f"{'TRY_CAST' if n.try_ else 'CAST'} AS {n.to}", [n.e]
    if isinstance(n, G.Between):
        return ("NOT BETWEEN" if n.neg else "BETWEEN"), [n.e, n.lo, n.hi]
    if isinstance(n, G.InList):
        return ("NOT IN" if n.neg else "IN"), [n.e, *n.items]
    if isinstance(n, G.IsNull):
        return ("IS NOT NULL" if n.neg else "IS NULL"), [n.e]
    if isinstance(n, G.Like):
        return ("NOT LIKE" if n.neg else "LIKE"), [n.e]
    if isinstance(n, G.CaseW):
        return "CASE", [x for w in n.whens for x in w] + ([n.els] if n.els else [])
    return None


def _nodes(obj):
    """Every AST node reachable from `obj`, through any dataclass field."""
    if isinstance(obj, G.Node):
        yield obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _nodes(getattr(obj, f.name))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _nodes(x)


def triples(query) -> set[tuple[str, str, str]]:
    out = set()
    for n in _nodes(query):
        app = _applications(n)
        if app is None:
            continue
        op, args = app
        for a in args:
            out.add((op, *_arg(a)))
    return out
