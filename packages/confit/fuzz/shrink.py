"""Greedy Case minimizer: keep any edit that preserves (kind, klass).

    uv run --directory packages/confit python -m fuzz.shrink <seed>

Passes run to a fixed point: drop query parts, shrink data, then fold
expression nodes into a child or a literal. Ends by printing the minimal SQL,
data, verdict, and a ready-to-paste pin snippet.
"""

from __future__ import annotations

import copy
import sys

from . import gen as G
from . import oracle


def _sig(case: G.Case):
    v = oracle.run_case(case)
    return (v.kind, v.klass)


def _try(case: G.Case, want, edit) -> G.Case:
    """`edit` applied to a COPY, kept only while the verdict signature still
    equals `want`. A rejected edit — or one that raised — returns the very
    object passed in, so callers detect "no progress" with `is`."""
    cand = copy.deepcopy(case)
    try:
        edit(cand)
        if _sig(cand) == want:
            return cand
    except Exception:  # noqa: BLE001, S110 — an edit may render invalid; discard
        pass
    return case


def _sites(case: G.Case):
    """Addressable expression slots on the BODY: (kind, idx). CTE and
    sub-select expressions are dropped whole by the structural passes rather
    than folded — a v1 shrinker trade."""
    b = case.query.body
    out = [
        ("item", i)
        for i, (e, _) in enumerate(b.items)
        if isinstance(e, G.Node) and not isinstance(e, G.Star)
    ]
    if b.where is not None:
        out.append(("where", 0))
    if b.joins and b.joins[0].on is not None:
        out.append(("on", 0))
    return out


def shrink(case: G.Case, rounds: int = 6) -> G.Case:
    """The smallest case this reaches whose verdict signature still equals
    `case`'s. `rounds` caps the passes; a round that renders the same SQL as
    the one before it stops early."""
    want = _sig(case)
    for _ in range(rounds):
        before = G.render(case.query)
        b = case.query.body

        # structural passes
        if len(b.items) > 1:
            for i in range(len(b.items)):

                def keep_one(c, i=i):
                    c.query.body.items = [c.query.body.items[i]]

                case = _try(case, want, keep_one)
                if len(case.query.body.items) == 1:
                    break
        for name in ("where", "qualify"):
            if getattr(case.query.body, name) is not None:
                case = _try(
                    case, want, lambda c, n=name: setattr(c.query.body, n, None)
                )
        for flag in ("distinct", "order_by", "limit", "top", "fetch"):
            if getattr(case.query.body, flag):
                dflt = False if flag == "distinct" else None
                case = _try(
                    case, want, lambda c, f=flag, d=dflt: setattr(c.query.body, f, d)
                )
        if case.query.body.joins:
            case = _try(case, want, lambda c: c.query.body.joins.clear())
        if case.query.ctes:
            case = _try(case, want, lambda c: c.query.ctes.clear())
        if case.query.body.group_by:
            case = _try(case, want, lambda c: c.query.body.group_by.clear())
        if case.query.body.sub is not None:

            def unsub(c):
                c.query.body = c.query.body.sub

            case = _try(case, want, unsub)

        # data passes
        while len(case.rows) > 1:
            fewer = _try(case, want, lambda c: c.rows.__delitem__(0))
            if fewer is case:
                break
            case = fewer
        for name in list(case.statics):
            if not any(j.table == name for j in case.query.body.joins):
                case = _try(case, want, lambda c, n=name: c.statics.pop(n))
        if case.udfs:
            case = _try(case, want, lambda c: c.udfs.clear())

        # expression folding: replace a node with one of its children
        for kind, idx in _sites(case):
            changed = True
            while changed:
                changed = False
                root = _root(case, kind, idx)
                if root is None:
                    break
                for path, node in _paths(root):
                    for ci in range(len(node.kids())):
                        cand = _try(
                            case,
                            want,
                            lambda c, k=kind, i=idx, p=path, x=ci: _fold(c, k, i, p, x),
                        )
                        if cand is not case:
                            case, changed = cand, True
                            break
                    if changed:
                        break
        if G.render(case.query) == before:
            break
    return case


def _fold(case: G.Case, kind: str, idx: int, path, child_i: int):
    node = _root(case, kind, idx)
    for i in path:
        node = node.kids()[i]
    child = copy.deepcopy(node.kids()[child_i])
    _replace(case, kind, idx, path, child)


def _paths(root: G.Node, path=()):
    yield path, root
    for i, k in enumerate(root.kids()):
        yield from _paths(k, (*path, i))


def _root(case: G.Case, kind: str, idx):
    b = case.query.body
    if kind == "item":
        return b.items[idx][0]
    if kind == "where":
        return b.where
    return b.joins[0].on if b.joins else None


def _replace(case: G.Case, kind: str, idx, path, new: G.Node):
    b = case.query.body
    if not path:
        if kind == "item":
            b.items[idx] = (new, b.items[idx][1])
        elif kind == "where":
            b.where = new
        elif b.joins:
            b.joins[0].on = new
        return
    node = _root(case, kind, idx)
    if node is None:
        raise IndexError
    for i in path[:-1]:
        node = node.kids()[i]
    node.swap(path[-1], new)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seed = int(sys.argv[1])
    case = G.gen(seed)
    v0 = oracle.run_case(case)
    print(f"seed {seed}: {v0.kind} [{v0.klass}]\n  {G.render(case.query)}")
    small = shrink(case)
    v1 = oracle.run_case(small)
    print(f"\nshrunk ({v1.kind} [{v1.klass}]):\n  {G.render(small.query)}")
    print(f"  row_schema = {small.row_schema}")
    print(f"  rows       = {small.rows}")
    for n, (sch, rows) in small.statics.items():
        print(f"  static {n}: {sch} {rows}")
    # A fresh finding is a divergence we intend to CLOSE, so it belongs in the
    # open file with a ticket - not in known_divergences/, which is behaviour
    # we decided to keep.
    print("\n-- pin sketch (tests/test_open_divergences.py, xfail-strict) --")
    print(f"""
def test_fuzz_seed_{seed}():
    duck_check(
        {G.render(small.query)!r},
        {small.row_schema!r},
        {small.rows!r},
    )""")


if __name__ == "__main__":
    main()
