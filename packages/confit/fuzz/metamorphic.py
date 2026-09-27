"""Metamorphic spelling checks: two spellings of one query must agree.

The oracle campaign compares confit with DuckDB, so it cannot see an
inconsistency inside confit itself: a query that confit serves in one
spelling and refuses in another. The rewrites here produce such spellings.
Each one is equivalent in DuckDB by construction (quoting, identifier case,
explicit qualification, the four ways to read a struct field, `<>` as
`NOT (=)`, and wrapping the whole query in a derived table or a CTE).

For each pair, the outcome must match: both refuse, both trap, or both serve
the same rows with the same types. The refusal messages themselves may
differ. A mismatch is a finding, unless `KNOWN` lists it by rewrite and
refusal text with a reason.

    uv run python -m fuzz.metamorphic --n 2000 --workers 4
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import dataclasses
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from fuzz import coverage, gen, oracle

# ------------------------------------------------------------------ outcomes


def outcome(case: gen.Case, sql: str) -> tuple:
    """`("refuse", msg)`, `("trap", msg)` or `("serve", rows, types)`."""
    try:
        fn = oracle._build(
            sql,
            oracle._arrow_schema(case.row_schema),
            {n: oracle._arrow_table(s, r) for n, (s, r) in case.statics.items()},
            [oracle.make_udf(u) for u in case.udfs],
            case.shape,
            False,
        )
    except Exception as e:  # noqa: BLE001 -- every build failure is a refusal here
        return ("refuse", f"{type(e).__name__}: {e}")
    try:
        out = fn.infer_arrow(oracle._arrow_table(case.row_schema, case.rows))
    except Exception as e:  # noqa: BLE001
        return ("trap", f"{type(e).__name__}: {e}")
    rows = repr([tuple(r.values()) for r in out.to_pylist()]).lower()
    return ("serve", rows, [str(f.type).lower() for f in out.schema])


def agree(a: tuple, b: tuple) -> bool:
    if a[0] != b[0]:
        return False
    return a[0] != "serve" or a[1:] == b[1:]


# ------------------------------------------------------------------ rewrites
#
# A rewrite takes a case and returns the rewritten SQL, or None when it does
# not apply (nothing to rewrite). It never mutates the case.


class _Skip(Exception):
    pass


@contextlib.contextmanager
def _ident(fn: Callable[[str], str]):
    orig = gen._ident
    gen._ident = fn
    try:
        yield
    finally:
        gen._ident = orig


def _with_ident(fn: Callable[[Callable[[str], str]], Callable[[str], str]]):
    def rw(case):
        with _ident(fn(gen._ident)):
            return gen.render(case.query)

    return rw


def _mutating(m: Callable[[gen.Q], None]):
    def rw(case):
        q = copy.deepcopy(case.query)
        try:
            m(q)
        except _Skip:
            return None
        return gen.render(q)

    return rw


def _quote_all(_orig):
    return lambda name: '"' + name.replace('"', '""') + '"'


def _upper(orig):
    # Unquoted identifiers are case-insensitive; a quoted one keeps its case.
    def f(name):
        s = orig(name)
        return s if s.startswith('"') else s.upper()

    return f


def _qualify_row_cols(q: gen.Q) -> None:
    """`x` -> `__THIS__.x` in a single-level query over the request table."""
    if q.ctes or q.body.frm != "__THIS__" or q.body.sub is not None:
        raise _Skip
    hit = False
    for n in coverage._nodes(q.body):
        if isinstance(n, gen.Col) and n.table is None:
            n.table = "__THIS__"
            hit = True
    if not hit:
        raise _Skip


def _ne_as_not_eq(q: gen.Q) -> None:
    """`a <> b` -> `NOT (a = b)`: equal under three-valued logic."""
    hit = False
    for n in list(coverage._nodes(q)):
        if isinstance(n, gen.Bin) and n.op == "<>":
            inner = gen.Bin("=", n.lhs, n.rhs)
            # Replace the node in place, so no parent needs rewiring.
            n.__class__ = gen.Un
            n.__dict__.clear()
            n.__dict__.update(op="NOT", e=inner)
            hit = True
    if not hit:
        raise _Skip


def _lane_spell(spell: str):
    def m(q: gen.Q) -> None:
        hit = False
        for n in coverage._nodes(q):
            if isinstance(n, gen.Col) and "." in n.name:
                n.spell = spell
                hit = True
        if not hit:
            raise _Skip

    return m


def _wrap(tmpl: str):
    def rw(case):
        if case.query.ctes:  # a CTE cannot nest inside a derived table here
            return None
        return tmpl.format(q=gen.render(case.query))

    return rw


REWRITES: dict[str, Callable[[gen.Case], str | None]] = {
    "quote-all": _with_ident(_quote_all),
    "upper-idents": _with_ident(_upper),
    "qualify-row-cols": _mutating(_qualify_row_cols),
    "ne-as-not-eq": _mutating(_ne_as_not_eq),
    "lane-dot": _mutating(_lane_spell("dot")),
    "lane-sub": _mutating(_lane_spell("sub")),
    "lane-paren": _mutating(_lane_spell("paren")),
    "lane-fn": _mutating(_lane_spell("fn")),
    "wrap-derived": _wrap("SELECT * FROM ({q})"),
    "wrap-cte": _wrap("WITH w AS ({q}) SELECT * FROM w"),
}

# Mismatches that are understood: (rewrite, substring of the side that
# refused) -> why. Each entry is a named, deliberate refusal that only one
# spelling reaches, not an engine inconsistency.
KNOWN: dict[tuple[str, str], str] = {
    ("wrap-derived", "struct- or list-valued column in a derived table"): (
        "derived tables carry scalar slots only (row-local subqueries design, "
        "phase 1); the unwrapped query serves the struct output directly"
    ),
    ("wrap-cte", "struct- or list-valued column in a derived table"): (
        "same refusal: a CTE read once binds as a derived table"
    ),
    ("wrap-derived", "a subquery under a shape='many' join"): (
        "shape='many' stays one join per query, not per level (design review "
        "condition); the unwrapped query's join is the only one"
    ),
    ("wrap-cte", "a subquery under a shape='many' join"): (
        "same refusal, reached through the CTE's derived-table binding"
    ),
}


def _known(rw: str, a: tuple, b: tuple) -> str | None:
    for side in (a, b):
        if side[0] == "refuse":
            for (k_rw, text), why in KNOWN.items():
                if k_rw == rw and text in side[1]:
                    return why
    return None


# ------------------------------------------------------------------ checking


@dataclasses.dataclass
class Finding:
    seed: int
    rewrite: str
    base_sql: str
    sql: str
    base: tuple
    other: tuple

    def line(self) -> str:
        def s(o):
            return o[0] if o[0] == "serve" else f"{o[0]}: {o[1][:100]}"

        return (
            f"seed {self.seed} [{self.rewrite}] {s(self.base)} -> {s(self.other)}\n"
            f"    {self.base_sql[:200]}\n    {self.sql[:200]}"
        )


def check(seed: int) -> tuple[collections.Counter, list[Finding]]:
    """Every rewrite of seed `seed`: (counts, unexplained findings)."""
    counts: collections.Counter = collections.Counter()
    findings: list[Finding] = []
    case = gen.gen(seed)
    base_sql = gen.render(case.query)
    base = None
    for name, rw in REWRITES.items():
        sql = rw(case)
        if sql is None or sql == base_sql:
            continue
        if base is None:
            base = outcome(case, base_sql)
        other = outcome(case, sql)
        counts[(name, "checked")] += 1
        if agree(base, other):
            continue
        # Qualifying a bare name resolves an ambiguity DuckDB also refuses:
        # the qualified query is a different query, not a respelling.
        if (
            name == "qualify-row-cols"
            and base[0] == "refuse"
            and "ambiguous" in base[1]
        ):
            counts[(name, "not-equivalent")] += 1
            continue
        if _known(name, base, other):
            counts[(name, "known")] += 1
            continue
        counts[(name, "finding")] += 1
        findings.append(Finding(seed, name, base_sql, sql, base, other))
    return counts, findings


def run(seeds, workers: int = 1):
    counts: collections.Counter = collections.Counter()
    findings: list[Finding] = []
    if workers <= 1:
        results = map(check, seeds)
    else:
        pool = ProcessPoolExecutor(workers)
        results = pool.map(check, seeds, chunksize=16)
    for c, f in results:
        counts.update(c)
        findings.extend(f)
    return counts, findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--show", type=int, default=5, help="findings to print per rewrite")
    a = ap.parse_args(argv)
    counts, findings = run(range(a.seed, a.seed + a.n), a.workers)
    for name in REWRITES:
        row = {
            k: counts[(name, k)]
            for k in ("checked", "known", "not-equivalent", "finding")
        }
        print(f"{name:18} " + "  ".join(f"{k} {v}" for k, v in row.items()))
    shown: collections.Counter = collections.Counter()
    for f in findings:
        if shown[f.rewrite] < a.show:
            shown[f.rewrite] += 1
            print(f.line())
    print(f"{len(findings)} unexplained finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
