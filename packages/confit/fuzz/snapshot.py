"""The subquery design's candidate snapshot: build it once, measure it after.

A seed names a case under one generator revision only, and the subquery work
adds generator productions, which shift the seed stream. So the candidates
are frozen as SQL plus inputs (`oracle.case_inputs`) before any generator
change, and recovery is measured by replaying that file, never by
re-generating seeds (docs/specs/2026-09-26-row-local-subqueries-design.md,
"Candidate snapshot").

    python -m fuzz.snapshot build --n 2000 --out fuzz/corpora/<name>.jsonl
    python -m fuzz.snapshot measure fuzz/corpora/<name>.jsonl

A candidate is a case DuckDB answers and confit refuses, whose query uses a
derived table or a CTE, with no whole-relation construct over request rows.
Phase 1: the subquery body reads the request table. Phase 2: it reads static
tables only and does not aggregate. A static-only body that aggregates is
neither (its own later ruling).
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import sys
from pathlib import Path

from . import coverage
from . import gen as G
from . import oracle as O
from .runner import CATEGORY, provenance


def _whole_relation(sel) -> bool:
    """DISTINCT, ORDER BY, a row limit, GROUP BY, QUALIFY, an aggregate or a
    window: the constructs that make a SELECT depend on other rows."""
    if sel is None:
        return False
    if (
        sel.distinct
        or sel.order_by
        or sel.limit is not None
        or sel.fetch is not None
        or sel.group_by
        or sel.qualify is not None
        or getattr(sel, "top", None)
    ):
        return True
    return any(
        isinstance(n, G.Call) and (n.name in G.AGGS or n.modifier == "over")
        for n in coverage._nodes(sel.items)
    )


def classify(case: G.Case) -> tuple[str, int] | None:
    """`(form, phase)` for a candidate query shape, else None. Reads only
    the AST; whether DuckDB answers and confit refuses is the caller's."""
    q, body = case.query, case.query.body
    if q.ctes:
        form, inner = "cte", [s for _, s in q.ctes]
    elif body.frm is None and body.sub is not None:
        form, inner = "derived", [body.sub]
    else:
        return None
    over_request = any(s.frm == "__THIS__" for s in inner)
    if _whole_relation(body) or any(
        _whole_relation(s) and s.frm == "__THIS__" for s in inner
    ):
        return None
    if over_request:
        return form, 1
    aggregates = any(
        s.group_by
        or any(
            isinstance(n, G.Call) and n.name in G.AGGS for n in coverage._nodes(s.items)
        )
        for s in inner
    )
    return None if aggregates else (form, 2)


def _candidate(seed: int) -> dict | None:
    case = G.gen(seed)
    shape = classify(case)
    if shape is None:
        return None
    v = O.run_case(case)
    if v.kind != "REFUSED" or v.oracle != "serves":
        return None
    form, phase = shape
    return {
        "seed": seed,
        "form": form,
        "phase": phase,
        "refusal": v.klass,
        "sql": G.render(case.query),
        "inputs": O.case_inputs(case),
    }


def _replay(line: dict) -> tuple[int, str, str]:
    case = O.case_from_inputs(line["seed"], line["sql"], line["inputs"])
    try:
        v = O.run_case(case)
    except Exception as e:  # noqa: BLE001 — the oracle's own bug, as in the campaign
        return line["phase"], "SKIP", f"oracle:{type(e).__name__}"
    return line["phase"], v.kind, v.klass


def build(start: int, n: int, out: Path, workers: int) -> None:
    with multiprocessing.Pool(workers) as pool:
        found = [c for c in pool.imap(_candidate, range(start, start + n), 16) if c]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        header = {"provenance": provenance(start, n)}
        f.write(json.dumps(header) + "\n")
        for c in found:
            f.write(json.dumps(c) + "\n")
    by = collections.Counter((c["phase"], c["form"]) for c in found)
    for (phase, form), k in sorted(by.items()):
        print(f"  phase {phase}  {form:8} {k}")
    print(f"{len(found)} candidates -> {out}")


def measure(path: Path, workers: int) -> dict:
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    cases = [ln for ln in lines if "seed" in ln]
    with multiprocessing.Pool(workers) as pool:
        got = list(pool.imap(_replay, cases, 8))
    table = collections.Counter((phase, kind) for phase, kind, _ in got)
    for phase in sorted({p for p, _, _ in got}):
        total = sum(k for (p, _), k in table.items() if p == phase)
        agree = sum(
            k
            for (p, kind), k in table.items()
            if p == phase and CATEGORY.get(kind) == "agreement"
        )
        print(f"phase {phase}: {agree} of {total} candidates agree")
        for (p, kind), k in sorted(table.items()):
            if p == phase:
                print(f"  {kind:14} {k}")
    return dict(table)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("measure")
    m.add_argument("path", type=Path)
    m.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.seed, a.n, a.out, a.workers)
    else:
        measure(a.path, a.workers)


if __name__ == "__main__":
    main()
