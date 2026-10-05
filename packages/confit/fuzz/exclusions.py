"""Every exclusion from parity, in one place.

An exclusion is not "do not compare these cases". It narrows the comparison
for one named mechanism, and each record states:

- `scope`: which cases it may look at, read off facts of the bound plan
  (`DuckDBInferFn.plan_facts`) or of the confit run, never off the SQL
  text. A scope only filters and never excuses alone, so it may be loose.
- the outcome pattern it accepts, beyond the ordinary comparison;
- `witness`: a second measurement that the difference is the claimed
  mechanism and nothing else. A case is excused only when the witness holds,
  so a different bug inside the scope is still a finding.
- `claim`: the soundness argument, meaning what remains observable inside
  the scope and why the witness pins it;
- `ruling` (the decision that admits it; `status="kept"` marks one in force
  before any ruling, as the ledger lists it) and `oracle` (the DuckDB
  version it was measured on). A record measured on another version is inactive and
  fails the suite until it is remeasured;
- `canaries`: cases it must still excuse, so a dead record is noticed, and
  `planted` faults it must not excuse (`tests/test_exclusions.py`).

The campaign keeps the raw verdict: an excused case is EXCLUDED with the
rule's id as its class and the verdict it would have had in `raw`, so a
report can be recomputed without a rule. The ledger table in
`docs/oracle/07-the-divergence-ledger.md` is generated from this module
(`python -m fuzz.exclusions --ledger`), and a test keeps the two in step.

Three kinds of record share the registry: verdict rules (applied inside
`fuzz.oracle.run_case`), rewrite tolerances (`fuzz.metamorphic`), and
input-source exclusions (the DuckDB test-corpus replay).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ORACLE = "1.5.5"  # the version every record below was measured on


# ------------------------------------------------------------------ evidence


@dataclass
class Evidence:
    """What a verdict rule may read: the confit run and the bound plan, plus
    (after the comparison) the baseline DuckDB reading."""

    sql: str
    facts: dict  # DuckDBInferFn.plan_facts of the cranelift build
    trap_cl: str | None  # confit's trap on each backend, or None for rows
    trap_in: str | None
    duck_off: tuple | None = None  # (table, phase, error), optimizer off
    case: object = None  # the fuzz.gen.Case, for a witness's re-run
    udf_objs: list = field(default_factory=list)


# ------------------------------------------------------------------ scopes


def joins_static_where(*, rows: int | None = None, kinds=("INNER", "CROSS")):
    """Scope: the plan joins a static table matching every given fact.

    `kinds` uses DuckDB's words: INNER is any inner join, CROSS an inner join
    with no equality key, LEFT a left join. A self-join builds from the batch
    and never qualifies. Returns the static names that match.
    """
    want = {"INNER": ("inner", None), "CROSS": ("inner", False), "LEFT": ("left", None)}

    def scope(facts: dict) -> list[str]:
        hits = []
        for j in facts.get("joins", []):
            if j["static"] is None:
                continue
            if rows is not None and j["rows"] != rows:
                continue
            ok = False
            for k in kinds:
                kind, keyed = want[k]
                if j["kind"] == kind and (keyed is None or (j["keys"] > 0) == keyed):
                    ok = True
            if ok:
                hits.append(j["static"])
        return hits

    return scope


def _category(msg: str | None) -> str | None:
    """DuckDB's error category inside a trap text, e.g. "out of range error"
    from "OutOfRangeException: Out of Range Error: Overflow ...": the field
    after the exception name, when it is one."""
    if not msg:
        return None
    parts = [p.strip() for p in msg.split(":")]
    if len(parts) >= 2 and parts[1].lower().endswith("error"):
        return parts[1].lower()
    return None


def same_trap(confit: str | None, duck: str | None) -> bool:
    """The ordinary trap comparison (both trap), tightened: when both texts
    name a DuckDB error category, the categories must be equal."""
    if confit is None or duck is None:
        return False
    a, b = _category(confit), _category(duck)
    return a is None or b is None or a == b


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class Rule:
    """A verdict rule. Exactly one of `pre` (decided from the confit run,
    before DuckDB runs) and `post` (decided after the raw verdict) is set;
    each returns the excusing detail, or None."""

    id: str
    ledger: str  # the anchor in docs/oracle/07-the-divergence-ledger.md
    ruling: str  # the decision that admits it (a path under docs/)
    claim: str
    canaries: tuple[int, ...]
    expected_per_100k: float  # the nightly flags a count far above this
    pre: Callable[[Evidence], str | None] | None = None
    post: Callable[[Evidence, object], str | None] | None = None
    oracle: str = ORACLE
    status: str = "ruled"  # or "kept": in force before a ruling, as the ledger says
    kind: str = "verdict"


@dataclass(frozen=True)
class RewriteTolerance:
    """A metamorphic mismatch that is a named refusal one spelling reaches."""

    id: str
    rewrite: str  # fuzz.metamorphic rewrite name
    refusal: str  # substring of the refusing side's message
    ledger: str
    ruling: str
    claim: str
    oracle: str = ORACLE
    status: str = "ruled"
    kind: str = "rewrite"


@dataclass(frozen=True)
class SourceExclusion:
    """A DuckDB test-corpus source whose answer no row-local engine can
    reproduce."""

    id: str
    source: str
    ledger: str
    ruling: str
    claim: str
    oracle: str = ORACLE
    status: str = "ruled"
    kind: str = "source"


# ------------------------------------------------------------------ rules

# The serving contract's resource ceilings, verbatim. Past one, DuckDB builds
# the gigabyte value the ceiling exists to refuse (minutes per case), so the
# rule decides before DuckDB runs, and only when BOTH backends hit it.
RESOURCE_CEILINGS = (
    "string builder result exceeds 1 GiB",
    "string column exceeds 2 GiB in one",
)


def _ceiling(ev: Evidence) -> str | None:
    if (
        ev.trap_cl is not None
        and ev.trap_in is not None
        and any(c in ev.trap_cl for c in RESOURCE_CEILINGS)
        and any(c in ev.trap_in for c in RESOURCE_CEILINGS)
    ):
        return ev.trap_cl
    return None


_EMPTY_INNER = joins_static_where(rows=0, kinds=("INNER", "CROSS"))


def _empty_static(ev: Evidence, raw) -> str | None:
    # The outcome pattern: confit traps on both backends, the optimizer-off
    # baseline returns zero rows.
    if raw.kind not in ("DIVERGE_TRAP", "OPT_EMULATED"):
        return None
    if ev.trap_cl is None or ev.trap_in is None or ev.duck_off is None:
        return None
    out, _, _ = ev.duck_off
    if out is None or out.num_rows != 0:
        return None
    empty = _EMPTY_INNER(ev.facts)
    if not empty:
        return None
    # The witness: the same query with one row in every empty static. A row
    # of plain non-NULL values (NULL keys would be filtered out of DuckDB's
    # hash table, which stays empty) makes DuckDB run the row side, and its
    # trap must match confit's. Whether the row matches does not matter: the
    # row side runs either way.
    err = _duck_with_padding(ev, empty)
    if err is None or not same_trap(ev.trap_cl, err):
        return None
    return f"padded DuckDB traps like confit: {err[:200]}"


def _plain_value(t):
    """A non-NULL value of Arrow type `t`: 0, 0.0, '', False, or a struct of
    those. None where there is no plain value (opaque types)."""
    import pyarrow as pa  # noqa: PLC0415

    if pa.types.is_integer(t):
        return 0
    if pa.types.is_floating(t):
        return 0.0
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return ""
    if pa.types.is_boolean(t):
        return False
    if pa.types.is_struct(t):
        return {
            t.field(i).name: _plain_value(t.field(i).type) for i in range(t.num_fields)
        }
    return None


def _duck_with_padding(ev: Evidence, statics: list[str]) -> str | None:
    """DuckDB's run-time error for `ev.sql` once each named static holds one
    row of plain values; None when it returns rows, refuses at build, or the
    row cannot be built."""
    import dataclasses  # noqa: PLC0415

    from fuzz import oracle as O  # noqa: PLC0415 -- oracle imports this module

    case = ev.case
    padded = dict(case.statics)
    for name in statics:
        sch, _ = case.statics[name]
        arrow = O._arrow_schema(sch)
        row = {f.name: _plain_value(f.type) for f in arrow}
        if any(v is None for v in row.values()):
            return None
        padded[name] = (sch, [row])
    case = dataclasses.replace(case, statics=padded)
    con = O._take()
    try:
        O._load(con, case, ev.udf_objs)
        _, phase, err = O._exec(con, ev.sql)
        return err if phase == "run" else None
    finally:
        O._give_back(con, [*case.statics, "__THIS__"], [u.name for u in ev.udf_objs])


RULES: tuple[Rule, ...] = (
    Rule(
        id="resource-ceiling",
        ledger="exclusion: resource-ceilings",
        ruling="specs/serving-contract.md",
        claim=(
            "Both backends hit a declared string ceiling. Past it the contract "
            "promises a refusal, not DuckDB's answer, so nothing is left to "
            "compare; a ceiling on one backend only still compares."
        ),
        canaries=(1003321,),
        expected_per_100k=120,
        pre=_ceiling,
    ),
    Rule(
        id="empty-static-trap-timing",
        ledger="divergence: empty-static-trap-timing",
        ruling="decisions/closed/empty-static-join-trap-timing.md",
        claim=(
            "An INNER or CROSS join on an empty static outputs no rows on "
            "either engine, so the only observable is whether the row side "
            "traps. DuckDB skips it in some pipeline shapes. The witness pads "
            "the static with one row of plain non-NULL values, which makes "
            "DuckDB run the row side, and requires its trap to match "
            "confit's; a different trap, rows, or a backend split stays a "
            "finding."
        ),
        canaries=(1994509, 2286807, 4001288, 4100483),
        expected_per_100k=1,
        post=_empty_static,
    ),
)

REWRITE_TOLERANCES: tuple[RewriteTolerance, ...] = tuple(
    RewriteTolerance(
        id=f"{rw}: {key}",
        rewrite=rw,
        refusal=refusal,
        ledger="divergence: wrapped-query-refusals",
        ruling="specs/2026-09-26-row-local-subqueries-design.md",
        claim=claim,
    )
    for rw, key, refusal, claim in (
        (
            "wrap-derived",
            "struct slot",
            "struct- or list-valued column in a derived table",
            "derived tables carry scalar slots only (phase 1 of the design); "
            "the unwrapped query serves the struct output directly",
        ),
        (
            "wrap-cte",
            "struct slot",
            "struct- or list-valued column in a derived table",
            "same refusal: a CTE read once binds as a derived table",
        ),
        (
            "wrap-derived",
            "many join",
            "a subquery under a shape='many' join",
            "shape='many' stays one join per query, not per level (a design "
            "review condition); the unwrapped query's join is the only one",
        ),
        (
            "wrap-cte",
            "many join",
            "a subquery under a shape='many' join",
            "same refusal, reached through the CTE's derived-table binding",
        ),
    )
)

SOURCE_EXCLUSIONS: tuple[SourceExclusion, ...] = (
    SourceExclusion(
        id="ilike-nul",
        source="test/sql/function/string/test_ilike_embedded_null.test",
        ledger="divergence: ilike-nul",
        ruling="",
        status="kept",
        claim=(
            "DuckDB's ILIKE over a NUL-containing row depends on its SIBLING "
            "rows (column statistics pick the kernel; pins-wave1/pins_like.json). "
            "No row-at-a-time engine can reproduce that; confit is "
            "NUL-transparent."
        ),
    ),
)


def every() -> list:
    return [*RULES, *REWRITE_TOLERANCES, *SOURCE_EXCLUSIONS]


def active(rec) -> bool:
    """A record applies only on the DuckDB version it was measured on."""
    from confit.oracle import Oracle  # noqa: PLC0415

    return rec.oracle == Oracle.VERSION


# ------------------------------------------------------------------ applying


def pre_oracle(ev: Evidence):
    """(rule id, detail) of the first pre-oracle rule that excuses `ev`."""
    for r in RULES:
        if r.pre is not None and active(r):
            detail = r.pre(ev)
            if detail is not None:
                return r.id, detail
    return None


def post_verdict(ev: Evidence, raw):
    """(rule id, detail) of the first post-verdict rule that excuses the
    raw verdict, or None. Only a mismatch is ever excused."""
    if raw.kind in ("AGREE", "AGREE_TRAP", "REFUSED", "UNSHIPPED"):
        return None
    for r in RULES:
        if r.post is not None and active(r):
            detail = r.post(ev, raw)
            if detail is not None:
                return r.id, detail
    return None


def rewrite_tolerance(rewrite: str, refusal: str) -> RewriteTolerance | None:
    for t in REWRITE_TOLERANCES:
        if active(t) and t.rewrite == rewrite and t.refusal in refusal:
            return t
    return None


def source_excluded(source: str | None) -> SourceExclusion | None:
    for s in SOURCE_EXCLUSIONS:
        if active(s) and s.source == source:
            return s
    return None


# ------------------------------------------------------------------ ledger

LEDGER = Path(__file__).parents[1] / "docs/oracle/07-the-divergence-ledger.md"
BEGIN, END = "<!-- exclusions:begin -->", "<!-- exclusions:end -->"


def ledger_table() -> str:
    rows = [
        "| id | kind | ledger entry | status | ruling | measured on | canaries |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in every():
        canaries = len(getattr(rec, "canaries", ())) or "-"
        ruling = f"`{rec.ruling}`" if rec.ruling else "-"
        rows.append(
            f"| `{rec.id}` | {rec.kind} | {rec.ledger} | {rec.status} | {ruling} | "
            f"DuckDB {rec.oracle} | {canaries} |"
        )
    return "\n".join(rows)


def ledger_block(text: str) -> str:
    m = re.search(re.escape(BEGIN) + r"\n(.*?)\n" + re.escape(END), text, re.S)
    return m.group(1) if m else ""


def write_ledger() -> None:
    text = LEDGER.read_text()
    new = f"{BEGIN}\n{ledger_table()}\n{END}"
    text = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: new, text, flags=re.S
    )
    LEDGER.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", action="store_true", help="rewrite the ledger table")
    args = ap.parse_args()
    if args.ledger:
        write_ledger()
    else:
        print(ledger_table())


if __name__ == "__main__":
    main()
