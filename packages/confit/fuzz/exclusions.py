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
- `ruling` (the decision that admits it, and `extended_by`, the later
  rulings that widen it; `status="kept"` marks one in force before any
  ruling, as the ledger lists it) and `oracle` (the DuckDB version it was
  measured on). A record measured on another version is inactive and
  fails the suite until it is remeasured;
- `canaries`: cases it must still excuse, so a dead record is noticed, and
  `planted` faults it must not excuse (`tests/test_exclusions.py`).

The campaign keeps the raw verdict: an excused case is EXCLUDED with the
rule's id as its class and the verdict it would have had in `raw`, so a
report can be recomputed without a rule. The ledger table in
`docs/oracle/07-the-divergence-ledger.md` is generated from this module
(`python -m fuzz.exclusions --ledger`), and a test keeps the two in step.

Three kinds of record share the registry: verdict rules (applied inside
`fuzz.oracle.run_case`, or by `fuzz.runner` once the oracle runs past the
time limit of a case), rewrite tolerances (`fuzz.metamorphic`), and
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


def joins_static_where(
    *, rows: int | None = None, built: int | None = None, kinds=("INNER", "CROSS")
):
    """Scope: the plan joins a static table matching every given fact.

    `kinds` uses DuckDB's words: INNER is any inner join, CROSS an inner join
    with no equality key, KEYED an inner join with one, LEFT a left join. A
    self-join builds from the batch and never qualifies. `rows` is the
    static's row count, and `built` the rows its build keeps: a row whose `=`
    key is NULL can match no request, and drops out. Returns the static
    names that match.
    """
    want = {
        "INNER": ("inner", None),
        "CROSS": ("inner", False),
        "KEYED": ("inner", True),
        "LEFT": ("left", None),
    }

    def scope(facts: dict) -> list[str]:
        hits = []
        for j in facts.get("joins", []):
            if j["static"] is None:
                continue
            if rows is not None and j["rows"] != rows:
                continue
            if built is not None and j.get("built") != built:
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


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class Rule:
    """A verdict rule. Exactly one of `pre` (decided from the confit run,
    before DuckDB runs), `post` (decided after the raw verdict) and `timeout`
    (decided in a new worker after the oracle ran past the time limit of the
    case, from the case alone) is set; each returns the excusing detail, or
    None."""

    id: str
    ledger: str  # the anchor in docs/oracle/07-the-divergence-ledger.md
    ruling: str  # the decision that admits it (under docs/, or a loops/ record)
    claim: str
    canaries: tuple[int, ...]
    expected_per_100k: float  # the nightly flags a count far above this
    pre: Callable[[Evidence], str | None] | None = None
    post: Callable[[Evidence, object], str | None] | None = None
    timeout: Callable[[object], str | None] | None = None  # over a fuzz.gen.Case
    oracle: str = ORACLE
    status: str = "ruled"  # or "kept": in force before a ruling, as the ledger says
    kind: str = "verdict"
    extended_by: tuple[str, ...] = ()  # later rulings that widen `ruling`


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
# An INNER join on a key whose static has no row that a request can match:
# DuckDB leaves a row whose key is NULL out of its hash table, which is then
# empty too (loops/confit/decisions/closed/all-null-key-static.md).
_NULL_KEYED = joins_static_where(built=0, kinds=("KEYED",))


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
    empty = list(dict.fromkeys([*_EMPTY_INNER(ev.facts), *_NULL_KEYED(ev.facts)]))
    if not empty:
        return None
    # The witness: the same query with one more row in each such static. A
    # row of plain non-NULL values (a NULL key would drop out of DuckDB's
    # hash table, which would stay empty) makes DuckDB run the row side, and
    # DuckDB must then trap. Any trap counts, as for AGREE_TRAP: DuckDB
    # evaluates one column over the batch at a time and confit one row at a
    # time, so with traps in two rows each engine can meet another one first
    # (loops/confit/decisions/closed/empty-static-witness-trap-order.md).
    # Whether the row matches does not matter: the row side runs either way.
    err = _duck_with_padding(ev, empty)
    if err is None:
        return None
    return f"padded DuckDB traps too: {err[:200]}"


def _plain_value(t):
    """A non-NULL value of Arrow type `t`: 0, 0.0, '', False, a decimal 0, the
    epoch, or a struct of those. None where there is no plain value (other
    opaque types)."""
    import datetime  # noqa: PLC0415

    import pyarrow as pa  # noqa: PLC0415

    # The generator's static tables have opaque timestamp columns (nightly
    # seed 8522528): without a value for one, no witness ran.
    if pa.types.is_timestamp(t):
        return datetime.datetime(1970, 1, 1)
    if pa.types.is_date(t):
        return datetime.date(1970, 1, 1)
    if pa.types.is_integer(t):
        return 0
    if pa.types.is_floating(t):
        return 0.0
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return ""
    if pa.types.is_boolean(t):
        return False
    if pa.types.is_decimal(t):
        from decimal import Decimal  # noqa: PLC0415

        return Decimal(0)
    if pa.types.is_struct(t):
        return {
            t.field(i).name: _plain_value(t.field(i).type) for i in range(t.num_fields)
        }
    return None


def _duck_with_padding(ev: Evidence, statics: list[str]) -> str | None:
    """DuckDB's run-time error for `ev.sql` once each named static holds one
    more row, of plain values; None when it returns rows, refuses at build,
    or the row cannot be built."""
    import dataclasses  # noqa: PLC0415

    from fuzz import oracle as O  # noqa: PLC0415 -- oracle imports this module

    case = ev.case
    padded = dict(case.statics)
    for name in statics:
        sch, rows = case.statics[name]
        arrow = O._arrow_schema(sch)
        row = {f.name: _plain_value(f.type) for f in arrow}
        if any(v is None for v in row.values()):
            return None
        padded[name] = (sch, [*rows, row])
    case = dataclasses.replace(case, statics=padded)
    con = O._take()
    try:
        if O._load(con, case, ev.udf_objs) is not None:
            return None  # DuckDB refuses a SQL function: a build refusal
        _, phase, err = O._exec(con, ev.sql)
        return err if phase == "run" else None
    finally:
        O._give_back(
            con,
            [*case.statics, "__THIS__"],
            [(u.name, hasattr(u, "sql_body")) for u in ev.udf_objs],
        )


# The witness of an oracle reading that ran past the time limit of a case:
# the oracle reads the query again with each length that a string builder
# takes capped at LENGTH_CAP. A temporary macro hides the builtin of its name
# (DuckDB looks in the temporary schema first, measured on 1.5.5, SQL
# functions' bodies included) and calls the builtin through `system.main`,
# with the length passed through a UDF that counts the lengths it caps.
LENGTH_CAP = 1000
_BUILDERS = (
    # name, parameters, the type of the length: lpad's count is INTEGER and
    # takes no BIGINT, repeat's is BIGINT
    ("lpad", ("s", "n", "c"), "INTEGER"),
    ("rpad", ("s", "n", "c"), "INTEGER"),
    ("repeat", ("s", "n"), "BIGINT"),
)


def _kind(msg: str | None) -> str | None:
    """The kind of a DuckDB error inside a trap text, such as "out of range
    error" in "OutOfRangeException: Out of Range Error: Overflow ...": the
    field after the exception name, when it names one. Confit's texts name
    one where they copy DuckDB's ("ValueError: Conversion Error: ..."), and
    not always ("ValueError: cannot take square root of a negative number")."""
    if not msg:
        return None
    parts = [p.strip() for p in msg.split(":")]
    if len(parts) >= 2 and parts[1].lower().endswith("error"):
        return parts[1].lower()
    return None


def _confit_traps(case, udf_objs) -> tuple[str | None, str | None] | None:
    """Confit's trap on each backend over the case's rows, read as
    `fuzz.oracle.run_case` reads it: `(cranelift, interpreter)`, None for
    rows; None when a backend does not build."""
    from fuzz import oracle as O  # noqa: PLC0415 -- oracle imports this module

    sql = O._sql(case)
    schema = O._arrow_schema(case.row_schema)
    statics = {n: O._arrow_table(sch, rows) for n, (sch, rows) in case.statics.items()}
    table = O._arrow_table(case.row_schema, case.rows)
    traps = []
    for force in (False, True):
        try:
            fn = O._build(sql, schema, statics, udf_objs, case.shape, force)
        except Exception:  # noqa: BLE001 -- a refusal or a build error: no witness
            return None
        try:
            fn.infer_arrow(table)
            traps.append(None)
        except Exception as e:  # noqa: BLE001 -- a trap, as run_case reads it
            traps.append(f"{type(e).__name__}: {e}")
    return traps[0], traps[1]


def _capped_oracle(case, udf_objs) -> tuple[str | None, str | None, int]:
    """The oracle's reading of the case with each string builder's length
    capped at LENGTH_CAP: `(phase, error, capped)`. `phase` is "build" or
    "run" when DuckDB raises, None for rows, and `capped` counts the lengths
    above the cap. The reading has a database of its own, which ends with
    the macros in it."""
    from confit.oracle import Oracle  # noqa: PLC0415

    from fuzz import oracle as O  # noqa: PLC0415 -- oracle imports this module

    capped = 0

    def cap(n):
        nonlocal capped
        if n is not None and n > LENGTH_CAP:
            capped += 1
            return LENGTH_CAP
        return n

    con = Oracle()
    try:
        for width in ("INTEGER", "BIGINT"):
            name = f"__confit_length_cap_{width.lower()}"
            con.create_function(name, cap, [width], width, null_handling="special")
        for name, params, width in _BUILDERS:
            cap_fn = f"__confit_length_cap_{width.lower()}"
            args = ", ".join(f"{cap_fn}({p})" if p == "n" else p for p in params)
            con.execute(
                f"CREATE TEMP MACRO {name}({', '.join(params)}) AS "
                f"system.main.{name}({args})"
            )
        if O._load(con, case, udf_objs) is not None:
            return "build", None, capped
        _, phase, err = O._exec(con, O._sql(case))
        return phase, err, capped
    finally:
        con.close()


def _long_string(case) -> str | None:
    # The scope: confit traps on both backends. The witness: the oracle,
    # with each length capped, caps at least one length and raises an error
    # of confit's kind (where both texts name one). Unlike the empty-static
    # witness, the kind stays: a cap changes the values that the query
    # computes, so the capped query can raise an error that the original
    # does not, such as a cast of a string that the cap made shorter
    # (loops/confit/decisions/closed/oracle-timeout-before-a-trap.md).
    from fuzz import oracle as O  # noqa: PLC0415 -- oracle imports this module

    udf_objs = O._udf_objs(case)
    traps = _confit_traps(case, udf_objs)
    if traps is None or None in traps:
        return None
    phase, err, capped = _capped_oracle(case, udf_objs)
    if phase != "run" or capped == 0:
        return None
    ours, theirs = _kind(traps[0]), _kind(err)
    if ours is not None and theirs is not None and ours != theirs:
        return None
    first = err.splitlines()[0] if err else ""
    return f"the oracle with lengths capped at {LENGTH_CAP} traps too: {first[:200]}"


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
        ruling="loops/confit/decisions/closed/empty-static-join-trap-timing.md",
        extended_by=(
            "loops/confit/decisions/closed/all-null-key-static.md",
            "loops/confit/decisions/closed/empty-static-witness-trap-order.md",
        ),
        claim=(
            "An INNER or CROSS join on an empty static, or an INNER join on "
            "`=` keys where each static row has a NULL in at least one key, "
            "outputs no rows on either engine, so the only observable is "
            "whether the row side "
            "traps. DuckDB's hash table is empty, and DuckDB skips the row "
            "side in some pipeline shapes. The witness adds one row of plain "
            "non-NULL values to the static, which makes DuckDB run the row "
            "side, and requires DuckDB to trap too: any trap, as AGREE_TRAP "
            "compares traps. Rows, or a backend split, stay a finding."
        ),
        canaries=(
            1994509,
            2286807,
            4001288,
            4100483,
            4313391,
            4889739,
            7242549,
            7829571,
        ),
        # 6 in the 50,000 seeds 31000000..31049999 on 5e641eb, each a comma
        # join over an empty static table
        expected_per_100k=12,
        post=_empty_static,
    ),
    Rule(
        id="oracle-timeout-before-a-trap",
        ledger="divergence: oracle-timeout-before-a-trap",
        ruling="loops/confit/decisions/closed/oracle-timeout-before-a-trap.md",
        claim=(
            "Confit traps on a request before it builds a long string. The "
            "oracle evaluates one expression over all the requests at a "
            "time, so it first builds the string (lpad, rpad or repeat over "
            "a computed length, up to 2^31 - 1 characters) and runs past the "
            "time limit of the case. The witness reads the oracle again "
            "with each such length capped at 1,000: it must cap at least one "
            "length, and the oracle must then raise an error of confit's "
            "kind where both texts name one. Rows, an error of another kind "
            "or a trap on one backend only stay a TIMEOUT."
        ),
        canaries=(946454, 4226438, 4657943, 8249704, 9195711),
        expected_per_100k=1,
        timeout=_long_string,
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
            "under shape='many' the joins lower as loops in one level only (a "
            "design review condition); the unwrapped query has one level",
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


def after_timeout(case):
    """(rule id, detail) of the first rule that excuses `case` once its
    oracle reading ran past the time limit of the campaign (`fuzz.runner`),
    or None."""
    for r in RULES:
        if r.timeout is not None and active(r):
            detail = r.timeout(case)
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
        later = getattr(rec, "extended_by", ())
        if later:
            ruling += ", extended by " + " and ".join(f"`{r}`" for r in later)
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
