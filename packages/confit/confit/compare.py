"""THE COMPARISON VOCABULARY: what "equal" means, said once.

Every site that checks an engine answer against an oracle answer -- tests,
corpus gates, the differential fuzzer -- has to decide whether row ORDER is
part of the claim. Left to each call site that decision is made by omission: a
`sorted()` that is there or is not, and a reader has to infer the strength of
a leg from what the code forgot to do. This module names it instead.

There is exactly ONE axis a caller declares, and it is order. What makes two
VALUES the same is not an option: `repr` is the contract everywhere, because
this project's whole claim is bit-exactness, and `repr` is stricter than `==`
in precisely the places bit-exactness differs from arithmetic agreement.

Ships in the wheel next to `confit.oracle` for the same reason that one does:
the fuzzer must not import from tests/, and the tests must not import from
fuzz/, so anything both of them compare with belongs to the PACKAGE. It
imports stdlib and pyarrow only -- no pytest, no duckdb -- and raises plain
AssertionError, which is what lets a non-pytest campaign runner use it. It is
deliberately absent from `confit/__init__.py`: `import confit` stays lean.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pyarrow as pa


def dedup_names(names: list[str]) -> list[str]:
    """The wave-5 client contract (pins-wave5/dup-names-client-contract.json,
    mirrored from frontend.rs::dedup_output_names): duplicate OUTPUT names
    rename left-to-right to `<name>_N`, smallest free N, case-insensitive,
    generated candidates included. DuckDB itself applies exactly this at
    every subquery/CTE/CTAS boundary and in .df(); its TOP-LEVEL arrow
    export keeps the duplicates instead, so the oracle leg normalizes
    DuckDB's names through the same rule before comparing -- the engine's
    deduped names are the DECIDED contract, not a divergence."""
    seen: set[str] = set()
    out = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
            continue
        i = 1
        while f"{n}_{i}".lower() in seen:
            i += 1
        seen.add(f"{n}_{i}".lower())
        out.append(f"{n}_{i}")
    return out


def rows(t: pa.Table) -> list[dict]:
    """`t` as dict rows, safe against duplicate column names.

    DuckDB's top-level arrow export keeps duplicate output names, and
    `to_pylist()` builds one dict per row -- so two columns named `a` collapse
    to a single key, last one wins, and half the answer disappears with no
    error at all (measured). Renaming through `dedup_names` first is not a
    repair invented here: those are the names the engine's own client contract
    gives this query, so the read lands where the comparison is anyway.
    """
    deduped = dedup_names(t.schema.names)
    if deduped != t.schema.names:
        t = t.rename_columns(deduped)
    return t.to_pylist()


def sequence(rows) -> list:
    """Row-order-PRESERVING canonical form, for the legs where order is part
    of the contract. `multiset` below is the order-insensitive form."""
    return [sorted((k, repr(v)) for k, v in r.items()) for r in rows]


def multiset(rows) -> list:
    """Order-INSENSITIVE canonical form: the same rows in any order compare
    equal, and only the values have to match.

    `repr` is the contract on those values. That is not the same claim as
    `==`, and it is the stricter one wherever a bit-exact answer differs from
    an arithmetically equal one:

        NaN is self-equal          `==` says no, and a bit-exact answer says
                                   yes -- the same NaN came back.
        -0.0 is not 0.0            `==` says they are the same number; the
                                   bits differ and so does 1/x.
        1, 1.0 and True differ     `==` collapses all three; the output TYPE
                                   is half of what is being checked.
        Decimal('0.50') is not     `==` compares numeric value, and the SCALE
        Decimal('0.5')             is exactly what a decimal contract owes.
    """
    return sorted(sequence(rows))


def assert_rows(got, want, *, ordered: bool = False, ctx: str = "") -> None:
    """`got` equals `want`, or AssertionError saying how not.

    `ordered=False` (the default) compares multisets: for a row-path leg the
    order is an accident of the join, and demanding it would fail on answers
    that are right. `ordered=True` compares sequences, for the legs where the
    order IS the contract -- a top-level ORDER BY, or the serving promise that
    output rows follow input rows.

    A future `order_by=<key>` would sit between the two: multiset-equal AND
    `got` sorted on that key, leaving ties free, which is what a query with an
    ORDER BY over a non-unique key actually promises. Nothing needs it yet.
    """
    axis = "ordered" if ordered else "unordered"
    g = sequence(got) if ordered else multiset(got)
    w = sequence(want) if ordered else multiset(want)
    if g == w:
        return

    where = f" [{ctx}]" if ctx else ""
    lines = [
        f"assert_rows {axis} mismatch{where}: "
        f"got {_n(len(g), 'row')}, want {_n(len(w), 'row')}"
    ]
    # The canonical entries said WHETHER; the source rows say what, so pair
    # each entry back with the row it came from. Failure path only.
    gp, wp = _paired(got, ordered), _paired(want, ordered)
    diffs = [i for i in range(max(len(gp), len(wp))) if _key(gp, i) != _key(wp, i)]
    for i in diffs[:_MAX_ROWS]:
        grow, wrow = _row(gp, i), _row(wp, i)
        lines.append(f"  row {i}:")
        lines.append(f"    got  {grow!r}")
        lines.append(f"    want {wrow!r}")
        lines.extend(_value_diffs(grow, wrow))
    if len(diffs) > _MAX_ROWS:
        lines.append(f"  ... {len(diffs) - _MAX_ROWS} more differing rows")
    raise AssertionError("\n".join(lines))


def assert_schema(got: pa.Schema, want: pa.Schema, *, ctx: str = "") -> None:
    """`got` equals `want`, or AssertionError naming the FIRST field that
    differs and the attribute that differs on it.

    One field and one attribute, never two whole schema dumps: a wide schema
    printed twice makes the reader diff it by eye, which is the work this is
    supposed to have already done.
    """
    if got.equals(want):
        return
    where = f" [{ctx}]" if ctx else ""
    for i in range(min(len(got), len(want))):
        gf, wf = got.field(i), want.field(i)
        label = f"field {i}" + (f" '{gf.name}'" if gf.name == wf.name else "")
        for attr in ("name", "type", "nullable"):
            a, b = getattr(gf, attr), getattr(wf, attr)
            if a != b:
                raise AssertionError(
                    f"assert_schema mismatch{where}: {label} differs on "
                    f"{attr}: got {a}, want {b}"
                )
    raise AssertionError(
        f"assert_schema mismatch{where}: got {_n(len(got), 'field')}, "
        f"want {_n(len(want), 'field')}"
    )


# --------------------------------------------------------------- internals

_MAX_ROWS = 20


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def _n(k: int, noun: str) -> str:
    return f"{k} {noun}" if k == 1 else f"{k} {noun}s"


def _paired(rows_, ordered: bool):
    """(canonical entry, source row) in the order the comparison saw them.

    Sorting the PAIRS on the entry reproduces `multiset` exactly -- it is the
    same list of keys, sorted the same way -- while keeping each source row
    reachable for the message.
    """
    pairs = list(zip(sequence(rows_), rows_, strict=True))
    return pairs if ordered else sorted(pairs, key=lambda p: p[0])


def _key(pairs, i):
    return pairs[i][0] if i < len(pairs) else _MISSING


def _row(pairs, i):
    return pairs[i][1] if i < len(pairs) else _MISSING


def _value_diffs(grow, wrow) -> list[str]:
    """One line per key whose reprs differ, each naming the contract property
    that separated the two values when there is one to name."""
    if not isinstance(grow, dict) or not isinstance(wrow, dict):
        return []
    out = []
    for k in sorted(set(grow) | set(wrow), key=str):
        a, b = grow.get(k, _MISSING), wrow.get(k, _MISSING)
        if repr(a) == repr(b):
            continue
        prop = _caught_by(a, b)
        out.append(f"    {k}: {a!r} != {b!r}" + (f"   [{prop}]" if prop else ""))
    return out


def _caught_by(a, b) -> str | None:
    """Which property of the repr contract separated `a` from `b`, or None
    when `==` would have separated them too and there is nothing to explain.
    """
    if _isnan(a) and _isnan(b):
        return "NaN"  # both NaN, spelled differently -- `==` calls these unequal
    if not _eq(a, b):
        return None
    if _eq(a, 0) and math.copysign(1.0, a) != math.copysign(1.0, b):
        return "signed zero"
    if isinstance(a, Decimal) and isinstance(b, Decimal):
        return "Decimal scale"
    return f"type {type(a).__name__} vs {type(b).__name__}"


def _eq(a, b) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 -- a value that will not compare is not equal
        return False


def _isnan(v) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError, OverflowError):
        return False
