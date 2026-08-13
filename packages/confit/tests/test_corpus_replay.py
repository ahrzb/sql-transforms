"""Corpus replay: the three-outcome contract over tests/corpus/duckdb_mined.jsonl.

Each case reconstructs its tables in a fresh DuckDB, feeds them to
DuckDBInferFn (driving table = the FROM'd table, all others static), and
classifies:

  match             -- engine output equals the mined rows
  clean-unsupported -- the engine rejects at BUILD time, naming the limit:
                       an "unsupported: ..." error, a parse error (DuckDB
                       dialect beyond sqlparser, e.g. `SELECT * LIKE`,
                       `COLUMNS(...)`), or the documented v0 static-data
                       contracts (unique join keys, no NULL in a value
                       column). Cases whose FROM is a table function
                       (`range(...)`) have no base tables and can never be
                       v0 -- also clean.
  FAIL              -- mismatch, wrong error, or crash

The gate requires zero FAILs. The match count is the growth ladder: every
construct the engine learns flips cases from clean-unsupported to match
(see scripts/mine_duckdb_corpus.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pyarrow as pa
from confit import DuckDBInferFn
from pydantic import create_model

CORPUS = Path(__file__).parent / "corpus" / "duckdb_mined.jsonl"

# Build-time errors that are documented v0 contract limits, not bugs.
_CLEAN = ("unsupported:", "parse error:", "duplicate map key", "NULL in value column")

# Documented oracle divergences (clean, not FAILs). Each entry must cite a
# measured reason the divergence is IRREPRODUCIBLE row-locally.
_KNOWN_DIVERGENT_SOURCES = {
    # DuckDB's ILIKE result for a NUL-containing row DEPENDS ON SIBLING
    # ROWS: pure-ASCII column stats select a NUL-safe ASCII kernel (row
    # matches itself -> TRUE), while any non-ASCII sibling selects the
    # generic kernel whose fold NUL-truncates (same row -> FALSE); measured
    # 2026-07-26, pins-wave1/pins_like.json. Statistics-dependent semantics
    # cannot be reproduced by a row-at-a-time engine even in principle; the
    # engine is NUL-transparent (the ASCII-kernel behavior).
    "test/sql/function/string/test_ilike_embedded_null.test",
}

# Corpus spellings whose DECLARED input schema our row surface cannot
# express: pydantic `int` is width-less, so this corpus INTEGER column
# binds as int64 — and round(f64, i64) then refuses HERE exactly as
# round(DOUBLE, BIGINT) refuses on DuckDB (TASK-97, probe 2026-08-13).
# Not a divergence: an input schema we cannot declare. Narrow ROW-input
# types are an open API question (statics get widths in TASK-96).
_INEXPRESSIBLE_INPUTS = {
    (
        "test/sql/function/numeric/test_round.test",
        "select round(a, b) from roundme",
    ),
}

_FROM_RE = re.compile(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

_PY_OF_ARROW = [
    (pa.types.is_boolean, bool),
    (pa.types.is_integer, int),
    (pa.types.is_floating, float),
    (pa.types.is_string, str),
    (pa.types.is_large_string, str),
]


def _py_type(t: pa.DataType):
    for pred, py in _PY_OF_ARROW:
        if pred(t):
            return py
    return None


def _struct_model(t: pa.DataType, name: str):
    """A nested pydantic model for an arrow struct type — the engine
    flattens it to scalar lanes (TASK-56). Non-identifier field names
    can't become model fields; the caller falls back to `object` (the
    engine's clean opaque rejection)."""
    fields = {}
    for i in range(t.num_fields):
        fld = t.field(i)
        if not fld.name.isidentifier():
            return None
        if pa.types.is_struct(fld.type):
            sub = _struct_model(fld.type, f"{name}_{fld.name}")
            if sub is None:
                return None
            fields[fld.name] = (sub | None, None)
        else:
            py = _py_type(fld.type)
            fields[fld.name] = (py | None, None) if py else (object, None)
    return create_model(f"S_{name}", **fields)


def _norm_row(row) -> tuple[str, ...]:
    # repr keeps int/float/str/bool/None apart and makes NaN self-equal.
    return tuple(repr(v) for v in row)


def _replay(case: dict) -> tuple[str, str]:
    """-> (outcome, detail)."""
    con = duckdb.connect()
    for stmt in case["setup"]:
        try:
            con.execute(stmt)
        except duckdb.CatalogException as e:
            # Miner limitation: a file that drops + re-creates a table via a
            # directive the line-parser skips records both CREATEs. Replaying
            # the re-create after a drop is exactly what the file did.
            m = re.match(r'\s*CREATE\s+TABLE\s+"?([A-Za-z_]\w*)"?', stmt, re.IGNORECASE)
            if m and "already exists" in str(e):
                con.execute(f'DROP TABLE "{m.group(1)}"')
                con.execute(stmt)
            else:
                return "FAIL", f"setup failed: {stmt[:80]}: {e}"
    named = con.execute(
        "SELECT schema_name, table_name FROM duckdb_tables()"
    ).fetchall()
    if not named:
        return "unsupported", "FROM is a table function; no base tables, never v0"

    m = _FROM_RE.search(case["sql"])
    by_lower = {t.lower(): (s, t) for s, t in named}
    driving = None
    if m and m.group(1).lower() in by_lower:
        driving = by_lower[m.group(1).lower()][1]
    if driving is None:
        driving = named[0][1]

    arrow = {
        t: con.execute(f'SELECT * FROM "{s}"."{t}"').to_arrow_table() for s, t in named
    }

    # f32 base tables cannot be emulated by the f64-only engine: widening
    # is value-exact, but every f32-GRID-sensitive op (nextafter's ulp
    # steps, FLOAT->VARCHAR shortest-round-trip, FLOAT rounding) computes
    # on the wrong grid. Comparisons happen to survive; the blanket rule
    # is the defensible one (wave-3: 3 sources, 5 cases).
    for t in arrow.values():
        if any(pa.types.is_float32(f.type) for f in t.schema):
            return "unsupported", "f32 base-table column (engine is f64-only)"

    # Row model from the driving table's schema; unmappable column types are
    # the engine's own clean rejection (it sees the same field as unsupported).
    fields = {}
    for f in arrow[driving].schema:
        py = _py_type(f.type)
        if py is None:
            sub = _struct_model(f.type, f.name) if pa.types.is_struct(f.type) else None
            # object = the engine's clean opaque rejection (on REFERENCE
            # since TASK-56; unreferenced columns no longer block).
            fields[f.name] = (sub | None, None) if sub else (object, None)
        else:
            fields[f.name] = (py | None, None)
    model = create_model("Row", **fields)
    statics = {t: a for t, a in arrow.items() if t != driving}

    try:
        fn = DuckDBInferFn(
            case["sql"], row_tables={driving: model}, static_tables=statics
        )
    except Exception as e:  # noqa: BLE001 -- classification, not control flow
        msg = str(e)
        # Stage-B (TASK-59): multiplicity constructs build only under the
        # opt-in shape='many'. The retry keeps proving the DEFAULT rejects
        # while letting the corpus exercise the multiplicity path; rows
        # compare as a sorted multiset below, which is exactly the pinned
        # parity contract (DuckDB's join order is a hash-join accident).
        if "duplicate map key" in msg or "dynamic table to itself" in msg:
            try:
                fn = DuckDBInferFn(
                    case["sql"],
                    row_tables={driving: model},
                    static_tables=statics,
                    shape="many",
                )
            except Exception as e2:  # noqa: BLE001
                msg2 = str(e2)
                if any(n in msg2 for n in _CLEAN):
                    return "unsupported", msg2
                return "FAIL", f"build error under shape='many': {msg2}"
        elif any(n in msg for n in _CLEAN):
            return "unsupported", msg
        elif (case.get("source"), case["sql"]) in _INEXPRESSIBLE_INPUTS:
            return "unsupported", f"inexpressible input schema: {msg}"
        else:
            return "FAIL", f"build error: {type(e).__name__}: {msg}"

    if case.get("source") in _KNOWN_DIVERGENT_SOURCES:
        return "unsupported", "known oracle divergence (see _KNOWN_DIVERGENT_SOURCES)"
    try:
        rows_in = [model(**r) for r in arrow[driving].to_pylist()]
        got = [list(r.model_dump().values()) for r in fn.infer({driving: rows_in})]
    except Exception as e:  # noqa: BLE001
        return "FAIL", f"run error: {type(e).__name__}: {e}"

    want = case["rows"]
    if sorted(map(_norm_row, got)) != sorted(map(_norm_row, want)):
        return "FAIL", f"mismatch: got {got!r}, want {want!r}"
    return "match", ""


def test_corpus_replay_three_outcomes():
    cases = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    counts = {"match": 0, "unsupported": 0, "FAIL": 0}
    fails: list[str] = []
    for i, case in enumerate(cases):
        outcome, detail = _replay(case)
        counts[outcome] += 1
        if outcome == "FAIL":
            fails.append(f"[{i}] {case['source']}: {case['sql']}\n    {detail}")

    print(
        f"\ncorpus replay: {counts['match']} match, "
        f"{counts['unsupported']} clean-unsupported, {counts['FAIL']} FAIL "
        f"of {len(cases)}"
    )
    assert not fails, (
        f"{len(fails)} corpus FAILs "
        f"({counts['match']} match / {counts['unsupported']} unsupported):\n"
        + "\n".join(fails[:25])
    )
