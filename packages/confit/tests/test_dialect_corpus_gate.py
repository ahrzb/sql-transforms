"""Dialect L2 gate: parse→print is invisible to the oracle, corpus-wide.

Law L2 of 2026-08-13-dialect-logical-plan-design.md: for every statement the
dialect frontend admits, `run_duck(sql) == run_duck(print_duck(parse_duck(sql)))`.
Each corpus case rebuilds its tables in a fresh DuckDB, hands the DESCRIBEd
catalog to the frontend, and classifies:

  match             -- printed SQL returns identical column names and the
                       identical row multiset (plan semantics are multisets,
                       design D4; scan order is not part of the contract)
  clean-unsupported -- the frontend or printer refuses by name
                       ("unsupported: ..."), including catalog types the
                       lattice boundary doesn't carry yet and FROMs that are
                       not a base table
  FAIL              -- a bind error on a statement DuckDB itself accepts, a
                       result mismatch, or a crash: the gate requires zero

The match count is the growth ladder, exactly like corpus replay: every
construct the frontend learns flips cases from clean-unsupported to match.
The floor below is the measured count at introduction — raise it when the
surface grows, never lower it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from confit import _engine
from confit.oracle import Oracle

CORPUS = Path(__file__).parent / "corpus" / "duckdb_mined.jsonl"

# Measured; general joins last raised it, 235 -> 288. A drop is a regression.
SUPPORTED_FLOOR = 288


def cases():
    with CORPUS.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(o: Oracle, sql: str):
    cur = o.execute(sql)
    names = [d[0] for d in cur.description]
    # repr keeps int/float/bool/Decimal apart and makes NaN self-equal —
    # plain == would count 1 == 1.0 == True as a match (review-confirmed
    # blind spot: a roundtrip changing result TYPES must FAIL).
    rows = sorted(tuple(repr(v) for v in row) for row in cur.fetchall())
    return names, rows


def test_dialect_corpus_gate():
    counts = {"match": 0, "clean-unsupported": 0}
    fails: list[str] = []

    for case in cases():
        with Oracle() as o:
            # Same miner limitation corpus replay handles: a skipped drop
            # directive records two CREATEs — drop and re-create.
            o.replay_setup(case["setup"])
            cat = o.catalog()
            try:
                plan_text = _engine.dialect_parse(case["sql"], cat)
                printed = _engine.dialect_print(plan_text, "duckdb", cat)
            except ValueError as e:
                if str(e).startswith("unsupported:"):
                    counts["clean-unsupported"] += 1
                else:
                    # A bind error on SQL DuckDB accepts is a frontend
                    # disagreement with the oracle — never clean.
                    fails.append(f"{case['source']}: {case['sql']!r}: {e}")
                continue
            original = run(o, case["sql"])
            reprinted = run(o, printed)
            if original == reprinted:
                counts["match"] += 1
            else:
                fails.append(
                    f"{case['source']}: {case['sql']!r} -> {printed!r}: "
                    f"{original} != {reprinted}"
                )

    total = counts["match"] + counts["clean-unsupported"] + len(fails)
    summary = (
        f"L2 gate: {counts['match']}/{total} match, "
        f"{counts['clean-unsupported']} clean-unsupported, {len(fails)} FAIL"
    )
    print(summary)
    # A warning survives `pytest -q` on green runs, so CI logs always carry
    # the measured count (the growth ladder needs it to ratchet floors).
    warnings.warn(summary, stacklevel=1)
    assert not fails, summary + "\n" + "\n".join(fails[:20])
    assert counts["match"] >= SUPPORTED_FLOOR, summary


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
