"""The campaign report, driven with synthetic verdict dicts.

`fuzz.runner.report` is the only place a campaign's verdicts become the
sections a reader acts on, so what it groups, separates and writes is pinned
here directly: nothing below needs a worker, a seed or DuckDB.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import runner  # noqa: E402


def _r(seed, kind, klass="", oracle="", detail="", tags=()):
    out = {
        "seed": seed,
        "sql": f"SELECT {seed}",
        "kind": kind,
        "klass": klass,
        "detail": detail,
        "tags": list(tags),
    }
    if oracle:
        out["oracle"] = oracle
    return out


def _section(text: str, title: str) -> str:
    """The lines under `== title ... ==` up to the next section header."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"== {title}"))
    body = []
    for ln in lines[start + 1 :]:
        if ln.startswith("=="):
            break
        body.append(ln)
    return "\n".join(body).strip("\n")


def test_refusals_are_summarized_by_oracle_outcome_and_class(tmp_path, capsys):
    results = [
        _r(1, "REFUSED", "unsupported: lpad", "serves"),
        _r(2, "REFUSED", "unsupported: lpad", "serves"),
        _r(3, "REFUSED", "unsupported: lpad", "traps"),
        _r(4, "REFUSED", "bind error: no column", "rejects"),
    ]
    runner.report(results, tmp_path / "f.jsonl")
    sec = _section(capsys.readouterr().out, "refusals by oracle outcome")
    assert sec.splitlines() == [
        f"  {'serves':14} 2",
        f"    {2:6}  unsupported: lpad",
        f"  {'traps':14} 1",
        f"    {1:6}  unsupported: lpad",
        f"  {'rejects':14} 1",
        f"    {1:6}  bind error: no column",
    ]
    # A refusal is reporting, not a finding: nothing is written for it.
    assert (tmp_path / "f.jsonl").read_text() == ""
