"""The campaign report, driven with synthetic verdict dicts.

`fuzz.runner.report` is the only place a campaign's verdicts become the
sections a reader acts on, so what it groups, separates and writes is pinned
here directly: nothing below needs a worker, a seed or DuckDB.
"""

from __future__ import annotations

import json
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


def test_findings_open_with_a_dated_provenance_header(tmp_path, capsys):
    prov = runner.provenance(start=10, n=3)
    for key in (
        "date",
        "engine_revision",
        "generator_revision",
        "reference",
        "seeds",
        "platform",
    ):
        assert key in prov, key
    assert prov["seeds"] == [10, 12]
    assert prov["reference"]["duckdb"] == prov["reference"]["oracle_version"]
    assert "optimizer-off" in prov["reference"]["baseline"]

    out = tmp_path / "f.jsonl"
    runner.report([_r(1, "DIVERGE_VALUE", "values")], out, provenance=prov)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[0] == {"provenance": prov}
    assert [x["kind"] for x in lines[1:]] == ["DIVERGE_VALUE"]


def test_a_dead_worker_is_blamed_with_its_sql_and_inputs():
    """The worker never returned, so the parent regenerates the case: the
    finding carries SQL and inputs, not a bare seed and an empty string."""
    r = runner.blame(3, "TIMEOUT", "stderr tail")
    assert r["kind"] == "TIMEOUT" and r["klass"] == "timeout"
    assert r["sql"].startswith("SELECT")
    assert "rows" in r["inputs"]
    assert r["detail"] == "stderr tail"
